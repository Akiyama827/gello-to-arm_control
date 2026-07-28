"""Dora node: drag-teach replay driver.

Turns a hand-guided recording (the CSV ``rt_handguide`` writes while the arm
floats) into an executable trajectory and feeds it to the trajectory
executor: resample -> smooth -> trim motionless head/tail -> uniform
time-scale under a velocity cap (times ``slow_factor``) -> prepend a min-jerk
JOIN from the arm's measured pose to the recording start (so the executor's
start-pose gate always passes and there is never a first-sample jump).

Triggers (file-based, like the operator gate — dora nodes have no stdin):

    echo float.csv > /tmp/arm_replay_go    # replay this recording
    touch /tmp/arm_replay_stop             # stop -> executor holds in place

An empty trigger file replays ``ARM_REPLAY_CSV`` from the environment. The
node never sends motor commands itself — everything goes through the
executor's acceptance gate, runaway abort, and the plant bridge's arm gate.
"""
from __future__ import annotations

# ruff: noqa: E402

import csv
import os
import time
from pathlib import Path

import numpy as np
from dora import Node

from arm_control.config import arm_joints, load_robot_config
from arm_control.messages import (
    pack_trajectory,
    unpack_json_message,
    unpack_motor_state,
)
from arm_control.node_utils import _load_mode_config

# Trigger files live in the user-private runtime dir (mode 0700), NOT /tmp:
# a world-writable trigger is motion authority for ANY local process/user,
# and a foreign pre-created /tmp file even crashes the unlink at startup.
_RUN_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
GO_FILE = _RUN_DIR / "arm_replay_go"
STOP_FILE = _RUN_DIR / "arm_replay_stop"
MAX_CSV_BYTES = 20 * 1024 * 1024  # a recording is minutes of 50 Hz rows, not GB


def load_recording(path: Path, n_arm: int) -> tuple[np.ndarray, np.ndarray]:
    """(t, q) from an rt_handguide CSV; columns matched by header name."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    t_i = header.index("t_mono")
    q_i = [header.index(f"q{j}") for j in range(n_arm)]
    t = np.array([float(r[t_i]) for r in data])
    q = np.array([[float(r[i]) for i in q_i] for r in data])
    keep = np.concatenate([[True], np.diff(t) > 0])  # drop duplicate stamps
    return t[keep] - t[keep][0], q[keep]


def build_replay(
    t: np.ndarray, q: np.ndarray, q_now: np.ndarray, rp: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """(times, positions, velocities, summary) — join + retimed recording."""
    hz = float(rp.get("resample_hz", 50))
    dt = 1.0 / hz
    tu = np.arange(0.0, t[-1], dt)
    qu = np.stack([np.interp(tu, t, q[:, j]) for j in range(q.shape[1])], axis=1)

    win = max(1, int(float(rp.get("smooth_window_s", 0.25)) * hz))
    kernel = np.ones(win) / win
    qs = np.stack(
        [np.convolve(qu[:, j], kernel, mode="same") for j in range(qu.shape[1])],
        axis=1,
    )
    qs[: win // 2] = qs[win // 2]          # convolve edge artefacts: clamp,
    qs[-(win // 2) or len(qs):] = qs[-(win // 2 + 1)]  # not taper toward 0

    speed = np.max(np.abs(np.gradient(qs, dt, axis=0)), axis=1)
    still = float(rp.get("still_speed", 0.02))
    moving = np.flatnonzero(speed > still)
    if len(moving) < int(0.5 * hz):
        raise ValueError("recording has <0.5 s of motion after trimming")
    qs = qs[moving[0] : moving[-1] + 1]

    v_raw = np.max(np.abs(np.gradient(qs, dt, axis=0)))
    cap = float(rp.get("vel_cap", 0.4))
    scale = max(1.0, v_raw / cap) * float(rp.get("slow_factor", 1.5))
    times = np.arange(len(qs)) * dt * scale
    vel = np.gradient(qs, times, axis=0)
    vel[0] = vel[-1] = 0.0

    # Min-jerk join from the measured pose to the recording start.
    dist = float(np.max(np.abs(q_now - qs[0])))
    t_join = max(float(rp.get("join_time_min_s", 2.5)), dist / 0.2)
    tj = np.arange(0.0, t_join, dt)
    s = 10 * (tj / t_join) ** 3 - 15 * (tj / t_join) ** 4 + 6 * (tj / t_join) ** 5
    sd = (30 * (tj / t_join) ** 2 - 60 * (tj / t_join) ** 3 + 30 * (tj / t_join) ** 4) / t_join
    qj = q_now[None, :] + s[:, None] * (qs[0] - q_now)[None, :]
    vj = sd[:, None] * (qs[0] - q_now)[None, :]

    out_t = np.concatenate([tj, times + t_join])
    out_q = np.concatenate([qj, qs])
    out_v = np.concatenate([vj, vel])
    summary = (
        f"join {dist:.3f} rad over {t_join:.1f}s + replay {times[-1]:.1f}s "
        f"({len(qs)} samples, time-scale x{scale:.2f}, "
        f"max vel {np.max(np.abs(out_v)):.2f} rad/s)"
    )
    return out_t, out_q, out_v, summary


def main() -> None:
    cfg = load_robot_config()
    rp = dict(_load_mode_config().get("replay") or {})
    n_arm = len(arm_joints(cfg))
    n = cfg.num_motors
    for f in (GO_FILE, STOP_FILE):
        f.unlink(missing_ok=True)  # stale triggers from a previous session

    node = Node()
    q_now: np.ndarray | None = None
    armed: bool | None = None  # None = no motor_health wired (sim graphs)
    faulted = False
    print(
        f"[trajectory_replay] ready — echo <recording.csv> > {GO_FILE} to replay, "
        f"touch {STOP_FILE} to stop",
        flush=True,
    )
    while True:
        event = node.next(timeout=0.2)
        if event is not None:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT" and event["id"] == "motor_state":
                q_now = unpack_motor_state(event["value"], n)["position"][:n_arm]
            elif event["type"] == "INPUT" and event["id"] == "motor_health":
                health = unpack_json_message(event["value"])
                armed = bool(health.get("armed", False))
                faulted = bool(health.get("any_fault", False))

        if STOP_FILE.exists():
            STOP_FILE.unlink(missing_ok=True)
            node.send_output(
                "trajectory",
                pack_trajectory(np.zeros(0), np.zeros((0, n_arm)), np.zeros((0, n_arm))),
            )
            print("[trajectory_replay] STOP sent — executor holds in place", flush=True)
        if not GO_FILE.exists():
            continue
        try:
            raw = GO_FILE.read_text().strip()
            GO_FILE.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[trajectory_replay] trigger unreadable ({exc})", flush=True)
            time.sleep(1.0)
            continue
        csv_path = Path(raw or os.environ.get("ARM_REPLAY_CSV", ""))
        if not csv_path.is_file():
            print(f"[trajectory_replay] REFUSED: no recording at '{csv_path}'", flush=True)
            continue
        if csv_path.stat().st_size > MAX_CSV_BYTES:
            print(
                f"[trajectory_replay] REFUSED: {csv_path.name} is "
                f"{csv_path.stat().st_size >> 20} MB — not a handguide recording",
                flush=True,
            )
            continue
        # This trigger is a motion-authority path with no page in front of it:
        # refuse anything the operator gate would refuse. (armed None = graph
        # wires no health — sim — where the plant applies its own gate.)
        if armed is False or faulted:
            print(
                "[trajectory_replay] REFUSED: "
                + ("server fault latched" if faulted else "DISARMED")
                + " — ARM on the teleop page first",
                flush=True,
            )
            continue
        if q_now is None:
            print("[trajectory_replay] REFUSED: no motor state yet", flush=True)
            continue
        try:
            t, q = load_recording(csv_path, n_arm)
            times, qs, vs, summary = build_replay(t, q, q_now, rp)
        except (ValueError, IndexError) as exc:
            print(f"[trajectory_replay] REFUSED: {exc}", flush=True)
            continue
        node.send_output("trajectory", pack_trajectory(times, qs, vs))
        print(f"[trajectory_replay] sent {csv_path.name}: {summary}", flush=True)
        time.sleep(0.2)  # debounce editor double-writes of the trigger file


if __name__ == "__main__":
    main()
