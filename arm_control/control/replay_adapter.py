"""Dora node: drag-teach replay driver.

Turns a hand-guided recording (the CSV ``rt_handguide`` writes while the arm
floats) into an executable trajectory and feeds it to the trajectory
executor: resample -> smooth -> trim motionless head/tail -> uniform
time-scale under a velocity cap (times ``slow_factor``) -> prepend a min-jerk
JOIN from the arm's measured pose to the recording start. The complete cubic
is then time-scaled to the velocity and acceleration caps, including the join.

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
import uuid
from pathlib import Path

import numpy as np
from dora import Node

from arm_control.config import arm_joints, load_robot_config
from arm_control.messages import (
    pack_control_update,
    pack_plan,
    unpack_json_message,
    unpack_motor_state,
)
from arm_control.node_utils import (
    ShutdownFlag,
    _load_mode_config,
    install_signal_handlers,
    resolve_gains,
)

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
    t: np.ndarray, q: np.ndarray, q_now: np.ndarray, rp: dict, *, max_acc
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
    from arm_control.motion import JointTrajectory
    from arm_control.planning.retiming import bound_trajectory

    amax = np.broadcast_to(np.asarray(max_acc, dtype=float), (q.shape[1],))
    if not np.isfinite(amax).all() or np.any(amax <= 0):
        raise ValueError("replay requires finite positive acceleration caps")
    curve = bound_trajectory(JointTrajectory(out_t, out_q, out_v),
                             np.full(q.shape[1], cap), amax)
    stretch = curve.duration_sec / out_t[-1]
    summary = (
        f"join {dist:.3f} rad over {t_join * stretch:.1f}s + replay {times[-1] * stretch:.1f}s "
        f"({len(qs)} samples, time-scale x{scale:.2f}, "
        f"curve stretch x{stretch:.2f}, "
        f"max vel {np.max(curve.bounds()['qd_abs_max']):.2f} rad/s)"
    )
    return curve.times, curve.positions, curve.velocities, summary


def _replay_acceleration_limits(cfg, mode_cfg):
    """Where replay's acceleration cap comes from, resolved in explicit order.

    This was one nested `.get()` with `mode_cfg['planner']['acc_limits']` as the
    default. Python evaluates that default BEFORE calling `.get()`, so a
    deployment carrying `execution_policy.acceleration_limits` and no mode
    profile raised KeyError on a fallback it never needed -- and
    `_load_mode_config()` legitimately returns {} when none is configured. A
    reusable package must not demand a mode-profile file to read a limit its
    caller already supplied.

    KEY PRESENCE, not truthiness: the same rule as the torque limits. A missing
    key may fall back; a key that is PRESENT and malformed is an operator error
    and must not silently resolve to some other number.
    """
    execution = cfg.get("execution_policy") or {}
    if "acceleration_limits" in execution:
        return execution["acceleration_limits"]
    planner = mode_cfg.get("planner") or {}
    if "acc_limits" in planner:
        return planner["acc_limits"]
    raise ValueError(
        "replay requires execution_policy.acceleration_limits (robot config) "
        "or planner.acc_limits (mode profile); neither is configured"
    )


def main() -> None:
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    cfg = load_robot_config()
    mode_cfg = _load_mode_config()
    rp = dict(mode_cfg.get("replay") or {})
    max_acc = _replay_acceleration_limits(cfg, mode_cfg)
    n_arm = len(arm_joints(cfg))
    # Gains travel WITH the plan (pack_plan carries kp/kd), resolved by the same
    # helper the controller's own node uses -- one number for one arm.
    _gains = resolve_gains(cfg, mode_cfg, list(cfg.joint_names or cfg.motor_names), n_arm)
    plan_kp, plan_kd = _gains["kp"][:n_arm], _gains["kd"][:n_arm]
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
    while not shutdown.stop_requested:
        event = node.next(timeout=0.2)
        if shutdown.stop_requested:
            break
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
                "control",
                pack_control_update(cancel=True, reason="replay stop file"),
            )
            print("[trajectory_replay] STOP sent — controller holds in place", flush=True)
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
            times, qs, vs, summary = build_replay(t, q, q_now, rp, max_acc=max_acc)
        except (ValueError, IndexError) as exc:
            print(f"[trajectory_replay] REFUSED: {exc}", flush=True)
            continue
        if shutdown.stop_requested:
            break
        # gated=False: the trigger file IS the operator's press. A gated plan
        # would wait for an `execute` that nothing here ever sends.
        node.send_output(
            "plan",
            pack_plan(
                plan_id=f"replay-{uuid.uuid4().hex[:8]}",
                phase="replay",
                gated=False,
                times=times,
                positions=qs,
                velocities=vs,
                kp=plan_kp,
                kd=plan_kd,
            ),
        )
        print(f"[trajectory_replay] sent {csv_path.name}: {summary}", flush=True)
        time.sleep(0.2)  # debounce editor double-writes of the trigger file


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
