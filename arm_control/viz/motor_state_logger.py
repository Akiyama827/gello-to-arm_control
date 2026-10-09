"""Dora node: append raw motor state samples to CSV."""
from __future__ import annotations

# ruff: noqa: E402

import csv
import time
from pathlib import Path

from dora import Node


import numpy as np

from arm_control.config import load_robot_config_dict
from arm_control.messages import unpack_motor_command, unpack_motor_state
from arm_control.node_utils import ShutdownFlag, install_signal_handlers


def _load_cfg() -> dict:
    return load_robot_config_dict()


#: Per-motor columns, in order. ``qdes`` is what the PC SENT (off
#: motor_command); ``pos_cmd``/``vel_cmd``/``tor_cmd`` are what the plant
#: actually SERVOED on the tick it reported. Keeping both is the point: the
#: difference between qdes and pos_cmd is the transport (a 100 Hz command
#: zero-order-held for ten 1 kHz ticks), and tor_cmd against torque is where
#: the clamp and the slew limiter show up. A post-mortem that has only
#: measured position cannot separate "the reference was wrong" from "the
#: tracking was wrong" -- which is exactly where the 2026-09-09 review ran out
#: of evidence.
_COLUMNS = ("pos", "vel", "torque", "qdes", "pos_cmd", "vel_cmd", "tor_cmd")


def _row(
    elapsed_s: float, motor_names: list[str], state: dict, q_des: np.ndarray
) -> dict[str, float]:
    row = {"elapsed_s": elapsed_s}
    for i, name in enumerate(motor_names):
        row[f"{name}_pos"] = float(state["position"][i])
        row[f"{name}_vel"] = float(state["velocity"][i])
        row[f"{name}_torque"] = float(state["torque"][i])
        row[f"{name}_qdes"] = float(q_des[i])  # nan until the first command
        row[f"{name}_pos_cmd"] = float(state["position_cmd"][i])
        row[f"{name}_vel_cmd"] = float(state["velocity_cmd"][i])
        row[f"{name}_tor_cmd"] = float(state["torque_cmd"][i])
    return row


def main(cfg: dict | None = None) -> None:
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    cfg = _load_cfg() if cfg is None else cfg
    n = int(cfg.get("num_motors", 7))
    motor_names = list(cfg.get("motor_names") or [f"motor_{i}" for i in range(n)])
    base = Path(str(cfg.get("motor_log_path") or "/tmp/arm_control_motor_state.csv"))
    # One file PER RUN: appending forever grows ~200 MB/hour at 100 Hz with
    # no ceiling — days-to-disk-full on an unattended box, and every past
    # session's rows pollute the current analysis anyway.
    path = base.with_name(
        f"{base.stem}_{time.strftime('%Y%m%d_%H%M%S')}{base.suffix}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["elapsed_s"] + [
        f"{name}_{suffix}" for name in motor_names for suffix in _COLUMNS
    ]

    node = Node()
    start = time.monotonic()
    last_flush = start
    q_des = np.full(n, np.nan)  # latest commanded target (graphs without one: nan)
    print(f"[motor_state_logger] logging to {path}", flush=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while not shutdown.stop_requested:
            event = node.next(timeout=0.05)
            if shutdown.stop_requested:
                break
            if event is None:
                continue
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            if event["id"] == "motor_command":
                q_des = unpack_motor_command(event["value"], n)["position"]
            elif event["id"] == "motor_state":
                state = unpack_motor_state(event["value"], n)
                writer.writerow(_row(time.monotonic() - start, motor_names, state, q_des))
                now = time.monotonic()
                if now - last_flush >= 1.0:  # not per-row: 100 syscalls/s
                    last_flush = now
                    f.flush()


#: trace/1 field per CSV column (``_COLUMNS`` order): the same measured/commanded split, from the supervisor's controller ticks
TRACE_FIELDS = dict(pos="q_rt", vel="dq_rt", torque="tau_meas_rt", qdes="q_cmd_pc", pos_cmd="q_cmd_rt", vel_cmd="dq_cmd_rt",
                    tor_cmd="tau_cmd_rt")
#: how often the supervisor's trace ring (TRACE_TICKS = 0.64 s at 100 Hz) is read: well inside it
TRACE_POLL_S = 0.1


def trace_row(elapsed_s: float, motor_names: list[str], sample: dict) -> dict[str, float]:
    """One controller tick of trace/1 as a CSV row with ``_row``'s columns; a field the tick lacks (no RT packet yet, no
    command sent, a stale RT stream) is nan, never zero or a repeat. ``qdes`` is the command computed on THIS tick (the Dora
    logger paired a state with the previous tick's command: ``qdes - pos`` now differs by about v*dt)."""
    row = {"elapsed_s": elapsed_s}
    stale = not (sample.get("health") or {}).get("state_fresh", True)
    for suffix, field in TRACE_FIELDS.items():
        # a tick that consumed no fresh RT packet re-reports the last one: its RT columns are nan, not a repeated measurement
        values = None if stale and field.endswith("_rt") else sample.get(field)
        for i, name in enumerate(motor_names):
            row[f"{name}_{suffix}"] = float("nan") if values is None else float(values[i])
    return row


def trace_main(cfg: dict | None = None) -> None:
    """The same CSV from a supervised graph, where no ``motor_command`` topic exists: the supervisor's trace/1 holds every
    controller tick's PC command and the RT packet it consumed. One row per tick; ticks that fell off the ring before a read are
    counted and reported (``dropped``), never invented. ``ARM_CONTROL_CONFIG`` names the arm (``rt.instance_id``),
    ``ARM_CONTROL_SOCKET`` its supervisor."""
    import os

    from arm_control.sdk import ArmClient, SupervisorError

    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    cfg = _load_cfg() if cfg is None else cfg
    arm = ArmClient(str((cfg.get("rt") or {}).get("instance_id", "")), socket_path=os.environ.get("ARM_CONTROL_SOCKET") or None)
    motor_names = list(arm.identity.joint_names)
    base = Path(str(cfg.get("motor_log_path") or "/tmp/arm_control_motor_state.csv"))
    path = base.with_name(f"{base.stem}_{time.strftime('%Y%m%d_%H%M%S')}{base.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["elapsed_s"] + [f"{name}_{suffix}" for name in motor_names for suffix in _COLUMNS]
    node = Node()
    cursor, start, dropped, errors, last_read = 0, None, 0, 0, 0.0
    print(f"[motor_state_logger] logging the supervisor's trace/1 to {path}", flush=True)
    try:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            def read() -> None:
                nonlocal cursor, start, dropped, errors
                try:
                    batch = arm.trace_samples(cursor)
                except (SupervisorError, TimeoutError, OSError) as exc:  # counted; the next read tries again
                    errors += 1
                    print(f"[motor_state_logger] trace read failed ({errors}): {exc}", flush=True)
                    return
                dropped += batch["dropped"]
                for sample in batch["samples"]:
                    start = sample["pc_monotonic"] if start is None else start
                    writer.writerow(trace_row(sample["pc_monotonic"] - start, motor_names, sample))
                cursor = batch["latest_seq"]
                f.flush()

            while not shutdown.stop_requested:
                event = node.next(timeout=0.05)
                if shutdown.stop_requested or (event is not None and event["type"] == "STOP"):
                    break
                if time.monotonic() - last_read >= TRACE_POLL_S:
                    last_read = time.monotonic()
                    read()
            read()  # the last ticks (the DISARM) before the supervisor stops
    finally:
        arm.close()
        print(f"[motor_state_logger] stopped ({dropped} controller ticks fell off the trace ring between reads)", flush=True)


def cli() -> None:
    main()


def trace_cli() -> None:
    trace_main()


def _self_check() -> None:
    """trace_row keeps the Dora logger's columns, maps each to its trace/1 field, and writes nan (never 0) for a missing one."""
    sample = dict(q_rt=[1., 2.], dq_rt=[.1, .2], tau_meas_rt=[3., 4.], q_cmd_pc=[1.5, 2.5], q_cmd_rt=None,
                  dq_cmd_rt=[0., 0.], tau_cmd_rt=[5., 6.])
    row = trace_row(.25, ["a", "b"], sample)
    assert set(row) == {"elapsed_s"} | {f"{n}_{c}" for n in "ab" for c in _COLUMNS}, row.keys()
    assert row["a_pos"] == 1. and row["b_qdes"] == 2.5 and row["b_tor_cmd"] == 6. and np.isnan(row["a_pos_cmd"]), row
    stale = trace_row(.26, ["a", "b"], dict(sample, health=dict(state_fresh=False)))
    assert np.isnan(stale["a_pos"]) and np.isnan(stale["b_tor_cmd"]) and stale["a_qdes"] == 1.5, stale  # the PC's own still holds
    print("motor_state_logger: trace/1 rows keep the CSV columns (qdes = the PC's command, pos_cmd = the RT's), nan when absent")


if __name__ == "__main__":
    import sys

    _self_check() if sys.argv[1:] == ["--self-check"] else cli()
