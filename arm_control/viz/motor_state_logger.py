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


def _row(
    elapsed_s: float, motor_names: list[str], state: dict, q_des: np.ndarray
) -> dict[str, float]:
    row = {"elapsed_s": elapsed_s}
    for i, name in enumerate(motor_names):
        row[f"{name}_pos"] = float(state["position"][i])
        row[f"{name}_vel"] = float(state["velocity"][i])
        row[f"{name}_torque"] = float(state["torque"][i])
        row[f"{name}_qdes"] = float(q_des[i])  # nan until the first command
    return row


def main() -> None:
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    cfg = _load_cfg()
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
        f"{name}_{suffix}"
        for name in motor_names
        for suffix in ("pos", "vel", "torque", "qdes")
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


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
