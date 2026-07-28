"""Dora node: Franka Hand — gripper-ONLY bridge for the RT-motion graph.

The RT machine owns the arm's one FCI session; a second ``franka.Robot``
from this host would fight it. The HAND is its own TCP service on the
control box and coexists with the RT servo — this node constructs ONLY
``fr.Gripper``. (``franka_interface`` still owns the Hand in the listen
graph, where it also owns the robot connection; never run both graphs.)

Input ``gripper``: motor_command_gripper format (finger metres, e.g. the
teleop slider). Latest-wins with a 2 mm width deadband; a worker thread
issues the BLOCKING move() calls so the node loop never stalls. Output
``gripper_state``: width + is_grasped at ~5 Hz (reads are skipped while a
move is in flight — read_once contends with a blocking motion).

Deliberately NOT gated on the arm: opening the jaws with a disarmed arm is
a legitimate bench act, and the slider is already an explicit human action
(grasp POLICY stays with the orchestrator's grasp_request path).
"""
from __future__ import annotations

# ruff: noqa: E402

import threading
import time

from dora import Node

from arm_control.config import load_robot_config
from arm_control.hardware.franka_backend import (
    FrankaBackendUnavailableError,
    FrankaConfig,
    _import_pylibfranka,
)
from arm_control.messages import pack_json_message, unpack_motor_command

DEADBAND_M = 0.002  # commanded-width change below this is slider noise
MOVE_SPEED = 0.08   # m/s — gentle; the Hand's max is 0.2


def main() -> None:
    cfg = load_robot_config()
    ip = FrankaConfig.from_config(cfg).ip
    try:
        fr = _import_pylibfranka()
        gripper = fr.Gripper(ip)
    except (FrankaBackendUnavailableError, Exception) as exc:
        raise SystemExit(f"[franka_gripper] no Hand at {ip}: {exc}") from exc

    target_w: list[float | None] = [None]  # latest-wins slot
    wake = threading.Event()
    busy = threading.Event()
    stop = threading.Event()

    def worker() -> None:
        last_cmd: float | None = None
        while not stop.is_set():
            wake.wait(0.1)
            wake.clear()
            w = target_w[0]
            if w is None or (last_cmd is not None and abs(w - last_cmd) < DEADBAND_M):
                continue
            last_cmd = w
            busy.set()
            try:
                gripper.move(w, MOVE_SPEED)  # blocking, seconds
            except Exception as exc:
                print(f"[franka_gripper] move failed: {exc}", flush=True)
            finally:
                busy.clear()

    threading.Thread(target=worker, daemon=True).start()
    node = Node()
    print(f"[franka_gripper] Hand at {ip} ready — finger slider drives width", flush=True)
    last_pub = 0.0
    while True:
        event = node.next(timeout=0.1)
        if event is not None:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT" and event["id"] == "gripper":
                finger_m = float(unpack_motor_command(event["value"], 2)["position"][0])
                target_w[0] = 2.0 * finger_m  # width = both fingers
                wake.set()
        now = time.monotonic()
        if now - last_pub >= 0.2 and not busy.is_set():
            last_pub = now
            try:
                st = gripper.read_once()
            except Exception:
                continue
            node.send_output(
                "gripper_state",
                pack_json_message(
                    "gripper_state",
                    {"width": float(st.width), "is_grasped": bool(st.is_grasped)},
                ),
            )
    stop.set()


if __name__ == "__main__":
    main()
