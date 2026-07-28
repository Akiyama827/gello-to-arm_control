"""Dora node: Franka Hand — gripper-ONLY bridge for the RT-motion graph.

The RT machine owns the arm's one FCI session; a second ``franka.Robot``
from this host would fight it. The HAND is its own TCP service on the
control box and coexists with the RT servo — this node constructs ONLY
``fr.Gripper``. (``franka_interface`` still owns the Hand in the listen
graph, where it also owns the robot connection; never run both graphs.)

Threading contract, learned the hard way: libfranka's Gripper is ONE TCP
request/response session and is NOT thread-safe — a read_once() from one
thread interleaved with a move() from another corrupts the protocol, the
Hand resets the socket, and every later call fails with "TCP send bytes:
I/O error". So exactly ONE thread here owns the Gripper object and does
everything (connect, reconnect, move, read); the dora loop only swaps
targets in and copies state out. The owner reconnects with backoff on any
I/O error — an appliance node must self-heal, not wedge.

Input ``gripper``: motor_command_gripper format (finger metres, e.g. the
teleop slider), latest-wins with a 2 mm width deadband. Output
``gripper_state``: width + is_grasped at ~5 Hz.

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
    FrankaConfig,
    _import_pylibfranka,
)
from arm_control.messages import pack_json_message, unpack_motor_command

DEADBAND_M = 0.002    # commanded-width change below this is slider noise
MOVE_SPEED = 0.08     # m/s — gentle; the Hand's max is 0.2
READ_PERIOD = 0.2     # state poll when idle
RECONNECT_S = 2.0     # backoff after an I/O error


class HandOwner(threading.Thread):
    """The one thread that talks TCP to the Hand. Everything else is memory."""

    def __init__(self, ip: str) -> None:
        super().__init__(daemon=True)
        self.ip = ip
        self.target: float | None = None  # desired width; GIL-atomic swap
        self.state: dict | None = None    # latest {width, is_grasped}
        self.stop_flag = threading.Event()

    def run(self) -> None:
        fr = _import_pylibfranka()
        gripper = None
        last_cmd: float | None = None
        last_read = 0.0
        warned = False
        while not self.stop_flag.is_set():
            if gripper is None:
                try:
                    gripper = fr.Gripper(self.ip)
                    print(f"[franka_gripper] Hand connected at {self.ip}", flush=True)
                    warned = False
                except Exception as exc:
                    if not warned:
                        warned = True
                        print(
                            f"[franka_gripper] no Hand at {self.ip}: {exc} — "
                            f"retrying every {RECONNECT_S:.0f}s",
                            flush=True,
                        )
                    self.stop_flag.wait(RECONNECT_S)
                    continue
            w = self.target
            try:
                if w is not None and (last_cmd is None or abs(w - last_cmd) >= DEADBAND_M):
                    last_cmd = w
                    gripper.move(w, MOVE_SPEED)  # blocking, seconds
                elif time.monotonic() - last_read >= READ_PERIOD:
                    last_read = time.monotonic()
                    st = gripper.read_once()
                    self.state = {
                        "width": float(st.width),
                        "is_grasped": bool(st.is_grasped),
                    }
                else:
                    self.stop_flag.wait(0.05)
            except Exception as exc:
                print(
                    f"[franka_gripper] Hand I/O error ({exc}) — reconnecting",
                    flush=True,
                )
                gripper = None       # session is poisoned after any I/O error
                last_cmd = None      # retry the current target after reconnect
                self.stop_flag.wait(RECONNECT_S)


def main() -> None:
    cfg = load_robot_config()
    owner = HandOwner(FrankaConfig.from_config(cfg).ip)
    owner.start()
    node = Node()
    print("[franka_gripper] up — finger slider drives width", flush=True)
    last_pub = 0.0
    while True:
        event = node.next(timeout=0.1)
        if event is not None:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT" and event["id"] == "gripper":
                finger_m = float(unpack_motor_command(event["value"], 2)["position"][0])
                owner.target = 2.0 * finger_m  # width = both fingers
        now = time.monotonic()
        state = owner.state
        if now - last_pub >= READ_PERIOD and state is not None:
            last_pub = now
            node.send_output(
                "gripper_state", pack_json_message("gripper_state", state)
            )
    owner.stop_flag.set()


if __name__ == "__main__":
    main()
