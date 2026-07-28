"""Dora node: Franka Hand client — talks to hand_bridge on the RT box.

Driving the Hand with pylibfranka FROM THE PC is impossible since the
network migration: the Hand mirrors the robot's protocol split (commands
over TCP, cyclic state PUSHED over UDP), and server-push UDP cannot cross
the PC-side NAT — moves worked, every read timed out (bench 2026-07-28).
Same physics that put the torque loop on the RT box, same fix: the Hand is
owned by ``rt/src/hand_bridge.cpp`` on the box (robot LAN, no NAT), and
this node speaks its dumb TCP line protocol over the direct link:

    -> "MOVE <width_m> <speed_mps>"  |  "HOME"  |  "GSTOP"
    <-  "STATE <width_m> <0|1>"      (~10 Hz, streams DURING moves too)

Input ``gripper``: motor_command_gripper format (finger metres, e.g. the
teleop slider), latest-wins with a 2 mm width deadband. Output
``gripper_state``: width + is_grasped. ``touch /tmp/arm_gripper_home``
requests a homing cycle (physical open-close — needed once per Hand
power-cycle; deliberately never automatic).

Deliberately NOT gated on the arm: opening the jaws with a disarmed arm is
a legitimate bench act, and the slider is already an explicit human action
(grasp POLICY stays with the orchestrator's grasp_request path).
"""
from __future__ import annotations

# ruff: noqa: E402

import socket
import threading
import time
from pathlib import Path

from dora import Node

from arm_control.config import load_robot_config
from arm_control.messages import pack_json_message, unpack_motor_command

DEADBAND_M = 0.002  # commanded-width change below this is slider noise
MOVE_SPEED = 0.10   # m/s — brisk but gentle; the Hand's max is 0.2
SETTLE_S = 0.15     # slider must rest this long before a goal is sent —
                    # one gesture becomes ONE move, not a queue of steps
HOME_FILE = Path("/tmp/arm_gripper_home")


class BridgeClient(threading.Thread):
    """One thread owns the bridge socket; the dora loop swaps memory only."""

    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.addr = (host, port)
        self.target: float | None = None  # desired width (GIL-atomic swap)
        self.want_home = False
        self.state: dict | None = None
        self.stop_flag = threading.Event()

    def run(self) -> None:
        sent: float | None = None
        seen: float | None = None
        stable_t = 0.0
        buf = b""
        sock: socket.socket | None = None
        warned = False
        while not self.stop_flag.is_set():
            if sock is None:
                try:
                    sock = socket.create_connection(self.addr, timeout=2.0)
                    sock.settimeout(0.2)
                    print(f"[franka_gripper] hand_bridge at {self.addr[0]}:"
                          f"{self.addr[1]} connected", flush=True)
                    warned = False
                    sent = None  # re-send the current target on a fresh session
                except OSError as exc:
                    if not warned:
                        warned = True
                        print(f"[franka_gripper] hand_bridge unreachable "
                              f"({exc}) — retrying; is it running on the RT "
                              "box?", flush=True)
                    self.stop_flag.wait(2.0)
                    continue
            try:
                w = self.target
                if w != seen:  # slider still moving — restart the settle clock
                    seen = w
                    stable_t = time.monotonic()
                if (
                    w is not None
                    and time.monotonic() - stable_t >= SETTLE_S
                    and (sent is None or abs(w - sent) >= DEADBAND_M)
                ):
                    sent = w
                    sock.sendall(f"MOVE {w:.5f} {MOVE_SPEED:.3f}\n".encode())
                if self.want_home:
                    self.want_home = False
                    sock.sendall(b"HOME\n")
                try:
                    data = sock.recv(256)
                    if not data:
                        raise OSError("bridge closed the connection")
                    buf += data
                except socket.timeout:
                    continue
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    parts = line.decode(errors="replace").split()
                    if len(parts) == 3 and parts[0] == "STATE":
                        self.state = {
                            "width": float(parts[1]),
                            "is_grasped": parts[2] == "1",
                        }
            except OSError as exc:
                print(f"[franka_gripper] bridge link lost ({exc}) — "
                      "reconnecting", flush=True)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                buf = b""
                self.stop_flag.wait(2.0)


def main() -> None:
    cfg = load_robot_config()
    rt = dict(cfg.get("rt") or {})
    client = BridgeClient(
        str(rt.get("host", "172.16.1.2")), int(rt.get("hand_port", 47802))
    )
    client.start()
    HOME_FILE.unlink(missing_ok=True)
    node = Node()
    print("[franka_gripper] up — finger slider drives width; "
          f"touch {HOME_FILE} to home", flush=True)
    last_pub = 0.0
    while True:
        event = node.next(timeout=0.1)
        if event is not None:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT" and event["id"] == "gripper":
                finger_m = float(unpack_motor_command(event["value"], 2)["position"][0])
                client.target = 2.0 * finger_m  # width = both fingers
        if HOME_FILE.exists():
            HOME_FILE.unlink(missing_ok=True)
            client.want_home = True
            print("[franka_gripper] homing requested (keep fingers clear)", flush=True)
        now = time.monotonic()
        state = client.state
        if now - last_pub >= 0.1 and state is not None:
            last_pub = now
            node.send_output(
                "gripper_state", pack_json_message("gripper_state", state)
            )
    client.stop_flag.set()


if __name__ == "__main__":
    main()
