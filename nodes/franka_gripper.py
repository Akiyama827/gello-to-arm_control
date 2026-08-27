"""Dora node: Franka Hand client — talks to hand_bridge on the RT box.

Driving the Hand with pylibfranka FROM THE PC is impossible since the
network migration: the Hand mirrors the robot's protocol split (commands
over TCP, cyclic state PUSHED over UDP), and server-push UDP cannot cross
the PC-side NAT — moves worked, every read timed out (bench 2026-07-28).
Same physics that put the torque loop on the RT box, same fix: the Hand is
owned by ``rt/src/hand_bridge.cpp`` on the box (robot LAN, no NAT), and
this node speaks its dumb TCP line protocol over the direct link:

    -> "MOVE <width_m> <speed_mps>"  |  "HOME"  |  "GSTOP"
    -> "GRASP <width_m> <speed_mps> <force_n> <eps_in_m> <eps_out_m>"
    <-  "STATE <width_m> <0|1>"      (~10 Hz, streams DURING moves too)
    <-  "GDONE <0|1>"                (once per completed GRASP)

Input ``gripper``: motor_command_gripper format (finger metres, e.g. the
teleop slider), latest-wins with a 2 mm width deadband. Output
``gripper_state``: width + is_grasped. ``touch /tmp/arm_gripper_home``
requests a homing cycle (physical open-close — needed once per Hand
power-cycle; deliberately never automatic).

Input ``grasp_request`` / output ``grasp_result`` (pick graphs): the
orchestrator's grasp contract, served by the Hand itself instead of the DM
path's GraspGate — ``close`` maps to the bridge GRASP verb (the Hand's own
width-band verdict is the GRASPED/MISSED call), ``release`` to a MOVE back
to the configured open width (acked immediately, like the DM release).
After a held grasp, ``is_grasped`` falling in the STATE stream is the drop
event: a second, failed result under the close request_id — the same LOST
semantics the orchestrator already freezes on. Grasp geometry/force come
from the ``franka.gripper`` config block. A GRASP killed mid-flight (GSTOP,
bridge reconnect) sends no GDONE; a timeout falls back to the freshest
``is_grasped`` sample so the orchestrator is never left hanging.

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

from arm_control.bridge.hand_grasp import GRASP_TIMEOUT_S, HandGraspFsm
from arm_control.config import load_robot_config
from arm_control.messages import (
    pack_grasp_result,
    pack_json_message,
    unpack_grasp_request,
    unpack_motor_command,
)

import os

DEADBAND_M = 0.002  # commanded-width change below this is slider noise
MOVE_SPEED = 0.10   # m/s — brisk but gentle; the Hand's max is 0.2
SETTLE_S = 0.15     # slider must rest this long before a goal is sent —
                    # one gesture becomes ONE move, not a queue of steps


def _grasp_line(fsm: HandGraspFsm) -> str:
    return (
        f"GRASP {fsm.grasp_width_m:.5f} {fsm.speed_mps:.3f} {fsm.force_n:.1f} "
        f"{fsm.epsilon_inner_m:.4f} {fsm.epsilon_outer_m:.4f}"
    )


def _open_line(fsm: HandGraspFsm) -> str:
    return f"MOVE {fsm.open_width_m:.5f} {fsm.speed_mps:.3f}"
# User-private runtime dir, not /tmp: this file triggers a PHYSICAL open-close
# sweep of the jaws — it must not be any local user's to touch.
HOME_FILE = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "arm_gripper_home"


class BridgeClient(threading.Thread):
    """One thread owns the bridge socket; the dora loop swaps memory only."""

    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.addr = (host, port)
        self.target: float | None = None  # desired width (GIL-atomic swap)
        self.want_home = False
        self.state: dict | None = None
        self.lines: list[str] = []  # one-shot commands (GRASP/MOVE); GIL-safe
        self.gdone_count = 0        # bumps per GDONE line received
        self.gdone_ok = False       # verdict behind gdone_count
        self.stop_flag = threading.Event()

    def send_line(self, line: str) -> None:
        self.lines.append(line)

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
                while self.lines:
                    sock.sendall((self.lines.pop(0) + "\n").encode())
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
                    elif len(parts) == 2 and parts[0] == "GDONE":
                        self.gdone_ok = parts[1] == "1"
                        self.gdone_count += 1
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
    grasp_fsm = HandGraspFsm(dict((cfg.get("franka") or {}).get("gripper") or {}))
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
            elif event["type"] == "INPUT" and event["id"] == "grasp_request":
                req = unpack_grasp_request(event["value"])
                action, immediate = grasp_fsm.on_request(
                    req, client.gdone_count, time.monotonic()
                )
                line = _grasp_line(grasp_fsm) if action == "grasp" else _open_line(grasp_fsm)
                print(f"[franka_gripper] grasp_request mode={req.get('mode')} "
                      f"-> {line}", flush=True)
                client.send_line(line)
                if immediate is not None:
                    node.send_output(
                        "grasp_result", pack_grasp_result(**immediate)
                    )
        if HOME_FILE.exists():
            HOME_FILE.unlink(missing_ok=True)
            client.want_home = True
            print("[franka_gripper] homing requested (keep fingers clear)", flush=True)
        now = time.monotonic()
        state = client.state
        result = grasp_fsm.poll(state, client.gdone_count, client.gdone_ok, now)
        if result is not None:
            print(f"[franka_gripper] grasp_result ok={result['ok']} "
                  f"({result['reason']})", flush=True)
            node.send_output("grasp_result", pack_grasp_result(**result))
        if now - last_pub >= 0.1 and state is not None:
            last_pub = now
            node.send_output(
                "gripper_state", pack_json_message("gripper_state", state)
            )
    client.stop_flag.set()


def _demo() -> None:
    """Self-check: FSM verdict flow + line framing over a real socket."""
    fsm = HandGraspFsm({})
    # close -> GDONE ok -> grasped -> is_grasped falling edge = LOST
    action, imm = fsm.on_request({"request_id": "r1", "module_id": "m"}, 0, 100.0)
    assert action == "grasp" and _grasp_line(fsm).startswith("GRASP ") and imm is None
    assert fsm.poll(None, 0, False, 100.5) is None  # still in flight
    r = fsm.poll(None, 1, True, 101.0)
    assert r["ok"] and r["reason"] == "grasped" and r["request_id"] == "r1"
    assert fsm.poll({"is_grasped": True}, 1, True, 101.5) is None
    r = fsm.poll({"is_grasped": False}, 1, True, 102.0)
    assert not r["ok"] and r["reason"] == "object lost" and r["request_id"] == "r1"
    assert fsm.poll({"is_grasped": False}, 1, True, 102.5) is None  # fires once
    # failed close (GDONE 0); the pre-grasp is_grasped=False must NOT re-fire LOST
    fsm.on_request({"request_id": "r2", "module_id": "m"}, 1, 103.0)
    r = fsm.poll({"is_grasped": False}, 2, False, 103.5)
    assert not r["ok"] and r["reason"] == "no object"
    # timeout fallback: no GDONE, freshest sample says held
    fsm.on_request({"request_id": "r3", "module_id": "m"}, 2, 104.0)
    r = fsm.poll({"is_grasped": True}, 2, False, 104.0 + GRASP_TIMEOUT_S + 1)
    assert r["ok"] and "fallback" in r["reason"]
    # release: immediate ack, held cleared (no LOST afterwards)
    action, imm = fsm.on_request(
        {"request_id": "r4", "module_id": "m", "mode": "release"}, 2, 105.0
    )
    assert action == "open" and _open_line(fsm).startswith("MOVE ")
    assert imm["ok"] and imm["reason"] == "released"
    assert fsm.poll({"is_grasped": False}, 2, False, 106.0) is None

    # Framing: BridgeClient against a scripted one-client bridge.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    got: list[bytes] = []

    def bridge() -> None:
        conn, _ = srv.accept()
        conn.sendall(b"STATE 0.07500 0\n")
        buf = b""
        while b"\n" not in buf:
            buf += conn.recv(256)
        got.append(buf)
        conn.sendall(b"GDONE 1\nSTATE 0.04510 1\n")
        time.sleep(0.3)
        conn.close()

    t = threading.Thread(target=bridge, daemon=True)
    t.start()
    client = BridgeClient("127.0.0.1", srv.getsockname()[1])
    client.start()
    deadline = time.monotonic() + 2.0
    while client.state is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.state is not None, "no STATE parsed"
    client.send_line("GRASP 0.04500 0.050 40.0 0.0200 0.0200")
    while client.gdone_count == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.gdone_count == 1 and client.gdone_ok, "no GDONE parsed"
    assert got and got[0].startswith(b"GRASP 0.04500"), got
    while not (client.state or {}).get("is_grasped") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.state["is_grasped"], "post-grasp STATE not parsed"
    client.stop_flag.set()
    srv.close()
    print("[franka_gripper] demo OK — FSM verdicts + wire framing")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        _demo()
    else:
        main()
