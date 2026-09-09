"""Dora node: Franka Hand client — talks to hand_bridge on the RT box.

Driving the Hand with pylibfranka FROM THE PC is impossible since the
network migration: the Hand mirrors the robot's protocol split (commands
over TCP, cyclic state PUSHED over UDP), and server-push UDP cannot cross
the PC-side NAT — moves worked, every read timed out (bench 2026-07-28).
Same physics that put the torque loop on the RT box, same fix: the Hand is
owned by ``rt/src/hand_bridge.cpp`` on the box (robot LAN, no NAT), and
this node speaks its dumb TCP line protocol over the direct link:

    -> "CMD <epoch> MOVE <width_m> <speed_mps>" | "CMD <epoch> HOME"
    -> "CMD <epoch> GRASP <width_m> <speed_mps> <force_n> <eps_in_m> <eps_out_m>"
    <-  "STATE <width_m> <0|1>"      (~10 Hz, streams DURING moves too)
    <-  "META 2 <epoch> <available> <busy> <measured> <sample_seq>"
    <-  "CAPS active_stop" | "STOPPING <epoch>"
    <-  "DONE <command_epoch> <0|1>" | "REJECT <command_epoch> <reason>"

Legacy STATE/GDONE alone cannot attest measured freshness or force capability.
Only one action is admitted from a fresh measured sample. With active_stop,
GSTOP and client disconnect interrupt outstanding moves/grasps. STOPPING only
acknowledges receipt; idle, available, fresh measured state must follow.

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
from contextlib import ExitStack
from pathlib import Path

from arm_control.end_effectors.franka_hand import (
    GRASP_TIMEOUT_S, HandGraspFsm, resolve_grasp_parameters,
)
from arm_control.end_effectors.hand_move import HandMove
from arm_control.config import load_robot_config
from arm_control.messages import (
    pack_grasp_result,
    pack_json_message,
    unpack_grasp_request,
    unpack_motor_command,
    unpack_json_message,
)
from arm_control.node_utils import (
    ShutdownFlag,
    install_signal_handlers,
    next_event_gil_friendly,
)

import os

DEADBAND_M = 0.002  # commanded-width change below this is slider noise
MOVE_SPEED = 0.10   # m/s total width; 50 mm/s per finger in the Hand manual
SETTLE_S = 0.15     # slider must rest this long before a goal is sent —
                    # one gesture becomes ONE move, not a queue of steps


def _grasp_line(fsm: HandGraspFsm) -> str:
    return _request_line(fsm.parameters, "close")


def _request_line(p: dict, mode: str) -> str:
    if mode == "release":
        return f"MOVE {p['width_m']:.5f} {p['speed_mps']:.3f}"
    return (
        f"GRASP {p['width_m']:.5f} {p['speed_mps']:.3f} {p['force_n']:.1f} "
        f"{p['epsilon_inner_m']:.4f} {p['epsilon_outer_m']:.4f}"
    )


def _open_line(fsm: HandGraspFsm) -> str:
    return _request_line(fsm.parameters, "release")


def submit_grasp_request(client, fsm: HandGraspFsm, req: dict, now: float, *,
                         require_authority=False, authorized=False, position_pending=False) -> dict | None:
    """Validate/admit before touching the FSM so rejection preserves active IDs."""
    try:
        if require_authority and not authorized:
            raise ValueError('grasp requires fresh arm authority')
        if require_authority and not client.snapshot()['active_stop']:
            raise ValueError('grasp requires active-stop Hand bridge update')
        if position_pending:
            raise ValueError('hand busy: position completion is pending')
        if fsm.awaiting_result:
            raise ValueError("hand busy: previous grasp result has not been consumed")
        p = resolve_grasp_parameters(fsm._config, req)
        line = _request_line(p, req.get("mode", "close"))
        gdone_base = client.gdone_count
        wire_req = dict(req, _grasp_operation=True, _requires_active_stop=True) if require_authority else req
        if not client.send_line(line, wire_req):
            raise ValueError("hand unavailable, busy, stale, or missing force protocol")
    except ValueError as exc:
        return dict(request_id=str(req.get("request_id", "")),
                    target_id=str(req.get("target_id", "")), ok=False, reason=str(exc))
    _, immediate = fsm.on_request(req, gdone_base, now)
    return immediate
def cancel_grasp_on_authority_loss(client, fsm, authorized):
    if authorized:
        return None
    result = fsm.fail('arm authority lost')
    if result is not None:
        client.cancel_move()
    return result


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
        self._lock = threading.RLock()
        self._connected = False
        self._epoch: int | None = None
        self._bridge_seq: int | None = None
        self._sample_seq = 0
        self._sample_at = 0.0
        self._admitted_seq = -1
        self._command: tuple | None = None
        self._inflight: tuple | None = None
        self._failures: list[dict] = []
        self._move_results: list[dict] = []
        self._cancel = False
        self._cancel_waiting = False
        self._stop_epoch: int | None = None
        self._stop_sample_seq = 0
        self._active_stop = False
        self.gdone_count = 0        # bumps per GDONE line received
        self.gdone_sample_seq = -1
        self.gdone_ok = False       # verdict behind gdone_count
        self.stop_flag = threading.Event()

    def snapshot(self) -> dict:
        with self._lock:
            state = dict(self.state or {})
            state.update(
                available=self._connected and state.get("available", False),
                force_grasp=self._epoch is not None,
                active_stop=self._connected and self._active_stop,
                busy=bool(self._cancel_waiting or self._inflight or state.get("busy", False)),
                measured=bool(state.get("measured") and "width" in state and self._connected
                              and not self._cancel_waiting
                              and time.monotonic() - self._sample_at < .6
                              and self._sample_seq > self._admitted_seq),
                sample_seq=self._sample_seq,
            )
            return state

    def send_line(self, line: str, request: dict | None = None) -> bool:
        """Admit one fresh action. Never queue behind an action or replay it."""
        parts = line.split()
        try:
            if len(parts) == 3 and parts[0] == "MOVE":
                resolve_grasp_parameters({}, dict(width_m=float(parts[1]), speed_mps=float(parts[2])))
            elif len(parts) == 6 and parts[0] == "GRASP":
                resolve_grasp_parameters({}, dict(zip(
                    ("width_m", "speed_mps", "force_n", "epsilon_inner_m", "epsilon_outer_m"),
                    map(float, parts[1:]))))
            elif parts != ["HOME"]:
                return False
        except (ValueError, OverflowError):
            return False
        with self._lock:
            state = self.snapshot()
            if ((request or {}).get('_position_move') or (request or {}).get('_requires_active_stop')) and not state['active_stop']:
                return False
            if self._cancel or not (state["available"] and state["force_grasp"] and state["measured"]) or state["busy"]:
                return False
            self._command = self._inflight = (self._epoch, line, dict(request or {}))
            self._admitted_seq = self._sample_seq
            self.target = None
            return True

    def _fail(self, reason: str) -> None:
        if self._inflight:
            request = self._inflight[2]
            if request.get('_position_move'):
                self._move_results.append(dict(request_id=request['request_id'],
                                               ok=False, reason=reason))
            elif request:
                self._failures.append(dict(request_id=str(request.get("request_id", "")),
                                           target_id=str(request.get("target_id", "")),
                                           ok=False, reason=reason))
        self._command = self._inflight = None
        self.target = None
        self.want_home = False

    def pop_failures(self) -> list[dict]:
        with self._lock:
            failures, self._failures = self._failures, []
            return failures

    def cancel_move(self):
        """Request active stop; stay busy until acknowledged and freshly measured."""
        with self._lock:
            if not self._cancel_waiting:
                self._cancel = self._cancel_waiting = True
                self._stop_epoch = None
            self._fail('hand operation canceled')

    def pop_move_results(self):
        with self._lock:
            results, self._move_results = self._move_results, []
            return results

    def _receive(self, line: bytes) -> None:
        parts = line.decode(errors="strict").split()
        with self._lock:
            if parts == ['CAPS', 'active_stop']:
                self._active_stop = True
            elif len(parts) == 2 and parts[0] == 'STOPPING':
                epoch = int(parts[1])
                if epoch < 0:
                    raise ValueError('invalid stop epoch')
                if self._cancel_waiting:
                    self._stop_epoch = epoch
                    self._stop_sample_seq = self._sample_seq
            elif len(parts) == 3 and parts[0] == "STATE":
                width = float(parts[1])
                resolve_grasp_parameters({}, {"width_m": width})
                if parts[2] not in ("0", "1"):
                    raise ValueError("invalid grasped bit")
                self.state = dict(self.state or {}, width=width, is_grasped=parts[2] == "1")
            elif len(parts) == 7 and parts[:2] == ["META", "2"]:
                epoch, available, busy, measured, seq = map(int, parts[2:])
                if epoch < 0 or seq < 0 or any(v not in (0, 1) for v in (available, busy, measured)):
                    raise ValueError("invalid bridge metadata")
                if self._inflight and (not available or epoch not in (
                        self._inflight[0], self._inflight[0] + 1)):
                    self._fail("hand disconnected or command canceled")
                self._epoch = epoch
                if self._bridge_seq is not None and seq < self._bridge_seq:
                    raise ValueError("bridge observation sequence regressed")
                if measured and seq != self._bridge_seq:
                    self._bridge_seq = seq
                    self._sample_seq += 1
                    self._sample_at = time.monotonic()
                self.state = dict(self.state or {}, available=bool(available), busy=bool(busy),
                                  measured=bool(measured))
                if (self._cancel_waiting and self._stop_epoch is not None
                        and epoch >= self._stop_epoch and available and not busy and measured
                        and self._sample_seq > self._stop_sample_seq):
                    self._cancel_waiting = False
                    self._stop_epoch = None
            elif len(parts) == 3 and parts[0] in ("DONE", "REJECT"):
                token = int(parts[1])
                if self._inflight and token == self._inflight[0]:
                    if parts[0] == "REJECT":
                        self._fail("hand rejected: " + parts[2])
                    elif self._inflight[1].startswith("GRASP ") or self._inflight[2].get("_grasp_operation"):
                        self.gdone_ok = parts[2] == "1"
                        self.gdone_count += 1
                        self.gdone_sample_seq = self._sample_seq
                        self._inflight = None
                    elif parts[2] != "1":
                        self._fail("hand action failed")
                    else:
                        if self._inflight[2].get('_position_move'):
                            self._move_results.append(dict(
                                request_id=self._inflight[2]['request_id'], ok=True))
                        self._inflight = None
                    self.target = None

    def close(self) -> None:
        self.stop_flag.set()
        # Connection establishment is bounded by 2 s; recv by 0.2 s.
        self.join(timeout=2.5)

    def run(self) -> None:
        sock: socket.socket | None = None
        try:
            sent: float | None = None
            seen: float | None = None
            stable_t = 0.0
            buf = b""
            warned = False
            while not self.stop_flag.is_set():
                if sock is None:
                    try:
                        sock = socket.create_connection(self.addr, timeout=2.0)
                        sock.settimeout(0.2)
                        print(f"[franka_gripper] hand_bridge at {self.addr[0]}:"
                              f"{self.addr[1]} connected", flush=True)
                        warned = False
                        with self._lock:
                            self._connected = True
                            self._epoch = self._bridge_seq = None
                            self._active_stop = False
                            self._cancel = self._cancel_waiting = False
                            self._stop_epoch = None
                            self.state = None
                            self.target = None
                        sent = seen = None
                    except OSError as exc:
                        if not warned:
                            warned = True
                            print(f"[franka_gripper] hand_bridge unreachable "
                                  f"({exc}) — retrying; is it running on the RT "
                                  "box?", flush=True)
                        self.stop_flag.wait(2.0)
                        continue
                if self.stop_flag.is_set():
                    break
                try:
                    w = self.target
                    state = self.snapshot()
                    if state["busy"] or not state["measured"]:
                        self.target = w = None
                    if w != seen:  # slider still moving — restart the settle clock
                        seen = w
                        stable_t = time.monotonic()
                    if (
                        w is not None
                        and time.monotonic() - stable_t >= SETTLE_S
                        and (sent is None or abs(w - sent) >= DEADBAND_M)
                    ):
                        sent = w
                        self.send_line(f"MOVE {w:.5f} {MOVE_SPEED:.3f}")
                    if self.want_home:
                        self.want_home = False
                        self.send_line("HOME")
                    with self._lock:
                        cancel, self._cancel = self._cancel, False
                        command, self._command = self._command, None
                        # Serialize cancellation with admission-to-wire. A local
                        # cancellation cannot clear inflight while a detached
                        # copy of its command is still waiting to be sent.
                        if cancel:
                            sock.sendall(b'GSTOP\n')
                        elif command:
                            sock.sendall(f"CMD {command[0]} {command[1]}\n".encode())
                    try:
                        data = sock.recv(256)
                        if not data:
                            raise OSError("bridge closed the connection")
                        buf += data
                    except socket.timeout:
                        continue
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._receive(line)
                    if len(buf) > 4096:
                        raise ValueError("oversized bridge response")
                except (OSError, ValueError, UnicodeError) as exc:
                    print(f"[franka_gripper] bridge link lost ({exc}) — "
                          "reconnecting", flush=True)
                    try:
                        sock.close()
                    except OSError:
                        pass
                    sock = None
                    buf = b""
                    with self._lock:
                        self._connected = False
                        self._epoch = None
                        self._fail("hand bridge disconnected")
                    self.stop_flag.wait(2.0)
        finally:
            if sock is not None:
                sock.close()
            with self._lock:
                self._connected = False
                self._fail("hand bridge stopped")


def main() -> None:
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    with ExitStack() as cleanup:
        _run(shutdown, cleanup)


def _run(shutdown: ShutdownFlag, cleanup: ExitStack) -> None:
    from dora import Node

    cfg = load_robot_config()
    rt = dict(cfg.get("rt") or {})
    client = BridgeClient(
        str(rt.get("host", "172.16.1.2")), int(rt.get("hand_port", 47802))
    )
    require_authority = bool(cfg.get('hand_move_requires_arm', False))
    grasp_fsm = HandGraspFsm(dict((cfg.get("franka") or {}).get("gripper") or {}),
                             measured_completion=require_authority)
    move = HandMove()
    permitted = not require_authority
    operator_armed = not require_authority
    health_at = 0.
    client.start()
    cleanup.callback(client.close)
    if not require_authority:
        HOME_FILE.unlink(missing_ok=True)
    node = Node()
    print('[franka_gripper] up — ' + (
        'gated grasp/release and position moves; file homing and sliders disabled'
        if require_authority else f'finger slider drives width; touch {HOME_FILE} to home'), flush=True)
    last_pub = 0.0
    while not shutdown.stop_requested:
        event = next_event_gil_friendly(node)
        if shutdown.stop_requested:
            break
        if event is not None:
            if event["type"] == "STOP":
                break
            if event["type"] == "INPUT" and event["id"] == "gripper":
                if require_authority:
                    continue  # Gated applications accept only correlated requests.
                finger_m = float(unpack_motor_command(event["value"], 2)["position"][0])
                client.target = 2.0 * finger_m  # width = both fingers
            elif event['type'] == 'INPUT' and event['id'] == 'motor_health':
                health = unpack_json_message(event['value'])
                health_at = time.monotonic()
                permitted = bool(health.get('armed') and health.get('state_fresh', False)
                                 and not health.get('any_fault') and not health.get('latched_fault'))
            elif event['type'] == 'INPUT' and event['id'] == 'arm':
                operator_armed = unpack_json_message(event['value']).get('armed') is True
            elif event['type'] == 'INPUT' and event['id'] == 'hand_move_request':
                req = unpack_json_message(event['value'], expected_schema='hand_move_request')
                try:
                    if require_authority and (not operator_armed or not permitted or time.monotonic() - health_at > 1.):
                        raise ValueError('position move requires fresh arm authority')
                    if grasp_fsm.awaiting_result:
                        raise ValueError('grasp completion is pending')
                    if not client.snapshot()['active_stop']:
                        raise ValueError('position move requires active-stop Hand bridge update')
                    line = move.start(req, client.snapshot(), time.monotonic())
                except ValueError as exc:
                    result = dict(request_id=str(req.get('request_id', '')), ok=False, reason=str(exc))
                else:
                    result = None
                    if not client.send_line(line, dict(req, _position_move=True)):
                        result = move.fail('hand admission refused')
                if result is not None:
                    node.send_output('hand_move_result', pack_json_message('hand_move_result', result))
            elif event["type"] == "INPUT" and event["id"] == "grasp_request":
                req = unpack_grasp_request(event["value"])
                now = time.monotonic()
                immediate = submit_grasp_request(
                    client, grasp_fsm, req, now, require_authority=require_authority,
                    authorized=operator_armed and permitted and now - health_at <= 1.,
                    position_pending=move.pending is not None)
                if immediate is not None:
                    node.send_output(
                        "grasp_result", pack_grasp_result(**immediate)
                    )
        if not require_authority and HOME_FILE.exists():
            HOME_FILE.unlink(missing_ok=True)
            client.want_home = True
            print("[franka_gripper] homing requested (keep fingers clear)", flush=True)
        now = time.monotonic()
        state = client.snapshot()
        state['effort_observed'] = False
        state['effort_observability'] = 'unavailable'
        state['hand_move_ready'] = bool(
            state['available'] and state['measured'] and not state['busy']
            and state['active_stop']
            and not state.get('is_grasped') and not move.pending
            and (not require_authority or (operator_armed and permitted and now - health_at <= 1.)))
        if require_authority and move.pending and (not operator_armed or not permitted or now - health_at > 1.):
            client.cancel_move()
            result = move.fail('arm authority lost')
            node.send_output('hand_move_result', pack_json_message('hand_move_result', result))
        if require_authority and (not operator_armed or not permitted or now - health_at > 1.):
            result = cancel_grasp_on_authority_loss(client, grasp_fsm, False)
            if result is not None:
                node.send_output('grasp_result', pack_grasp_result(**result))
        for completed in client.pop_move_results():
            result = move.done(completed, state)
            if result is not None:
                node.send_output('hand_move_result', pack_json_message('hand_move_result', result))
        result = move.poll(state, now)
        if result is not None:
            if not result['ok']:
                client.cancel_move()
            node.send_output('hand_move_result', pack_json_message('hand_move_result', result))
        for failure in client.pop_failures():
            grasp_fsm.fail(failure["reason"], failure["request_id"])
            node.send_output("grasp_result", pack_grasp_result(**failure))
        result = (grasp_fsm.poll(state if state["measured"] else None,
                                client.gdone_count, client.gdone_ok, now, done_sample_seq=client.gdone_sample_seq)
                  if state["available"] else grasp_fsm.fail("hand unavailable"))
        if result is not None:
            if require_authority and not result["ok"]:
                client.cancel_move()
            print(f"[franka_gripper] grasp_result ok={result['ok']} "
                  f"({result['reason']})", flush=True)
            node.send_output("grasp_result", pack_grasp_result(**result))
        if now - last_pub >= 0.1:
            last_pub = now
            node.send_output(
                "gripper_state", pack_json_message("gripper_state", state)
            )


def _demo() -> None:
    """Self-check: FSM verdict flow + line framing over a real socket."""
    fsm = HandGraspFsm({})
    # close -> GDONE ok -> grasped -> is_grasped falling edge = LOST
    action, imm = fsm.on_request({"request_id": "r1", "target_id": "m"}, 0, 100.0)
    assert action == "grasp" and _grasp_line(fsm).startswith("GRASP ") and imm is None
    assert fsm.poll(None, 0, False, 100.5) is None  # still in flight
    r = fsm.poll(None, 1, True, 101.0)
    assert r["ok"] and r["reason"] == "grasped" and r["request_id"] == "r1"
    assert fsm.poll({"is_grasped": True}, 1, True, 101.5) is None
    r = fsm.poll({"is_grasped": False}, 1, True, 102.0)
    assert not r["ok"] and r["reason"] == "object lost" and r["request_id"] == "r1"
    assert fsm.poll({"is_grasped": False}, 1, True, 102.5) is None  # fires once
    # failed close (GDONE 0); the pre-grasp is_grasped=False must NOT re-fire LOST
    fsm.on_request({"request_id": "r2", "target_id": "m"}, 1, 103.0)
    r = fsm.poll({"is_grasped": False}, 2, False, 103.5)
    assert not r["ok"] and r["reason"] == "no object"
    # timeout fallback: no GDONE, freshest sample says held
    fsm.on_request({"request_id": "r3", "target_id": "m"}, 2, 104.0)
    r = fsm.poll({"is_grasped": True}, 2, False, 104.0 + GRASP_TIMEOUT_S + 1)
    assert r["ok"] and "fallback" in r["reason"]
    # release: immediate ack, held cleared (no LOST afterwards)
    action, imm = fsm.on_request(
        {"request_id": "r4", "target_id": "m", "mode": "release"}, 2, 105.0
    )
    assert action == "open" and _open_line(fsm).startswith("MOVE ")
    assert imm["ok"] and imm["reason"] == "released"
    assert fsm.poll({"is_grasped": False}, 2, False, 106.0) is None

    from tools.bench.check_hand_grasp import check_active_stop, check_parameters, check_socket
    check_parameters()
    check_active_stop()
    check_socket()
    print("[franka_gripper] demo OK — FSM verdicts + wire framing")


def cli() -> None:
    import sys

    if "--demo" in sys.argv:
        _demo()
    else:
        main()


if __name__ == "__main__":
    cli()
