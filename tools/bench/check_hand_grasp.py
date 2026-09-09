"""No-hardware assert checks for Franka Hand force requests and TCP admission."""
from __future__ import annotations

import socket
import time

from arm_control.contracts.gripper import pack_grasp_request, unpack_grasp_request
from arm_control.end_effectors import franka_hand


def check_parameters():
    legacy = unpack_grasp_request(pack_grasp_request(request_id="r", target_id="hand"))
    assert legacy == dict(schema="grasp_request", request_id="r", target_id="hand", mode="close", gripper_body="gripper")
    assert hasattr(franka_hand, "resolve_grasp_parameters"), "missing request parameter resolver"
    resolve = franka_hand.resolve_grasp_parameters
    cfg = {"force_n": 40.0, "grasp_width_m": 0.045}
    custom = unpack_grasp_request(pack_grasp_request(
        request_id="r", target_id="hand", width_m=0.03, force_n=55.0))
    assert resolve(cfg, custom)["width_m"] == 0.03
    assert resolve(cfg, custom)["force_n"] == 55.0
    assert resolve(cfg, {})["force_n"] == 40.0 and cfg["grasp_width_m"] == 0.045
    assert resolve(cfg, {"mode": "release"})["width_m"] == 0.075
    for override in ({"width_m": .03}, {"speed_mps": .1}):
        try:
            resolve(cfg, {"mode": "release", **override})
        except ValueError:
            pass
        else:
            raise AssertionError("Open accepted an override of configured width/speed")
    for key, values in {
        "force_n": [True, "40", None, float("nan"), float("inf"), 29.9, 70.1],
        "width_m": [False, "0.04", None, float("nan"), -0.01, 0.081],
        "speed_mps": [0, -1, 0.101, float("inf")],
        "epsilon_inner_m": [-0.01, 0.081, float("nan")],
        "epsilon_outer_m": [-0.01, 0.081, float("inf")],
        "mode": ["typo", "", None],
    }.items():
        for value in values:
            try:
                resolve(cfg, {key: value})
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {key}={value!r}")
    fsm = franka_hand.HandGraspFsm(cfg)
    action, result = fsm.on_request(custom, 0, 1.0)
    assert action == "grasp" and result is None
    assert fsm.parameters["force_n"] == 55 and fsm.force_n == 40
    action, result = fsm.on_request({"request_id": "bad", "force_n": 100}, 0, 1.1)
    assert action is None and not result["ok"] and result["request_id"] == "bad"
    assert fsm.poll(None, 1, True, 1.2)["request_id"] == "r"
    fsm.on_request({"request_id": "next"}, 1, 2.0)
    assert fsm.parameters["force_n"] == 40
    print("PASS hand request validation and parameter isolation")


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.01)
    assert predicate(), "fake peer deadline"


def check_active_stop():
    from arm_control.end_effectors.franka_adapter import BridgeClient
    client = BridgeClient("127.0.0.1", 1)
    client._connected = True
    client._receive(b"STATE 0.075 0")
    client._receive(b"META 2 8 1 0 1 10")
    request = dict(request_id="empty-move", _position_move=True)
    assert not client.send_line("MOVE 0.045 0.05", request), \
        "position-only pipeline accepted a bridge without active STOP"
    client._receive(b"CAPS active_stop")
    assert client.snapshot()["active_stop"]
    assert client.send_line("MOVE 0.045 0.05", request)
    client.cancel_move()
    assert client.pop_move_results() == [dict(
        request_id="empty-move", ok=False, reason="hand operation canceled")]
    assert client._command is None and client.snapshot()["busy"]
    client._cancel = False  # Socket worker has sent GSTOP.
    client._receive(b"DONE 8 1")  # Late success cannot resurrect cancellation.
    assert client.pop_move_results() == []
    client._receive(b"META 2 8 1 0 1 11")  # Buffered before GSTOP arrived.
    assert client.snapshot()["busy"]
    assert not client.send_line("MOVE 0.075 0.05")
    client._receive(b"STOPPING 10")
    client._receive(b"META 2 10 1 1 0 11")
    assert client.snapshot()["busy"]
    client._receive(b"META 2 10 0 0 0 11")  # Failed stop remains unavailable.
    assert not client.snapshot()["available"]
    client._receive(b"STATE 0.060 0")
    client._receive(b"META 2 11 1 0 1 12")  # Recovered + fresh post-stop read.
    assert not client.snapshot()["busy"] and client.snapshot()["measured"]
    assert client.send_line("MOVE 0.075 0.05", dict(request, request_id="new"))
    print("PASS active STOP capability, cancellation barrier and late DONE rejection")


def check_gated_grasp():
    from arm_control.end_effectors.franka_adapter import (
        BridgeClient, submit_grasp_request, cancel_grasp_on_authority_loss,
    )

    def setup():
        client = BridgeClient('127.0.0.1', 1)  # Never started: no socket.
        client._connected = True
        client._receive(b'CAPS active_stop')
        client._receive(b'STATE 0.075 0')
        client._receive(b'META 2 8 1 0 1 1')
        return client, franka_hand.HandGraspFsm({}, measured_completion=True)

    req = dict(request_id='close', target_id='module', mode='close')
    client, fsm = setup()
    for authorized, pending, capability in ((False, False, True), (True, True, True), (True, False, False)):
        client._active_stop = capability
        result = submit_grasp_request(client, fsm, req, 1., require_authority=True,
                                      authorized=authorized, position_pending=pending)
        assert not result['ok'] and client._command is None and not fsm.awaiting_result
    client._active_stop = True
    assert submit_grasp_request(client, fsm, req, 1., require_authority=True, authorized=True) is None
    assert client._command[1].startswith('GRASP ')
    assert client._command[2]['_requires_active_stop']
    assert not submit_grasp_request(client, fsm, dict(req, request_id='duplicate'), 1.,
                                   require_authority=True, authorized=True)['ok']
    def poll(now=1.1):
        state = client.snapshot()
        return fsm.poll(state if state['measured'] else None, client.gdone_count,
                        client.gdone_ok, now, done_sample_seq=client.gdone_sample_seq)
    client._receive(b'DONE 7 1')
    assert client.gdone_count == 0 and poll() is None
    client._receive(b'DONE 8 1')
    assert poll() is None, 'completion without fresh measurement passed'
    client._receive(b'STATE 0.045 1')
    client._receive(b'META 2 9 1 0 1 2')
    assert poll()['ok']
    client._receive(b'DONE 8 1')
    assert poll() is None
    # Release is measured completion, never an immediate successful grasp_result.
    release = dict(request_id='release', target_id='module', mode='release')
    assert submit_grasp_request(client, fsm, release, 2., require_authority=True, authorized=True) is None
    assert client._command[1].startswith('MOVE ')
    client._receive(b'DONE 9 1')
    assert poll(2.1) is None
    client._receive(b'STATE 0.075 0')
    client._receive(b'META 2 10 1 0 1 3')
    result = poll(2.2)
    assert result['ok'] and result['request_id'] == 'release'
    for held in (False, True):
        client, fsm = setup()
        submit_grasp_request(client, fsm, req, 1., require_authority=True, authorized=True)
        if held:
            client._receive(b'DONE 8 1')
            client._receive(b'STATE 0.045 1')
            client._receive(b'META 2 9 1 0 1 2')
            assert poll()['ok']
        result = cancel_grasp_on_authority_loss(client, fsm, False)
        assert not result['ok'] and client._cancel_waiting and client._command is None
        client._receive(b'DONE 8 1')
        assert poll() is None, 'late completion resurrected canceled grasp'
    client, fsm = setup()
    submit_grasp_request(client, fsm, req, 1., require_authority=True, authorized=True)
    # Unlike the legacy FSM, a held observation without matching DONE cannot pass.
    client._receive(b'STATE 0.045 1')
    client._receive(b'META 2 9 1 0 1 2')
    assert not poll(1. + franka_hand.GRASP_TIMEOUT_S + 1.)['ok']
    print('PASS gated grasp/release admission, measured completion, authority cancellation and late DONE')


def check_socket():
    from arm_control.end_effectors.franka_adapter import BridgeClient, submit_grasp_request
    class FastClient:
        gdone_count = 0
        def send_line(self, line, request):
            assert "55.0" in line
            self.gdone_count += 1
            return True
    fsm = franka_hand.HandGraspFsm({})
    fast = FastClient()
    assert submit_grasp_request(fast, fsm, {"request_id": "fast", "force_n": 55}, 1.0) is None
    assert fsm.poll(None, fast.gdone_count, True, 1.1)["ok"], "fast completion lost"
    # DONE can reach the socket thread before the Dora loop publishes its result.
    ready = BridgeClient("127.0.0.1", 1)
    ready._connected = True
    ready._receive(b"STATE 0.075 0")
    ready._receive(b"META 2 8 1 0 1 10")
    fsm = franka_hand.HandGraspFsm({})
    assert submit_grasp_request(ready, fsm, {"request_id": "first"}, 1.0) is None
    ready._receive(b"DONE 8 1")
    ready._receive(b"STATE 0.04 1")
    ready._receive(b"META 2 9 1 0 1 11")
    rejected = submit_grasp_request(ready, fsm, {"request_id": "second"}, 1.1)
    assert rejected and not rejected["ok"] and rejected["request_id"] == "second", \
        "new request overwrote the unpublished first result"
    assert fsm.poll(ready.snapshot(), ready.gdone_count, True, 1.2)["request_id"] == "first"
    assert fsm.poll({"is_grasped": True}, ready.gdone_count, True, 1.3) is None
    ready._receive(b"META 2 9 0 0 0 11")
    assert not submit_grasp_request(ready, fsm, {"request_id": "offline"}, 1.4)["ok"]
    assert fsm.poll({"is_grasped": False}, ready.gdone_count, True, 1.5)["reason"] == "object lost"
    ready._receive(b"STATE 0.04 0")
    ready._receive(b"META 2 9 1 0 1 12")
    assert submit_grasp_request(ready, fsm, {"request_id": "next"}, 1.6) is None
    client = BridgeClient("127.0.0.1", 1)
    assert hasattr(client, "snapshot"), "missing measured freshness metadata"
    assert not client.send_line("GRASP 0.04 0.05 40 0.02 0.02"), "offline request accepted"
    client._connected = True
    client._receive(b"META 2 1 1 0 1 10")
    assert not client.snapshot()["measured"], "metadata without width treated as measured"
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    server.settimeout(4)
    client = BridgeClient("127.0.0.1", server.getsockname()[1])
    client.start()
    conn, _ = server.accept()
    conn.settimeout(1)
    try:
        conn.sendall(b"STATE 0.075 0\nMETA 2 8 1 0 1 10\n")
        wait_for(lambda: client.snapshot()["measured"])
        first = client.snapshot()["sample_seq"]
        conn.sendall(b'CAPS active_stop\n')
        wait_for(lambda: client.snapshot()['active_stop'])
        conn.sendall(b"STATE 0.075 0\nMETA 2 8 1 0 1 10\n")
        time.sleep(.2)
        assert client.snapshot()["sample_seq"] == first
        assert client.send_line("GRASP 0.04 0.05 55 0.02 0.02", {"request_id": "r", "target_id": "hand"})
        assert not client.send_line("MOVE 0.075 0.05"), "busy action queued"
        assert conn.recv(256) == b"CMD 8 GRASP 0.04 0.05 55 0.02 0.02\n"
        client.target = .08 # old slider gesture must be dropped while busy
        conn.sendall(b"ACCEPT 8 accepted\nSTATE 0.05 0\nMETA 2 9 1 1 0 10\n")
        time.sleep(.2)
        assert client.snapshot()["sample_seq"] == first and not client.snapshot()["measured"]
        conn.sendall(b"GDONE 1\nDONE 8 1\nSTATE 0.0401 1\nMETA 2 9 1 0 1 11\n")
        wait_for(lambda: client.gdone_count == 1)
        assert client.gdone_ok and client.snapshot()["sample_seq"] > first
        # Cached repetition must not revive freshness.
        for _ in range(5):
            conn.sendall(b"STATE 0.0401 1\nMETA 2 9 1 0 1 11\n")
            time.sleep(.15)
        assert not client.snapshot()["measured"]
        assert not client.send_line("MOVE 0.075 0.05"), "stale action accepted"
        conn.sendall(b"STATE 0.04 1\nMETA 2 9 1 0 1 12\n")
        wait_for(lambda: client.snapshot()["measured"])
        assert not client.send_line("MOVE nan 0.05"), "invalid command accepted"
        assert client.send_line("GRASP 0.04 0.05 40 0.02 0.02", {"request_id": "lost", "target_id": "hand"})
        assert conn.recv(256).startswith(b"CMD 9 GRASP")
        conn.close()
        wait_for(lambda: not client.snapshot()["available"])
        assert client.pop_failures()[0]["request_id"] == "lost"
        client.target = .08 # offline slider must not replay after reconnect
        conn, _ = server.accept()
        conn.settimeout(.4)
        conn.sendall(b"STATE 0.04 1\nMETA 2 20 1 0 1 1\n")
        wait_for(lambda: client.snapshot()["measured"])
        assert client.snapshot()["sample_seq"] > first
        try:
            assert not conn.recv(256), "stale command replayed after reconnect"
        except socket.timeout:
            pass
        assert not client.snapshot()['active_stop'], 'capability leaked across reconnect'
        conn.sendall(b'CAPS active_stop\n')
        wait_for(lambda: client.snapshot()['active_stop'])
        conn.settimeout(1)
        assert client.send_line('MOVE 0.045 0.05', dict(
            request_id='cancel-wire', _position_move=True))
        assert conn.recv(256) == b'CMD 20 MOVE 0.045 0.05\n'
        client.cancel_move()
        assert conn.recv(256) == b'GSTOP\n'
        client.cancel_move()  # Coalesced while waiting, never queues another action.
        conn.sendall(b'DONE 20 1\nSTATE 0.045 0\nMETA 2 21 1 0 1 2\n')
        wait_for(lambda: client._epoch == 21)
        assert client.snapshot()['busy'], 'pre-STOP buffered state cleared cancellation'
        results = client.pop_move_results()
        assert len(results) == 1 and not results[0]['ok']
        conn.sendall(b'STOPPING 22\nMETA 2 22 1 1 0 2\n')
        wait_for(lambda: client._stop_epoch == 22)
        assert client.snapshot()['busy']
        conn.sendall(b'STATE 0.060 0\nMETA 2 22 1 0 1 3\n')
        wait_for(lambda: not client.snapshot()['busy'])
        assert client.snapshot()['measured'] and not client.pop_move_results()
        conn.settimeout(.3)
        try:
            assert not conn.recv(256), 'duplicate STOP or canceled command replayed'
        except socket.timeout:
            pass
        # Cancel an admitted command before run() can move it to the wire.
        with client._lock:
            assert client.send_line('MOVE 0.075 0.05', dict(
                request_id='cancel-unsent', _position_move=True))
            client.cancel_move()
        conn.settimeout(1)
        assert conn.recv(256) == b'GSTOP\n', 'unsent MOVE escaped cancellation'
        conn.sendall(b'STOPPING 23\nSTATE 0.060 0\nMETA 2 23 1 0 1 4\n')
        wait_for(lambda: not client.snapshot()['busy'])
        assert client.pop_move_results() == [dict(
            request_id='cancel-unsent', ok=False, reason='hand operation canceled')]
    finally:
        conn.close()
        client.close()
        server.close()
    print("PASS actual TCP force framing, freshness, reconnect, GSTOP and post-stop barrier")


if __name__ == "__main__":
    check_parameters()
    check_active_stop()
    check_gated_grasp()
    check_socket()
