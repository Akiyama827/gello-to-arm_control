"""The simulated Hand serves the real hand_bridge protocol: admission, replies and finger semantics, against a fake plant.

A fake two-finger plant (fingers servo toward their targets, blocked by an object of a given width, loaded while blocked)
steps at 250 Hz and feeds ``SimHandBridge.on_sample`` exactly as ``ScenePlant`` does. The protocol is driven two ways: a raw
socket (every reply checked) and the project's own client (``BridgeClient`` + ``HandGraspFsm``).
"""
from __future__ import annotations

import socket
import threading
import time

import numpy as np

from arm_control.end_effectors.franka_adapter import BridgeClient, submit_grasp_request
from arm_control.end_effectors.franka_hand import HandGraspFsm
from arm_control.end_effectors.sim_hand_bridge import SimHandBridge, parse_command


class FakeFingers:
    def __init__(self) -> None:
        self.q = np.full(2, .0375)
        self.target = self.q.copy()
        self.force = 0.
        self.object_w: float | None = None
        self.lock = threading.Lock()

    def apply(self, targets, force_n):
        with self.lock:
            self.target, self.force = np.asarray(targets, float), float(force_n)

    def step(self, dt):
        with self.lock:
            floor = 0. if self.object_w is None else self.object_w / 2
            self.q += np.clip(self.target - self.q, -.3 * dt, .3 * dt)   # a fast servo
            blocked = self.q < floor
            self.q = np.maximum(self.q, floor)
            pushing = blocked & (self.target < floor)
            effort = np.where(pushing, self.force / 2 if self.force else 6., 0.)
            return self.q.copy(), effort


PAUSED = threading.Event()   # set: the plant sends no samples (a stalled device)


def plant(bridge, fingers, stop):
    period, last = .004, time.monotonic()
    while not stop.is_set():
        now = time.monotonic()
        if PAUSED.is_set():
            last = now
            time.sleep(period)
            continue
        positions, efforts = fingers.step(now - last)
        last = now
        command = bridge.on_sample(positions, efforts, now)
        if command is not None:
            fingers.apply(*command)
        time.sleep(period)


def wait(predicate, timeout=5., what=''):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError(f'timed out waiting for {what}')


class Raw:
    """A raw protocol client: send lines, collect replies."""

    def __init__(self, port):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=2)
        self.sock.settimeout(.05)
        self.lines: list[str] = []

    def pump(self):
        """Drain everything waiting (the Hand streams STATE/META at 10 Hz; a single read lags behind)."""
        data = b''
        while True:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            data += chunk
        self._partial = getattr(self, '_partial', '') + data.decode()
        *complete, self._partial = self._partial.split('\n')
        self.lines += [line for line in complete if line]

    def meta(self):
        """The newest META: (epoch, available, busy, measured, seq)."""
        self.pump()
        metas = [line.split() for line in self.lines if line.startswith('META 2 ')]
        return tuple(int(v) for v in metas[-1][2:]) if metas else None

    def send(self, line):
        self.sock.sendall((line + '\n').encode())

    def reply(self, prefix):
        try:
            return wait(lambda: self.pump() or next((ln for ln in self.lines if ln.startswith(prefix)), None), 3, prefix)
        except AssertionError:
            raise AssertionError(f'no {prefix}; replies: {[ln for ln in self.lines if not ln.startswith(("STATE", "META", "CAPS"))]}')

    def take(self, prefix):
        line = self.reply(prefix)
        self.lines.remove(line)
        return line

    def idle_epoch(self):
        """The epoch once the Hand reports available, not busy and measured."""
        return wait(lambda: (m := self.meta()) and m[1:4] == (1, 0, 1) and m[0], 5, 'an idle measured Hand')

    def close(self):
        self.sock.close()


def parser_cases():
    assert parse_command('CMD 3 MOVE 0.04 0.05')['w'] == .04
    assert parse_command('CMD 3 HOME')['kind'] == 'HOME' and parse_command('CMD 3 HOME 1') is None
    assert parse_command('CMD 3 GRASP 0.04 0.05 55 0.02 0.02')['f'] == 55
    for bad in ('CMD x MOVE 0.04 0.05', 'CMD 3 MOVE -0.000002 0.05', 'CMD 3 MOVE 0.080002 0.05', 'CMD 3 MOVE 0.04 0',
                'CMD 3 GRASP 0.04 0.05 29 0.02 0.02', 'CMD 3 GRASP 0.04 0.05 71 0.02 0.02', 'CMD 3 MOVE 0.04 0.05 1',
                'MOVE 0.04 0.05', 'CMD 3 JUMP', 'CMD 3 GRASP 0.04 0.05 55 0.02 nan'):
        assert parse_command(bad) is None, bad


def raw_cases(bridge, fingers):
    raw = Raw(bridge.addr[1])
    try:
        epoch = raw.idle_epoch()
        raw.send('CMD 1 MOVE 0.07 0.05' if epoch != 1 else 'CMD 0 MOVE 0.07 0.05')
        assert 'stale_epoch' in raw.take('REJECT'), raw.lines
        raw.send('BOGUS')
        assert 'invalid_command' in raw.take('REJECT')
        # MOVE: accepted, busy while it runs, a second command is refused busy, DONE ok at the goal
        raw.send(f'CMD {epoch} MOVE 0.06 0.05')
        assert raw.take('ACCEPT') == f'ACCEPT {epoch} accepted'
        raw.send(f'CMD {epoch + 1} MOVE 0.07 0.05')
        assert 'busy' in raw.take('REJECT')
        assert raw.take('DONE') == f'DONE {epoch} 1', raw.lines
        assert abs(sum(fingers.q) - .06) < .001
        time.sleep(.2)
        raw.pump()
        assert not any(ln.startswith('GDONE') for ln in raw.lines), 'a MOVE reported a grasp verdict'
        # no measured sample for longer than the freshness bound: refused stale_state, and META says not measured
        epoch = raw.idle_epoch()
        PAUSED.set()
        time.sleep(.8)
        raw.send(f'CMD {epoch} MOVE 0.07 0.05')
        assert 'stale_state' in raw.take('REJECT'), raw.lines
        assert raw.meta()[3] == 0, raw.meta()
        PAUSED.clear()
        # GRASP on an object: GDONE 1, held under force, STATE grasped
        fingers.object_w = .057
        epoch = raw.idle_epoch()
        raw.send(f'CMD {epoch} GRASP 0.045 0.05 40 0.02 0.02')
        raw.take('ACCEPT')
        assert raw.take('GDONE') == 'GDONE 1' and raw.take('DONE') == f'DONE {epoch} 1'
        assert fingers.force == 40., fingers.force
        wait(lambda: raw.pump() or any(ln == 'STATE 0.05700 1' for ln in raw.lines), 2, 'STATE grasped')
        # a disconnect with nothing committed keeps the grasp holding
        raw.close()
        raw = Raw(bridge.addr[1])
        epoch = raw.idle_epoch()
        assert fingers.force == 40. and bridge.real_grasped, 'a disconnect let go of a completed grasp'
        # GSTOP while holding: STOPPING, the grasp lets go
        raw.send('GSTOP')
        stop_epoch = int(raw.take('STOPPING').split()[1])
        assert stop_epoch > epoch
        wait(lambda: fingers.force == 0. and not bridge.real_grasped, 2, 'the grasp released by GSTOP')
        # GRASP on nothing: GDONE 0 (an empty close is not blocked short of the command)
        fingers.object_w = None
        epoch = raw.idle_epoch()
        raw.send(f'CMD {epoch} MOVE 0.075 0.1')
        raw.take('ACCEPT')
        raw.take('DONE')
        epoch = raw.idle_epoch()
        raw.send(f'CMD {epoch} GRASP 0.045 0.05 40 0.02 0.02')
        raw.take('ACCEPT')
        assert raw.take('GDONE') == 'GDONE 0' and raw.take('DONE') == f'DONE {epoch} 0'
        # GSTOP mid-move: the action ends where the fingers are and reports nothing
        epoch = raw.idle_epoch()
        raw.send(f'CMD {epoch} MOVE 0.0 0.02')
        raw.take('ACCEPT')
        time.sleep(.3)
        raw.send('GSTOP')
        raw.take('STOPPING')
        time.sleep(.4)
        raw.pump()
        assert not any(ln.startswith('DONE') for ln in raw.lines), raw.lines
        width = sum(fingers.q)
        assert .02 < width < .045, width
        time.sleep(.3)
        assert abs(sum(fingers.q) - width) < .001, 'the fingers kept moving after GSTOP'
        # a disconnect DURING a committed move stops it
        epoch = raw.idle_epoch()
        raw.send(f'CMD {epoch} MOVE 0.08 0.02')
        raw.take('ACCEPT')
        time.sleep(.2)
        raw.close()
        time.sleep(.3)
        width = sum(fingers.q)
        time.sleep(.3)
        assert abs(sum(fingers.q) - width) < .001 and width < .075, 'a disconnect left a committed move running'
        # a fresh sample is required after an action: the next command is admitted only once META says measured
        raw = Raw(bridge.addr[1])
        raw.idle_epoch()
    finally:
        raw.close()


def client_cases(bridge, fingers):
    """The project's own client and grasp FSM, unchanged, against the simulated Hand."""
    fingers.object_w = .057
    client = BridgeClient('127.0.0.1', bridge.addr[1])
    client.start()
    try:
        wait(lambda: client.snapshot()['measured'], 5, 'the client sees a measured Hand')
        fsm = HandGraspFsm({'grasp_width_m': .045, 'speed_mps': .05, 'force_n': 40,
                            'epsilon_inner_m': .02, 'epsilon_outer_m': .02, 'open_width_m': .075})
        assert submit_grasp_request(client, fsm, {'request_id': 'g1', 'target_id': 'module'}, time.monotonic()) is None
        result = wait(lambda: fsm.poll(client.snapshot(), client.gdone_count, client.gdone_ok, time.monotonic()),
                      10, 'the grasp verdict')
        assert result['ok'] and result['request_id'] == 'g1', result
        wait(lambda: client.snapshot()['is_grasped'], 2, 'the client sees the Hand grasping')
    finally:
        client.close()


def main():
    parser_cases()
    fingers = FakeFingers()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    bridge = SimHandBridge(port, default_force_n=45.)
    stop = threading.Event()
    bridge.start()
    thread = threading.Thread(target=plant, args=(bridge, fingers, stop), daemon=True)
    thread.start()
    try:
        raw_cases(bridge, fingers)
        client_cases(bridge, fingers)
    finally:
        stop.set()
        thread.join(2)
        bridge.close()
    print('sim Hand bridge: PASS (parser bounds, epochs and busy/stale/invalid rejects, MOVE/GRASP DONE and GDONE, held grasp '
          'survives a disconnect, GSTOP releases and stops mid-move silently, disconnect stops committed work, '
          'BridgeClient + HandGraspFsm grasp)')


if __name__ == '__main__':
    main()
