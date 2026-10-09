"""The simulated Hand serves the real hand_bridge protocol: admission, replies and finger semantics, against a fake plant.

A fake two-finger plant (fingers servo toward their targets, blocked by an object of a given width, loaded while blocked)
steps at 250 Hz and feeds ``SimHandBridge.on_sample`` exactly as ``SimLink`` does (for every sim plant). The protocol is driven two ways: a raw
socket (every reply checked) and the project's own client (``BridgeClient`` + ``HandGraspFsm``).
"""
from __future__ import annotations

import socket
import threading
import time

import numpy as np

from arm_control.end_effectors.franka_adapter import BridgeClient, submit_grasp_request
from arm_control.end_effectors.franka_hand import FRANKA_HAND, HandGraspFsm, HandLimits, parse_limits
from arm_control.end_effectors.hand_move import HandMove
from arm_control.end_effectors.sim_hand_bridge import SimHandBridge, parse_command


class FakeFingers:
    def __init__(self) -> None:
        self.q = np.full(2, .0375)
        self.target = self.q.copy()
        self.force = 0.
        self.object_w: float | None = None
        self.unloaded = False            # the object rides without loading the servos (a dropping module)
        self.report: float | None = None  # a forced per-finger reading (the envelope cases)
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
            effort = np.where(pushing & (not self.unloaded), self.force / 2 if self.force else 6., 0.)
            return (self.q.copy() if self.report is None else np.full(2, self.report)), effort


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
            raise AssertionError(f'no {prefix}; replies: {[ln for ln in self.lines if not ln.startswith(("STATE", "META", "CAPS", "LIMITS"))]}')

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
    # a stated envelope replaces the Franka Hand's: a wider stroke admits wider widths; force_max 0 admits no GRASP
    wide = HandLimits(.0878, .2, 0., 0.)
    assert parse_command('CMD 3 MOVE 0.085 0.15', wide)['w'] == .085 and parse_command('CMD 3 MOVE 0.085 0.05') is None
    assert parse_command('CMD 3 GRASP 0.04 0.05 55 0.02 0.02', wide) is None
    assert parse_command('CMD 3 MOVE 0.04 0.21', wide) is None
    for bad in ('LIMITS 0.08 0.1 30', 'LIMITS 0 0.1 30 70', 'LIMITS 0.08 0.1 70 30', 'LIMITS 0.08 nan 30 70',
                'LIMITS 0.08 0.1 -1 70'):
        try:
            parse_limits(bad.split())
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad}')


def client_limits_cases():
    """The client validates against the envelope the bridge states (absent: the Franka Hand's), and refuses a force grasp
    a gripper without one cannot do -- before anything reaches the wire."""
    def fed(*lines):
        """A never-started client, connected in name, fed these lines (then a fresh measured Hand)."""
        c = BridgeClient('127.0.0.1', 1)
        c._connected = True
        for line in (*lines, b'STATE 0.050 0', b'CAPS active_stop', b'META 2 3 1 0 1 5'):
            c._receive(line)
        c._sample_at = time.monotonic()
        return c

    no_force = b'LIMITS 0.08780 0.20000 0.000 0.000'
    # no force grasp refuses a close ONLY: position moves, HOME and release still go out
    assert fed(no_force).snapshot()['force_grasp'], 'force_grasp is the epoch protocol, not the gripper'
    assert fed(no_force).send_line('MOVE 0.08500 0.150') and fed(no_force).send_line('HOME')
    move = HandMove()
    move.limits = HandLimits(.0878, .2, 0., 0.)
    assert move.start(dict(request_id='m', width_m=.085, speed_mps=.15), fed(no_force).snapshot(), 0.).startswith('MOVE')
    c = fed(no_force)
    fsm = HandGraspFsm({'grasp_width_m': .045, 'speed_mps': .05, 'force_n': 40, 'open_width_m': .075})
    fsm.limits = c.limits
    released = submit_grasp_request(c, fsm, {'request_id': 'r', 'target_id': 'm', 'mode': 'release'}, time.monotonic())
    assert released and released['ok'] and c._inflight is not None, (released, c._inflight)
    assert fsm.parameters['force_n'] == 0.
    assert fed().send_line('MOVE 0.06000 0.100') and not fed().send_line('MOVE 0.08500 0.100')  # absent: the Franka Hand

    client = BridgeClient('127.0.0.1', 1)  # never started: lines are fed to the receiver directly
    assert client.limits == FRANKA_HAND
    try:
        client._receive(b'STATE 0.085 0')
    except ValueError:
        pass
    else:
        raise AssertionError('a width past the Franka stroke was accepted before any LIMITS')
    client._receive(b'LIMITS 0.08780 0.20000 0.000 0.000')
    client._receive(b'STATE 0.085 0')
    assert client.state['width'] == .085 and not client.snapshot()['force_grasp']
    assert client.snapshot()['limits'] == HandLimits(.0878, .2, 0., 0.)._asdict()
    fsm = HandGraspFsm({'grasp_width_m': .045, 'speed_mps': .05, 'force_n': 40})
    fsm.limits = client.limits
    result = submit_grasp_request(client, fsm, {'request_id': 'g', 'target_id': 'm'}, time.monotonic())
    assert result and not result['ok'] and 'no calibrated force grasp' in result['reason'], result
    assert not client.send_line('GRASP 0.04 0.05 40 0.02 0.02') and client._inflight is None


def raw_cases(bridge, fingers):
    raw = Raw(bridge.addr[1])
    try:
        epoch = raw.idle_epoch()
        raw.send('CMD 1 MOVE 0.07 0.05' if epoch != 1 else 'CMD 0 MOVE 0.07 0.05')
        assert 'stale_epoch' in raw.take('REJECT'), raw.lines
        raw.send('BOGUS')
        assert raw.take('REJECT') == 'REJECT 0 invalid_command'
        raw.send('CMD 7 BOGUS')
        assert raw.take('REJECT') == 'REJECT 7 invalid_command'
        raw.send('CMD \u00b2 HOME')        # a Unicode digit: refused, and the server lives on
        assert raw.take('REJECT') == 'REJECT 0 invalid_command'
        raw.send('CMD 1 MOVE \u0660.\u0660\u0664 0.05')   # Unicode digits in a number
        assert raw.take('REJECT') == 'REJECT 1 invalid_command'
        assert raw.idle_epoch()
        # MOVE: accepted, busy while it runs, a second command is refused busy, DONE ok at the goal
        raw.send(f'CMD {epoch} MOVE 0.06 0.05')
        assert raw.take('ACCEPT') == f'ACCEPT {epoch} accepted'
        raw.send(f'CMD {epoch + 1} MOVE 0.07 0.05')
        assert 'busy' in raw.take('REJECT')
        time.sleep(.15)
        assert raw.meta()[2] == 1 and raw.meta()[3] == 0, f'META while acting: {raw.meta()}'
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
        # the hold breaks when the servos unload (the module slipping) and when the object is gone
        fingers.unloaded = True
        wait(lambda: not bridge.real_grasped, 2, 'grasped dropping with the servo load')
        fingers.unloaded = False
        wait(lambda: bridge.real_grasped, 2, 'grasped back with the load')
        fingers.object_w = .02           # still loaded, but closed past the band [0.025, 0.065]
        wait(lambda: abs(sum(fingers.q) - .02) < 1e-4, 2, 'the fingers settling on the smaller object')
        time.sleep(.1)
        for _ in range(5):               # loaded and settled: only the band can say "not grasped"
            assert not bridge.real_grasped, 'a loaded grasp outside the epsilon band reported grasped'
            time.sleep(.05)
        fingers.object_w = .057
        wait(lambda: bridge.real_grasped, 2, 'grasped back inside the band')
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
        # a MOVE blocked by the object: DONE 0, and the fingers stop where they are
        epoch = raw.idle_epoch()
        raw.send(f'CMD {epoch} MOVE 0.052 0.1')   # 5 mm short of the object: blocked
        raw.take('ACCEPT')
        assert raw.take('DONE') == f'DONE {epoch} 0', raw.lines
        assert abs(bridge._fingers.cmd_w - sum(fingers.q)) < .001, 'a failed MOVE keeps squeezing'
        # HOME opens fully
        epoch = raw.idle_epoch()
        raw.send(f'CMD {epoch} HOME')
        raw.take('ACCEPT')
        assert raw.take('DONE') == f'DONE {epoch} 1' and abs(sum(fingers.q) - .08) < .001, sum(fingers.q)
        # the sample envelope: an overshoot is clamped, a reading past it is no STATE and not measured
        fingers.report = .04025          # 80.5 mm
        time.sleep(.2)
        raw.pump()
        raw.lines.clear()                # HOME's own 'STATE 0.08000' must not answer for the clamp
        wait(lambda: raw.pump() or 'STATE 0.08000 0' in raw.lines, 2, 'an overshoot clamped to 0.08')
        fingers.report = -.0001          # -0.2 mm
        time.sleep(.3)
        raw.pump()                       # drain what was sent before the bad reading
        raw.lines.clear()
        time.sleep(.35)
        raw.pump()
        assert not any(ln.startswith('STATE') for ln in raw.lines) and raw.meta()[3] == 0, raw.lines
        fingers.report = None
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
        # an oversized line disconnects
        raw.send('X' * 255)              # one 256-byte read holding a complete line at the 255-byte limit
        wait(lambda: not bridge._adm.client, 2, 'the oversized line disconnecting')
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


def reconnect_cases():
    """A reconnect forgets the previous bridge's envelope: one that states none is the Franka Hand."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    wide = SimHandBridge(port, limits=HandLimits(.0878, .2, 0., 0.))
    wide.start()
    client = BridgeClient('127.0.0.1', port)
    client.start()
    try:
        wait(lambda: client.limits.width_max_m == .0878, 3, 'the stated envelope')
        # the reject reason, as the C++ serve(): GRASP on a no-force gripper, malformed or not
        assert wide._handle('CMD 5 GRASP 0.04 0.05 40 0.02 0.02') == 'REJECT 5 uncalibrated_force\n'
        assert wide._handle('CMD 5 GRASP 0.04') == 'REJECT 5 uncalibrated_force\n'
        assert SimHandBridge(0)._handle('CMD 5 GRASP 0.04') == 'REJECT 5 invalid_command\n'
        wide.close()
        wait(lambda: not client.snapshot()['available'], 3, 'the client sees the bridge gone')
        old = socket.socket()  # an old bridge: META and CAPS, no LIMITS
        old.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        old.bind(('127.0.0.1', port))
        old.listen(1)
        old.settimeout(5)
        conn, _ = old.accept()
        conn.sendall(b'CAPS active_stop\nMETA 2 1 1 0 0 0\n')
        wait(lambda: client.snapshot()['active_stop'], 3, 'the client on the old bridge')
        assert client.limits == FRANKA_HAND, client.limits
        conn.close()
        old.close()
    finally:
        client.close()


def main():
    parser_cases()
    client_limits_cases()
    fingers = FakeFingers()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    bridge = SimHandBridge(port)
    stop = threading.Event()
    bridge.start()
    # before any sample: META only, no STATE (the C++ withholds it until a usable sample)
    first = Raw(port)
    time.sleep(.35)
    first.pump()
    assert first.meta() is not None and not any(ln.startswith('STATE') for ln in first.lines), first.lines
    assert first.lines[0] == 'LIMITS 0.08000 0.10000 30.000 70.000', first.lines  # the envelope leads every report
    first.close()
    thread = threading.Thread(target=plant, args=(bridge, fingers, stop), daemon=True)
    thread.start()
    try:
        report = Raw(port)
        wait(lambda: report.pump() or any(ln.startswith('STATE') for ln in report.lines), 2, 'a report with STATE')
        report.close()
        at = next(i for i, ln in enumerate(report.lines) if ln.startswith('STATE'))
        assert report.lines[at - 1].startswith('LIMITS '), report.lines  # a client validates STATE against it
        raw_cases(bridge, fingers)
        client_cases(bridge, fingers)
        reconnect_cases()
    finally:
        stop.set()
        thread.join(2)
        bridge.close()
    print('sim Hand bridge: PASS (parser bounds, stated LIMITS (wire, client validation, no-force-grasp refusal), epochs and busy/stale/invalid rejects, MOVE/GRASP DONE and GDONE, held grasp '
          'survives a disconnect, GSTOP releases and stops mid-move silently, disconnect stops committed work, '
          'BridgeClient + HandGraspFsm grasp)')


if __name__ == '__main__':
    main()
