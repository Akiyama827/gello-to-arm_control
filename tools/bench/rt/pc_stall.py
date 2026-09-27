"""Offline counterpart of the CTRL-034 PC-stall injection (fake backend only).

Pre-registers what the hardware trial must show: with valid_for/phase_window at
the 50 ms ceiling, a 30 ms PC send-pause never expires the reference envelope, a
60/150 ms pause expires it at ~50 ms and the arm holds its measured position
with zero feedforward until a fresh envelope arrives, and none of them latches
CMD_LOST because fault_ms is 1000 ms.

Runs one fake `arm_rt_server` with --enable-reference, --hold-ms 100 and
--fault-ms 1000 (the deployed defaults; the unit passes neither flag) and
--state-hz 1000 so the hold duration is measurable to ~1 ms. No hardware, no
deployment, no service action.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as p
from arm_control.plants.remote_rt.client import RtBackend, RtConfig

STALLS_MS = (30, 60, 150)
FRESHNESS_MS = 50      # min(hold_ms, valid_for_us); valid_for/phase_window = 50 ms
HOLD_MS, FAULT_MS = 100, 1000
POLL_S = 0.001
# Expected hold duration is stall - FRESHNESS_MS; allow for one pump period and
# one state frame at each edge rather than pretending to sub-ms precision.
SLACK_MS = 25


def unused_port(kind):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        out = pred()
        if out:
            return out
        time.sleep(0.005)
    raise AssertionError('timed out')


class Pump:
    """Native v7 envelopes only; `pause` is the injected PC stall."""

    def __init__(self, rt):
        self.rt, self.stop = rt, threading.Event()
        self.pause = False
        self.errors = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        command = dict(position=np.zeros(3), velocity=np.zeros(3),
                       torque=np.zeros(3), kp=np.ones(3) * 2, kd=np.ones(3) * .1)
        while not self.stop.is_set():
            if not self.pause:
                state, rx = self.rt.latest_state()
                try:
                    if state and state.reference and state.reference['reference_id']:
                        self.rt.apply_reference_envelope(
                            command, reference_id=state.reference['reference_id'],
                            revision=state.reference['revision'],
                            phase=state.reference['phase'], sample_received_at=rx,
                            valid_for_s=.05, phase_window_s=.05)
                except (ValueError, OSError) as exc:
                    self.errors.append(str(exc))
            self.stop.wait(.005)

    def close(self):
        self.stop.set()
        self.thread.join(2)
        assert not self.errors, self.errors


@contextmanager
def server(binary):
    udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    argv = [binary, '--backend', 'fake', '--n', '3', '--instance-id', 'fake',
            '--joint-names', 'j0,j1,j2', '--udp-port', str(udp), '--tcp-port', str(tcp),
            '--bind', '127.0.0.1', '--hold-ms', str(HOLD_MS), '--fault-ms', str(FAULT_MS),
            '--state-hz', '1000', '--enable-reference']
    with tempfile.TemporaryFile(mode='w+') as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=log)
        rt = RtBackend(RtConfig(host='127.0.0.1', udp_port=udp, tcp_port=tcp,
                                instance_id='fake'), ['j0', 'j1', 'j2'])
        try:
            time.sleep(.2)
            rt.open()
            assert rt.negotiate_reference('pc-stall-fixture-model')
            rt.enable_all()
            yield rt
        except BaseException:
            log.seek(0)
            print(log.read())
            raise
        finally:
            rt.close()
            proc.terminate()
            proc.wait(3)


def commit_running(rt, reference_id):
    load = rt.upload_reference(
        JointTrajectory([0, 3.0], np.array([[0] * 3, [.1] * 3]), np.zeros((2, 3))),
        reference_id=reference_id, terminal=False)
    rt.commit_reference(load, reference_id=reference_id)
    return wait_for(lambda: (rt.latest_state()[0] or None)
                    if (rt.latest_state()[0] and rt.latest_state()[0].reference
                        and rt.latest_state()[0].reference['status'] == p.REFERENCE_RUNNING)
                    else None)


def inject(rt, pump, stall_ms):
    """Pause the PC stream for `stall_ms` and measure the hold it causes."""
    assert not rt.latest_state()[0].holding, 'already holding before the stall'
    pump.pause = True
    t0 = time.monotonic()
    released = False
    first = last = None
    faulted = False
    while True:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if not released and elapsed_ms >= stall_ms:
            pump.pause = False
            released = True
        state = rt.latest_state()[0]
        if state is not None:
            faulted = faulted or state.faulted
            if state.holding:
                if first is None:
                    first = elapsed_ms
                last = elapsed_ms
        if released and elapsed_ms >= stall_ms + 4 * SLACK_MS:
            break
        time.sleep(POLL_S)
    return first, last, faulted


def main(binary):
    expected = {30: (False, 0.0), 60: (True, 60.0 - FRESHNESS_MS),
                150: (True, 150.0 - FRESHNESS_MS)}
    with server(binary) as rt:
        pump = Pump(rt)
        try:
            commit_running(rt, 1)
            wait_for(lambda: not rt.latest_state()[0].holding)
            rows = []
            for stall in STALLS_MS:
                first, last, faulted = inject(rt, pump, stall)
                want_hold, want_ms = expected[stall]
                state = rt.latest_state()[0]
                assert state.reference['status'] == p.REFERENCE_RUNNING, state.reference
                assert not faulted, f'{stall} ms stall latched CMD_LOST (fault_ms={FAULT_MS})'
                if want_hold:
                    assert first is not None, f'{stall} ms stall produced no hold'
                    held_ms = last - first
                    assert abs(first - FRESHNESS_MS) <= SLACK_MS, \
                        f'hold began at {first:.1f} ms, expected ~{FRESHNESS_MS}'
                    assert abs(held_ms - want_ms) <= 2 * SLACK_MS, \
                        f'{stall} ms stall held {held_ms:.1f} ms, expected ~{want_ms}'
                    rows.append((stall, f'hold {first:.0f}..{last:.0f} ms'))
                else:
                    assert first is None, \
                        f'{stall} ms stall (< {FRESHNESS_MS} ms ceiling) caused a hold'
                    rows.append((stall, 'no hold'))
                # Tracking must resume before the next stall.
                wait_for(lambda: not rt.latest_state()[0].holding)
            assert rt.latest_state()[0].reference['status'] == p.REFERENCE_RUNNING
            assert not rt.latest_state()[0].faulted
            for stall, what in rows:
                print(f'  {stall:>3} ms -> {what}')
        finally:
            pump.close()
    print(f'pc stall: {len(STALLS_MS)} injections PASS '
          f'(ceiling {FRESHNESS_MS} ms, hold_ms {HOLD_MS}, fault_ms {FAULT_MS})')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
