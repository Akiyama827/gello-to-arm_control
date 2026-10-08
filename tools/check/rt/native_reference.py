"""Native selection/dynamic-envelope fake-only E2E; pass reference_runtime_selfcheck."""
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
from arm_control.plants.remote_rt.client import RtBackend, RtConfig, RtLinkError


def unused_port(kind):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@contextmanager
def fake(binary):
    udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    with tempfile.TemporaryFile(mode='w+') as log:
        proc = subprocess.Popen([binary, str(udp), str(tcp)], stdout=log, stderr=log)
        rt = RtBackend(RtConfig(host='127.0.0.1', udp_port=udp, tcp_port=tcp,
                                instance_id='fake'), ['j0', 'j1', 'j2'])
        try:
            time.sleep(.2)
            rt.open()
            assert rt.negotiate_reference('software-fixture-model')
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
            assert proc.returncode == 0


class Pump:
    def __init__(self, rt):
        self.rt, self.stop = rt, threading.Event()
        self.pause = False
        self.phase_zero = False
        self.wrong_revision = False
        self.errors = []
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        r = self.rt
        while not self.stop.is_set():
            if not self.pause:
                state, rx = r.latest_state()
                command = dict(position=np.zeros(3), velocity=np.zeros(3), torque=np.zeros(3),
                               kp=np.ones(3)*2, kd=np.ones(3)*.1)
                try:
                    ref = state and state.reference
                    if ref and ref['reference_id']:
                        r.apply_reference_envelope(
                            command, reference_id=ref['reference_id'],
                            revision=ref['revision'] + int(self.wrong_revision),
                            phase=0 if self.phase_zero else ref['phase'], sample_received_at=rx,
                            valid_for_s=.1, phase_window_s=.1)
                    else:
                        r.apply_command(command)
                except (ValueError, OSError) as exc:
                    self.errors.append(str(exc))
            self.stop.wait(.005)

    def close(self):
        self.stop.set()
        self.thread.join(2)
        assert not self.errors, self.errors


def wait_state(rt, predicate, timeout=2):
    until = time.monotonic()+timeout
    while time.monotonic()<until:
        state, _ = rt.latest_state()
        if state and state.reference and predicate(state):
            return state
        time.sleep(.005)
    raise AssertionError(('state timeout', state))


def curve(end=.3):
    return JointTrajectory([0, end], np.array([[0]*3, [.1]*3]), np.zeros((2,3)))


def expect_refusal(fn):
    try:
        fn()
    except RtLinkError:
        return
    raise AssertionError('reference operation unexpectedly accepted')


def main(binary):
    with fake(binary) as rt:
        pump = Pump(rt)
        try:
            # Terminal end retains the planned endpoint, including nonzero
            # stored terminal velocity: execution policy selects zero qd/qdd.
            tr = curve()
            tr.velocities[-1] = .2
            u = rt.upload_reference(tr, reference_id=1)
            rt.commit_reference(u, reference_id=1)
            rt.commit_reference(u, reference_id=1)  # exact value retry, new correlation
            end = wait_state(rt, lambda s: s.reference['status']==p.REFERENCE_END)
            assert end.armed and not end.faulted and end.q_cmd==[.1]*3
            assert end.reference['qd_ref']==end.reference['qdd_ref']==[0.]*3
            assert end.last_cmd_seq and end.reference['envelope_seq']
            rt.cancel_reference(1,1)
            cancelled = wait_state(rt, lambda s: s.reference['status']==p.REFERENCE_CANCELLED)
            held = cancelled.q_cmd.copy()
            rt.cancel_reference(1,1)
            time.sleep(.025)
            assert rt.latest_state()[0].q_cmd==held
            expect_refusal(lambda: rt.commit_reference(u, reference_id=1))
            # Nonterminal exhaustion is distinct from command loss and stays
            # retired even while matching fresh envelope packets keep arriving.
            u = rt.upload_reference(curve(), reference_id=2, terminal=False)
            rt.commit_reference(u, reference_id=2)
            under = wait_state(rt, lambda s: s.reference['status']==p.REFERENCE_UNDERFLOW)
            assert under.armed and under.holding and not under.faulted
            held = under.q_cmd.copy()
            time.sleep(.08)
            assert rt.latest_state()[0].q_cmd==held and not rt.latest_state()[0].faulted
            expect_refusal(lambda: rt.commit_reference(u, reference_id=2, revision=2,
                                                       expected_revision=1, protected_until=.3,
                                                       latest_phase=.29))
            # Reference underflow must not suspend the independent command timeout.
            pump.pause=True
            timeout = wait_state(rt, lambda s: s.faulted)
            assert timeout.fault_code==p.FAULT_CMD_LOST
            assert timeout.reference['status']==p.REFERENCE_UNDERFLOW
        finally:
            pump.close()
    # CTRL-034 F2: an ORDINARY PC cancel keeps the pre-(a) behaviour. The
    # retired selection retains its id and phase, so the envelopes the PC keeps
    # streaming stay fresh and hold the command deadman: the arm holds and never
    # latches CMD_LOST. Only a reference the RT retired on envelope EXPIRY loses
    # liveness (covered by the invalid cases below). The reviewer's probe:
    # cancel a RUNNING reference while the PC keeps streaming -> no fault on
    # 0755f01, CMD_LOST at fault_ms when CANCELLED was excluded from freshness.
    with fake(binary) as rt:
        pump = Pump(rt)
        try:
            u = rt.upload_reference(curve(3.), reference_id=1, terminal=False)
            rt.commit_reference(u, reference_id=1)
            wait_state(rt, lambda s: s.reference['phase']>.15 and not s.holding)
            ref = rt.latest_state()[0].reference
            rt.cancel_reference(ref['reference_id'], ref['revision'])
            wait_state(rt, lambda s: s.reference['status']==p.REFERENCE_CANCELLED)
            time.sleep(1.2)  # deployed default hold/fault are 100/1000 ms
            state, _ = rt.latest_state()
            assert state.armed and not state.faulted, (state.faulted, state.fault_code)
            assert state.holding, 'cancel must keep the measured compensated hold'
            assert state.reference['reference_id']==1
        finally:
            pump.close()
    for invalid in ('phase_zero', 'wrong_revision', 'pause'):
        with fake(binary) as rt:
            pump = Pump(rt)
            try:
                u = rt.upload_reference(curve(2.), reference_id=1)
                rt.commit_reference(u, reference_id=1)
                wait_state(rt, lambda s: s.reference['phase']>.15 and not s.holding)
                setattr(pump, invalid, True)
                held = wait_state(rt, lambda s: s.holding)
                # CTRL-034 decision (a): an expired envelope RETIRES the
                # reference; the arm holds and only a new reference resumes it.
                assert not held.faulted and held.reference['status']==p.REFERENCE_CANCELLED
                fault = wait_state(rt, lambda s: s.faulted)
                assert fault.fault_code==p.FAULT_CMD_LOST
            finally:
                pump.close()
    print('PASS native fake E2E: terminal, exact commit/cancel retry, retired late commit, '
          'underflow independent of command timeout, plain cancel keeps liveness, '
          'stale phase/revision/wall age')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
