"""Armed-only mode-selection E2E against the fake RT runtime (offline).

Drives one fake `arm_rt_server` twice: with and without --enable-reference, so
the only difference is the flag. No hardware, no deployment.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
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

TOKEN = 'software-fixture-model'


def unused_port(kind):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_for(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        out = pred()
        if out:
            return out
        time.sleep(0.01)
    raise AssertionError('timed out')


class Pump:
    """Streams v5 joint commands while armed."""

    def __init__(self, rt, position):
        self.rt, self.position, self.stop = rt, np.asarray(position, float), threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        command = dict(position=self.position, velocity=np.ones(3),
                       torque=np.zeros(3), kp=np.ones(3) * 60, kd=np.ones(3) * 10)
        while not self.stop.is_set():
            try:
                self.rt.apply_command(command)
            except (ValueError, OSError, RtLinkError) as exc:
                self.errors.append(str(exc))
            self.stop.wait(0.01)

    def close(self):
        self.stop.set()
        self.thread.join(2)


def tracked(rt, timeout=2.0):
    """True once the servo echoes the streamed target rather than the hold."""
    def check():
        # qd_cmd is the SELECTED servo velocity target: zero in every hold,
        # the command's own value while the stream is authority.
        state, _ = rt.latest_state()
        return state is not None and abs(state.qd_cmd[0]) > 0.5
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.01)
    return False


@contextmanager
def serve(binary, *, enable_reference):
    """One fake server; --enable-reference is the ONLY difference."""
    udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    argv = [binary, '--backend', 'fake', '--n', '3', '--instance-id', 'fake',
            '--joint-names', 'j0,j1,j2', '--udp-port', str(udp), '--tcp-port', str(tcp),
            '--bind', '127.0.0.1', '--hold-ms', '100', '--fault-ms', '1500']
    if enable_reference:
        argv.append('--enable-reference')
    with tempfile.TemporaryFile(mode='w+') as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=log)
        rt = RtBackend(RtConfig(host='127.0.0.1', udp_port=udp, tcp_port=tcp,
                                instance_id='fake'), ['j0', 'j1', 'j2'])
        try:
            time.sleep(.2)
            rt.open()
            yield rt
        except BaseException:
            log.seek(0)
            print(log.read())
            raise
        finally:
            rt.close()
            proc.terminate()
            proc.wait(3)


def raw_select(rt, **overrides):
    """Send a mode-select frame directly, bypassing the client's bookkeeping."""
    fields = dict(session_id=rt.session_id, generation=rt._mode_generation + 1,
                  expected_predecessor=rt._mode_generation, arm_epoch=rt._arm_epoch,
                  desired_mode=p.MODE_STREAMED)
    fields.update(overrides)
    packet = p.pack_mode_select(**fields)
    while not rt._mode_q.empty():
        rt._mode_q.get_nowait()
    rt._send_frame(packet)
    return rt._mode_q.get(timeout=3)


class NativePump:
    """Native envelopes, then a v5 stream once switched (mode-transition cases)."""

    def __init__(self, rt):
        self.rt, self.stop = rt, threading.Event()
        self.soft = None
        self.errors = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            state, rx = self.rt.latest_state()
            try:
                if self.soft is not None:
                    self.rt.apply_command(dict(
                        position=self.soft, velocity=np.ones(3), torque=np.zeros(3),
                        kp=np.ones(3) * 60, kd=np.ones(3) * 10))
                elif state and state.reference and state.reference['reference_id']:
                    self.rt.apply_reference_envelope(
                        dict(position=np.zeros(3), velocity=np.zeros(3),
                             torque=np.zeros(3), kp=np.ones(3) * 2, kd=np.ones(3) * .1),
                        reference_id=state.reference['reference_id'],
                        revision=state.reference['revision'],
                        phase=state.reference['phase'], sample_received_at=rx,
                        valid_for_s=.05, phase_window_s=.05)
            except (ValueError, OSError, RtLinkError) as exc:
                self.errors.append(str(exc))
            self.stop.wait(.005)

    def stream_soft(self, position):
        self.soft = np.asarray(position, float)

    def close(self):
        self.stop.set()
        self.thread.join(2)
        assert not self.errors, self.errors


def commit_running(rt, reference_id, *, end=3.0):
    """Commit a nonterminal reference and wait until RT reports it RUNNING."""
    load = rt.upload_reference(
        JointTrajectory([0, end], np.array([[0] * 3, [.1] * 3]), np.zeros((2, 3))),
        reference_id=reference_id, terminal=False)
    rt.commit_reference(load, reference_id=reference_id)
    state = wait_for(lambda: (rt.latest_state()[0] or None)
                     if (rt.latest_state()[0] and rt.latest_state()[0].reference
                         and rt.latest_state()[0].reference['reference_id'] == reference_id
                         and rt.latest_state()[0].reference['status'] == p.REFERENCE_RUNNING)
                     else None)
    assert state.selected_mode == p.MODE_REFERENCE, state.flags
    return state


def case_native_to_soft(binary):
    token = TOKEN
    # A. Native -> Soft AFTER a real trajectory. retire() keeps the reference
    # id, so the liveness predicate must not drop the NEW law's envelopes just
    # because one was used: the streamed arm must track and stay unfaulted past
    # fault_ms.
    with serve(binary, enable_reference=True) as rt:
        assert rt.negotiate_reference(token)
        rt.enable_all()
        pump = NativePump(rt)
        commit_running(rt, 1)
        reply = rt.select_mode(p.MODE_STREAMED)
        assert reply['result'] == p.MODE_SEL_OK and reply['selected_mode'] == p.MODE_STREAMED
        pump.stream_soft([1.0, 1.0, 1.0])
        time.sleep(2.0)                      # well past --fault-ms 1500
        state = rt.latest_state()[0]
        assert state.selected_mode == p.MODE_STREAMED, state.flags
        # The retained reference id must not drop the NEW law's envelopes: the
        # PC is streaming, so there is no CMD_LOST.
        assert not state.faulted, 'CMD_LOST after native->Soft with a retained reference id'
        assert tracked(rt), 'streamed law did not actuate after native->Soft'
        pump.close()



def case_order_with_running_reference(binary):
    token = TOKEN
    # B. Mis-ordered request while a reference is RUNNING: ORDER, and the
    # running reference is untouched because nothing is published for a
    # generation that is not the confirmed one plus one.
    with serve(binary, enable_reference=True) as rt:
        assert rt.negotiate_reference(token)
        rt.enable_all()
        pump = NativePump(rt)
        commit_running(rt, 1)
        reply = rt.select_mode(p.MODE_REFERENCE, model_token=token)   # gen 1 confirmed
        assert reply['result'] == p.MODE_SEL_OK and reply['generation'] == 1
        commit_running(rt, 2)                                         # RUNNING under gen 1
        reply = raw_select(rt, generation=7, expected_predecessor=6)
        assert reply['result'] == p.MODE_SEL_ORDER, reply
        state = rt.latest_state()[0]
        assert state.reference and state.reference['status'] == p.REFERENCE_RUNNING, state.reference
        pump.close()



def case_retry_with_running_reference(binary):
    token = TOKEN
    # C. Exact retry of the confirmed generation while a reference is RUNNING:
    # OK, and the running reference is untouched (no side effect on a retry).
    with serve(binary, enable_reference=True) as rt:
        assert rt.negotiate_reference(token)
        rt.enable_all()
        pump = NativePump(rt)
        commit_running(rt, 1)
        reply = rt.select_mode(p.MODE_REFERENCE, model_token=token)   # gen 1 confirmed
        assert reply['result'] == p.MODE_SEL_OK and reply['generation'] == 1
        commit_running(rt, 2)
        digest = hashlib.sha256(token.encode()).digest()
        reply = raw_select(rt, generation=1, expected_predecessor=0,
                           desired_mode=p.MODE_REFERENCE, model_digest=digest)
        assert reply['result'] == p.MODE_SEL_OK, reply
        state = rt.latest_state()[0]
        assert state.reference and state.reference['status'] == p.REFERENCE_RUNNING, state.reference
        pump.close()


def main(binary):
    token = TOKEN 
    with serve(binary, enable_reference=True) as rt:
        assert rt.identity.capabilities & p.CAP_MODE_SELECT, 'capability not advertised'
        assert rt.negotiate_reference(token)          # disarmed opt-in
        rt.enable_all()                               # ARM seeds generation 0

        # ARM seeds MODE_REFERENCE from the opt-in: Q1, and a v5 stream is not
        # authority in that law (the arm holds its measured position).
        state = wait_for(lambda: (rt.latest_state()[0] or None)
                         if (rt.latest_state()[0] and rt.latest_state()[0].armed
                             and rt.latest_state()[0].selected_mode == p.MODE_REFERENCE)
                         else None)
        assert state.armed and state.selected_mode == p.MODE_REFERENCE, state.flags
        pump = Pump(rt, [1.0, 1.0, 1.0])
        time.sleep(.4)
        assert not tracked(rt, timeout=.3), 'v5 command actuated in MODE_REFERENCE'
        assert pump.errors == [], pump.errors

        # Soft selection: confirmed, generation advances, and ONLY THEN does a
        # fresh v5 stream actuate.
        reply = rt.select_mode(p.MODE_STREAMED)
        assert reply['result'] == p.MODE_SEL_OK and reply['selected_mode'] == p.MODE_STREAMED
        assert reply['generation'] == 1
        assert tracked(rt), 'streamed command did not actuate after confirmation'

        # Same-mode selection is allowed and allocates a new generation.
        reply = rt.select_mode(p.MODE_STREAMED)
        assert reply['result'] == p.MODE_SEL_OK and reply['generation'] == 2
        assert rt._mode_generation == 2

        # Invalid Soft never retires execution: a wrong digest is IDENTITY and
        # the confirmed law keeps actuating.
        reply = rt.select_mode(p.MODE_REFERENCE, model_token='wrong-model')
        assert reply['result'] == p.MODE_SEL_IDENTITY, reply
        assert (rt.latest_state()[0]).selected_mode == p.MODE_STREAMED
        assert tracked(rt), 'invalid request retired the confirmed law'

        # Selection with a stale ARM epoch is refused, never confirmed.
        reply = raw_select(rt, arm_epoch=rt._arm_epoch + 1)
        assert reply['result'] == p.MODE_SEL_REFUSED, reply
        assert (rt.latest_state()[0]).selected_mode == p.MODE_STREAMED

        # Soft -> native with the bound digest confirms and fences the stream.
        reply = rt.select_mode(p.MODE_REFERENCE, model_token=token)
        assert reply['result'] == p.MODE_SEL_OK and reply['selected_mode'] == p.MODE_REFERENCE
        time.sleep(.3)
        assert not tracked(rt, timeout=.3), 'v5 command actuated after Soft->native'

        # A latched fault refuses selection outright.
        pump.close()
        wait_for(lambda: (rt.latest_state()[0] or None) and rt.latest_state()[0].faulted,
                 timeout=3)
        reply = raw_select(rt)
        assert reply['result'] == p.MODE_SEL_REFUSED, reply
    case_native_to_soft(binary)
    case_order_with_running_reference(binary)
    case_retry_with_running_reference(binary)
    with serve(binary, enable_reference=False) as rt:
        # enable_reference is off: no capability, and the state stream carries
        # neither mode flag (byte-identical flags to the old build).
        assert not rt.identity.capabilities & p.CAP_MODE_SELECT
        rt.enable_all()
        state = wait_for(lambda: (rt.latest_state()[0] or None)
                         if (rt.latest_state()[0] and rt.latest_state()[0].armed)
                         else None)
        assert state.selected_mode == p.MODE_UNSELECTED
        assert not state.flags & (p.FLAG_MODE_STREAMED | p.FLAG_MODE_REFERENCE), hex(state.flags)

    print('mode select E2E: seed/reject/confirm/same-mode/invalid/latched/stale/bare '
          '+ native->Soft/native-running ORDER+retry PASS')


if __name__ == '__main__':
    # Optional case selector keeps the new transition cases runnable alone,
    # so a before/after result can be reported per case.
    cases = {'native-to-soft': case_native_to_soft,
             'order': case_order_with_running_reference,
             'retry': case_retry_with_running_reference}
    binary = str(Path(sys.argv[1]).resolve())
    if len(sys.argv) > 2:
        cases[sys.argv[2]](binary)
    else:
        main(binary)
