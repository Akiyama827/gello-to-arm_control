"""A cancel that races a pending stream COMMIT must free the candidate slot.

CTRL-034: a preset change cancels a stream while its COMMIT is pending. If the
horizon never learns the outcome, ``candidate_selection`` stays 'pending', so
``busy`` stays true, the supervisor sends nothing (its tail returns None while
busy), and the RT latches CMD_LOST.

Both orderings are forced with explicit gates, not timing:
  before_dispatch  the cancel is seen before COMMIT is sent. The RT never saw
                   COMMIT, so cancel() must ABORT the sealed upload (not CANCEL).
  in_transport     the cancel lands after the client's own check, with the COMMIT
                   frame about to be sent: COMMIT reaches the RT, the RT selects,
                   and the queued CANCEL retires it before any PC poll can
                   observe RUNNING (the PC's view is blinded to model that).
After each: the slot is free, the link is not faulted, and the RT holds no
sealed upload (a COMMIT of it is refused).
"""
from pathlib import Path
import sys
import threading
import time

import numpy as np

from arm_control.execution.contract import Phase
from arm_control.execution.stream import StreamOpen, SelectionState
from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.client import ReferenceRefused
from controller_extension import configure
from reference_supervisor import until
from stream_races import main


class CancelToken(threading.Event):
    """The horizon's cancel Event; blocks the upload worker at its first check."""

    def __init__(self):
        super().__init__()
        self.reached, self.go, self._armed = threading.Event(), threading.Event(), True

    def is_set(self):
        if self._armed and threading.current_thread().name.startswith('reference-upload'):
            self._armed = False
            self.reached.set()
            assert self.go.wait(2.), 'fixture gate not released'
        return super().is_set()


def cases(arm, runtime, scene, service):
    h, link = runtime._native, runtime.link
    seq, retire_calls, hold_gate = 0, [], threading.Event()

    for name in ('abort_reference_upload', 'cancel_reference'):
        def traced(*args, _real=getattr(link, name), _name=name, **kwargs):
            retire_calls.append(_name)
            return _real(*args, **kwargs)
        setattr(link, name, traced)
    stage = h.stage
    # While set, the runtime may not start its measured-hold upload, which would
    # replace the RT's upload slot before it can be inspected.
    h.stage = lambda key, *a, **k: None if key is None and hold_gate.is_set() else stage(key, *a, **k)

    def stream():
        nonlocal seq
        seq += 1
        q = arm.state().q
        curve = JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))
        return arm.open_stream(StreamOpen(f'r{seq}', seq, 'fake', tuple(runtime.joints),
            scene.facts.model_token, scene.facts.scene_token,
            scene.facts.policy.stream_policy_id, curve))

    def recovered(name, expected_call):
        until(lambda: not h.busy, 5)
        assert retire_calls == [expected_call], (name, retire_calls)
        # The RT keeps no sealed upload: COMMIT of it is refused. A leftover one
        # would be accepted and select the reference.
        try:
            link.commit_reference(h._upload, reference_id=h.id)
        except ReferenceRefused as exc:
            assert exc.result != rtp.REF_OK, (name, exc)
        else:
            raise AssertionError(f'{name}: RT still held a sealed upload')
        hold_gate.clear()
        retire_calls.clear()
        # The PC keeps the RT alive: an RT that hears nothing latches CMD_LOST
        # once --fault-ms (1 s here) passes.
        time.sleep(1.5)
        health = link.motor_health()
        assert not health['any_fault'], (name, health['latched_fault'])
        arm.select_control_preset('track')

    # Ordering 1: the cancel is seen before COMMIT is dispatched.
    token, approve = CancelToken(), h.approve
    def gated_approve():
        h._interrupt = token
        h.approve = approve
        approve()
    h.approve = gated_approve
    hold_gate.set()
    eid = stream().execution.execution_id
    assert token.reached.wait(2.)
    assert arm.stream_revision(eid, 'r1').selection_state is SelectionState.PENDING
    arm.select_control_preset('float')
    assert arm.status(eid).phase is Phase.CANCELLED
    token.go.set()
    recovered('before_dispatch', 'abort_reference_upload')

    # Ordering 2: COMMIT is being sent when the cancel lands. The gate sits in
    # _send_frame, after the client's cancel check; the frame type is the one
    # most recently packed on this thread.
    entered, resume, blind, gate = (threading.Event() for _ in range(4))
    packed, pack, send, poll = threading.local(), rtp.pack_reference_control, link._send_frame, h.poll
    def packing(*args, **kwargs):
        packed.type = kwargs.get('ctl_type')
        return pack(*args, **kwargs)
    def sending(frame, **kwargs):
        if gate.is_set() and h.stream and getattr(packed, 'type', None) == rtp.CTL_REFERENCE_COMMIT:
            gate.clear()
            entered.set()
            assert resume.wait(2.), 'fixture gate not released'
        return send(frame, **kwargs)
    rtp.pack_reference_control, link._send_frame = packing, sending
    h.poll = lambda state, at: poll(None if blind.is_set() else state, at)
    hold_gate.set()
    gate.set()
    try:
        eid = stream().execution.execution_id
        assert entered.wait(2.)
        blind.set()
        arm.select_control_preset('float')
        assert arm.status(eid).phase is Phase.CANCELLED
        resume.set()
        until(lambda: h._job is None and h._cancel_job is None, 5)
        blind.clear()
        recovered('in_transport', 'cancel_reference')
    finally:
        resume.set()
        rtp.pack_reference_control, h.poll, h.stage = pack, poll, stage
        del link._send_frame
    print('PASS: a cancel racing a pending stream COMMIT frees the slot and resumes commands '
          '(ABORT before dispatch; CANCEL once COMMIT is in transport)')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
