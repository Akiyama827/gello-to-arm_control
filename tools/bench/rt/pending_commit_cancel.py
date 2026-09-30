"""A cancel that races a pending stream COMMIT must free the candidate slot.

CTRL-034: a preset change cancels a stream while its COMMIT is pending. If the
horizon never learns the outcome, ``candidate_selection`` stays 'pending', so
``busy`` stays true, the supervisor sends nothing (its tail returns None while
busy), and the RT latches CMD_LOST.

Every ordering is forced with explicit gates, not timing:
  before_dispatch  the cancel is seen before COMMIT is sent. The RT never saw
                   COMMIT, so cancel() must ABORT the sealed upload (not CANCEL),
                   and the slot settles from the failed COMMIT job alone, before
                   the queued ABORT has even returned.
  in_transport     the cancel lands after the client's own check, with the COMMIT
                   frame about to be sent: COMMIT reaches the RT, the RT selects,
                   and the queued CANCEL retires it before any PC poll can
                   observe RUNNING (the PC's view is blinded to model that).
  replacement_*    a not-yet-sent replacement is cancelled while revision 1 is
                   selected: ABORT the replacement, then CANCEL(1). The slot
                   settles first; CANCEL(1) succeeds, is refused with a code the
                   existing path tolerates (no error), or is refused with any
                   other code (error surfaced, slot still settled).
After each: the slot is free, the link is not faulted, and the RT holds no
sealed upload (a COMMIT of it is refused).
"""
from pathlib import Path
import sys
import threading
import time

import numpy as np

from arm_control.execution.contract import Phase
from arm_control.execution.stream import StreamOpen, StreamRevision, SelectionState
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
    holding, held, release = threading.Event(), threading.Event(), threading.Event()
    refuse, errors = {}, []
    real_cancel = link.cancel_reference
    operation = {'abort_reference_upload': rtp.CTL_REFERENCE_ABORT,
                 'cancel_reference': rtp.CTL_CANCEL_REFERENCE}

    for name in operation:
        def traced(*args, _real=getattr(link, name), _name=name, **kwargs):
            retire_calls.append(_name)
            if holding.is_set():  # the worker is inside the queued ABORT/CANCEL
                held.set()
                assert release.wait(2.), 'fixture gate not released'
            if _name in refuse:  # scripted: the RT is not contacted
                raise ReferenceRefused(operation[_name], refuse.pop(_name))
            return _real(*args, **kwargs)
        setattr(link, name, traced)
    stage, poll0 = h.stage, h.poll
    # While set, the runtime may not start its measured-hold upload, which would
    # replace the RT's upload slot before it can be inspected.
    h.stage = lambda key, *a, **k: None if key is None and hold_gate.is_set() else stage(key, *a, **k)
    def recording_poll(state, at):
        result = poll0(state, at)
        if h.error and (not errors or errors[-1] != h.error):
            errors.append(h.error)
        return result
    h.poll = recording_poll

    def curve():
        q = arm.state().q
        return JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))

    def stream():
        nonlocal seq
        seq += 1
        return arm.open_stream(StreamOpen(f'r{seq}', seq, 'fake', tuple(runtime.joints),
            scene.facts.model_token, scene.facts.scene_token,
            scene.facts.policy.stream_policy_id, curve()))

    def gate_retire():
        held.clear(), release.clear()
        holding.set()

    def settled_before_retire():
        # The failed COMMIT job alone settles the slot: the queued ABORT is still
        # inside the worker, so the cancel job cannot have succeeded or failed.
        assert held.wait(2.), 'queued retire never started'
        until(lambda: h.candidate_selection == 'not_requested', 5)
        assert h._cancel_job is not None
        release.set()
        holding.clear()

    def recovered(name, expected_calls, cleanup=None):
        until(lambda: not h.busy, 5)
        assert retire_calls == expected_calls, (name, retire_calls)
        # The RT keeps no sealed upload: COMMIT of it is refused. A leftover one
        # would be accepted and select the reference.
        try:
            cut = {'protected_until': h._cut, 'latest_phase': h._cut} if h._revision > 1 else {}
            link.commit_reference(h._upload, reference_id=h.id, revision=h._revision,
                                  expected_revision=h._revision-1, **cut)
        except ReferenceRefused as exc:
            assert exc.result != rtp.REF_OK, (name, exc)
        else:
            raise AssertionError(f'{name}: RT still held a sealed upload')
        if cleanup is not None:
            cleanup()
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
    # No retire of the runtime's own can be in flight now: busy includes _cancel_job.
    retire_calls.clear()
    gate_retire()
    arm.select_control_preset('float')
    assert arm.status(eid).phase is Phase.CANCELLED
    token.go.set()
    settled_before_retire()
    recovered('before_dispatch', ['abort_reference_upload'])

    # Ordering 2: COMMIT is being sent when the cancel lands. The gate sits in
    # _send_frame, after the client's cancel check; the frame type is the one
    # most recently packed on this thread.
    entered, resume, blind, gate = (threading.Event() for _ in range(4))
    packed, pack, send = threading.local(), rtp.pack_reference_control, link._send_frame
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
    h.poll = lambda state, at: recording_poll(None if blind.is_set() else state, at)
    hold_gate.set()
    gate.set()
    try:
        eid = stream().execution.execution_id
        assert entered.wait(2.)
        retire_calls.clear()
        blind.set()
        arm.select_control_preset('float')
        assert arm.status(eid).phase is Phase.CANCELLED
        resume.set()
        until(lambda: h._job is None and h._cancel_job is None, 5)
        blind.clear()
        recovered('in_transport', ['cancel_reference'])
    finally:
        resume.set()
        rtp.pack_reference_control = pack
        del link._send_frame
        h.poll = recording_poll

    # Replacement: revision 1 is selected and a replacement, not yet sent, is
    # cancelled. cancel() must ABORT the replacement, then CANCEL revision 1.
    def replacement(name, script):
        eid = stream().execution.execution_id
        until(lambda: arm.stream_state(eid).selected_revision == 1, 10)
        until(lambda: not h.busy, 5)
        anchor = arm.stream_anchor(eid)
        suffix = JointTrajectory([anchor.cut_phase, anchor.cut_phase+.6],
                                 [anchor.q, np.array(anchor.q)+.002], [anchor.qd, np.zeros(3)])
        token, approve = CancelToken(), h.approve
        def gated():
            h._interrupt = token
            h.approve = approve
            approve()
        h.approve = gated
        if script is not None:
            refuse['cancel_reference'] = script
        errors.clear()
        hold_gate.set()
        arm.revise_stream(StreamRevision(f'rev{seq}', eid, 100+seq, 1, anchor.anchor_id,
                                         scene.facts.scene_token, suffix))
        assert token.reached.wait(2.)
        retire_calls.clear()
        gate_retire()
        arm.select_control_preset('float')  # cancels the stream, revision 1 selected
        assert arm.status(eid).phase is Phase.CANCELLED
        token.go.set()
        settled_before_retire()
        def cleanup():
            if script is not None:
                real_cancel(h.id, 1)  # the scripted refusal never reached the RT
        try:
            recovered(name, ['abort_reference_upload', 'cancel_reference'], cleanup)
        finally:
            refuse.clear()
        if script is None or script in (rtp.REF_REVISION, rtp.REF_RETIRED):
            assert not errors, (name, errors)
        else:
            assert any('refused' in e for e in errors), (name, errors)

    try:
        replacement('replacement_cancel_ok', None)
        replacement('replacement_cancel_tolerated', rtp.REF_RETIRED)
        replacement('replacement_cancel_refused', rtp.REF_BUSY)
    finally:
        h.poll, h.stage = poll0, stage
    print('PASS: a cancel racing a pending stream COMMIT frees the slot, resumes commands and '
          'leaves no sealed upload (ABORT before dispatch; CANCEL once COMMIT is in transport; '
          'ABORT then CANCEL(r) for a replacement, tolerated or reported refusal)')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
