"""A cancel that races a pending stream COMMIT must free the candidate slot.

CTRL-034: a preset change cancels a stream while its COMMIT is pending. If the
horizon never learns the outcome, ``candidate_selection`` stays 'pending', so
``busy`` stays true, the supervisor sends nothing (its tail returns None while
busy), the RT's ``last_cmd_seq`` freezes and CMD_LOST follows.

Both orderings are forced with explicit gates, not timing:
  before_dispatch  the worker sees the cancel at commit()'s own check
  in_transport     the cancel lands after that check; the transport clears its
                   cancel flag, COMMIT reaches the RT, the RT selects, and the
                   queued CANCEL retires the reference before any PC poll can
                   observe RUNNING (the PC's view is blinded to model that).
"""
from pathlib import Path
import sys
import threading
import time

import numpy as np

from arm_control.execution.contract import Phase
from arm_control.execution.stream import StreamOpen, SelectionState
from arm_control.motion import JointTrajectory
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
    seq = 0

    def stream():
        nonlocal seq
        seq += 1
        q = arm.state().q
        curve = JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))
        return arm.open_stream(StreamOpen(f'r{seq}', seq, 'fake', tuple(runtime.joints),
            scene.facts.model_token, scene.facts.scene_token,
            scene.facts.policy.stream_policy_id, curve))

    def recovered(name):
        # The slot is free and the PC keeps the RT alive: an RT that hears
        # nothing latches CMD_LOST once --fault-ms (1 s here) passes.
        until(lambda: not h.busy, 5)
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
    eid = stream().execution.execution_id
    assert token.reached.wait(2.)
    assert arm.stream_revision(eid, 'r1').selection_state is SelectionState.PENDING
    arm.select_control_preset('float')
    assert arm.status(eid).phase is Phase.CANCELLED
    token.go.set()
    recovered('before_dispatch')

    # Ordering 2: COMMIT is in the transport when the cancel lands.
    entered, resume, blind = threading.Event(), threading.Event(), threading.Event()
    commit, poll = link.commit_reference, h.poll
    def held(*args, **kwargs):
        entered.set()
        assert resume.wait(2.), 'fixture gate not released'
        return commit(*args, **kwargs)
    link.commit_reference = held
    h.poll = lambda state, at: poll(None if blind.is_set() else state, at)
    try:
        eid = stream().execution.execution_id
        assert entered.wait(2.)
        blind.set()
        arm.select_control_preset('float')
        assert arm.status(eid).phase is Phase.CANCELLED
        resume.set()
        until(lambda: h._job is None and h._cancel_job is None, 5)
        blind.clear()
        recovered('in_transport')
    finally:
        resume.set()
        link.commit_reference, h.poll = commit, poll
    print('PASS: a cancel racing a pending stream COMMIT frees the slot and resumes commands '
          '(seen before dispatch; COMMIT already in transport)')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
