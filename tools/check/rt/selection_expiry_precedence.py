"""An expired reference is reported as REFERENCE_EXPIRED, never as 'selection unresolved'.

CTRL-033 4b review. The PC never observes the RT selecting a stream's reference (its
poll is fed nothing until the RT reports the reference CANCELLED) and sends no
envelopes, so the RT retires it on expiry (~100 ms). The poll that finally sees the
expiry also finds the candidate pending past 250 ms; the unresolved rule must not
override that evidence, or an implicit hold would heal an expiry (the CTRL-034
decision (a) rule says it never does). Explicit hooks, no timing: the pending age is
advanced by fiat at exactly that poll.
"""
from pathlib import Path
import sys
import time

import numpy as np

from arm_control.execution.contract import Phase
from arm_control.execution.stream import StreamOpen
from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt import reference
from controller_extension import configure
from reference_supervisor import until
from stream_races import main


def cases(arm, runtime, scene, service):
    h, link = runtime._native, runtime.link
    poll = h.poll
    link.apply_reference_envelope = lambda *args, **kwargs: None  # nothing feeds the reference

    def evidence_only(state, at):
        ref = None if state is None else state.reference
        if ref is not None and ref['reference_id'] == h.id and ref['status'] == rtp.REFERENCE_CANCELLED:
            if h.candidate_selection == 'pending':
                h._pending_since -= 1.  # the candidate has now been pending for over a second
            return poll(state, at)
        return poll(None, at)  # RUNNING/END are never observed: the slot stays pending

    h.poll = evidence_only
    q = arm.state().q
    curve = JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))
    eid = arm.open_stream(StreamOpen('e1', 1, 'fake', tuple(runtime.joints), scene.facts.model_token,
        scene.facts.scene_token, scene.facts.policy.stream_policy_id, curve)).execution.execution_id
    until(lambda: arm.status(eid).terminal, 6)
    status = arm.status(eid)
    assert status.phase is Phase.FAULTED and 'REFERENCE_EXPIRED' in status.reason, status
    assert 'unresolved' not in status.reason, status
    assert h.candidate_selection == 'pending'  # never reported unresolved: the evidence won
    assert time.monotonic() - h._pending_since > reference.SELECTION_UNRESOLVED_S  # the rule was eligible
    assert runtime._native_failed, 'expiry must latch the native failure, not be healed'
    stream_reference = h.id
    time.sleep(.8)  # no implicit hold reference may be staged after an expiry
    assert link.latest_state()[0].reference['reference_id'] == stream_reference
    print('PASS: an expiry seen while pending ends FAULTED REFERENCE_EXPIRED (not unresolved), latched, no implicit hold')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
