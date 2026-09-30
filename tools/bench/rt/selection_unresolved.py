"""A stuck 'pending' selection can no longer starve commands until CMD_LOST.

CTRL-033 4b. A stream's COMMIT succeeds at the RT but the PC never observes the
selection (its poll is fed no RT state, the ~1 ms RUNNING window made certain).
Before, ``candidate_selection`` stayed 'pending', ``busy`` stayed true, the
supervisor sent nothing, and the RT latched CMD_LOST. Now, after 250 ms the
horizon raises ``SelectionUnresolved``; the runtime fails the execution with a
receipt, sends an explicit CANCEL, and the ordinary tail publishes the measured
hold. The control run disables the timeout and shows the starvation is real.
"""
from pathlib import Path
import sys
import threading
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
    blind, retire_calls, seq = threading.Event(), [], 0
    for name in ('abort_reference_upload', 'cancel_reference'):
        def traced(*args, _real=getattr(link, name), _name=name, **kwargs):
            retire_calls.append(_name)
            return _real(*args, **kwargs)
        setattr(link, name, traced)
    poll, seen = h.poll, set()
    def watching(state, at):
        try:
            return poll(None if blind.is_set() else state, at)
        finally:
            seen.add(h.candidate_selection)
    h.poll = watching

    def stream():
        nonlocal seq
        seq += 1
        q = arm.state().q
        curve = JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))
        return arm.open_stream(StreamOpen(f'u{seq}', seq, 'fake', tuple(runtime.joints),
            scene.facts.model_token, scene.facts.scene_token,
            scene.facts.policy.stream_policy_id, curve))

    # Fixed: the selection stays unobserved, yet the slot is reported and freed.
    blind.set()
    started = time.monotonic()
    eid = stream().execution.execution_id
    until(lambda: h.candidate_selection == 'pending', 5)
    until(lambda: arm.status(eid).terminal, 5)
    took = time.monotonic() - started
    status = arm.status(eid)
    assert status.phase is Phase.REJECTED and 'selection unresolved' in status.reason, status
    assert time.monotonic() - h._pending_since >= reference.SELECTION_UNRESOLVED_S
    assert took < 3., took  # reported by ~250 ms plus polling, not left to CMD_LOST
    assert {'pending', 'unresolved'} <= seen, seen  # the slot was reported, not left pending
    assert 'cancel_reference' in retire_calls, retire_calls  # explicit retire of the maybe-selected reference
    stream_reference = h.id
    blind.clear()
    # The measured hold is published: a newer reference is selected and the RT stays alive.
    until(lambda: link.latest_state()[0].reference['reference_id'] > stream_reference, 5)
    time.sleep(1.5)  # longer than --fault-ms (1 s)
    health = link.motor_health()
    assert not health['any_fault'], health['latched_fault']
    assert link.latest_state()[0].reference['status'] in (rtp.REFERENCE_RUNNING, rtp.REFERENCE_END)
    arm.select_control_preset('track')

    # Control: with the timeout disabled the same stuck pending starves to CMD_LOST.
    reference.SELECTION_UNRESOLVED_S = 1e9
    try:
        blind.set()
        eid = stream().execution.execution_id
        until(lambda: h.candidate_selection == 'pending', 5)
        until(lambda: link.motor_health()['any_fault'], 6)
        assert h.candidate_selection == 'pending' and h.busy
        assert not arm.status(eid).terminal or 'unresolved' not in arm.status(eid).reason
    finally:
        reference.SELECTION_UNRESOLVED_S = 0.25
        blind.clear()
        h.poll = poll
    print('PASS: a stuck pending selection is reported after 250 ms (receipt, explicit CANCEL, measured '
          'hold, no CMD_LOST); with the timeout disabled the same setup starves to CMD_LOST')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
