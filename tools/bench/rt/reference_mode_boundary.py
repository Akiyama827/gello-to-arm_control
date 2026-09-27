"""Probe v7 retirement -> legacy command selection on the actual fake RT loop.

This establishes the shared v5/v6 arbitration boundary, not Soft physics. The
fake backend has no Cartesian model; no pose command is sent or qualified.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np

from arm_control.execution.contract import Phase, TrajectoryRequest
from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as p
from reference_supervisor import until
from stream_races import main


def cases(arm, runtime, scene, service):
    q = arm.state().q
    req = TrajectoryRequest('native', 'fake', tuple(runtime.joints),
        JointTrajectory([0., 1.], [q, q+.001], np.zeros((2, 3))), model_token=scene.facts.model_token)
    eid = arm.execute(req).execution_id
    until(lambda: arm.status(eid).phase is Phase.APPLIED)
    # Only substitute the PC sender: continue valid v5 joint commands after
    # explicit native cancel, with the existing session/ARM epoch/sequence.
    runtime._call(setattr, runtime, '_send_command', runtime.link.apply_command)
    arm.cancel(eid)
    until(lambda: runtime.link.latest_state()[0].reference['status'] == p.REFERENCE_CANCELLED)
    before = runtime.link.latest_state()[0]
    sent_before = runtime.link._cmd_seq
    time.sleep(.12)  # exceeds hold_ms=50, below independent fault_ms=500
    after = runtime.link.latest_state()[0]
    assert runtime.link._cmd_seq > sent_before
    assert after.state_seq > before.state_seq
    assert after.last_cmd_seq == before.last_cmd_seq
    assert after.reference['reference_id'] == before.reference['reference_id'] != 0
    assert after.holding and after.armed and not after.faulted
    print(json.dumps(dict(result='CONFIRMED', reference_id=after.reference['reference_id'],
        status=after.reference['status'], sent_commands=runtime.link._cmd_seq-sent_before,
        applied_before=before.last_cmd_seq, applied_after=after.last_cmd_seq,
        armed=after.armed, holding=after.holding, faulted=after.faulted,
        scope='v5 packet rejection after native retirement; same RT predicate covers v6')))


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases)
