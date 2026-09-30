"""What a peer or client can observe of one arm's motion state, pinned.

CTRL-021 transition work: armed-hold vs executing must be read from signals that already
exist, and this bench fixes which. For one arm on the fake RT:
  armed idle / staged  authority ARMED, holding=True, no RT reference (id 0)
  executing            holding=False, RT reference RUNNING
  after a cancel       a measured-hold reference (newer id, RUNNING then END) on the RT;
                       `holding` was observed False there, so it alone is NOT "armed-hold":
                       read the reference id/status
  disarmed             CLAIMED, arm_epoch 0, RT not armed
  re-armed             ARMED, arm_epoch + 1, holding=True, reference id 0 again
"""
from pathlib import Path
import sys

import numpy as np

from arm_control.execution.contract import Lifecycle, Phase, Start, TrajectoryRequest
from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as rtp
from controller_extension import configure
from reference_supervisor import until
from stream_races import main


def cases(arm, runtime, scene, service):
    link = runtime.link

    def rt():
        return link.latest_state()[0]

    a = arm.authority()
    assert a.lifecycle is Lifecycle.ARMED and a.holding is True and rt().reference['reference_id'] == 0
    epoch = a.arm_epoch
    q = arm.state().q
    plan = arm.execute(TrajectoryRequest(
        'life', 'fake', tuple(runtime.joints),
        JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3))),
        model_token=scene.facts.model_token, start=Start.ON_RELEASE))
    until(lambda: arm.status(plan.execution_id).phase is Phase.ADMITTED, 5)
    assert arm.authority().holding is True and rt().reference['reference_id'] == 0  # staged: still held
    arm.release(plan.execution_id)
    until(lambda: arm.status(plan.execution_id).phase is Phase.APPLIED, 8)
    assert arm.authority().holding is False and rt().reference['status'] == rtp.REFERENCE_RUNNING
    moving = rt().reference['reference_id']
    arm.cancel(plan.execution_id)
    until(lambda: rt().reference['reference_id'] > moving, 5)  # the measured hold is a reference
    assert arm.authority().lifecycle is Lifecycle.ARMED  # `holding` may read False here (seen)
    assert rt().reference['status'] in (rtp.REFERENCE_RUNNING, rtp.REFERENCE_END)
    arm.disarm()
    a = arm.authority()
    assert a.lifecycle is Lifecycle.CLAIMED and a.arm_epoch == 0 and a.holding is False
    until(lambda: not rt().armed, 5)
    arm.arm()
    a = arm.authority()
    assert a.lifecycle is Lifecycle.ARMED and a.arm_epoch == epoch + 1 and a.holding is True
    until(lambda: rt().reference['reference_id'] == 0, 5)
    print('PASS: armed-hold (holding, no reference), executing (RUNNING reference), post-cancel '
          '(measured-hold reference, holding False), disarmed, re-armed (new epoch) are distinguishable')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
