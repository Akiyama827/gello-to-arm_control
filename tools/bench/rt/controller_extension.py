"""controller/1 admission/retirement over the actual SDK and native fake RT."""
from pathlib import Path
import sys
import threading

import numpy as np

from arm_control.execution.contract import Phase, Start, TrajectoryRequest
from arm_control.execution.controller import PayloadValue
from arm_control.execution.stream import StreamOpen, SelectionState
from arm_control.motion import JointTrajectory
from arm_control.sdk import ArmClient
from reference_supervisor import until
from stream_supervisor import refusal
from stream_races import main


def configure(cfg):
    cfg.raw['control_mode'] = dict(gain_presets={
        'track': {}, 'float': dict(kp=0., kd=20.),
        'soft': dict(pose_hold=dict(kc=[100.]*3+[10.]*3, dc=[20.]*3+[2.]*3,
                                    nullspace_kp=1., nullspace_kd=1.))})


def cases(arm, runtime, scene, service):
    assert arm.extensions['controller'] == 1
    assert [(p.id, p.law) for p in arm.controller_presets()] == [
        ('track', 'joint'), ('float', 'joint'), ('soft', 'soft')]
    original = arm.controller_state()
    assert original.preset_id == 'track' and original.payload.revision == 0
    with ArmClient('fake', socket_path=service.path) as observer:
        assert observer.controller_state() == original
        refusal(lambda: observer.select_control_preset('float'))
        refusal(lambda: observer.set_payload(PayloadValue(0)))
    payload = arm.set_payload(PayloadValue(.5, [.01, .02, .03]))
    assert payload.revision == 1 and payload.value.com_ee == (.01, .02, .03)
    assert arm.set_payload(PayloadValue(.5)).revision == 2
    assert runtime.controller.executor._payload == (.5, None)
    assert arm.set_payload(PayloadValue(0)).revision == 3

    def curve():
        q = arm.state().q
        return JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))

    def planned(rid, start=Start.ON_RELEASE):
        return arm.execute(TrajectoryRequest(rid, 'fake', tuple(runtime.joints), curve(),
                                             model_token=scene.facts.model_token, start=start))

    staged = planned('staged')
    assert staged.phase is Phase.ADMITTED
    assert arm.receipts(staged.execution_id)[-1].evidence['payload_revision'] == 3
    refusal(lambda: arm.set_payload(PayloadValue(0)))
    refusal(lambda: arm.select_control_preset('missing'))
    assert arm.status(staged.execution_id) == staged
    # Calibrated/vendor distinction still refuses, before retiring the plan.
    runtime._call(setattr, runtime.controller, 'pose_provenance', 'calibrated')
    refusal(lambda: arm.select_control_preset('soft'))
    assert arm.status(staged.execution_id) == staged
    runtime._call(setattr, runtime.controller, 'pose_provenance', 'nominal')
    changed = arm.select_control_preset('float')
    assert changed.law == 'joint' and changed.preset_id == 'float' and changed.kp == (0.,)*3
    assert arm.status(staged.execution_id).phase is Phase.CANCELLED
    arm.select_control_preset('track')
    running = planned('running', Start.ON_ADMIT)
    until(lambda: arm.status(running.execution_id).phase is Phase.APPLIED)
    refusal(lambda: arm.set_payload(PayloadValue(0)))
    refusal(lambda: arm.select_control_preset('soft'))  # running, invalid Soft costs nothing
    assert not arm.status(running.execution_id).terminal
    arm.select_control_preset('float')
    assert arm.status(running.execution_id).phase is Phase.CANCELLED
    before_soft = arm.controller_state()
    refusal(lambda: arm.select_control_preset('soft'))
    assert arm.controller_state() == before_soft
    arm.set_payload(PayloadValue(0))
    arm.select_control_preset('track')
    print('PASS: named presets, owner gates, invalid Soft preserves execution, valid mode retires staged/running native execution')

    def stream(rid, seq):
        return arm.open_stream(StreamOpen(rid, seq, 'fake', tuple(runtime.joints),
            scene.facts.model_token, scene.facts.scene_token, scene.facts.policy.stream_policy_id, curve()))

    scene.entered.clear()
    scene.resume.clear()
    opened = stream('worker', 1)
    assert scene.entered.wait(1.)
    before = arm.controller_state().payload
    refusal(lambda: arm.set_payload(PayloadValue(1)))
    assert arm.controller_state().payload == before
    arm.select_control_preset('float')
    assert arm.status(opened.execution.execution_id).phase is Phase.REJECTED
    scene.resume.set()
    until(lambda: runtime._stream_job is None)
    assert arm.status(opened.execution.execution_id).terminal
    arm.select_control_preset('track')

    # Selection uncertainty does not let a load update cross the execution.
    entered, resume = threading.Event(), threading.Event()
    commit = runtime.link.commit_reference
    def delayed(*args, **kwargs):
        if not runtime._native.stream:
            # The runtime also commits its own measured-hold references. Under
            # load one can reach this gate before the stream is approved (the
            # attempt is still NOT_REQUESTED); only the stream's COMMIT is held.
            return commit(*args, **kwargs)
        entered.set()
        assert resume.wait(2.)
        return commit(*args, **kwargs)
    runtime.link.commit_reference = delayed
    opened = stream('pending', 2)
    eid = opened.execution.execution_id
    try:
        assert entered.wait(1.)
        assert arm.stream_revision(eid, 'pending').selection_state is SelectionState.PENDING
        refusal(lambda: arm.set_payload(PayloadValue(0)))
        arm.select_control_preset('float')
        assert arm.status(eid).phase is Phase.CANCELLED
    finally:
        resume.set()
        runtime.link.commit_reference = commit
    until(lambda: not runtime._native.busy)
    assert arm.status(eid).phase is Phase.CANCELLED
    arm.set_payload(PayloadValue(.5))
    arm.select_control_preset('track')
    loaded = stream('loaded', 3)
    assert arm.wait(loaded.execution.execution_id, 1.).phase is Phase.REJECTED
    assert arm.controller_state().payload.value.mass_kg == .5
    arm.set_payload(PayloadValue(0))
    print('PASS: payload immutable under worker/open/PENDING; mode retires late jobs and COMMIT; canonical nonzero load refuses stream')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
