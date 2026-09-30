"""Console `control` updates reach mode/payload only through the SDK owner.

CTRL-033. The console emits gains, Soft (`pose_hold`) and payload on the Dora
`control` topic. ConsoleControlAdapter turns exactly those into `controller/1` SDK
calls on the real RobotRuntime + Service over the fake RT, forwards every other
field unchanged, and reports each SDK refusal. Parity: the state after the adapter
equals what the named preset defines, i.e. what `select_control_preset` gives.
"""
from pathlib import Path
import sys

import numpy as np

from arm_control.contracts.motion import pack_control_update, unpack_control_update
from arm_control.execution.contract import Start, TrajectoryRequest
from arm_control.motion import JointTrajectory
from arm_control.supervisor.console_adapter import ConsoleControlAdapter
from controller_extension import configure
from stream_races import main


def console(**fields):
    """A decoded update exactly as the controller node receives it."""
    return unpack_control_update(pack_control_update(**fields))


def cases(arm, runtime, scene, service):
    presets = runtime._presets
    adapter = ConsoleControlAdapter(arm, presets)
    runtime._call(setattr, runtime.controller, 'pose_provenance', 'nominal')

    def state():
        return arm.controller_state()

    # Gains: the console sends a named preset's kp/kd; the SDK owner applies it.
    for name in ('float', 'track'):
        p = presets[name]
        result = adapter.apply(console(gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist()),
                                       reason=f'preset {name}'))
        assert result.applied == (name,) and not result.refusals, result
        assert result.forward == {'reason': f'preset {name}'}, result.forward
        now = state()
        assert (now.law, now.preset_id) == ('joint', name), now
        assert np.array_equal(now.kp, p.kp) and np.array_equal(now.kd, p.kd), now

    # Not a named preset: refused, nothing applied, state unchanged.
    before = state()
    p = presets['track']
    result = adapter.apply(console(gains=dict(kp=(p.kp * 1.5).tolist(), kd=p.kd.tolist())))
    assert not result.applied and any('not a named controller preset' in r for r in result.refusals), result
    assert state() == before

    # Soft: the console's pose_hold spec (plus its request id) resolves to the Soft
    # preset and reaches the SDK; the fake plant is not Soft-capable, so the SDK's own
    # refusal comes back (same text as a direct call), and nothing changes.
    spec = dict(presets['soft'].pose_hold, id=7)
    try:
        arm.select_control_preset('soft')
    except Exception as exc:  # noqa: BLE001 - only the text is compared
        direct = str(exc)
    else:
        raise AssertionError('fake plant unexpectedly Soft-capable')
    result = adapter.apply(console(pose_hold=spec))
    assert not result.applied and result.refusals == (f'preset soft: {direct}',), (result, direct)
    assert state() == before
    result = adapter.apply(console(pose_hold=dict(spec, kc=[1.] * 6)))
    assert any('not a named controller preset' in r for r in result.refusals), result

    # Payload: applied through the SDK with a new revision; invalid values are refused.
    revision = state().payload.revision
    result = adapter.apply(console(payload={'mass_kg': .5, 'com_ee': [.01, .02, .03]}))
    assert result.applied == ('payload',) and not result.refusals, result
    assert state().payload.revision == revision + 1 and state().payload.value.mass_kg == .5
    result = adapter.apply(console(payload={'mass_kg': -1.}))
    assert not result.applied and result.refusals[0].startswith('payload:'), result
    q = arm.state().q
    staged = arm.execute(TrajectoryRequest(
        'staged', 'fake', tuple(runtime.joints),
        JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3))),
        model_token=scene.facts.model_token, start=Start.ON_RELEASE))
    result = adapter.apply(console(payload={'mass_kg': 0.}))
    assert not result.applied and result.refusals[0].startswith('payload:'), result  # execution current
    arm.cancel(staged.execution_id)
    result = adapter.apply(console(payload={'mass_kg': 0.}))
    assert result.applied == ('payload',), result

    # Everything else is forwarded unchanged, with no SDK call; mixed updates are split.
    rest = console(execute=5, cancel=True, reason='operator stop')
    result = adapter.apply(rest)
    assert result.forward == rest and not result.applied and not result.refusals
    mixed = adapter.apply(console(arm=True, gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist()),
                                  payload={'mass_kg': 0.}))
    assert mixed.forward == {'arm': True} and mixed.applied == ('track', 'payload'), mixed
    print('PASS: console gain/Soft/payload updates become SDK calls with named-preset parity; '
          'refusals are reported; other fields are forwarded unchanged')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
