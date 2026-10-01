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
from arm_control.sdk import Refused, SupervisorError
from arm_control.supervisor.console_adapter import ConsoleControlAdapter
from controller_extension import configure
from reference_supervisor import until
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
        assert result.forward == (), result.forward  # reason travels only with cancel/hold/stop
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

    # Same-preset gains during a running leg cancel it (the Dora path cancels on any
    # gains message; the SDK alone returns early when the preset is unchanged).
    from arm_control.execution.contract import Phase
    from arm_control.supervisor.console_adapter import ConsoleControlAdapter as Adapter
    q = arm.state().q
    running = arm.execute(TrajectoryRequest(
        'running', 'fake', tuple(runtime.joints),
        JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3))),
        model_token=scene.facts.model_token, start=Start.ON_ADMIT))
    until(lambda: arm.status(running.execution_id).phase is Phase.APPLIED, 5)
    result = Adapter(arm, presets).apply(console(gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist())))
    assert not arm.status(running.execution_id).terminal, 'default live() cancels nothing'
    result = Adapter(arm, presets, live=lambda: [running.execution_id]).apply(
        console(gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist())))
    assert result.applied == ('track',) and arm.status(running.execution_id).phase is Phase.CANCELLED

    # Malformed fields are refusals or no-ops, never exceptions.
    for bad in (dict(pose_hold=None), dict(pose_hold=[1]), dict(payload={'mass_kg': None}),
                dict(gains=dict(kp='x', kd=[1.]))):
        result = adapter.apply(console(**bad))
        assert result.refusals and not result.applied, (bad, result)
    for noop in (dict(gains=None), dict(gains={})):
        assert adapter.apply(console(**noop)) == adapter.apply(console()), noop
    assert adapter.apply(console(payload=None)).applied == ('payload',)

    # Everything else is forwarded unchanged and in on_control order; mixed updates are
    # split, each field applied independently.
    rest = console(stop=True, cancel=True, execute=5, hold=True, arm=True, reason='operator stop')
    result = adapter.apply(rest)
    assert result.forward == ({'arm': True}, {'execute': 5}, {'cancel': True, 'reason': 'operator stop'},
                              {'hold': True, 'reason': 'operator stop'},
                              {'stop': True, 'reason': 'operator stop'}), result.forward
    assert not result.applied and not result.refusals
    mixed = adapter.apply(console(arm=True, gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist()),
                                  payload={'mass_kg': 0.}))
    assert mixed.forward == ({'arm': True},) and mixed.applied == ('payload', 'track'), mixed
    independent = adapter.apply(console(gains=dict(kp=presets['float'].kp.tolist(),
                                                   kd=presets['float'].kd.tolist()),
                                        payload={'mass_kg': -1.}))
    assert independent.applied == ('float',) and len(independent.refusals) == 1, independent
    assert state().preset_id == 'float'  # the refused payload did not undo the preset
    assert adapter.apply(console(gains=dict(kp=None, kd=None))) == adapter.apply(console())
    adapter.apply(console(gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist())))
    # Gate (b): a real live() supplier. LiveExecutions tracks what this client started and drops finished ones.
    import time
    from arm_control.supervisor.console_adapter import LiveExecutions
    arm.select_control_preset('track')
    tracker = LiveExecutions(arm)
    q = arm.state().q
    run = tracker.track(arm.execute(TrajectoryRequest(
        'tracked', 'fake', tuple(runtime.joints),
        JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3))),
        model_token=scene.facts.model_token, start=Start.ON_ADMIT)).execution_id)
    until(lambda: arm.status(run).phase is Phase.APPLIED, 5)
    assert tracker() == [run]
    result = Adapter(arm, presets, live=tracker).apply(console(gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist())))
    assert result.applied == ('track',) and arm.status(run).phase is Phase.CANCELLED, result
    assert tracker() == [], 'a finished execution stayed in the live set'

    class Unreachable:
        def status(self, execution_id):
            raise Refused('supervisor unreachable')
    ghost = LiveExecutions(Unreachable())
    ghost.track('x')
    assert ghost() == ['x'], 'an execution whose end cannot be shown was dropped from the live set'

    class Dead:
        def status(self, execution_id):
            raise SupervisorError('connection closed')
    dead = LiveExecutions(Dead())
    dead.track('y')
    assert dead() == ['y']  # a lost connection never makes live() raise

    class Forgotten:
        def status(self, execution_id):
            raise Refused(f'unknown execution {execution_id!r}')
    forgotten = LiveExecutions(Forgotten())
    forgotten.track('old')
    assert forgotten() == [], 'an execution the supervisor forgot stayed live forever'

    # Gate (d): the cancel wait is bounded by one budget per apply(); an exhausted budget does NOT change the law.
    until(lambda: not runtime._native.busy, 5)
    q = arm.state().q
    run = arm.execute(TrajectoryRequest(
        'budget', 'fake', tuple(runtime.joints),
        JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3))),
        model_token=scene.facts.model_token, start=Start.ON_ADMIT)).execution_id
    until(lambda: arm.status(run).phase is Phase.APPLIED, 5)
    t0 = time.monotonic()
    result = Adapter(arm, presets, live=lambda: [run], cancel_budget_s=0.).apply(
        console(gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist())))
    assert not result.applied and 'budget' in result.refusals[0], result
    assert time.monotonic() - t0 < 1. and not arm.status(run).terminal, 'a refused budget still cancelled the leg'
    arm.cancel(run)

    # A positive budget is spent from the first cancel and passed on as the cancel timeout; when a slow cancel uses
    # it up, the remaining legs are not cancelled and the refusal says how many were.
    class SlowArm:
        def __init__(self, real):
            self.real, self.timeouts = real, []
        def controller_state(self):
            return self.real.controller_state()
        def cancel(self, execution_id, reason='', timeout_s=2.):
            self.timeouts.append(round(timeout_s, 1))
            time.sleep(.3)
        def select_control_preset(self, name):
            raise AssertionError('the preset was selected after the budget ran out')
    slow = SlowArm(arm)
    time.sleep(.3)  # time before the first cancel must not count against the budget
    result = Adapter(slow, presets, live=lambda: ['a', 'b'], cancel_budget_s=.2).apply(
        console(gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist())))
    assert slow.timeouts == [.2] and 'after 1 cancel' in result.refusals[0], (slow.timeouts, result)

    # Gate (c): with send(), a mixed message keeps the on_control order (arm, payload, execute, gains, cancel, hold,
    # stop); without it the groups come back for the caller. A raising send is reported and the rest still runs.
    log = []
    real_set_payload, real_select = arm.set_payload, arm.select_control_preset
    arm.set_payload = lambda *a, **k: (log.append('payload'), real_set_payload(*a, **k))[1]
    arm.select_control_preset = lambda *a, **k: (log.append('gains'), real_select(*a, **k))[1]
    try:
        arm.select_control_preset('float')
        log.clear()
        sent = adapter.apply(console(stop=True, cancel=True, execute=5, hold=True, arm=True, reason='r',
                                     payload={'mass_kg': 0.}, gains=dict(kp=p.kp.tolist(), kd=p.kd.tolist())),
                             send=lambda group: log.append(next(k for k in group if k != 'reason')))
        assert sent.forward == () and sent.applied == ('payload', 'track'), sent
        assert log == ['arm', 'payload', 'execute', 'gains', 'cancel', 'hold', 'stop'], log
        def boom(group):
            raise RuntimeError('down')
        failed = adapter.apply(console(arm=True, payload={'mass_kg': 0.}, stop=True), send=boom)
        assert failed.applied == ('payload',) and len(failed.refusals) == 2, failed
    finally:
        arm.set_payload, arm.select_control_preset = real_set_payload, real_select
    # Gate (e), what the fake plant can show: a payload during a live execution is refused with the SDK's reason
    # (asserted above); a same-Soft re-request is refused only on a Soft-capable plant: NOT exercised here.
    print('PASS: console gain/Soft/payload updates become SDK calls with named-preset parity; '
          'refusals are reported; other fields are forwarded unchanged')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
