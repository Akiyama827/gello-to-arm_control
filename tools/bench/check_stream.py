"""stream/1: the optional codec, negotiation and lightweight SDK import (run first: it asserts the SDK pulled in no supervisor
or plant), then the analytical target strategy with explicit fixture values. No devices."""
from dataclasses import replace
import json
import sys

import numpy as np

from arm_control.execution.controller import PayloadValue, PayloadState, ControllerState
from arm_control.execution import codec
from arm_control.execution.contract import CONTRACT_VERSION, Clock, Stamp, Start
from arm_control.execution.stream import StreamOpen, StreamRevision, SpliceAnchor
from arm_control.motion import JointTrajectory
from arm_control.sdk import ArmClient, Refused


class Channel:
    def __init__(self, extensions):
        self.extensions = extensions
        self.calls = []

    def call(self, op, **args):
        self.calls.append(op)
        assert op == 'hello', 'unnegotiated stream operation reached transport'
        return dict(contract_version=1, session='s', extensions=self.extensions,
                    identity=dict(instance_id='fake', joint_names=['j'], backend='rt', model_id='m'),
                    capabilities=dict(contract_version=1, profiles=['joint_impedance'],
                                      protocol_version=5, pose_hold=False, observer=False, rt_setup='ok'))

    def close(self):
        pass


def check():
    curve = JointTrajectory([0., 1.], [[0.], [.01]], [[0.], [0.]])
    request = StreamOpen('r', 7, 'fake', ('j',), 'model', 'scene', 'policy', curve, Start.ON_RELEASE)
    curve.positions[0] = .8
    assert request.initial.positions[0, 0] == 0.  # producer storage is detached
    wire = json.loads(json.dumps(codec.encode(request)))
    assert codec.encode(codec.stream_open_from(wire)) == wire
    at = Stamp(1, Clock.PC, 'pc')
    anchor = SpliceAnchor('a', 1, .5, (0.,), (0.,), at, replace(at, ns=100))
    assert codec.anchor_from(codec.encode(anchor)) == anchor
    revision = StreamRevision('r2', 'e', 20, 1, 'a', 'scene',
                              JointTrajectory([.5, 1.], [[0.], [.01]], [[0.], [0.]]), True)
    assert codec.encode(codec.stream_revision_from(codec.encode(revision))) == codec.encode(revision)
    for value in (0, -1, 2**64, True, 1.5):
        try:
            replace(request, open_seq=value)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid uint64 sequence')
    for extension in ({}, {'stream': 2}):
        channel = Channel(extension)
        arm = ArmClient('fake', channel=channel)
        for fn in (lambda: arm.open_stream(request), lambda: arm.stream_state('e'),
                   lambda: arm.stream_anchor('e'), lambda: arm.revise_stream(revision),
                   lambda: arm.stream_revision('e', 'r'), lambda: arm.abort_stream_revision('e', 'r')):
            try:
                fn()
            except Refused:
                pass
            else:
                raise AssertionError('unnegotiated extension accepted')
        assert channel.calls == ['hello']
    payload = PayloadValue(.5, [1., 2., 3.])
    state = ControllerState('joint', None, [1.], [2.], PayloadState(3, payload))
    assert codec.controller_state_from(codec.encode(state)) == state
    for mass, com in ((-1, None), (float('nan'), None), (float('inf'), None),
                      (.5, [1, 2]), (.5, [1, 2, 3, 4]), (.5, [0, 0, float('nan')])):
        try:
            PayloadValue(mass, com)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid payload accepted')
    assert PayloadValue(0, [float('nan')]).com_ee is None
    original = [1., 2., 3.]
    value = PayloadValue(.5, original)
    original[0] = 100.
    assert value.com_ee == (1., 2., 3.)
    for extension in ({}, {'controller': 2}):
        channel = Channel(extension)
        arm = ArmClient('fake', channel=channel)
        for fn in (arm.controller_presets, arm.controller_state,
                   lambda: arm.select_control_preset('track'), lambda: arm.set_payload(payload)):
            try:
                fn()
            except Refused:
                pass
            else:
                raise AssertionError('unnegotiated controller extension accepted')
        assert channel.calls == ['hello']
    assert CONTRACT_VERSION == 1
    assert not any(name.split('.')[0] in {'mujoco', 'pinocchio', 'pyarrow', 'dora', 'rerun', 'ompl'} for name in sys.modules)
    assert not any(name.startswith(('arm_control.supervisor', 'arm_control.plants')) for name in sys.modules)
    assert np.array_equal(request.initial.velocities, [[0.], [0.]])
    print('PASS: stream/controller codecs, complete payload, immutable input, uint64 syntax, absent extension refuses all six methods, SDK device-free, execution v1 unchanged')



def target():
    from arm_control.control.execution_policy import ExecutionPolicy  # after check(): these pull in the supervisor
    from arm_control.planning.stream_target import TargetStopGrid
    from arm_control.supervisor.admission import StreamPolicy

    policy = StreamPolicy('analytical-envelope-1', .05, .1, .9, (.01,))
    strategy = TargetStopGrid((.1, .2, .4), (1., .5, .25))
    limits = ExecutionPolicy(command_rate_hz=100, state_timeout_sec=.25,
                             health_timeout_sec=1., start_pos_tol_rad=.1,
                             abort_pos_err_rad=.35, hold_relatch_rad=.3,
                             torque_limits=np.array([87.]), position_lower=np.array([-3.]),
                             position_upper=np.array([3.]), velocity_limits=np.array([.4]),
                             acceleration_limits=np.array([.8]))
    args = dict(cut_phase=.2, q=[.002], qd=[.01], observed_phase=0., stream_policy=policy,
                execution_policy=limits, kp=np.array([100.]), kd=np.array([10.]), check=lambda _: '')
    curve, alpha, duration = strategy.toward([.022], **args)
    assert (alpha, duration) == (1., .4)
    for phase in (.01, .03):
        another, _, _ = strategy.toward([.022], **{**args, 'observed_phase': phase})
        assert another.positions.tobytes() == curve.positions.tobytes()
        assert another.times.tobytes() == curve.times.tobytes()
    partial = strategy.toward([.022], **{**args, 'check': lambda t: 'collision' if any(q[0] > .013 for q in t.densify(.001)) else ''})
    assert partial[1] == .5
    for invalid in (lambda: replace(strategy, duration_grid_s=(.1, .1)),
                    lambda: replace(strategy, progress_fractions=(.5, .25)),
                    lambda: replace(policy, max_candidate_age_s=float('nan')),
                    lambda: strategy.toward([10.], **args),
                    lambda: strategy.toward([.022], **{**args, 'check': lambda _: 'collision'})):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid/infeasible declaration passed')
    print('PASS: production target strategy uses explicit grid/envelope, continuous bounds, partial progress, scene refusal, no arrival-derived timing')


if __name__ == '__main__':
    check()
    target()
