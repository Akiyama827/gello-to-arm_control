"""Optional stream/1 codec, negotiation and lightweight SDK import checks."""
from dataclasses import replace
import json
import sys

import numpy as np

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
    assert CONTRACT_VERSION == 1
    assert not any(name.split('.')[0] in {'mujoco', 'pinocchio', 'pyarrow', 'dora', 'rerun', 'ompl'} for name in sys.modules)
    assert not any(name.startswith(('arm_control.supervisor', 'arm_control.plants')) for name in sys.modules)
    assert np.array_equal(request.initial.velocities, [[0.], [0.]])
    print('PASS: stream codec/immutable input, uint64 syntax, absent extension refuses all six methods, SDK device-free, execution v1 unchanged')


if __name__ == '__main__':
    check()
