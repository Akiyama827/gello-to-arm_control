"""SceneCore (plants/mujoco/scene_core): the world rules the Dora plant node and the supervisor's scene link share.

The rules moved out of the Dora ``mujoco_interface`` node unchanged; this checks them directly, without Dora: a silent slice's
gains are zeroed after the idle timeout (and only that slice's), time the host spent blocked is credited back, the single-model
ARM gate drops effort and refuses commands while disarmed, Soft mode expires with its commander, a stale or scene-less scene
change is refused, and the drift warning is rate-limited.
"""
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sim_supervisor import plant, write_config  # noqa: E402  (the 3-joint fixture)

from arm_control.plants.mujoco.scene_core import SceneCore  # noqa: E402


def sub(value, n, kp=100.):
    return dict(position=np.full(n, value), velocity=np.zeros(n), torque=np.zeros(n), kp=np.full(n, kp), kd=np.full(n, 5.),
                cartesian=None)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='scene_core_'))
    write_config(tmp, 0, 0)  # writes three.urdf
    logs = []

    def log(*args, **_):
        logs.append(" ".join(map(str, args)))

    # Slices: two actors in one vector, each with its own deadman.
    p = plant(tmp / 'three.urdf')
    p.load()
    core = SceneCore(p, {'a': dict(start=0, n=2), 'b': dict(start=2, n=1), 'never': dict(start=2, n=0)}, 3, period=.004,
                     idle_timeout=.1, log=log)
    assert core.command['kp'].tolist() == [60.] * 3          # the gentle spawn hold until commands arrive
    assert core.set_slice('a', sub(.1, 2), 10.0) and core.set_slice('b', sub(.2, 1), 10.0)
    assert not core.set_slice('nobody', sub(0, 1), 10.0)
    core.tick(10.05)
    assert core.command['kp'].tolist() == [100.] * 3, core.command['kp']
    core.set_slice('a', sub(.1, 2), 10.15)                    # a keeps talking, b falls silent
    core.tick(10.2)
    assert core.command['kp'].tolist() == [100., 100., 0.], core.command['kp']
    assert sum('b: no command' in m for m in logs) == 1
    for k in range(5):  # a keeps sending an EE-level Cartesian target (shared by the vector), so b stays "active": still once
        core.set_slice('a', dict(sub(.1, 2), cartesian=dict(k=[1.] * 6)), 10.25 + k * .01)
        assert core.command['cartesian'] == dict(k=[1.] * 6)  # a slice write carries the EE-level term
        core.tick(10.25 + k * .01)
        assert core.command['cartesian'] is None             # ...and a zeroed slice clears it: no limp arm on a Cartesian spring
    assert sum('b: no command' in m for m in logs) == 1      # said once per gap
    # A blocked host (a recompile) is credited: a slice silent only while the host was not listening keeps its gains.
    core.set_slice('a', sub(.1, 2), 20.0)
    core.credit_stall(.5)
    core.tick(20.3)
    assert core.command['kp'][:2].tolist() == [100., 100.], core.command['kp']
    core.tick(20.75)                                         # 0.75 s after the command, minus 0.5 s credited: past 0.1 s
    assert core.command['kp'][:2].tolist() == [0., 0.], core.command['kp']
    assert core.last_cmd_time['never'] == 0.0                # a stall credit never makes a silent slice look commanded
    p.close()

    # Single model: the ARM gate and Soft mode.
    p = plant(tmp / 'three.urdf')
    p.load()
    core = SceneCore(p, {}, 3, period=.004, idle_timeout=.1, log=log)
    core.set_single_armed(False)
    assert core.command['kp'].tolist() == [0.] * 3 and not core.set_single(sub(.3, 3), 1.0)
    core.set_single_armed(True)
    assert core.set_single(dict(sub(.3, 3), velocity=np.ones(3), pose_hold=dict(id=1), cartesian=dict(k=[1.] * 6)), 1.0)
    core.set_single_armed(False)                             # disarm drops every effort term, Soft and Cartesian included
    assert core.command['velocity'].tolist() == [0.] * 3 and core.command['pose_hold'] is None, core.command
    assert core.command['cartesian'] is None
    core.set_single_armed(True)
    # Soft mode expires with its commander. The rule is the core's, not the physics': a stub plant records what it is given
    # (the fixture has no EE for a real pose hold).
    from types import SimpleNamespace

    class Stub:
        def __init__(self):
            self.data, self._qadr, self.given = SimpleNamespace(qpos=np.array([.7, .8, .9])), np.arange(3), []
            self.scene_state, self.applied = SimpleNamespace(revision=5), []

        def motor_state(self):
            return dict(position=self.data.qpos.copy())

        def apply_scene_state(self, state):
            self.applied.append(state)

        def step(self, command):
            self.given.append(dict(command))
            return dict(position=self.data.qpos.copy())

    stub = Stub()
    core = SceneCore(stub, {}, 3, period=.004, idle_timeout=.1, log=log)
    core.set_single(dict(sub(.3, 3), velocity=np.ones(3), pose_hold=dict(id=1)), 2.0)
    core.tick(2.05)
    assert stub.given[-1].get('pose_hold') is not None
    core.tick(2.2)                                           # the Soft commander went quiet: hold where the arm is
    assert stub.given[-1]['pose_hold'] is None and stub.given[-1]['position'].tolist() == [.7, .8, .9], stub.given[-1]
    assert stub.given[-1]['velocity'].tolist() == [0.] * 3
    # A stale scene change is refused and nothing is applied.
    for revision in (5, 4):
        try:
            core.apply_scene(dict(revision=revision, state={}))
        except ValueError as exc:
            assert 'stale' in str(exc), exc
        else:
            raise AssertionError(f'stale scene revision {revision} was accepted')
    assert not stub.applied
    # The drift warning is rate-limited (at most one per 5 s of wall time), not said every step.
    before = len(logs)
    for k in range(40):
        core.tick(10.0 + k)                                  # 1 s of wall per 4 ms step: far behind
    assert 1 <= len(logs) - before <= 8, logs[before:]
    core = SceneCore(p, {}, 3, period=.004, idle_timeout=.1, log=log)
    try:
        core.apply_scene(dict(revision=1, state={}))
    except ValueError as exc:
        assert 'no scene state' in str(exc), exc
    else:
        raise AssertionError('a scene change was applied to a world without a scene')
    p.close()
    print('SceneCore: per-slice deadmans (said once, stall credited), single-model ARM gate, Soft expiry, stale/scene-less refusal, drift rate limit: PASS')


if __name__ == '__main__':
    main()
