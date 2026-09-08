"""Assert console controls without HTTP sockets, Dora graphs or hardware."""
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from arm_control.ui import arm_console as console


class FK:
    def poses(self, q, grip):
        return {'geoms': [], 'ee': {'p': list(q[:3]), 'q': [0, 0, 0, 1]}}


def refused(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid input accepted')


def main():
    assert 'gain_presets' in inspect.signature(console.ControlPanel).parameters, 'missing confirmed preset state'
    runtime = inspect.getsource(console._run)
    assert 'panel.jog_command()' in runtime and 'panel.apply_pose(ik)' in runtime, 'runtime does not consume new controls'
    assert 'panel.can_execute(pending_revision)' in runtime, 'runtime does not guard stale Execute'
    clock = [100.0]
    gains = {'track': (np.ones(7), np.ones(7)), 'float': (np.zeros(7), np.ones(7))}
    with patch.object(console, 'ConsoleServer'), patch.object(console.time, 'monotonic', side_effect=lambda: clock[0]):
        panel = console.ControlPanel(
            [f'j{i}' for i in range(7)], [-3]*7, [3]*7, 0, FK(),
            extra_buttons=('Gains: track', 'Gains: float', 'Gains: soft'),
            gain_presets=gains,
        )
        assert panel._state()['mode']['selected'] is None
        snapshot = {'kind': 'mode', 'ok': True, 'reason': 'joint',
                    'mode_state': {'law': 'joint', 'kp': [1]*7, 'kd': [1]*7}}
        panel.accept_mode(snapshot)
        assert panel._state()['mode']['selected'] == 'track'
        panel._post('click', {'button': 'Gains: float'})
        assert panel._state()['mode']['selected'] == 'track'
        assert panel._state()['mode']['pending'] == 'float'
        panel.accept_mode(dict(snapshot, ok=False, reason='refused'))
        assert panel._state()['mode'] == {'selected': 'track', 'pending': None, 'reason': 'refused'}
        panel.accept_mode(dict(snapshot, mode_state={'law': 'joint', 'kp': [0]*7, 'kd': [1]*7}))
        assert panel._state()['mode']['selected'] == 'float'
        panel.accept_mode(dict(snapshot, mode_state={'law': 'joint', 'kp': [0.5]*7, 'kd': [1]*7}))
        assert panel._state()['mode']['selected'] == 'custom'
        panel.accept_mode(dict(snapshot, mode_state={'law': 'soft', 'kp': [0.5]*7, 'kd': [1]*7}))
        assert panel._state()['mode']['selected'] == 'soft'
        panel.accept_mode(snapshot)

        panel._post('jog_speed', {'speed_m_s': .005, 'joint_speed_rad_s': .05})
        for bad in (0, -.01, float('nan'), float('inf'), .010001, True, '', None):
            refused(lambda: panel._post('jog_speed', {'speed_m_s': bad, 'joint_speed_rad_s': .05}))
        refused(lambda: panel._post('jog_speed', {'speed_m_s': .005, 'joint_speed_rad_s': .150001}))
        panel.set_jog('x', 1, True)
        initial = panel.jog_command()
        assert initial[2:4] == (.005, .05)
        panel._post('jog_speed', {'speed_m_s': .01, 'joint_speed_rad_s': .15})
        panel.set_jog('x', 1, True)  # heartbeat, not another press
        assert panel.jog_command() == initial
        panel.set_jog('x', 1, False)
        assert panel.jog_command() is None
        panel.set_jog('x', 1, True)
        assert panel.jog_command()[2:4] == (.01, .15)
        assert panel.jog_command()[4] > initial[4]
        clock[0] += 1.1
        assert panel.jog_command() is None
        assert panel._state()['mode']['selected'] is None

        assert panel._state()['pose']['current'] is None
        panel.set_measured([.1, .2, .3, 0, 0, 0, 0])
        np.testing.assert_allclose(panel._state()['pose']['current']['xyz_mm'], [100, 200, 300])
        panel.set_sliders([.2, .3, .4, 0, 0, 0, 0])
        np.testing.assert_allclose(panel._state()['pose']['target']['xyz_mm'], [200, 300, 400])
        np.testing.assert_allclose(panel._state()['pose']['current']['xyz_mm'], [100, 200, 300])
        before = panel.sliders()
        for xyz in ([1, 2], [1, 2, float('nan')], [True, 2, 3], ['', 2, 3], [10**400, 2, 3]):
            refused(lambda: panel._post('pose', {'xyz_mm': xyz, 'rpy_deg': [0, 0, 0]}))
        pose = {'xyz_mm': [123, -456, 789], 'rpy_deg': [20, 30, 40]}
        T = console.pose_matrix(pose)
        np.testing.assert_allclose(T[:3, 3], [.123, -.456, .789])
        np.testing.assert_allclose(T[:3, :3], pin.rpy.rpyToMatrix(np.deg2rad([20, 30, 40])))
        for rpy in ([20, 30, 40], [40, 90, 20], [-70, -90, 150]):
            T = console.pose_matrix(dict(pose, rpy_deg=rpy))
            fields = console.pose_fields({'p': T[:3, 3], 'q': pin.Quaternion(T[:3, :3]).coeffs()})
            np.testing.assert_allclose(console.pose_matrix(fields), T, atol=1e-8)
        revision = panel.target_revision
        assert panel.set_plan([0, 1], [], revision=revision)
        panel._post('pose', pose)
        assert panel.planning_target() is None, 'planning used a target whose pose edit is still queued'
        assert panel.target_revision > revision
        assert not panel.set_plan([0, 1], [], revision=revision), 'stale preview accepted'
        assert not panel.can_execute(revision), 'stale target remains executable'
        panel.apply_pose(SimpleNamespace(solve=lambda T, q: None))
        np.testing.assert_array_equal(panel.sliders(), before)
        assert 'unreachable' in panel._state()['pose']['status'].lower()
        panel._post('pose', pose)
        panel.apply_pose(SimpleNamespace(solve=lambda T, q: np.full(7, .5)))
        np.testing.assert_array_equal(panel.sliders(), np.full(7, .5))
        target, generation = panel.planning_target()
        np.testing.assert_array_equal(target, panel.sliders())
        assert generation == panel.target_revision
        sent = []
        assert panel.execute_if_current(generation, lambda: sent.append('execute'))
        panel._post('pose', pose)
        panel._click('DISARM')
        assert not panel._state()['pose']['status'].startswith('Applying'), 'cancelled pose is still applying'
        assert not panel.execute_if_current(generation, lambda: sent.append('stale'))
        assert sent == ['execute']
        panel.set_cart_pending(0, delta=.001)
        panel.set_sliders([.6]*7)
        assert panel.pop_cart() is None, 'old gizmo edit survived a newer slider target'
        assert panel._armed is None and not panel.clicked('ARM') and not panel.clicked('Execute')
        clock[0] += 1.1
        assert panel._state()['pose']['current'] is None
        panel.set_measured([float('nan')]*7)
        assert panel._state()['pose']['current'] is None
    print('console controls: confirmed modes, speed bounds/latching, pose and target revisions OK')


if __name__ == '__main__':
    main()
