"""Deterministic command/integral capture for ownership refactors; no plant I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from arm_control.config import load_config_tree, load_robot_config
from arm_control.control.factory import arm_urdf, build_arm_controller
from arm_control.control.inprocess import QueueControllerNode
from arm_control.motion import JointState, TrajectoryPoint


def capture(config, mode):
    cfg = load_robot_config(config)
    mode_cfg = load_config_tree(mode) if mode else cfg.get('control_mode') or {}
    model = Path(arm_urdf(cfg))
    out = dict(urdf_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), commands=[])
    for plant_comp in (False, True):
        cfg.raw['arm']['plant_gravity_comp'] = plant_comp
        c = build_arm_controller(cfg, QueueControllerNode(), arm_id='capture', mode_cfg=mode_cfg)
        ex, n = c.executor, c.n_arm
        clock = [10.]
        q = np.linspace(-.1, .1, n)
        ex.set_payload(.5, com_offset=[.01, -.02, .12])
        for i in range(140):
            clock[0] += .5 if i == 90 else .01
            error = np.full(n, .005 if i < 100 else -.009)
            error[0] = .02 if i in (50, 51) else error[0]
            if i == 100:
                ex.set_payload(0.)
            if i == 110:
                ex.set_gains(kp=np.zeros(n))
            if i == 120:
                c.integral.reset()
            state = JointState(q.copy(), np.full(n, .01))
            point = TrajectoryPoint(i*.01, q+error, np.full(n, .02), np.full(n, .03))
            bias = c.integral.update(i, clock[0], point.position, state.position, ex.kp)
            command = (ex.hold_command(state, q_des=point.position, integral_bias=bias) if i < 30
                       else ex.step(point, state, integral_bias=bias))
            row = dict(plant_gravity_comp=plant_comp, cycle=i,
                command={key: np.asarray(getattr(command, key), dtype='<f8').tobytes().hex()
                         for key in ('q_des', 'qd_des', 'tau_ff', 'kp', 'kd')},
                integral=c.integral.bias.tolist())
            out['commands'].append(row)
            if i == 80:
                again = ex.step(point, state, integral_bias=bias)
                assert again.tau_ff.tobytes() == command.tau_ff.tobytes()
        assert np.any(np.asarray(out['commands'][-140+40]['integral']) != 0), 'fixture integral was off'
    return out


def check_factory_inputs():
    from check_sdk_supervisor import write_config
    with tempfile.TemporaryDirectory() as directory:
        cfg = load_robot_config(write_config(Path(directory), 1, 2))
        names = ['j1', 'j2', 'j3', 'finger']
        cfg.raw.update(joint_names=names, motor_names=names, num_motors=4)
        for key in ('ki', 'i_limit'):
            mode = {'controller': {key: [1., 2., 3., 4.]}}
            controller = build_arm_controller(cfg, QueueControllerNode(), arm_id='inputs', mode_cfg=mode)
            assert np.array_equal(getattr(controller.integral, key), [1., 2., 3.])
            for bad in ([1., 2., 3., 4., 5.], [1., 2., 3., np.nan], [[1., 2., 3., 4.]]):
                try:
                    build_arm_controller(cfg, QueueControllerNode(), arm_id='inputs',
                                         mode_cfg={'controller': {key: bad}})
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'factory admitted {key}={bad!r}')
    print('factory integral admission: complete motor vector checked before arm projection')



def check_update_order():
    """Expose the current construction-side updates; evidence, not a new law."""
    from arm_control.control.arm_controller import ArmController
    from arm_control.control.execution_policy import ExecutionPolicy
    from arm_control.control.trajectory_executor import JointTrajectoryExecutor
    from arm_control.control.integral import IntegralState

    class NoDynamics:
        def gravity(self, q):
            return np.zeros(len(q))

        def rnea(self, q, qd, qdd):
            return np.zeros(len(q))

    now = [0.]
    ex = JointTrajectoryExecutor('trace', ['j1', 'j2'], NoDynamics(),
        np.full(2, 100.), np.full(2, 10.), np.full(2, 50.))
    integral = IntegralState(2, ki=np.full(2, 200.), i_limit=np.ones(2))
    policy = ExecutionPolicy(100., .1, .1, .01, .05, .1, np.full(2, 50.))
    controller = ArmController(QueueControllerNode(), ex,
        gripper={'n_motors': 2, 'open_finger_m': 0., 'gains': (0., 0.), 'mimic': None}, integral_state=integral, execution_policy=policy, clock=lambda: now[0], plant_reports_health=False)
    state = controller.last_state
    integral.update(0, now[0], state.position, state.position, ex.kp)
    # One joint exceeds tracking admission; another is in the integral window.
    now[0] = .01
    point = TrajectoryPoint(.01, np.array([.1, .005]), np.zeros(2), np.zeros(2))
    candidate = ex.step(point, state, integral_bias=integral.update(1, now[0], point.position, state.position, ex.kp))
    before_send = integral.bias.copy()
    now[0] += .0001
    assert controller._send(candidate, None) is False
    held = controller.last_command
    assert np.array_equal(held.q_des, state.position)
    assert np.allclose(before_send, [0., .01])
    assert np.array_equal(integral.bias, before_send)
    assert np.array_equal(held.tau_ff, before_send)
    print('tracking-refused candidate: integral before send', before_send.tolist(),
          'measured-hold output', held.tau_ff.tolist(),
          'one update using final measured hold would yield [0.0, 0.0]')
    # Pure construction removes the historical extra 0.0001 Nm from a second call.
    integral.reset()
    target = np.full(2, .005)
    integral.update(0, now[0], target, state.position, ex.kp)
    now[0] += .01
    bias = integral.update(1, now[0], target, state.position, ex.kp)
    first = ex.hold_command(state, q_des=target, integral_bias=bias).tau_ff.copy()
    now[0] += .0001
    second = ex.hold_command(state, q_des=target, integral_bias=integral.bias).tau_ff.copy()
    assert np.array_equal(second, first)
    print('double construction: old extra [0.0001, 0.0001] Nm; new', (second-first).tolist())

    # Trace real tick paths, including construction that never becomes output.
    from arm_control.control.reference import TrajectoryReference
    from arm_control.messages import pack_motor_state
    from arm_control.motion import JointTrajectory

    calls = []
    integrate = integral.update

    def trace(cycle_id, cycle_time, q_des, q, kp):
        result = integrate(cycle_id, cycle_time, q_des, q, kp)
        calls.append((np.asarray(q_des)-q).tolist())
        return result

    integral.update = trace  # fixture instrumentation, not a second state owner
    integral.reset()
    controller._set_arm(True)
    controller.last_command = controller._hold_anchor = None
    now[0] = 1.
    controller._last_state_at = now[0]
    controller.tick()
    assert calls == [[0., 0.]]
    print('initial measured hold: 1 explicit update, old 2 command-side updates')
    # Keep one still-fresh observation across two controller cycles.
    controller._hold_anchor.q_des[:] = .005
    calls.clear()
    now[0] = 1.01
    controller.tick()
    first = integral.bias.copy()
    now[0] = 1.02
    controller.tick()
    assert len(calls) == 2 and np.allclose(integral.bias-first, .01)
    assert controller._last_state_at == 1.
    print('two cycles reuse one fresh observation: 1 update per tick, bias grows')
    calls.clear()
    saved = integral.bias.copy()
    controller.tick()  # identical clock: law invoked again, zero dt
    assert len(calls) == 1 and np.array_equal(integral.bias, saved)
    print('duplicate tick at identical time: integral called, zero dt')
    calls.clear()
    now[0] = 1.2
    controller.tick()
    assert not calls and np.array_equal(integral.bias, saved)
    zeros = np.zeros(2)
    controller.on_motor_state(pack_motor_state(np.full(2, np.nan), *[zeros]*7))
    assert not calls and np.array_equal(integral.bias, saved)
    print('stale/nonfinite observation: no update; ordinary pause retains bias')
    controller._set_arm(False)
    assert np.array_equal(integral.bias, zeros) and integral._time is None
    print('DISARM: bias and integral time reset')

    # Exercise the complete tick -> jog command -> tracking fallback chain too.
    controller._set_arm(True)
    controller._last_state_at = now[0] = 2.
    controller.tick()
    calls.clear()
    controller._jog.set(np.array([.1, .005]), now[0])
    now[0] = 2.01
    controller.tick()
    assert calls == [[.1, .005]]
    assert np.allclose(controller.last_command.tau_ff, [0., .01])
    print('tracking-abort tick: candidate updated once; fallback reads same bias')

    # Successful settling constructs an endpoint hold and then rebuilds it.
    calls.clear()
    controller._last_state_at = now[0] = 2.02
    goal = np.full(2, .0005)
    trajectory = JointTrajectory([0., .1], np.stack([goal, goal]), np.zeros((2, 2)))
    controller._reference = TrajectoryReference(trajectory, 1.)
    controller._plan = dict(plan_id='settled', phase='settled', positions=trajectory.positions)
    controller._running = True
    controller._settling.begin(1.1)
    controller._settling._settled_since = 1.1
    controller.tick()
    assert calls == [goal.tolist()] and not controller._running
    print('ordinary completion tick: 1 endpoint update, old 2')

    # Timeout selects measured hold before integration, not the retired endpoint.
    calls.clear()
    saved = integral.bias.copy()
    controller._last_state_at = now[0] = 9.
    goal = np.full(2, .005)  # inside integral window but outside done tolerance
    trajectory = JointTrajectory([0., .1], np.stack([goal, goal]), np.zeros((2, 2)))
    controller._reference = TrajectoryReference(trajectory, 1.)
    controller._plan = dict(plan_id='timeout', phase='timeout', positions=trajectory.positions)
    controller._running = True
    controller._settling.begin(1.1)
    controller.tick()
    assert calls == [[0., 0.]] and controller.frozen
    assert np.array_equal(integral.bias, saved)
    assert np.array_equal(controller.last_command.q_des, state.position)
    print('settling timeout: measured target selected before update; no retired-target error')

    # The event-driven compatibility path caches hold commands between states.
    controller.execution_policy = None
    controller.last_command = controller._hold_anchor = None
    calls.clear()
    controller.on_motor_state(pack_motor_state(*[zeros]*8))
    assert len(calls) == 1
    now[0] += .01
    controller.on_motor_state(pack_motor_state(*[zeros]*8))
    assert len(calls) == 1
    print('event-driven cached hold: only first state constructs/integrates hold')

    # Preserve both established Soft transitions, without invoking an RT plant.
    controller.frozen = False
    controller.supports_pose_hold = True
    controller._set_pose_hold(dict(id=7, kc=[300]*3+[30]*3, dc=[35]*3+[8]*3,
                                   nullspace_kp=.5, nullspace_kd=2.))
    assert controller._pose_hold is not None and np.array_equal(integral.bias, zeros)
    integral.update(0, now[0], target, state.position, ex.kp)
    integral.update(1, now[0]+.01, target, state.position, ex.kp)
    assert np.any(integral.bias)
    controller._cancel('leave Soft fixture')
    assert controller._pose_hold is None and np.array_equal(integral.bias, zeros)
    print('Soft entry/exit: existing bias reset semantics preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--check-update-order', action='store_true')
    parser.add_argument('--mode')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.check_update_order:
        check_update_order()
        raise SystemExit(0)
    if args.check_inputs:
        check_factory_inputs()
        raise SystemExit(0)
    if not args.config or not args.out:
        parser.error("capture requires --config and --out")
    output = capture(args.config, args.mode)
    Path(args.out).write_text(json.dumps(output, sort_keys=True, indent=2)+'\n')
    print(f"{len(output['commands'])} command records, model {output['urdf_sha256']}, "
          f"trace {hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}")
