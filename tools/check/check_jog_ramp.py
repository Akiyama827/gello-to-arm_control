#!/usr/bin/env python3
"""The controller's jog saturation never resumes from a ramp a cleared jog left behind (setpoint saturation review, 2026-10-09).

    PYTHONPATH=. python tools/check/check_jog_ramp.py

A real ArmController (fake executor, fake clock, a policy with velocity/acceleration limits). S1: a jog in flight, a plan
released (which clears the jog), the plan carries the arm, the plan cancelled and a jog arriving BEFORE the next tick: the
commanded target starts at the measured pose (it used to jump 0.25 rad back toward the pre-plan ramp). S2: a jog in flight at
speed, `clear_jog` (an owner left), a new jog before the next tick: it starts at rest at the measured pose. S3: a setpoint ON
a joint limit is reached and never passed (the arm stays ARMed). Plus the ramp's own self-check (control/jog_hold.py).
"""
from __future__ import annotations

import numpy as np

from arm_control.control.arm_controller import ArmController
from arm_control.control.doubles import _FakeExecutor, _FakeNode
from arm_control.control.execution_policy import ExecutionPolicy
from arm_control.control.jog_hold import _self_check as jog_hold_self_check
from arm_control.contracts._arrow import pack_json_message
from arm_control.contracts.motion import pack_control_update, pack_jog, pack_plan
from arm_control.contracts.motor import pack_motor_state, unpack_motor_command

HEALTH = pack_json_message("motor_health", dict(armed=True, state_fresh=True, any_fault=False))


def state(q):
    z = np.zeros(7)
    return pack_motor_state(np.full(7, float(q)), z, z, z, z, z, z, z)


def controller():
    clock = [10.0]
    policy = ExecutionPolicy(command_rate_hz=100, state_timeout_sec=.25, health_timeout_sec=1, start_pos_tol_rad=.1,
                             abort_pos_err_rad=.35, hold_relatch_rad=.3, torque_limits=np.array([87.] * 4 + [12.] * 3),
                             velocity_limits=np.full(7, 1.), acceleration_limits=np.full(7, 2.))
    c = ArmController(_FakeNode(), _FakeExecutor(), clock=lambda: clock[0], execution_policy=policy)
    c._set_arm(True)
    c.on_motor_health(HEALTH)
    c.on_motor_state(state(0.))
    c.tick()
    assert c.ready_sent
    return c, clock


def commanded(c) -> float:
    return [unpack_motor_command(v, 7) for t, v in c.node.sent if t == "motor_command"][-1]["position"][0]


def tick(c, clock, measured):
    clock[0] += .01
    c.tick()
    c.on_motor_state(state(measured))
    c.on_motor_health(HEALTH)


def main() -> int:
    jog_hold_self_check()
    # S1
    c, clock = controller()
    for _ in range(15):
        c.on_jog(pack_jog(q=np.full(7, .3)))
        tick(c, clock, commanded(c) if c.node.sent else 0.)
    q0 = c.last_state.position[0]
    c.on_plan(pack_plan(plan_id="p", phase="move", gated=True, times=[0, 1], positions=np.array([[q0] * 7, [q0 + .25] * 7]),
                        velocities=np.zeros((2, 7)), kp=[600] * 7, kd=[20] * 7))
    c.on_control(pack_control_update(execute="p"))
    assert c._running
    for k in range(60):
        tick(c, clock, q0 + .25 * (k + 1) / 60)
    c.on_control(pack_control_update(cancel=True, cancel_plan_id="p"))
    measured = c.last_state.position[0]
    c.on_jog(pack_jog(q=np.full(7, measured + .05)))  # before the next tick
    tick(c, clock, measured)
    assert abs(commanded(c) - measured) < .001, ("S1: resumed a stale ramp", commanded(c), measured)
    # S2
    c, clock = controller()
    for _ in range(40):  # the arm does not follow: the ramp runs ahead at speed
        c.on_jog(pack_jog(q=np.full(7, .3)))
        tick(c, clock, 0.)
    assert c._jog_ramp.v[0] > .5, c._jog_ramp.v
    c.clear_jog()
    c.on_jog(pack_jog(q=np.full(7, 0.)))  # a new owner holds where the arm is
    tick(c, clock, 0.)
    assert abs(commanded(c)) < 1e-9 and abs(c._jog_ramp.v[0]) < 1e-9, ("S2: not at rest at the measured pose", commanded(c),
                                                                      c._jog_ramp.v)
    # S3: a setpoint ON a joint limit (the supervisor admits it), from several distances at the FR3's 0.4/0.8: the ramp brakes
    # onto the limit and never commands past it (round 2 overshot ~1e-5 rad past the envelope: stopped and DISARMed)
    for distance in (.0123, .05, .1337, .21, .3):
        clock = [10.0]
        policy = ExecutionPolicy(command_rate_hz=100, state_timeout_sec=.25, health_timeout_sec=1, start_pos_tol_rad=.1,
                                 abort_pos_err_rad=.35, hold_relatch_rad=.3, torque_limits=np.array([87.] * 4 + [12.] * 3),
                                 velocity_limits=np.full(7, .4), acceleration_limits=np.full(7, .8),
                                 position_lower=np.full(7, -2.), position_upper=np.full(7, 1.))
        c = ArmController(_FakeNode(), _FakeExecutor(), clock=lambda: clock[0],
                          execution_policy=policy)
        c._set_arm(True)
        c.on_motor_health(HEALTH)
        c.on_motor_state(state(1. - distance))
        c.tick()
        for _ in range(150):
            assert c.on_jog(pack_jog(q=np.full(7, 1.))) is None and not c.stopped, (distance, c.stopped)
            tick(c, clock, commanded(c))
            assert commanded(c) <= 1., (distance, commanded(c))
        assert commanded(c) == 1. and c.armed and not c.stopped, (distance, commanded(c))
    print("jog ramp: a new jog starts at the measured pose at rest, also within the tick a jog was cleared in "
          "(plan release+cancel, clear_jog); a setpoint on a limit is reached, never passed; the ramp saturates velocity/acceleration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
