"""Opt-in execution guards: direct fake-clock assertions, no hardware/pytest."""
from __future__ import annotations

import numpy as np
from dataclasses import replace

from arm_control.control.arm_controller import ArmController, _FakeExecutor, _FakeNode, _GRIPPER
from arm_control.messages import (
    pack_control_update, pack_json_message, pack_motor_state, pack_plan,
    pack_jog, unpack_json_message, unpack_motor_command,
)


def state(q=0.0, velocity=0.0):
    z = np.zeros(7)
    return pack_motor_state(np.full(7, q), np.full(7, velocity), z, z, z, z, z, z)


def plan(ident="p", start=0.0, width=7, **changes):
    q = np.full((2, width), start)
    fields = dict(plan_id=ident, phase="move", gated=True, times=[0, 1],
                  positions=q, velocities=np.zeros_like(q), kp=[600]*width, kd=[20]*width)
    return pack_plan(**(fields | changes))


def setup(*, strict=True, health=True):
    from arm_control.control.execution_policy import ExecutionPolicy

    clock = [10.0]
    policy = ExecutionPolicy(
        command_rate_hz=100, state_timeout_sec=.25, health_timeout_sec=1,
        start_pos_tol_rad=.1, abort_pos_err_rad=.35, hold_relatch_rad=.3,
        torque_limits=np.array([87]*4+[12]*3),
    ) if strict else None
    c = ArmController(_FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
                      clock=lambda: clock[0], plant_reports_health=health,
                      execution_policy=policy)
    c._set_arm(True)
    if health:
        c.on_motor_health(pack_json_message("motor_health", dict(
            armed=True, state_fresh=True, any_fault=False, supports_pose_hold=True)))
    c.on_motor_state(state())
    c.tick()
    assert c.ready_sent
    c.node.sent.clear()
    return c, clock


def commands(c):
    return [unpack_motor_command(v, 7) for t, v in c.node.sent if t == "motor_command"]


def main():
    c, clock = setup()
    old_kp = c.executor.kp.copy()
    for bad in (plan(start=.2), plan(width=6), plan(start=float("nan")),
                plan(kp=[float("inf")]*7), plan(times=[0, float("nan")])):
        c.on_plan(bad)
        assert c._plan is None and np.array_equal(c.executor.kp, old_kp)
    c.on_plan(plan())
    assert not c._running
    c.on_control(pack_control_update(execute="stale-id"))
    assert not c._running
    c.on_motor_state(state(.11))
    c.on_control(pack_control_update(execute="p"))
    assert not c._running and c._plan is None, "Execute must recheck measured start"

    c, clock = setup()
    c.on_plan(plan())
    c.on_control(pack_control_update(execute="p"))
    assert c._running
    c.on_motor_state(state(.4))
    assert not commands(c), "strict commands are deadline-driven, not event-driven"
    c.tick()
    assert not c._running
    assert np.allclose(commands(c)[-1]["position"], .4), "abort must hold measured before output"
    results = [unpack_json_message(v) for t, v in c.node.sent if t == "controller_event"]
    assert len([r for r in results if r["kind"] == "leg_result"]) == 1
    assert not results[-1]["ok"], "abort cannot also report success"

    for running in (False, True):
        c, clock = setup()
        c.on_plan(plan())
        if running:
            c.on_control(pack_control_update(execute="p"))
        clock[0] += .26
        c.tick()
        assert not commands(c) and c._plan is None and not c._running
        assert c.armed and not c.ready_sent, "state outage must not auto-disarm"
        c.on_motor_state(state(.2))
        c.tick()
        assert np.allclose(commands(c)[-1]["position"], .2)
        assert not c._running, "recovery must never resume old motion"
    c, clock = setup()
    c.on_plan(plan())
    clock[0] += .26
    c.on_motor_state(state())
    assert c._plan is None, "gap recovery must catch a stalled event loop"
    # Recovery has fresh observations but has not yet reached a servo tick.
    # A reviewed plan accepted here must still be dropped by a second outage.
    c.on_plan(plan("between-outages"))
    assert c._plan is not None and c._observation_paused
    clock[0] += .26
    c.on_motor_state(state())
    assert c._plan is None, "a second outage retained a newly pending plan"

    c, clock = setup()
    c.on_motor_state(state(float("nan")))
    c.tick()
    assert not commands(c) and not c.ready_sent
    c.on_motor_state(state())
    c.tick()
    assert commands(c)

    c, clock = setup()
    c.tick()
    c.on_motor_state(state(.002))
    c.tick()
    assert np.allclose(commands(c)[-1]["position"], 0), "small displacement retains anchor"
    c.on_motor_state(state(.04))
    c.tick()
    assert np.allclose(commands(c)[-1]["position"], .04), "torque cap must force relatch"

    c, clock = setup()
    c.on_plan(plan())
    c.on_motor_health(pack_json_message("motor_health", dict(
        armed=True, state_fresh=False, any_fault=True, latched_fault="stale")))
    c.tick()
    assert not commands(c) and c._plan is None and not c.ready_sent
    c, clock = setup()
    clock[0] += 1.01
    c.on_motor_state(state())
    c.tick()
    assert not commands(c), "fresh states cannot conceal expired health"

    c, clock = setup(strict=False)
    c.on_plan(plan(start=.2))
    c.on_control(pack_control_update(execute="p"))
    assert c._running, "profile-off behavior changed"
    clock[0] += .6
    c.on_motor_state(state())
    assert c._running and commands(c), "profile-off scheduling/recovery changed"
    check_deadlines()
    check_interpolated_limits()
    check_modes_and_validation()
    check_cartesian_lifecycle()
    check_completion_order()
    print("execution policy: start/release, tracking, outage, finite state, relatch, health, opt-out OK")


def check_deadlines():
    # A 1 kHz input source must still produce ~100 Hz commands, including
    # non-state traffic. One 40 ms scheduler stall must not cause a burst.
    c, clock = setup(health=False)
    times = []
    original_send = c.node.send_output
    start = clock[0]

    class Node:
        polls = 0

        def next(self, *, timeout):
            self.polls += 1
            clock[0] += min(timeout, .001)
            if self.polls == 50:
                clock[0] += .04
            if clock[0] >= start + .3:
                return {"type": "STOP"}
            if self.polls % 2:
                return {"type": "INPUT", "id": "motor_state", "value": state()}
            return {"type": "INPUT", "id": "unwired", "value": None}

        def send_output(self, topic, value):
            if topic == "motor_command":
                times.append(clock[0])
            original_send(topic, value)

    c.node = Node()
    c.run()
    assert 24 <= len(times) <= 28, times
    assert np.min(np.diff(times)) >= .009 - 1e-9, times
    # Idle inputs still wake on the deadline; invalid clock configuration
    # cannot silently select a second, incompatible trajectory timebase.
    try:
        ArmController(_FakeNode(), _FakeExecutor(), gripper=_GRIPPER,
                      execution_policy=c.execution_policy, state_period_s=.01)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed wall/plant clocks accepted")


def check_interpolated_limits():
    """A plan legal at every WAYPOINT and CHORD, illegal in flight.

    The executor flies a cubic Hermite between waypoints, so the samples a
    policy inspects are a proxy for the curve, not the curve. Here both
    endpoint velocities are exactly zero and the chord is 0.05 rad/s, while
    the cubic actually peaks at 0.075 rad/s halfway along -- 50% over the
    chord. Under a 0.06 cap every sampled check passes and the arm would fly
    past the limit. This is what ExecutionPolicy.plan_error's bounds() pass
    exists for; without it the case below is admitted.
    """
    from arm_control.control.execution_policy import ExecutionPolicy
    from arm_control.contracts.motion import unpack_plan

    policy = ExecutionPolicy(
        command_rate_hz=100, state_timeout_sec=.25, health_timeout_sec=1,
        start_pos_tol_rad=.1, abort_pos_err_rad=.35, hold_relatch_rad=.3,
        torque_limits=np.array([87.0]), velocity_limits=np.array([0.06]),
        acceleration_limits=np.array([1.0]),
    )
    over = unpack_plan(plan(width=1, times=[0, 1],
                            positions=np.array([[0.0], [0.05]]),
                            velocities=np.zeros((2, 1)), kp=[600], kd=[20]))
    # The proxies really are clean, or this proves nothing.
    assert np.abs(over["velocities"]).max() <= 0.06
    assert abs(np.diff(over["positions"], axis=0).max()) / 1.0 <= 0.06
    reason = policy.plan_error(over, np.zeros(1), 1)
    assert reason and "between waypoints" in reason, reason

    # A curve genuinely inside the cap is still admitted -- the check must not
    # simply refuse anything with curvature.
    ok = unpack_plan(plan(width=1, times=[0, 1],
                          positions=np.array([[0.0], [0.02]]),
                          velocities=np.zeros((2, 1)), kp=[600], kd=[20]))
    assert policy.plan_error(ok, np.zeros(1), 1) is None, \
        policy.plan_error(ok, np.zeros(1), 1)
    print("interpolated limits: curve certified, not just its samples OK")


def check_modes_and_validation():
    c, clock = setup()
    for key in ("command_rate_hz", "state_timeout_sec", "health_timeout_sec",
                "start_pos_tol_rad", "abort_pos_err_rad", "hold_relatch_rad"):
        for value in (0, -1, float("nan"), float("inf")):
            try:
                replace(c.execution_policy, **{key: value})
            except ValueError:
                pass
            else:
                raise AssertionError((key, value))
    for limits in ([0]*7, [-1]*7, [float("nan")]*7, []):
        try:
            replace(c.execution_policy, torque_limits=limits)
        except ValueError:
            pass
        else:
            raise AssertionError(limits)
    # Feedforward is re-evaluated at measured q while the TARGET stays fixed.
    seen = []
    hold = c.executor.hold_command

    def measured_hold(state, q_des=None):
        seen.append(state.position.copy())
        return hold(state, q_des)

    c.executor.hold_command = measured_hold
    c.tick()
    c.on_motor_state(state(.002))
    c.tick()
    assert np.allclose(seen[-1], .002)
    assert np.allclose(commands(c)[-1]["position"], 0)

    # A held jog expires independently of planning; corrupt setpoints never
    # reach encoding. Lost observation drops a jog/Soft, not just a plan.
    c, clock = setup()
    c.on_jog(pack_jog(q=[.02]*7))
    c.tick()
    assert np.allclose(commands(c)[-1]["position"], .02)
    clock[0] += .21
    c.on_motor_state(state(.01))
    c.tick()
    assert c._jog_q is None and np.allclose(commands(c)[-1]["position"], .01)
    c.on_jog(pack_jog(q=[float("nan")]*7))
    count = len(commands(c))
    c.tick()
    assert c.stopped and len(commands(c)) == count

    c, clock = setup()
    spec = dict(id=1, kc=[300]*3+[30]*3, dc=[35]*3+[8]*3,
                nullspace_kp=.5, nullspace_kd=2)
    c.on_control(pack_control_update(pose_hold=spec))
    c.on_motor_state(state(.5))
    c.tick()
    assert c._pose_hold is not None and not c.stopped, "Soft nullspace must stay compliant"
    assert commands(c)[-1]["pose_hold"] == spec
    clock[0] += .26
    c.tick()
    assert c._pose_hold is None and not c.ready_sent
    c.on_motor_state(state(.5))
    c.tick()
    assert commands(c)[-1]["pose_hold"] is None
    assert np.allclose(commands(c)[-1]["position"], .5)


def cartesian_plan(**changes):
    spec = dict(task_R=np.eye(3), kc=np.ones(6), dc=np.ones(6))
    poses = np.array([[.4, 0, .3, 1, 0, 0, 0]] * 2)
    return plan(**(dict(cartesian=spec, cartesian_poses=poses) | changes))


def check_cartesian_lifecycle():
    c, clock = setup()
    bad = dict(task_R=np.eye(3), kc=np.full(6, float("nan")), dc=np.ones(6))
    for value in (cartesian_plan(cartesian=bad),
                  cartesian_plan(cartesian_poses=np.full((2, 7), float("nan"))),
                  cartesian_plan(cartesian_poses=None)):
        c.on_plan(value)
        assert c._plan is None, "malformed Cartesian tail accepted"
    # The packer enforces some shapes, but a received plan must be checked
    # independently of whichever producer constructed its JSON.
    from arm_control.messages import unpack_plan
    body = unpack_plan(cartesian_plan())
    body["cartesian"]["kc"] = np.ones(5)
    assert c.execution_policy.plan_error(body, c.last_state.position, 7)
    body = unpack_plan(cartesian_plan())
    body["cartesian_poses"] = np.zeros((1, 7))
    assert c.execution_policy.plan_error(body, c.last_state.position, 7)

    for stop in ("outage", "abort", "cancel"):
        c, clock = setup()
        c.on_plan(cartesian_plan())
        c.on_control(pack_control_update(execute="p"))
        c.tick()
        assert commands(c)[-1]["cartesian"] is not None
        if stop == "outage":
            clock[0] += .26
            c.tick()
            c.on_motor_state(state(.2))
        elif stop == "abort":
            c.on_motor_state(state(.4))
        else:
            c.on_control(pack_control_update(cancel=True))
        c.tick()
        assert commands(c)[-1]["cartesian"] is None, (stop, commands(c)[-1])
        assert c._cartesian_now is None and c._cartesian_poses is None
        assert c._last_cartesian_pose is None


def check_completion_order():
    c, clock = setup(health=False)
    c.on_plan(cartesian_plan())
    c.on_control(pack_control_update(execute="p"))
    c.executor.done = lambda *args, **kwargs: True
    c.on_motor_state(state(.0005))
    c.tick()
    assert c.executor.steps == 0, "completed trajectory was stepped again"
    assert not c._running
    assert np.allclose(commands(c)[-1]["position"], .0005)
    assert commands(c)[-1]["cartesian"] is None
    c, clock = setup(health=False)
    c.on_plan(plan(completion={"vel_tol": 100, "goal_q": [0]*7, "residual_max": 100}))
    c.on_control(pack_control_update(execute="p"))
    # The permissive per-plan completion path passes infinite tolerances;
    # the legacy safety profile must still use the executor's own tolerances.
    c.executor.done = lambda *args, **kwargs: bool(kwargs)
    c.on_motor_state(state(.4))
    c.tick()
    results = [unpack_json_message(v) for t, v in c.node.sent if t == "controller_event"]
    assert results[-1]["kind"] == "leg_result" and not results[-1]["ok"]
    assert np.allclose(commands(c)[-1]["position"], .4)


if __name__ == "__main__":
    main()
