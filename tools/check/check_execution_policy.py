"""Opt-in execution guards: direct fake-clock assertions, no hardware/pytest."""
from __future__ import annotations

import numpy as np
from dataclasses import replace

from arm_control.control.arm_controller import ArmController
from arm_control.control.doubles import _FakeExecutor, _FakeNode
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


def setup(*, strict=True, health=True, **kw):
    from arm_control.control.execution_policy import ExecutionPolicy

    clock = [10.0]
    policy = ExecutionPolicy(
        command_rate_hz=100, state_timeout_sec=.25, health_timeout_sec=1,
        start_pos_tol_rad=.1, abort_pos_err_rad=.35, hold_relatch_rad=.3,
        torque_limits=np.array([87]*4+[12]*3), cartesian_kp_min=20.,
    ) if strict else None
    c = ArmController(_FakeNode(), _FakeExecutor(), clock=lambda: clock[0], plant_reports_health=health,
                      execution_policy=policy, **kw)
    c._set_arm(True)
    if health:
        c.on_motor_health(pack_json_message("motor_health", dict(
            armed=True, state_fresh=True, any_fault=False, supports_pose_hold=True)))
    c.on_motor_state(state())
    c.tick()
    assert c.ready_sent
    c.node.sent.clear()
    return c, clock


def advance(c, clock, seconds, q=0.0, dt=.1):
    """Run the controller forward, feeding state at a policy-legal rate.

    Jumping the clock in one step trips ``state_timeout_sec`` and CANCELS the
    leg -- which from outside is indistinguishable from completion (not
    running, no further steps, holding the measured pose). Anything asserting
    on completion has to get there without a state gap.
    """
    end = clock[0] + seconds
    while clock[0] < end:
        clock[0] = min(clock[0] + dt, end)
        c.on_motor_state(state(q))
        c.tick()
        assert not c._observation_paused, "state gap: the leg was cancelled, not completed"


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

    for running, velocity in ((False, 0.), (True, 0.), (False, .02)):
        c, clock = setup()
        c.on_plan(plan())
        if running:
            c.on_control(pack_control_update(execute="p"))
        c.on_motor_state(state(.002, velocity=velocity))
        c.on_control(pack_control_update(cancel=True, cancel_plan_id="p"))
        c.tick()
        assert c._plan is None and c._pending_traj is None and not c._running
        expected = .002 if running or velocity else 0.
        assert np.allclose(commands(c)[-1]["position"], expected), (
            "cancel must retain a staged hold, but reanchor an active motion"
        )

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
    check_clock_rule()
    check_interpolated_limits()
    check_goal_on_limit()
    check_measured_outside()
    check_factory_envelope()
    check_modes_and_validation()
    check_cartesian_lifecycle()
    check_completion_order()
    print("execution policy: start/release, tracking, outage, finite state, relatch, health, opt-out OK")


def check_clock_rule():
    # The 100 Hz command schedule is the supervisor's servo loop now (ArmController.run, the Dora loop, is gone);
    # here only the controller's own clock rule remains.
    c, _ = setup(health=False)
    # An invalid clock configuration cannot silently select a second, incompatible trajectory timebase.
    try:
        ArmController(_FakeNode(), _FakeExecutor(), execution_policy=c.execution_policy, state_period_s=.01)
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


def check_goal_on_limit():
    """A joint held ON its limit is flown, not faulted (measured live: joint 4 at -0.1518 faulted the second move in 6 of 9 runs).

    The evaluator returns a still segment's knot exactly, and the plan, curve and servo checks share one rounding allowance
    (ENVELOPE_TOL), so a command one ulp past the limit is legal and one a micro-radian past it is not.
    """
    from arm_control.control.execution_policy import ENVELOPE_TOL, ExecutionPolicy
    from arm_control.motion.types import JointServoCommand, JointTrajectory

    hi = -0.1518
    flat = JointTrajectory(times=np.array([0.0, 2.0]), positions=np.array([[0.0, hi], [0.3, hi]]),
                           velocities=np.zeros((2, 2)))
    held = {float(flat.sample_at(t).position[1]) for t in np.linspace(0, 2, 2001)}
    assert held == {hi}, sorted(held)[-3:]
    linear = JointTrajectory(times=np.array([0.0, 2.0]), positions=np.array([[hi], [hi]]))
    assert {float(linear.sample_at(t).position[0]) for t in np.linspace(0, 2, 2001)} == {hi}
    policy = ExecutionPolicy(
        command_rate_hz=100, state_timeout_sec=.25, health_timeout_sec=1,
        start_pos_tol_rad=.1, abort_pos_err_rad=.35, hold_relatch_rad=.3,
        torque_limits=np.array([87.0]), position_lower=np.array([-3.0]), position_upper=np.array([hi]),
    )

    def command(q):
        return JointServoCommand(np.array([q]), np.zeros(1), np.zeros(1), np.array([600.0]), np.array([20.0]))

    assert policy.command_valid(command(hi), 1)
    assert policy.command_valid(command(np.nextafter(hi, 0)), 1)            # one ulp past: rounding, legal
    assert policy.command_valid(command(hi + 0.5 * ENVELOPE_TOL), 1)
    assert not policy.command_valid(command(hi + 1e-6), 1)                 # a micro-radian past: refused
    from arm_control.contracts.motion import unpack_plan

    def waypoints(end):
        return unpack_plan(plan(width=1, times=[0, 1], positions=np.array([[hi - 0.01], [end]]),
                                velocities=np.zeros((2, 1)), kp=[600], kd=[20]))
    assert policy.plan_error(waypoints(np.nextafter(hi, 0)), np.array([hi - 0.01]), 1) is None   # a knot one ulp past: legal
    assert policy.plan_error(waypoints(hi + 1e-6), np.array([hi - 0.01]), 1)                     # a micro-radian past: refused
    # Never further out (operator 2026-10-09): a joint MEASURED past its limit may be held there and brought back in, never
    # driven further out; inside the envelope nothing changes.
    out = hi + 0.02
    assert policy.command_valid(command(out), 1, np.array([out]))               # hold where gravity rested it
    assert policy.command_valid(command(hi + 0.01), 1, np.array([out]))         # back toward the envelope
    assert not policy.command_valid(command(out + 1e-6), 1, np.array([out]))    # further out: refused
    assert not policy.command_valid(command(hi + 1e-6), 1, np.array([hi - 0.1]))  # measured inside: the envelope as before
    assert not policy.command_valid(command(out), 1)                            # no measurement: the envelope as before
    back = unpack_plan(plan(width=1, times=[0, 1], positions=np.array([[out], [hi - 0.05]]),
                            velocities=np.zeros((2, 1)), kp=[600], kd=[20]))
    assert policy.plan_error(back, np.array([out]), 1) is None                  # a plan out of the stop
    further = unpack_plan(plan(width=1, times=[0, 1], positions=np.array([[out], [out + 0.01]]),
                               velocities=np.zeros((2, 1)), kp=[600], kd=[20]))
    assert "envelope" in policy.plan_error(further, np.array([out]), 1)        # deeper: refused
    print("goal on a joint limit: held exactly, one rounding allowance for plan, curve and servo OK")


def check_measured_outside():
    """Never further out (operator 2026-10-09): a joint gravity rested past a limit ARMs and is held where it rests -- as it
    creeps back in under the hold (a command built from an older sample) or settles a little further out on its stop --
    may be planned or jogged back in, and never driven beyond where it has been. A joint inside at ARM never widens (an
    overshoot grants nothing); re-ARM starts over."""
    from arm_control.control.execution_policy import ExecutionPolicy
    from arm_control.motion import JointServoCommand

    def controller():
        clock = [10.0]
        policy = ExecutionPolicy(
            command_rate_hz=100, state_timeout_sec=.25, health_timeout_sec=1,
            start_pos_tol_rad=.1, abort_pos_err_rad=.35, hold_relatch_rad=.3,
            torque_limits=np.array([87]*4+[12]*3), position_lower=np.full(7, -1.0), position_upper=np.full(7, 1.0))
        c = ArmController(_FakeNode(), _FakeExecutor(), clock=lambda: clock[0],
                          plant_reports_health=True, execution_policy=policy)
        c._set_arm(True)
        c.on_motor_health(pack_json_message("motor_health", dict(
            armed=True, state_fresh=True, any_fault=False, supports_pose_hold=True)))
        return c, clock, policy

    def feed(c, clock, *qs):
        for q in qs:
            clock[0] += .01
            c.on_motor_state(state(q))
            c.tick()
            assert not c.stopped, (q, "a command where the arm rested at ARM faulted")

    def hold(q):
        return JointServoCommand(np.full(7, q), np.zeros(7), np.zeros(7), np.full(7, 600.), np.full(7, 20.))

    # ARMed where gravity rested it, 0.02 rad past the lower limit; then the hold pulls it back in, a sample behind
    c, clock, policy = controller()
    feed(c, clock, -1.02, -1.0199, -1.0198, -1.0197)
    assert np.allclose(commands(c)[-1]["position"], -1.02, atol=1e-3)
    # A jog (no hold anchor) built from a sample older than the latest, as the arm keeps creeping in.
    assert c.on_jog(pack_jog(q=[-1.0197] * 7)) is None
    feed(c, clock, -1.0196, -1.0195, -1.0194)
    assert c._hold_anchor is None
    assert policy.command_valid(hold(-1.02), 7, c._where_been())        # where it rested at ARM
    assert not policy.command_valid(hold(-1.021), 7, c._where_been())   # further out: never
    # A plan out of the stop, its first knot where the arm rested (older than the latest sample): admitted.
    c.on_control(pack_control_update(cancel=True))
    feed(c, clock, -1.0193)
    c.on_plan(plan("out", positions=np.array([[-1.0195] * 7, [-.95] * 7])))
    assert c._plan is not None, "a plan out of the stop from where the arm rested was refused"
    c.on_plan(plan("deeper", positions=np.array([[-1.0195] * 7, [-1.03] * 7])))
    assert c._plan is None or c._plan.get("plan_id") != "deeper", "a plan further out was admitted"
    # Resting on its stop, the joint settles a little further out under the hold: holding it where it now is is no fault.
    feed(c, clock, -1.0205)
    assert policy.command_valid(hold(-1.0205), 7, c._where_been())
    assert not policy.command_valid(hold(-1.0206), 7, c._where_been())  # beyond anywhere it has been: never
    c._set_arm(False)
    c.on_motor_state(state(-1.01))     # re-ARMed resting outside again, but less far
    c._set_arm(True)
    assert policy.command_valid(hold(-1.01), 7, c._where_been())
    assert not policy.command_valid(hold(-1.02), 7, c._where_been())    # a new ARM grants nothing from before it

    # Resting past its LOWER limit widens only that side: a traverse past the upper limit grants nothing there.
    c, clock, policy = controller()
    feed(c, clock, -1.02)
    clock[0] += .01
    c.on_motor_state(state(1.003))
    assert not policy.command_valid(hold(1.003), 7, c._where_been())
    assert policy.command_valid(hold(-1.02), 7, c._where_been())

    # Inside at ARM: an overshoot past the limit while armed widens nothing (no ratchet).
    c, clock, policy = controller()
    feed(c, clock, -.99)
    clock[0] += .01
    c.on_motor_state(state(-1.003))
    clock[0] += .01
    c.on_motor_state(state(-.999))
    assert not policy.command_valid(hold(-1.003), 7, c._where_been())


def check_factory_envelope():
    """The position envelope comes from the DESCRIPTION, never from YAML.

    A deployment that restates the machine's own limits gets a second copy
    with its own way of going stale -- real-add carried a hand-derived inner
    envelope ([-2.3093, ...]) narrower than the FR3's factory bounds and
    justified only by a comment. build_execution_policy fills it from the
    URDF, and an explicit declaration can only NARROW the machine deliberately.
    """
    from arm_control.control.execution_policy import build_execution_policy

    base = dict(command_rate_hz=100, state_timeout_sec=.25, health_timeout_sec=1,
                start_pos_tol_rad=.1, abort_pos_err_rad=.35, hold_relatch_rad=.3)
    tau = np.array([87.0, 12.0])
    described = (np.array([-2.9007, -3.0508]), np.array([2.9007, 3.0508]))

    filled = build_execution_policy(base, torque_limits=tau, joint_limits=described)
    assert np.allclose(filled.position_lower, described[0]), filled.position_lower
    assert np.allclose(filled.position_upper, described[1]), filled.position_upper

    # An explicit narrowing wins; it is an operating choice, not an invention.
    narrowed = build_execution_policy(
        dict(base, position_lower=[-1.0, -1.0], position_upper=[1.0, 1.0]),
        torque_limits=tau, joint_limits=described)
    assert np.allclose(narrowed.position_lower, [-1.0, -1.0])

    # Neither side may expand the description. A half-declaration must be
    # refused rather than silently overwritten by the inherited bounds.
    for lo, hi in (([-3.0, -1.0], [1.0, 1.0]),
                   ([-1.0, -1.0], [3.0, 1.0]),
                   (None, [1.0, 1.0]), ([-1.0, -1.0], None)):
        try:
            build_execution_policy(dict(base, position_lower=lo, position_upper=hi),
                                   torque_limits=tau, joint_limits=described)
        except ValueError as exc:
            expected = "both position bounds" if lo is None or hi is None else "only narrow"
            assert expected in str(exc), exc
        else:
            raise AssertionError((lo, hi))

    equal = build_execution_policy(
        dict(base, position_lower=described[0], position_upper=described[1]),
        torque_limits=tau, joint_limits=described)
    for name in ("position_lower", "position_upper", "torque_limits"):
        assert getattr(equal, name).tobytes() == getattr(filled, name).tobytes()
        assert not getattr(equal, name).flags.writeable

    # A continuous joint reports infinite bounds; leave them unset rather
    # than fabricate a number the machine never stated.
    infinite = build_execution_policy(
        base, torque_limits=tau,
        joint_limits=(np.array([-np.inf, -1.0]), np.array([np.inf, 1.0])))
    assert infinite.position_lower is None

    # Explicit bounds on a continuous joint are permitted, while the other
    # joint's finite factory bound remains authoritative.
    mixed = (np.array([-np.inf, -1.0]), np.array([np.inf, 1.0]))
    build_execution_policy(dict(base, position_lower=[-10., -.5], position_upper=[10., .5]),
                           torque_limits=tau, joint_limits=mixed)
    try:
        build_execution_policy(dict(base, position_lower=[-10., -.5], position_upper=[10., 1.1]),
                               torque_limits=tau, joint_limits=mixed)
    except ValueError as exc:
        assert "only narrow" in str(exc)
    else:
        raise AssertionError("continuous joint concealed another joint's factory bound")

    assert build_execution_policy(None, torque_limits=tau) is None
    print("factory envelope: read from the description, narrowing honoured OK")


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

    def measured_hold(state, q_des=None, *, integral_bias=0.0):
        seen.append(state.position.copy())
        return hold(state, q_des, integral_bias=integral_bias)

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
    assert not c._jog.active and np.allclose(commands(c)[-1]["position"], .01)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        c, clock = setup()
        c.on_jog(pack_jog(q=[invalid]*7))
        count = len(commands(c))
        def no_update(*args):
            raise AssertionError("invalid jog reached integral update")
        c.integral.update = no_update
        c.tick()
        assert c.stopped and not c.armed and len(commands(c)) == count
        events = [unpack_json_message(v) for t, v in c.node.sent if t == "controller_event"]
        assert events[-1]["reason"] == "non-finite or invalid servo command"

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
    spec = dict(frame="base", task_R=np.eye(3), kc=np.ones(6), dc=np.ones(6))
    poses = np.array([[.4, 0, .3, 1, 0, 0, 0]] * 2)
    return plan(**(dict(cartesian=spec, cartesian_poses=poses) | changes))


def check_cartesian_lifecycle():
    c, clock = setup()
    # The packer refuses a malformed spec, but a received plan is checked independently of whichever producer built its JSON.
    from arm_control.messages import unpack_plan
    measured = c.last_state.position
    assert not c.execution_policy.plan_error(unpack_plan(cartesian_plan()), measured, 7)
    assert not c.execution_policy.plan_error(unpack_plan(cartesian_plan(cartesian_poses=None)), measured, 7), \
        "poses are visualisation only: the plant targets FK(q_des)"
    for key, value in (("kc", np.ones(5)), ("kc", np.full(6, float("nan"))), ("kc", np.full(6, 1001.)),
                       ("frame", "world"), ("task_R", -np.eye(3))):
        body = unpack_plan(cartesian_plan())
        body["cartesian"][key] = value
        assert "Cartesian impedance spec refused" in c.execution_policy.plan_error(body, measured, 7), (key, value)
    body = unpack_plan(cartesian_plan())
    body["cartesian_poses"] = np.zeros((1, 7))
    assert c.execution_policy.plan_error(body, measured, 7)
    # Operator 2026-10-09: the RT's fault hold keeps only the joint law, so a Cartesian leg carries kp >= the floor everywhere.
    low = cartesian_plan(kp=[600] * 6 + [19.9])
    c.on_plan(low)
    assert c._plan is None, "a Cartesian leg under the joint-kp floor was accepted"
    assert "joint kp >= 20" in c.execution_policy.plan_error(unpack_plan(low), measured, 7)
    assert not c.execution_policy.plan_error(unpack_plan(plan(kp=[600] * 6 + [19.9])), measured, 7), \
        "the floor binds Cartesian legs only"
    from dataclasses import replace
    floorless = replace(c.execution_policy, cartesian_kp_min=None)
    assert "no cartesian_kp_min" in floorless.plan_error(unpack_plan(cartesian_plan()), measured, 7)

    for stop in ("outage", "abort", "cancel"):
        c, clock = setup()
        c.on_plan(cartesian_plan())
        c.on_control(pack_control_update(execute="p"))
        c.tick()
        sent = commands(c)[-1]["cartesian"]
        assert sent is not None and sent["frame"] == "base" and "pose" not in sent, sent  # the plant targets FK(q_des)
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
        assert c._cartesian_now is None and c.reference is None

    # The law lives with its leg and nowhere else: every hold is the joint law only (the RT box's own hold), so no later event
    # can leave a spring on under joint gains the floor never admitted. With and without a policy.
    for strict in (True, False):
        for after in ("complete", "disarm_rearm", "gains"):
            c, clock = setup(strict=strict, health=False, settle_dwell_s=0.)
            c.on_plan(cartesian_plan())
            c.on_control(pack_control_update(execute="p"))
            advance(c, clock, .1)
            assert commands(c)[-1]["cartesian"] is not None, (strict, after)
            if after == "complete":
                c.executor.at_goal = lambda *args, **kwargs: True
                advance(c, clock, 1.2)
                assert not c._running, (strict, after)
            elif after == "disarm_rearm":
                c._set_arm(False)
                c._set_arm(True)
            else:
                c.on_control(pack_control_update(gains=dict(kp=[5.] * 7, kd=[1.] * 7)))
            sent = len(commands(c))
            for _ in range(3):
                clock[0] += .01
                c.on_motor_state(state())
                c.tick()
            later = commands(c)[sent:]
            assert later and all(m["cartesian"] is None for m in later), (strict, after, [m["cartesian"] for m in later])

    # A jog while a gated Cartesian plan is staged (loaded, not released) is the joint law: the spec belongs to the leg.
    c, clock = setup()
    c.on_plan(cartesian_plan())
    assert c._plan is not None and not c._running
    c.on_jog(pack_jog(q=[.02] * 7))
    c.tick()
    assert np.allclose(commands(c)[-1]["position"], .02) and commands(c)[-1]["cartesian"] is None

    # A plan that names itself but does not decode is refused with its reason; the controller lives on.
    import json
    body = json.loads(json.dumps(unpack_json_message(cartesian_plan(), expected_schema="plan")))
    body["cartesian"]["kc"] = [2000.] * 6
    c, clock = setup()
    c.on_plan(pack_json_message("plan", body))
    assert c._plan is None and not c.stopped
    events = [unpack_json_message(v) for t, v in c.node.sent if t == "controller_event"]
    assert any(e.get("kind") == "leg_result" and not e["ok"] and "malformed plan" in e["reason"] for e in events), events
    c.on_plan(cartesian_plan())
    assert c._plan is not None, "a good plan after a malformed one must load"


def check_completion_order():
    # dwell 0 isolates the ORDERING invariant this stanza is about, the way
    # arm_controller._controller() does for the same reason. The production
    # dwell is exercised below.
    c, clock = setup(health=False, settle_dwell_s=0.)
    c.on_plan(cartesian_plan())
    c.on_control(pack_control_update(execute="p"))
    # Completion is gated on the trajectory clock since bounded settling landed
    # (fffec10): before _leg_end there is nothing to complete and the executor
    # is SUPPOSED to keep stepping. Play the reference out first.
    advance(c, clock, .9, q=.0005)
    assert c._running and c.executor.steps, "the reference must play out first"
    c.executor.at_goal = lambda *args, **kwargs: True
    stepped = c.executor.steps
    clock[0] += .1                           # t == _leg_end; dwell 0 completes now
    c.on_motor_state(state(.0005))
    c.tick()
    assert c.executor.steps == stepped, "completed trajectory was stepped again"
    assert not c._running and c.reference is None, "a finished leg releases its reference"
    assert np.allclose(commands(c)[-1]["position"], 0), "successful joint goal must retain its endpoint"
    assert commands(c)[-1]["cartesian"] is None
    c.on_motor_state(state(.0075))
    c.on_control(pack_control_update(cancel=True))
    c.tick()
    assert np.allclose(commands(c)[-1]["position"], .0075), (
        "ordinary Cancel after completion must still hold the measured pose"
    )
    c, clock = setup(health=False)
    c.on_plan(plan(completion={"vel_tol": 100, "goal_q": [0]*7, "residual_max": 100}))
    c.on_control(pack_control_update(execute="p"))
    # A permissive per-plan completion rule does not override the policy's
    # tracking abort: far from the reference, the leg fails and holds.
    c.on_motor_state(state(.4))
    c.tick()
    results = [unpack_json_message(v) for t, v in c.node.sent if t == "controller_event"]
    assert results[-1]["kind"] == "leg_result" and not results[-1]["ok"]
    assert np.allclose(commands(c)[-1]["position"], .4)
    check_settle_dwell()
    # Task-specific completion and timeout still freeze at measured q.
    for custom in (True, False):
        c, clock = setup(health=False, settle_dwell_s=0)
        fields = {"completion": {"vel_tol": .01, "goal_q": [0]*7, "residual_max": .01}} if custom else {}
        c.on_plan(plan(**fields))
        c.on_control(pack_control_update(execute="p"))
        if not custom:
            c.executor.at_goal = lambda *args, **kwargs: False
        advance(c, clock, 6.5, q=.005)
        assert not c._running
        assert np.allclose(commands(c)[-1]["position"], .005)
        assert c.frozen is (not custom)


def check_settle_dwell():
    """Settling is not completion: the dwell streams the GOAL, then finishes.

    ``done()`` true only ends the reference; the leg completes once the arm has
    held that verdict for ``settle_dwell_s``. Stepping during the dwell is
    correct BECAUSE sample_at clamps past times[-1] -- the command is the final
    waypoint, not a cubic extrapolated off the end of the plan.
    """
    c, clock = setup(health=False)          # production settle_dwell_s
    assert c._settling.dwell_s > 0, "this check is meaningless at dwell 0"
    c.on_plan(plan(start=.0, ident="dwell"))
    c.on_control(pack_control_update(execute="dwell"))
    advance(c, clock, .9, q=.0005)
    c.executor.at_goal = lambda *args, **kwargs: True
    clock[0] += .1                           # t == _leg_end: settling starts
    c.on_motor_state(state(.0005))
    c.tick()
    assert c._running, "dwell not elapsed: the leg must not be complete yet"
    assert c.executor.steps, "the dwell must keep streaming, not go silent"
    # Stepping during the dwell is safe only because sample_at CLAMPS past
    # times[-1]: the command is the final waypoint, not a cubic extrapolated
    # off the end of the plan. That is what this asserts.
    assert np.allclose(commands(c)[-1]["position"], 0.), commands(c)[-1]["position"]

    advance(c, clock, c._settling.dwell_s + .1, q=.0005)
    assert not c._running, "dwell elapsed: the leg must complete"
    after = c.executor.steps
    clock[0] += .1
    c.on_motor_state(state(.0005))
    c.tick()
    assert c.executor.steps == after, "completed trajectory was stepped again"
    assert np.allclose(commands(c)[-1]["position"], 0), "settle residual became the next hold target"
    assert np.allclose(commands(c)[-1]["velocity"], 0)
    print("Joint-goal completion retains endpoint; task completion and timeout hold measured pose: PASS")


if __name__ == "__main__":
    main()
