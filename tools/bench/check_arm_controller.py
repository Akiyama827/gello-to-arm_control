"""Contract checks for ArmController: clock policies and the whole lifecycle.

Ran as ``python -m`` inside arm_controller until 2026-09-10. Moved out with the
doubles it uses: a production module should not carry the harness that proves
it, and this one was a third of the file.

    PYTHONPATH=. python tools/bench/check_arm_controller.py
"""
from __future__ import annotations

import numpy as np

from arm_control.control.arm_controller import ArmController
from arm_control.messages import (
    pack_control_update,
    pack_controller_event,
    pack_json_message,
    pack_jog,
    unpack_controller_event,
)
from arm_control.control.doubles import (
    _controller,
    _FakeExecutor,
    _FakeNode,
    _GRIPPER,
    _plan_msg,
    _state,
)


def _check_clock_policies() -> None:
    """Either clock must actually advance, and a bad period must be refused.

    The regression this pins: _plant_t advanced only when a period was
    configured, while exec_time always returned it -- so a config WITHOUT
    motor_state_period_s (which is every real/*.yaml here) froze trajectory
    time at 0.0 and no leg could ever finish. It passed every sim run because
    the sim config happens to set the key.
    """
    # WALL policy: no period configured. Time must move without any state.
    ticks = iter([10.0, 10.5, 11.25])
    c = ArmController(
        _FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER), clock=lambda: next(ticks)
    )
    assert c.exec_time == 10.0
    assert c.exec_time == 10.5
    assert c.exec_time == 11.25, "wall clock must advance with no motor_state"

    # PLANT policy: time moves per state message and NOT with wall time.
    frozen = ArmController(
        _FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
        state_period_s=0.02, clock=lambda: 999.0,
    )
    assert frozen.exec_time == 0.0
    frozen.on_motor_state(_state())
    frozen.on_motor_state(_state())
    assert abs(frozen.exec_time - 0.04) < 1e-12, frozen.exec_time

    # A period that is present but nonsense is a config error, not a fallback:
    # silently switching clocks on a typo is how the original bug felt normal.
    for bad in (0.0, -0.01, float("inf"), float("nan")):
        try:
            ArmController(_FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
                          state_period_s=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"motor_state_period_s={bad} must be refused")


def _self_check() -> None:
    import inspect

    assert 'qd' in inspect.signature(pack_controller_event).parameters, \
        'ready event cannot report measured velocity'
    old = unpack_controller_event(pack_controller_event(kind='ready', q=[.1, .2]))
    assert 'qd' not in old and np.array_equal(old['q'], [.1, .2])
    measured = unpack_controller_event(pack_controller_event(
        kind='ready', q=[.1, .2], qd=[-.3, .4]))
    assert np.array_equal(measured['qd'], [-.3, .4])
    for q, qd in (([0., 0.], [np.nan, 0.]), ([0., 0.], [0., np.inf]),
                  ([0., 0.], [0.]), ([0., 0.], [[0., 0.]]),
                  ([0., 0.], 'invalid'), ([0.], 0.), ([], []),
                  (None, [0.]), ([[0., 0.]], [0., 0.])):
        for decode in (False, True):
            try:
                if decode:
                    unpack_controller_event(pack_json_message('controller_event', dict(
                        kind='ready', q=q, qd=qd)))
                else:
                    pack_controller_event(kind='ready', q=q, qd=qd)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f'invalid ready velocity accepted: q={q!r}, qd={qd!r}')

    # 1. A gated plan does NOT run until an execute names it -- and while it
    #    waits, the controller still streams. That stream is the whole point of
    #    the split: it is what the plant's deadman sees while a human reviews.
    c = _controller()
    c.on_plan(_plan_msg("p1", gated=True))
    c.on_motor_state(_state())
    assert c._running is False, "a gated plan ran without an execute"
    assert c.node.topics().count("motor_command") == 1, c.node.topics()
    assert c.executor.steps == 0, "the executor stepped a gated plan"

    # 2. An execute naming a DIFFERENT plan is refused: a plan reviewed and then
    #    superseded by a re-plan must never run.
    c.on_control(pack_control_update(execute="p-stale"))
    assert c._running is False
    c.on_control(pack_control_update(execute="p1"))
    assert c._running is True

    # 3. A plan arriving mid-flight is refused, not swapped.
    c.on_plan(_plan_msg("p2", gated=False))
    assert c._plan["plan_id"] == "p1", c._plan["plan_id"]

    # 4. Running: the executor steps, and the leg reports back under ITS id.
    c.on_motor_state(_state())
    assert c.executor.steps == 1
    c._plant_t = 5.0  # past the 1 s trajectory
    c.on_motor_state(_state())
    events = [unpack_controller_event(a) for t, a in c.node.sent if t == "controller_event"]
    legs = [e for e in events if e["kind"] == "leg_result"]
    assert len(legs) == 1 and legs[0]["plan_id"] == "p1" and legs[0]["ok"], legs
    assert c._running is False and c._plan is None

    # 5. The settle rule, with the tolerances the LEG carried in: the plan
    #    playing out is not enough -- a still-moving arm is not settled, and a
    #    residual beyond the bound is not either.
    settle = {"vel_tol": 0.002, "goal_q": np.zeros(7), "residual_max": 0.01}
    c = _controller()
    c.on_plan(_plan_msg("s1", gated=False, completion=settle))
    c._plant_t = 5.0
    c.on_motor_state(_state(vel=0.05))          # moving: 25x the settle gate
    assert c._running is True, "settle leg completed while still moving"
    c.on_motor_state(_state(vel=0.0))
    assert c._running is False, "stopped settle leg never completed"

    # A settle leg parked far from its plan must NOT be called done.
    c = _controller()
    c.on_plan(_plan_msg("s2", gated=False,
                        completion=dict(settle, goal_q=np.full(7, 0.5))))
    c._plant_t = 5.0
    c.on_motor_state(_state(vel=0.0))
    assert c._running is True, "settle leg completed 0.5 rad from its plan"

    # 6. Payload crosses as a control message, not as config.
    c = _controller()
    c.on_control(pack_control_update(payload={"mass_kg": 0.4051, "com_ee": [0.013, 0.0, 0.0]}))
    assert abs(c.executor.payload[0] - 0.4051) < 1e-9, c.executor.payload

    # 7. Disarmed, no commands stream (the bridge zero-holds); mode feedback
    #    still reports gains. Commands wait for the health ACK after arming.
    node = _FakeNode()
    c = ArmController(node, _FakeExecutor(), gripper=dict(_GRIPPER),
                      state_period_s=0.01)
    c.on_motor_state(_state())
    assert node.topics() == ["controller_event"], node.topics()
    assert unpack_controller_event(node.sent[-1][1])["kind"] == "mode"
    c._set_arm(True)
    node.sent.clear()
    c.on_motor_state(_state(pos=-.4, vel=.7))
    assert node.topics() == [], "streamed before the plant ACKed the arm"
    c.bridge_armed = True
    c.on_motor_state(_state(pos=.2, vel=-.3))
    readies = [unpack_controller_event(a) for t, a in node.sent if t == "controller_event"]
    assert [e["kind"] for e in readies] == ["ready"], readies
    assert np.array_equal(readies[0]['q'], np.full(7, .2))
    assert np.array_equal(readies[0]['qd'], np.full(7, -.3))
    assert np.array_equal(readies[0]['q'], c.last_state.position)
    assert np.array_equal(readies[0]['qd'], c.last_state.velocity)
    assert "motor_command" in node.topics(), "no hold streaming after ready"

    # 8. Disarming is a lifecycle reset, not a flag flip: the running leg fails
    #    back (the planner is blocked on it), the loaded plan is dropped, and
    #    re-arming re-earns both the plant ACK and `ready` before anything runs.
    c = _controller()
    c.on_plan(_plan_msg("d1", gated=False))
    assert c._running is True
    c.node.sent.clear()
    c.on_control(pack_control_update(arm=False))
    legs = [unpack_controller_event(a) for t, a in c.node.sent
            if t == "controller_event"]
    assert [(e["plan_id"], e["ok"]) for e in legs] == [("d1", False)], legs
    assert c._plan is None and c._running is False
    assert not c.ready_sent and not c.bridge_armed, "re-arm would skip the ACK"

    c.on_control(pack_control_update(arm=True))
    c.node.sent.clear()
    c.on_motor_state(_state())
    assert c.node.topics() == [], "streamed on a re-arm without a fresh ACK"
    c.bridge_armed = True
    c.on_motor_state(_state())
    kinds = [unpack_controller_event(a)["kind"] for t, a in c.node.sent
             if t == "controller_event"]
    assert kinds == ["ready"], kinds

    # 9. An execute that lands while disarmed is refused AND fails the leg, so
    #    the planner learns instead of waiting on a leg that will never run.
    #    (Flag set directly: a real disarm would have dropped the plan already,
    #    which is check 8 -- this isolates the release guard itself.)
    c = _controller()
    c.on_plan(_plan_msg("x1", gated=True))
    c.armed = False
    c.node.sent.clear()
    c.on_control(pack_control_update(execute="x1"))
    assert c._running is False, "started a leg while disarmed"
    legs = [unpack_controller_event(a) for t, a in c.node.sent
            if t == "controller_event"]
    assert [(e["kind"], e["ok"]) for e in legs] == [("leg_result", False)], legs

    _check_clock_policies()

    # N. Cancel is the OPERATOR stop, and it is none of the other three.
    #    A running leg aborts, the plan is dropped, and the arm stays armed and
    #    still accepts the next plan -- unlike `hold` (frozen forever) and
    #    `stop` (terminal), both of which refuse everything after.
    c = _controller()
    c.on_plan(_plan_msg("run1", gated=False))
    c.on_motor_state(_state())
    c.on_motor_state(_state(vel=0.5))
    assert c._running, "leg should be running before the cancel"
    c.on_control(pack_control_update(cancel=True, reason="operator stop"))
    assert not c._running and c._plan is None, (c._running, c._plan)
    assert c.armed and not c.stopped and not c.frozen, (c.armed, c.stopped, c.frozen)
    # It reports the abort rather than letting a caller block on a leg that
    # will never finish.
    kinds = [unpack_controller_event(a)["kind"] for _t, a in c.node.sent
             if _t == "controller_event"]
    assert "leg_result" in kinds, kinds
    # And the arm is still usable: the next plan loads and runs.
    c.on_plan(_plan_msg("run2", gated=False))
    c.on_motor_state(_state())
    assert c._running, "a cancelled controller must still accept a new plan"

    # Contrast, so the three cannot quietly converge: `hold` refuses the next
    # plan outright.
    c = _controller()
    c.on_motor_state(_state())
    c.on_control(pack_control_update(hold=True, reason="milestone"))
    c.on_plan(_plan_msg("after_hold", gated=False))
    assert c._plan is None, "a frozen controller must not load a plan"

    # N+1. A plant with no health topic still streams; one WITH a bridge still
    #      waits for the armed edge. Getting this backwards is silent either
    #      way -- an arm that never moves, or one anchored on a pre-arm pose.
    # (built raw, not via _controller(), which pre-sets bridge_armed for the
    #  checks above -- that shortcut is the very thing under test here)
    c = ArmController(
        _FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
        state_period_s=0.01, plant_reports_health=False,
    )
    c._set_arm(True)
    c.on_motor_state(_state())
    assert c.ready_sent, "a bridgeless plant must not wait for a health edge"
    assert any(t == "motor_command" for t, _a in c.node.sent), c.node.topics

    c = ArmController(
        _FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER), state_period_s=0.01,
    )  # the default: there IS a bridge
    c._set_arm(True)
    c.on_motor_state(_state())
    assert not c.ready_sent, "a bridged plant must wait for the armed edge"
    c.on_motor_health(pack_json_message("motor_health", {"armed": True}))
    c.on_motor_state(_state())
    assert c.ready_sent, "the armed edge should have released it"

    # N+2. The console's gripper slider reaches the jaws. Needs an arm whose
    #      gripper IS a motor slot (mimic_cfg=None means the jaws are their own
    #      device and no slot is packed at all -- the FR3 case). Default is the
    #      configured open width: the assembly path never sends this input,
    #      because the bridge's grasp gate owns the slot there and two writers
    #      would fight over it.
    from arm_control.joint_motor_map import gripper_finger_to_motor
    from arm_control.messages import pack_motor_command
    from arm_control.messages import unpack_motor_command as _umc

    mimic = {"source": "Gripper", "motor_open": 0.0, "motor_closed": -4.985,
             "lower": 0.0, "upper": 0.0439}
    node = _FakeNode()
    c = ArmController(
        node, _FakeExecutor(n=6), state_period_s=0.01,
        gripper={"n_motors": 7, "mimic": mimic, "open_finger_m": 0.04,
                 "gains": (40.0, 2.0)},
    )
    c._set_arm(True)
    c.bridge_armed = True
    c.on_motor_state(_state(n=7))
    held = _umc([a for t, a in node.sent if t == "motor_command"][-1], 7)
    assert held["position"][6] == gripper_finger_to_motor(0.04, mimic), held["position"][6]

    zeros2 = np.zeros(2)
    c.on_gripper(pack_motor_command([0.01, 0.01], zeros2, zeros2, zeros2, zeros2))
    c.on_motor_state(_state(n=7))
    moved = _umc([a for t, a in node.sent if t == "motor_command"][-1], 7)
    assert moved["position"][6] == gripper_finger_to_motor(0.01, mimic), moved["position"][6]
    assert moved["position"][6] != held["position"][6]

    # N+3. Gains change the control law, so a running leg is cancelled first --
    #      float (kp=0) applied to a moving trajectory would drop the arm
    #      through the rest of its path.
    c = _controller()
    c.on_plan(_plan_msg("gainrun", gated=False))
    c.on_motor_state(_state())
    c.on_motor_state(_state(vel=0.5))
    assert c._running
    c.on_control(pack_control_update(gains={"kp": [0.0] * 7, "kd": [1.0] * 7}))
    assert not c._running, "a gains change must not ride on top of a running leg"
    assert float(np.max(np.abs(c.executor.kp))) == 0.0, c.executor.kp
    assert c.armed and not c.stopped, "and it must leave the arm armed"

    # N+4. Jog: it moves the arm, and it DIES IF IT STOPS ARRIVING. That
    #      expiry is the whole safety story -- a closed tab, a wedged console
    #      and a cut network are indistinguishable here, and all three must
    #      stop the arm without anything having to notice and send a stop.

    now = [1000.0]
    c = _controller(clock=lambda: now[0])
    c.on_jog(pack_jog(q=[0.4] * 7))
    c.on_motor_state(_state())
    cmd = [a for t, a in c.node.sent if t == "motor_command"][-1]
    served = _umc(cmd, 7)["position"][:7]
    assert float(np.max(np.abs(served - 0.4))) < 1e-9, served

    # still fresh just inside the timeout
    now[0] += 0.19
    c.node.sent.clear()
    c.on_motor_state(_state())
    served = _umc([a for t, a in c.node.sent if t == "motor_command"][-1], 7)["position"][:7]
    assert float(np.max(np.abs(served - 0.4))) < 1e-9, "a fresh jog stopped early"

    # ...and dead just past it: the arm holds where it IS (measured = zeros),
    # not where the last setpoint was still asking it to go.
    now[0] += 0.5
    c.node.sent.clear()
    c.on_motor_state(_state())
    served = _umc([a for t, a in c.node.sent if t == "motor_command"][-1], 7)["position"][:7]
    assert float(np.max(np.abs(served))) < 1e-9, f"a stale jog kept commanding: {served}"
    assert c.armed, "expiry holds the arm; it does not disarm it"

    # A jog must never fight a reviewed plan for the same joints.
    c = _controller(clock=lambda: now[0])
    c.on_plan(_plan_msg("owns", gated=False))
    c.on_motor_state(_state())
    assert c._running
    c.on_jog(pack_jog(q=[0.9] * 7))
    assert not c._jog.active, "a jog was accepted while a plan was running"

    print("arm_controller self-check ok")


def main() -> None:
    _check_clock_policies()
    _self_check()


if __name__ == "__main__":
    main()
