"""Controller mode snapshots and bounded heartbeat; no hardware or services."""
import numpy as np

from arm_control.control.arm_controller import ArmController, _FakeExecutor, _FakeNode, _GRIPPER, _state
from arm_control.contracts.motion import pack_controller_event, unpack_controller_event
from arm_control.messages import pack_control_update, pack_json_message, pack_plan


def events(c):
    return [unpack_controller_event(v) for t, v in c.node.sent if t == "controller_event"]


def snapshot(c, law, kp, kd, *, ok=True):
    event = events(c)[-1]
    assert event["kind"] == "mode" and event["ok"] is ok, event
    assert event["mode_state"] == dict(law=law, kp=list(kp), kd=list(kd)), event
    return event


def main():
    clock = [0.0]
    c = ArmController(_FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
                      clock=lambda: clock[0], plant_reports_health=False)
    c.on_motor_state(_state())
    assert events(c), "fresh disarmed state must publish actual controller gains"
    snapshot(c, "joint", [1200.] * 7, [30.] * 7)
    assert "motor_command" not in c.node.topics()
    for tick in range(1, 250):
        clock[0] = tick / 1000
        c.on_motor_state(_state())
    assert len(events(c)) == 1, "heartbeat exceeded 4 Hz"
    clock[0] = .25
    c.on_motor_state(_state())
    assert len(events(c)) == 2
    clock[0] = .5
    c.on_motor_state(_state(vel=float("nan")))
    assert len(events(c)) == 2, "non-finite measurement refreshed mode feedback"
    c.on_motor_state(_state())
    assert len(events(c)) == 3
    c._set_arm(True)
    c.on_motor_state(_state())
    c.node.sent.clear()

    c.on_control(pack_control_update(gains=dict(kp=[0.] * 7, kd=[1.] * 7)))
    snapshot(c, "joint", [0.] * 7, [1.] * 7)
    c.on_control(pack_control_update(gains=dict(kp=list(range(7)))))
    snapshot(c, "joint", range(7), [1.] * 7)
    soft = dict(id=1, kc=[100.] * 6, dc=[10.] * 6, nullspace_kp=1., nullspace_kd=1.)
    c.on_control(pack_control_update(pose_hold=soft))
    refused = snapshot(c, "joint", range(7), [1.] * 7, ok=False)
    assert "capable" in refused["reason"]
    c.supports_pose_hold = True
    c.on_control(pack_control_update(pose_hold=soft))
    snapshot(c, "soft", [1200.] * 7, [30.] * 7)
    c.on_control(pack_control_update(pose_hold=dict(soft, id=0)))
    snapshot(c, "soft", [1200.] * 7, [30.] * 7, ok=False)
    c.on_control(pack_control_update(cancel=True))
    snapshot(c, "joint", [1200.] * 7, [30.] * 7)
    c.on_control(pack_control_update(pose_hold=soft))
    c.on_control(pack_control_update(gains=dict(kp=[0.] * 7, kd=[2.] * 7)))
    snapshot(c, "joint", [0.] * 7, [2.] * 7)

    q = np.zeros((2, 7))
    c.on_plan(pack_plan(plan_id="custom", phase="move", gated=True,
                        times=[0, 1], positions=q, velocities=q,
                        kp=[321.] * 7, kd=[12.] * 7))
    clock[0] += .25
    c.on_motor_state(_state())
    snapshot(c, "joint", [321.] * 7, [12.] * 7)
    c.on_control(pack_control_update(pose_hold=soft))
    c.on_control(pack_control_update(arm=False))
    c.node.sent.clear()
    clock[0] += .25
    c.on_motor_state(_state())
    snapshot(c, "joint", [1200.] * 7, [30.] * 7)
    assert "motor_command" not in c.node.topics()

    # Outage fallback must report its joint law and retain deadline scheduling.
    from arm_control.control.execution_policy import ExecutionPolicy
    policy = ExecutionPolicy(command_rate_hz=100, state_timeout_sec=.25,
                             health_timeout_sec=1, start_pos_tol_rad=.1,
                             abort_pos_err_rad=.35, hold_relatch_rad=.3,
                             torque_limits=np.full(7, 10.))
    c = ArmController(_FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
                      clock=lambda: clock[0], plant_reports_health=False,
                      execution_policy=policy)
    c._set_arm(True)
    c.on_motor_state(_state())
    assert "motor_command" not in c.node.topics()
    c.tick()
    c.supports_pose_hold = True
    c.on_control(pack_control_update(pose_hold=soft))
    clock[0] += .26
    c.node.sent.clear()
    c.tick()
    snapshot(c, "joint", [1200.] * 7, [30.] * 7)
    assert "motor_command" not in c.node.topics()
    c.node.sent.clear()
    clock[0] += 10
    c.tick()
    assert not events(c), "no fresh state must mean no heartbeat"
    for bad_kp in ([float("nan")] * 7, [1.], [[1.] * 7]):
        c.executor.kp = np.asarray(bad_kp)
        clock[0] += .25
        c.on_motor_state(_state())
        assert not events(c), "malformed executor gains must not refresh feedback"
    c.executor.kp = np.full(7, 1200.)
    c.on_motor_state(_state())
    snapshot(c, "joint", [1200.] * 7, [30.] * 7)
    c._stop("check complete")
    c.node.sent.clear()
    clock[0] += 1
    c.on_motor_state(_state())
    assert not events(c), "stopped controller emitted heartbeat"

    old = unpack_controller_event(pack_controller_event(kind="mode", ok=False, reason="old"))
    assert "mode_state" not in old and old["reason"] == "old" and not old["ok"]
    valid = dict(law="joint", kp=[1., 2.], kd=[3., 4.])
    for law in ("joint", "soft"):
        expected = dict(valid, law=law)
        assert unpack_controller_event(pack_controller_event(
            kind="mode", mode_state=expected))["mode_state"] == expected
    invalid = [dict(valid, law="float"), dict(valid, kp=[1.]),
               dict(valid, kd=[float("nan"), 1.]),
               dict(valid, kp=[float("inf"), 1.]),
               dict(valid, kp=[[1., 2.]]), dict(valid, kp=[], kd=[]),
               dict(law="joint", kp=[1.]), [1, 2], "joint"]
    for bad in invalid:
        for decode in (False, True):
            try:
                if decode:
                    unpack_controller_event(pack_json_message("controller_event", dict(
                        kind="mode", ok=True, reason="joint", mode_state=bad)))
                else:
                    pack_controller_event(kind="mode", mode_state=bad)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError(f"invalid snapshot accepted: {bad!r}, decode={decode}")
    print("controller mode feedback: snapshots, lifecycle, 4 Hz heartbeat, validation PASS")


if __name__ == "__main__":
    main()
