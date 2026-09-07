"""Controller-only pose hold assertions: no FK, IK, planner or plant."""
import numpy as np

from arm_control.control.arm_controller import _controller, _state, _plan_msg
from arm_control.messages import pack_control_update, unpack_motor_command, pack_jog


def main():
    spec = dict(id=7, kc=[300]*3+[30]*3, dc=[35]*3+[8]*3,
                nullspace_kp=0.5, nullspace_kd=2)
    c = _controller()
    c.on_control(pack_control_update(pose_hold=spec))
    assert c._pose_hold is None, "unsupported plant must refuse"
    c.supports_pose_hold = True
    track_kp = c.executor.kp.copy()
    c.on_control(pack_control_update(gains={"kp": [0]*7, "kd": [5]*7}))
    c.on_control(pack_control_update(pose_hold=spec))
    assert c._pose_hold == spec
    assert np.array_equal(c.executor.kp, track_kp), "Float must not leak into watchdog fallback"
    c.on_motor_state(_state())
    wire = [v for t, v in c.node.sent if t == "motor_command"][-1]
    assert unpack_motor_command(wire, 7)["pose_hold"] == spec
    c.on_jog(pack_jog(q=np.ones(7)*0.02, reason="test"))
    assert c._jog_q is None
    c.on_plan(_plan_msg("must-not-execute", gated=False))
    assert not c._running and c._pose_hold is not None
    c.on_control(pack_control_update(gains={"kp": [10]*7, "kd": [1]*7}))
    assert c._pose_hold is None and c.last_command is None
    c.on_motor_state(_state())
    wire = [v for t, v in c.node.sent if t == "motor_command"][-1]
    assert unpack_motor_command(wire, 7)["pose_hold"] is None
    c.on_control(pack_control_update(pose_hold=spec))
    c.on_control(pack_control_update(cancel=True))
    assert c._pose_hold is None and c.last_command is None
    c.on_control(pack_control_update(arm=False))
    c.on_control(pack_control_update(pose_hold=spec))
    assert c._pose_hold is None
    clock = [0.0]
    c = _controller(clock=lambda: clock[0])
    c.supports_pose_hold = True
    clock[0] = 2.0
    c.on_control(pack_control_update(pose_hold=spec))
    assert c._pose_hold is None, "stale state must refuse"
    c.on_motor_state(_state(vel=float("nan")))
    c.on_control(pack_control_update(pose_hold=spec))
    assert c._pose_hold is None, "nonfinite state must refuse"
    c.on_motor_state(_state(vel=0.1))
    c.on_control(pack_control_update(pose_hold=spec))
    assert c._pose_hold is None, "moving plant must refuse"
    print("pose hold controller: capability, authority, exclusion and exit OK")


if __name__ == "__main__":
    main()
