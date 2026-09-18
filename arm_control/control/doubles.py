"""Test doubles for the arm controller: a fake node, executor and gripper.

Lifted out of ``arm_control.control.arm_controller`` on 2026-09-10. These were
never private -- five checks under ``tools/`` already imported them through the
underscore names -- so ~200 lines of scaffolding sat inside the module that
actually servos an arm. A shared double library belongs beside the thing it doubles, not inside it --
the same shape as ``numpy.testing`` or ``django.test``. It is NOT under
``tools/`` because both this repo and its parent ship a ``tools/`` directory,
so ``tools.doubles`` resolves to whichever landed on the path first.

Public names now. The old underscore spellings stay as aliases so a check that
has not been updated keeps working.
"""
from __future__ import annotations

import numpy as np

from arm_control.control.arm_controller import ArmController
from arm_control.messages import pack_plan
from arm_control.motion import JointServoCommand


class _FakeNode:
    """Collects outputs; iterating it ends the run loop immediately."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []

    def send_output(self, topic: str, arrow) -> None:
        self.sent.append((topic, arrow))

    def topics(self) -> list[str]:
        return [t for t, _ in self.sent]


class _FakeExecutor:
    """Just enough executor to exercise the controller's own decisions."""

    completion_tolerances = (0.01, 0.05)

    def __init__(self, n: int = 7) -> None:
        self.n_joints = n
        self._traj = None
        self._t_start = 0.0
        self.kp = np.full(n, 1200.0)
        self.kd = np.full(n, 30.0)
        self.payload = (0.0, None)
        self.steps = 0
        # The real executor reads these off the URDF it loaded; a double with
        # no model states an honest, permissive envelope rather than omitting
        # the property, so control/adapter.py can keep requiring it.
        self.joint_limits = (np.full(n, -np.pi), np.full(n, np.pi))

    def set_gains(self, kp=None, kd=None) -> None:
        if kp is not None:
            self.kp = np.asarray(kp, dtype=float)
        if kd is not None:
            self.kd = np.asarray(kd, dtype=float)

    def set_payload(self, mass_kg, frame_name="", com_offset=None) -> None:
        self.payload = (float(mass_kg), com_offset)

    @property
    def has_trajectory(self) -> bool:
        return self._traj is not None

    def load_trajectory(self, traj, t_start: float) -> None:
        self._traj, self._t_start = traj, float(t_start)

    def elapsed(self, t_now: float) -> float:
        return float(t_now) - self._t_start

    def clear_trajectory(self) -> None:
        self._traj = None

    def step(self, t_now, state) -> JointServoCommand:
        self.steps += 1
        n = self.n_joints
        return JointServoCommand(
            q_des=np.zeros(n), qd_des=np.zeros(n), tau_ff=np.zeros(n),
            kp=self.kp, kd=self.kd,
        )

    def hold_command(self, state, q_des=None) -> JointServoCommand:
        n = self.n_joints
        q = np.asarray(state.position, dtype=float) if q_des is None else np.asarray(q_des)
        return JointServoCommand(
            q_des=q, qd_des=np.zeros(n), tau_ff=np.zeros(n), kp=self.kp, kd=self.kd
        )

    def done(self, t_now, state, pos_tol=None, vel_tol=None) -> bool:
        return self._traj is not None and t_now - self._t_start >= self._traj.duration_sec


_GRIPPER = {"n_motors": 7, "mimic": None, "open_finger_m": 0.04, "gains": (100.0, 1.0)}


def _controller(**kw) -> "ArmController":
    node = _FakeNode()
    # PLANT clock by default: these checks fast-forward by assigning _plant_t,
    # which only means anything under that policy. The WALL policy -- what a
    # config without motor_state_period_s gets -- has its own check below.
    kw.setdefault("state_period_s", 0.01)
    kw.setdefault('settle_dwell_s', 0.)  # Legacy lifecycle checks; dwell has its own replay.
    c = ArmController(node, _FakeExecutor(), gripper=dict(_GRIPPER), **kw)
    c._set_arm(True)
    c.bridge_armed = True
    c.on_motor_state(_state())   # the post-arm state that earns `ready`
    node.sent.clear()
    return c


def _state(n: int = 7, vel: float = 0.0, pos: float = 0.0):
    from arm_control.messages import pack_motor_state_dict

    return pack_motor_state_dict(
        {k: np.zeros(n) for k in
         ("position", "velocity", "position_cmd", "velocity_cmd",
          "torque_cmd", "kp", "kd", "torque")}
        | {"position": np.full(n, pos), "velocity": np.full(n, vel)}
    )


def _plan_msg(plan_id: str, *, gated: bool, completion=None):

    times = np.array([0.0, 1.0])
    q = np.zeros((2, 7))
    return pack_plan(
        plan_id=plan_id, phase="move", gated=gated, times=times,
        positions=q, velocities=q, kp=np.full(7, 1200.0), kd=np.full(7, 30.0),
        completion=completion,
    )


FakeNode = _FakeNode
FakeExecutor = _FakeExecutor
GRIPPER = _GRIPPER
controller = _controller
state = _state
plan_msg = _plan_msg
