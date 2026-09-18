"""Explicit deployment-selected execution limits, independent of planning.

The start/tracking/relatch guards preserve the retired standalone executor's
contract. No defaults enable this policy in existing event-driven graphs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from arm_control.control.gains import validate_torque_limits
from arm_control.motion import JointTrajectory


@dataclass(frozen=True)
class ExecutionPolicy:
    command_rate_hz: float
    state_timeout_sec: float
    health_timeout_sec: float
    start_pos_tol_rad: float
    abort_pos_err_rad: float
    hold_relatch_rad: float
    torque_limits: np.ndarray
    position_lower: np.ndarray | None = None
    position_upper: np.ndarray | None = None
    velocity_limits: np.ndarray | None = None
    acceleration_limits: np.ndarray | None = None

    def __post_init__(self):
        for field in fields(self):
            if field.name in ("torque_limits", "velocity_limits", "acceleration_limits", "position_lower", "position_upper"):
                continue
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"execution_policy.{field.name} must be finite and positive")
            object.__setattr__(self, field.name, value)
        limits = validate_torque_limits(
            self.torque_limits, where="execution_policy.torque_limits"
        )
        object.__setattr__(self, "torque_limits", limits)
        for name in ("velocity_limits", "acceleration_limits"):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.asarray(value, dtype=float).copy()
            if value.shape != limits.shape or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError(f"execution_policy.{name} must match the positive joint limits")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        if (self.position_lower is None) != (self.position_upper is None):
            raise ValueError("execution_policy requires both position bounds")
        if self.position_lower is not None:
            for name in ("position_lower", "position_upper"):
                value = np.asarray(getattr(self, name), dtype=float).copy()
                if value.shape != limits.shape or not np.isfinite(value).all():
                    raise ValueError(f"execution_policy.{name} must match joint limits")
                value.setflags(write=False)
                object.__setattr__(self, name, value)
            if np.any(self.position_lower >= self.position_upper):
                raise ValueError("execution_policy position bounds must be ordered")

    @property
    def period(self):
        return 1.0 / self.command_rate_hz

    def plan_error(self, plan, measured, n_arm):
        q, v, t = plan["positions"], plan["velocities"], plan["times"]
        if q.ndim != 2 or q.shape[1] != n_arm:
            return f"trajectory must have {n_arm} joint columns"
        if t.ndim != 1 or len(t) < 2 or len(t) != len(q) or v.shape != q.shape:
            return "trajectory has invalid sample shapes"
        for key in ("times", "positions", "velocities", "kp", "kd"):
            if not np.isfinite(plan[key]).all():
                return "trajectory contains non-finite values"
        if np.any(np.diff(t) <= 0):
            return "trajectory times must be strictly increasing"
        if self.position_lower is not None:
            if np.any(q < self.position_lower) or np.any(q > self.position_upper):
                return "trajectory outside configured position/velocity safety envelope"
        dt = np.diff(t)[:, None]
        if self.velocity_limits is not None:
            if (np.any(np.abs(v) > self.velocity_limits + 1e-8)
                    or np.any(np.abs(np.diff(q, axis=0) / dt) > self.velocity_limits + 1e-8)):
                return "trajectory exceeds configured joint velocity limits"
        if self.acceleration_limits is not None:
            if np.any(np.abs(np.diff(v, axis=0) / dt) > self.acceleration_limits + 1e-8):
                return "trajectory exceeds configured joint acceleration limits"

        # Certify the CURVE, not its samples. The executor flies a cubic
        # Hermite between waypoints (JointTrajectory.sample_at), and a cubic
        # can hold an interior velocity extremum with both endpoint
        # velocities legal -- so chords and waypoints are a proxy that can
        # pass a trajectory whose flown speed exceeds the limit. bounds() is
        # closed-form over every segment, so this is a statement about all t,
        # not about the points that happened to be sampled.
        #
        # Runs AFTER the sampled checks, not before: those are cheaper and
        # name which proxy failed, so a plan that violates both should be
        # reported in the more specific terms. This block therefore speaks
        # only when the samples were CLEAN -- which is exactly the hole it
        # exists to close. It is additive rather than a replacement, since
        # the sampled checks also catch a malformed plan (a v that disagrees
        # with its own chord) that a well-formed curve would hide. Measured before wiring: on real retimed legs this
        # changes no verdict -- a straight leg lands exactly on both caps and
        # a cornered one was already refused -- so it closes the hole without
        # newly refusing anything the retimer produces today.
        curve = JointTrajectory(times=t, positions=q, velocities=v).bounds()
        if self.position_lower is not None:
            if (np.any(curve["q_min"] < self.position_lower - 1e-8)
                    or np.any(curve["q_max"] > self.position_upper + 1e-8)):
                return ("interpolated trajectory leaves the position envelope "
                        "between waypoints")
        if self.velocity_limits is not None:
            if np.any(curve["qd_abs_max"] > self.velocity_limits + 1e-8):
                return ("interpolated trajectory exceeds joint velocity limits "
                        "between waypoints")
        if self.acceleration_limits is not None:
            if np.any(curve["qdd_abs_max"] > self.acceleration_limits + 1e-8):
                return ("interpolated trajectory exceeds joint acceleration "
                        "limits between waypoints")
        for key in ("kp", "kd"):
            if plan[key].shape != (n_arm,) or np.any(plan[key] < 0):
                return f"trajectory {key} must be a nonnegative joint vector"
        cartesian, poses = plan.get("cartesian"), plan.get("cartesian_poses")
        if cartesian is not None or poses is not None:
            if cartesian is None or poses is None:
                return "Cartesian trajectory needs both poses and impedance settings"
            if poses.shape != (len(t), 7) or not np.isfinite(poses).all():
                return "Cartesian trajectory poses must be finite sample-by-7 values"
            for key, shape in (("task_R", (3, 3)), ("kc", (6,)), ("dc", (6,))):
                value = np.asarray(cartesian.get(key), dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    return f"Cartesian trajectory {key} must be finite with shape {shape}"
                if key != "task_R" and np.any(value < 0):
                    return f"Cartesian trajectory {key} must be nonnegative"
        error = float(np.max(np.abs(q[0] - measured)))
        if not error <= self.start_pos_tol_rad:
            return f"start point {error:.3f} rad from measured pose (tol {self.start_pos_tol_rad:g}) — re-plan"
        return None

    def relatch_tolerance(self, kp):
        return np.minimum(self.hold_relatch_rad, .25 * self.torque_limits / np.maximum(kp, 1e-9))

    def tracking_error_exceeded(self, command, measured):
        return not float(np.max(np.abs(command.q_des - measured))) <= self.abort_pos_err_rad

    def command_valid(self, command, n_arm):
        for key in ("q_des", "qd_des", "tau_ff", "kp", "kd"):
            value = np.asarray(getattr(command, key))
            if value.shape != (n_arm,) or not np.isfinite(value).all():
                return False
        if self.position_lower is not None:
            if np.any(command.q_des < self.position_lower) or np.any(command.q_des > self.position_upper):
                return False
        if self.velocity_limits is not None and np.any(np.abs(command.qd_des) > self.velocity_limits + 1e-8):
            return False
        return bool(np.all(command.kp >= 0) and np.all(command.kd >= 0)
                    and np.all(np.abs(command.tau_ff) <= self.torque_limits))


def build_execution_policy(policy_config, *, torque_limits, joint_limits=None):
    """Assemble the policy, taking factory facts from the robot description.

    One constructor for every caller, because the interesting part is not the
    dataclass but WHICH numbers are allowed to come from YAML. The position
    envelope is a factory limit: the URDF owns it and a deployment reads it.
    A deployment may still declare both bounds to NARROW the machine
    deliberately -- an operating choice -- and that declaration wins here.

    ``joint_limits`` is the description's ``(lower, upper)``; non-finite
    bounds (a continuous joint) are left unset rather than fabricated.
    Returns None when the deployment configures no policy at all.
    """
    if policy_config is None:
        return None
    config = dict(policy_config)
    if config.get("position_lower") is None and joint_limits is not None:
        lower, upper = (np.asarray(v, dtype=float) for v in joint_limits)
        if np.isfinite(lower).all() and np.isfinite(upper).all():
            config["position_lower"], config["position_upper"] = lower, upper
    return ExecutionPolicy(**config, torque_limits=torque_limits)
