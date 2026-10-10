"""Joint trajectory executor producing servo commands."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from arm_control.control.gains import validate_torque_limits
from arm_control.dynamics import PinocchioDynamics
from arm_control.motion import JointServoCommand, JointState, TrajectoryPoint


def gain_error(kp, kd) -> str | None:
    """Reject positive stiffness with zero damping; None when the law is sane.

    An undamped spring on a 1 kHz plant is an oscillator with nothing to take
    energy out of it. Real-add shipped kp=[600...] with kd=[0...] for one run
    (the YAML has since been corrected) and the arm rang until the plant's
    own velocity reflex fired -- a CONFIGURATION reaching the servo through an
    admission path that checked only `>= 0`.

    kp == 0 stays legal at any kd: that is Float and the pose-hold law, where
    the joint spring is meant to be absent. The rule is only that a joint
    which is being SPRUNG must also be damped.
    """
    kp = np.asarray(kp, dtype=float).ravel()
    kd = np.asarray(kd, dtype=float).ravel()
    if kp.shape != kd.shape:
        return f"kp shape {kp.shape} != kd shape {kd.shape}"
    if not (np.isfinite(kp).all() and np.isfinite(kd).all()):
        return "gains must be finite"
    if np.any(kp < 0) or np.any(kd < 0):
        return "gains must be nonnegative"
    undamped = np.flatnonzero((kp > 0) & (kd <= 0))
    if undamped.size:
        return (f"joint(s) {undamped.tolist()} have stiffness with zero damping "
                f"(kp={kp[undamped].tolist()}, kd=0) — an undamped spring "
                f"rings; set kd > 0 or kp = 0")
    return None


class JointTrajectoryExecutor:
    """Turns a sampled reference point into a joint servo command.

    Integral bias is an explicit input; constructing commands never updates it.
    It holds no trajectory and no trajectory time: the controller's
    ``control.reference.TrajectoryReference`` owns both and passes the point.
    The feedforward torque is RNEA(q_meas, qd_des, qdd_des), all three taken
    off the SAME trajectory polynomial. This adds gravity, Coriolis, and
    inertia feedforward in one call. The downstream PD law (which lives in the
    simulator / motor firmware) applies kp*(q_des - q) + kd*(qd_des - qd).
    """

    def __init__(
        self,
        arm_id: str,
        joint_names: Sequence[str],
        dynamics: PinocchioDynamics,
        kp_default: np.ndarray,
        kd_default: np.ndarray,
        max_torque: np.ndarray,
        done_pos_tol: float = 1e-3,
        done_vel_tol: float = 1e-2,
        gravity_comp: bool = False,
        *,
        payload_frame: str = "",
        done_pose_tol: tuple[float, float] | None = None,
    ) -> None:
        n = len(list(joint_names))
        for arr, name in ((kp_default, "kp_default"), (kd_default, "kd_default")):
            if np.asarray(arr).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        # The SAME contract set_gains() enforces. Without this the constructor
        # is a way in to exactly the undamped kp>0/kd=0 pair that method exists
        # to reject, and this is a reusable public class -- the invariant
        # belongs at the class boundary, not in whichever factory calls it.
        reason = gain_error(kp_default, kd_default)
        if reason:
            raise ValueError(reason)
        max_torque = validate_torque_limits(max_torque, n, where="max_torque")
        self.arm_id = str(arm_id)
        self._joints = list(joint_names)
        self._dyn = dynamics
        # RNEA/gravity source: the construction model, or a load model installed later (set_dynamics). Joint limits and the
        # payload frame stay the construction model's.
        self._rigid = dynamics
        self._kp = np.asarray(kp_default, dtype=float).copy()
        self._kd = np.asarray(kd_default, dtype=float).copy()
        self._max_tau = np.array(max_torque, dtype=float)
        # Held-payload feedforward: RNEA knows only the bare arm, so a grasped
        # module's weight otherwise lands on the PD as steady-state sag (the
        # softened contact-phase gains make it centimeters at the EE).
        self._payload_frame = str(payload_frame)  # model/config owns this frame
        self._payload: tuple[float, np.ndarray | None] = (0.0, None)
        # Plant compensates gravity itself (FR3 control box): ship RNEA MINUS
        # gravity. The subtraction happens HERE, before the torque clamp — a
        # post-clamp subtraction downstream let |tau| reach clamp+|g| and made
        # the configured torque_limits decorative (audit 2026-07-29).
        self._gravity_comp = bool(gravity_comp)
        # Leg-completion tolerances. Joint stiction (frictionloss vs the soft
        # wrist kp) parks a settle short of the target — 1e-3 rad assumes a
        # frictionless plant; a sim with the noslip pass (or a real geared
        # joint) stalls at ~frictionloss/kp and completion would never fire.
        self._done_pos_tol = float(done_pos_tol)
        self._done_vel_tol = float(done_vel_tol)
        # A Cartesian leg's completion (metres, radians at the hand): its K_c may leave the joints anywhere in the null space
        # while the hand is on target (bench 2026-10-10, rung 6: j1 +32 / j3 -38 mrad at 0.00 mm), and with the joint
        # integral frozen its static hand error is load/K_c (twin: 5.7 mm at 500 N/m). So it completes on SETTLING, at rest
        # with the hand inside a SANITY bound -- precision is K_c's, not completion's (operator 2026-10-10). The hand is
        # the payload frame, the config's ee_frame. Default 20 mm / 5 deg wherever the model has that frame; a body-list
        # arm has none and judges no Cartesian leg (its controller refuses one without a rule).
        if done_pose_tol is None and self._payload_frame and hasattr(dynamics, "frame_pose"):
            done_pose_tol = (0.02, np.radians(5.0))
        if done_pose_tol is not None:
            done_pose_tol = tuple(float(v) for v in done_pose_tol)
            if len(done_pose_tol) != 2 or not all(np.isfinite(v) and v > 0 for v in done_pose_tol):
                raise ValueError("done_pose_tol must be (metres, radians), finite and positive")
            if not self._payload_frame or not hasattr(dynamics, "frame_pose"):
                raise ValueError("done_pose_tol needs a URDF model and its ee frame (payload_frame)")
        self._done_pose_tol = done_pose_tol

    @property
    def n_joints(self) -> int:
        return len(self._joints)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) FACTORY limits off the URDF this executor loaded."""
        return self._dyn.joint_limits

    @property
    def kp(self) -> np.ndarray:
        return self._kp.copy()

    @property
    def kd(self) -> np.ndarray:
        return self._kd.copy()

    def hold_command(
        self, state: JointState, q_des: np.ndarray | None = None, *,
        integral_bias: np.ndarray | float = 0.0,
    ) -> JointServoCommand:
        """Static hold: zero desired velocity, gravity (+ payload, + the clamped
        integral when ki > 0) feedforward, current gains, torque-clamped.

        ``state`` is the MEASURED state — gravity is always evaluated at the
        real pose (evaluating it at a latched anchor while a downstream stage
        subtracted it at the measured pose left a spurious g(anchor)-g(meas)
        feedforward that grew with the latch offset). ``q_des`` is the anchor
        to servo toward; default is the measured pose itself.

        The safe freeze/keepalive primitive — unlike replaying the last
        trajectory sample it can never carry a nonzero qd_des or a
        motion-computed tau_ff into a hold.
        """
        q = np.asarray(state.position, dtype=float).copy()
        anchor = q if q_des is None else np.asarray(q_des, dtype=float).copy()
        tau = self._rigid.gravity(q) + self._payload_tau(q) + integral_bias
        if self._gravity_comp:
            tau = tau - self._rigid.gravity(q)  # plant adds gravity itself
        tau = np.clip(tau, -self._max_tau, self._max_tau)
        return JointServoCommand(
            q_des=anchor,
            qd_des=np.zeros_like(q),
            tau_ff=tau,
            kp=self._kp.copy(),
            kd=self._kd.copy(),
        )

    def set_payload(
        self,
        mass_kg: float,
        *,
        com_offset: Sequence[float] | None = None,
    ) -> None:
        """Replace the complete payload in the construction-owned EE frame.

        CoM is local to that frame; None means zero offset, never previous CoM.
        Zero mass clears both mass and CoM. Copy the value so callers cannot
        change the active payload by mutating their array after declaration.
        """
        mass = float(mass_kg)
        if not np.isfinite(mass):
            raise ValueError("payload mass must be finite")
        mass = max(0.0, mass)
        if mass > 0.0 and self._rigid is not self._dyn:
            raise ValueError("a load model is installed: its bodies carry the load, not a payload")
        com = None
        if mass > 0.0:
            if not self._payload_frame:
                raise ValueError("positive payload requires a construction-owned payload_frame")
            if com_offset is not None:
                com = np.array(com_offset, dtype=float, copy=True)
                if com.shape != (3,) or not np.all(np.isfinite(com)):
                    raise ValueError("payload CoM must have shape (3,) and be finite")
        self._payload = (mass, com)

    def set_dynamics(self, dynamics) -> None:
        """Install a load model's RNEA (``arm_control.dynamics.BodyListDynamics``, over this executor's joint order) in place
        of the construction model's: the base with the modules grafted onto it. Joint limits stay the construction model's.
        Refused while a payload is declared (the payload's frame belongs to the construction model)."""
        if list(getattr(dynamics, "joints", [])) != self._joints:
            raise ValueError(f"load model joints {getattr(dynamics, 'joints', None)} != executor joints {self._joints}")
        if self._payload[0] > 0.0:
            raise ValueError("clear the payload before installing a load model")
        self._rigid = dynamics

    def _payload_tau(self, q: np.ndarray) -> np.ndarray:
        mass, com = self._payload
        if mass <= 0.0:
            return np.zeros(len(self._joints))
        return self._dyn.payload_gravity(
            q, mass, self._payload_frame, com
        )

    def set_gains(self, kp: np.ndarray | None = None, kd: np.ndarray | None = None) -> None:
        """Install a control law. Refuses stiffness without damping.

        THE chokepoint: plans (ArmController.on_plan), operator presets
        (_set_gains) and config all arrive here, so the guard lives here once
        rather than in each caller. See gain_error for why the rule is what
        it is.
        """
        reason = gain_error(self._kp if kp is None else kp,
                            self._kd if kd is None else kd)
        if reason:
            raise ValueError(reason)
        if kp is not None:
            kp = np.asarray(kp, dtype=float)
            if kp.shape != self._kp.shape:
                raise ValueError(f"kp shape {kp.shape} != expected {self._kp.shape}")
            self._kp = kp
        if kd is not None:
            kd = np.asarray(kd, dtype=float)
            if kd.shape != self._kd.shape:
                raise ValueError(f"kd shape {kd.shape} != expected {self._kd.shape}")
            self._kd = kd

    def step(self, pt: TrajectoryPoint, state: JointState, *,
             integral_bias: np.ndarray | float = 0.0) -> JointServoCommand:
        # qdd comes off the SAME polynomial as q and qd (JointTrajectory is a
        # cubic Hermite). It used to be a 1 ms forward difference of qd, which
        # both lagged half a step and straddled knots -- and differenced a qd
        # that was not the derivative of the q being commanded anyway.
        tau_ff = self._rigid.rnea(state.position, pt.velocity, pt.acceleration)
        tau_ff = tau_ff + self._payload_tau(np.asarray(state.position, dtype=float))
        tau_ff = tau_ff + integral_bias
        if self._gravity_comp:
            tau_ff = tau_ff - self._rigid.gravity(state.position)
        tau_ff = np.clip(tau_ff, -self._max_tau, self._max_tau)
        return JointServoCommand(
            q_des=pt.position.copy(),
            qd_des=pt.velocity.copy(),
            tau_ff=tau_ff,
            kp=self._kp.copy(),
            kd=self._kd.copy(),
        )

    def at_goal(
        self,
        goal: np.ndarray,
        state: JointState,
        pos_tol: float | None = None,
        vel_tol: float | None = None,
    ) -> bool:
        """At ``goal`` and at rest, within this arm's completion tolerances."""
        pos_tol = self._done_pos_tol if pos_tol is None else float(pos_tol)
        vel_tol = self._done_vel_tol if vel_tol is None else float(vel_tol)
        pos_ok = bool(np.max(np.abs(state.position - goal)) <= pos_tol)
        vel_ok = bool(np.max(np.abs(state.velocity)) <= vel_tol)
        return pos_ok and vel_ok

    @property
    def pose_tolerances(self) -> tuple[float, float] | None:
        """Hand position (m) and rotation (rad) sanity bound of a Cartesian leg's completion, or None (not configured)."""
        return self._done_pose_tol

    def hand_error(self, goal: np.ndarray, q: np.ndarray) -> tuple[float, float]:
        """Hand position (m) and rotation (rad) distance between the joint poses ``q`` and ``goal``, on the construction
        model (a load model has no ee frame). Both sides on one model, so a model offset common to both cancels."""
        a = self._dyn.frame_pose(q, self._payload_frame)
        b = self._dyn.frame_pose(goal, self._payload_frame)
        cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
        return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.arccos(np.clip(cos, -1.0, 1.0)))

    def hand_at_goal(self, goal: np.ndarray, state: JointState) -> bool:
        """At rest with the hand within ``pose_tolerances`` of the hand at ``goal`` (a Cartesian leg's completion)."""
        if self._done_pose_tol is None:
            raise ValueError("this executor has no done_pose_tol")
        if np.max(np.abs(state.velocity)) > self._done_vel_tol:
            return False
        pos, rot = self.hand_error(goal, state.position)
        return pos <= self._done_pose_tol[0] and rot <= self._done_pose_tol[1]

    @property
    def completion_tolerances(self) -> tuple[float, float]:
        """Joint position and velocity limits used by default completion."""
        return self._done_pos_tol, self._done_vel_tol
