"""Joint trajectory executor producing servo commands."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from arm_control.control.gains import validate_torque_limits
from arm_control.dynamics import PinocchioDynamics
from arm_control.motion import JointServoCommand, JointState, JointTrajectory


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
    """Samples a JointTrajectory and produces joint servo commands.

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
        self._kp = np.asarray(kp_default, dtype=float).copy()
        self._kd = np.asarray(kd_default, dtype=float).copy()
        self._max_tau = np.array(max_torque, dtype=float)
        # Held-payload feedforward: RNEA knows only the bare arm, so a grasped
        # module's weight otherwise lands on the PD as steady-state sag (the
        # softened contact-phase gains make it centimeters at the EE).
        self._payload_mass = 0.0
        self._payload_frame = ""
        self._payload_com: np.ndarray | None = None
        # Plant compensates gravity itself (FR3 control box): ship RNEA MINUS
        # gravity. The subtraction happens HERE, before the torque clamp — a
        # post-clamp subtraction downstream let |tau| reach clamp+|g| and made
        # the configured torque_limits decorative (audit 2026-07-29).
        self._gravity_comp = bool(gravity_comp)
        self._traj: JointTrajectory | None = None
        self._t_start: float = 0.0
        # Leg-completion tolerances. Joint stiction (frictionloss vs the soft
        # wrist kp) parks a settle short of the target — 1e-3 rad assumes a
        # frictionless plant; a sim with the noslip pass (or a real geared
        # joint) stalls at ~frictionloss/kp and done() would never fire.
        self._done_pos_tol = float(done_pos_tol)
        self._done_vel_tol = float(done_vel_tol)

    @property
    def n_joints(self) -> int:
        return len(self._joints)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) FACTORY limits off the URDF this executor loaded."""
        return self._dyn.joint_limits

    @property
    def has_trajectory(self) -> bool:
        return self._traj is not None

    @property
    def kp(self) -> np.ndarray:
        return self._kp.copy()

    @property
    def kd(self) -> np.ndarray:
        return self._kd.copy()

    def hold_command(
        self, state: JointState, q_des: np.ndarray | None = None
    ) -> JointServoCommand:
        """Static hold: zero desired velocity, gravity-only feedforward,
        current gains, torque-clamped.

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
        tau = self._dyn.gravity(q) + self._payload_tau(q)
        if self._gravity_comp:
            tau = tau - self._dyn.gravity(q)  # plant adds gravity itself
        tau = np.clip(tau, -self._max_tau, self._max_tau)
        anchor = q if q_des is None else np.asarray(q_des, dtype=float).copy()
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
        frame_name: str = "",
        com_offset: Sequence[float] | None = None,
    ) -> None:
        """Enable (mass > 0, with the EE frame) or clear (mass 0) payload FF.

        ``com_offset``: payload CoM in the frame's LOCAL coordinates — the
        carried module's weight acts ~12 cm out of the flange, not at it.
        """
        self._payload_mass = max(0.0, float(mass_kg))
        if frame_name:
            self._payload_frame = str(frame_name)
        if com_offset is not None:
            self._payload_com = np.asarray(com_offset, dtype=float).ravel()

    def _payload_tau(self, q: np.ndarray) -> np.ndarray:
        if self._payload_mass <= 0.0 or not self._payload_frame:
            return np.zeros(len(self._joints))
        return self._dyn.payload_gravity(
            q, self._payload_mass, self._payload_frame, self._payload_com
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

    def load_trajectory(self, traj: JointTrajectory, t_start: float) -> None:
        if traj.num_joints != self.n_joints:
            raise ValueError(
                f"trajectory has {traj.num_joints} joints, executor expects {self.n_joints}"
            )
        self._traj = traj
        self._t_start = float(t_start)

    def elapsed(self, t_now: float) -> float:
        """Seconds into the loaded trajectory -- the executor's OWN clock.

        The controller needs this to index a Cartesian pose alongside the joint
        reference, and used to read ``executor._t_start`` directly. Both sides
        must agree on the anchor, so the anchor stays private and the elapsed
        time is what crosses the seam.
        """
        return float(t_now) - self._t_start

    def clear_trajectory(self) -> None:
        self._traj = None

    def step(self, t_now: float, state: JointState) -> JointServoCommand:
        if self._traj is None:
            raise RuntimeError("step() called before load_trajectory()")
        tau_local = float(t_now) - self._t_start
        pt = self._traj.sample_at(tau_local)
        # qdd comes off the SAME polynomial as q and qd (JointTrajectory is a
        # cubic Hermite). It used to be a 1 ms forward difference of qd, which
        # both lagged half a step and straddled knots -- and differenced a qd
        # that was not the derivative of the q being commanded anyway.
        tau_ff = self._dyn.rnea(state.position, pt.velocity, pt.acceleration)
        tau_ff = tau_ff + self._payload_tau(np.asarray(state.position, dtype=float))
        if self._gravity_comp:
            tau_ff = tau_ff - self._dyn.gravity(state.position)
        tau_ff = np.clip(tau_ff, -self._max_tau, self._max_tau)
        return JointServoCommand(
            q_des=pt.position.copy(),
            qd_des=pt.velocity.copy(),
            tau_ff=tau_ff,
            kp=self._kp.copy(),
            kd=self._kd.copy(),
        )

    def done(
        self,
        t_now: float,
        state: JointState,
        pos_tol: float | None = None,
        vel_tol: float | None = None,
    ) -> bool:
        if self._traj is None:
            return True
        pos_tol = self._done_pos_tol if pos_tol is None else float(pos_tol)
        vel_tol = self._done_vel_tol if vel_tol is None else float(vel_tol)
        tau_local = float(t_now) - self._t_start
        if tau_local < self._traj.duration_sec:
            return False
        last = self._traj.sample_at(self._traj.duration_sec)
        pos_ok = bool(np.max(np.abs(state.position - last.position)) <= pos_tol)
        vel_ok = bool(np.max(np.abs(state.velocity)) <= vel_tol)
        return pos_ok and vel_ok

    @property
    def completion_tolerances(self) -> tuple[float, float]:
        """Joint position and velocity limits used by default completion."""
        return self._done_pos_tol, self._done_vel_tol
