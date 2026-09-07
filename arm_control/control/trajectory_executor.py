"""Joint trajectory executor producing servo commands."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from arm_control.dynamics import PinocchioDynamics
from arm_control.motion import JointServoCommand, JointState, JointTrajectory


class JointTrajectoryExecutor:
    """Samples a JointTrajectory and produces joint servo commands.

    The feedforward torque is RNEA(q_meas, qd_des, qdd_des) where qdd_des is
    a finite-difference of qd_des. This adds gravity, Coriolis, and inertia
    feedforward in one call. The downstream PD law (which lives in the
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
        for arr, name in (
            (kp_default, "kp_default"),
            (kd_default, "kd_default"),
            (max_torque, "max_torque"),
        ):
            if np.asarray(arr).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        self.arm_id = str(arm_id)
        self._joints = list(joint_names)
        self._dyn = dynamics
        self._kp = np.asarray(kp_default, dtype=float).copy()
        self._kd = np.asarray(kd_default, dtype=float).copy()
        self._max_tau = np.asarray(max_torque, dtype=float).copy()
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

    def clear_trajectory(self) -> None:
        self._traj = None

    def step(self, t_now: float, state: JointState) -> JointServoCommand:
        if self._traj is None:
            raise RuntimeError("step() called before load_trajectory()")
        tau_local = float(t_now) - self._t_start
        pt = self._traj.sample_at(tau_local)
        # Finite-difference qdd. Avoid sampling beyond duration (where vel = 0
        # by construction) - clamp to within [0, duration_sec].
        dt = 1e-3
        t1 = min(tau_local + dt, self._traj.duration_sec)
        pt_next = self._traj.sample_at(t1)
        actual_dt = max(t1 - tau_local, 1e-9)
        qdd = (pt_next.velocity - pt.velocity) / actual_dt
        tau_ff = self._dyn.rnea(state.position, pt.velocity, qdd)
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
