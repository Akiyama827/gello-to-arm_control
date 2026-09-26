"""Joint trajectory executor producing servo commands."""
from __future__ import annotations

import time
from typing import Callable, Sequence

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
        ki: np.ndarray | None = None,
        i_limit: np.ndarray | None = None,
        clock: Callable[[], float] = time.monotonic,
        *,
        payload_frame: str = "",
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
        # Clamped integral torque on (q_des - q), riding in tau_ff. A pure PD
        # parks the arm off target by (unmodelled torque)/kp: on the real FR3
        # that residual friction/model torque is 1-2 N.m, several mrad per
        # joint, 2-3 mm at a shared object (2026-09-24). Zero ki = off, which
        # every plant except the real FR3 keeps. The term is stored as a torque
        # so the clamp is a torque bound and gain changes cannot make it jump.
        self._ki = np.zeros(n) if ki is None else np.asarray(ki, dtype=float).ravel()[:n]
        self._i_limit = (np.zeros(n) if i_limit is None
                         else np.asarray(i_limit, dtype=float).ravel()[:n])
        if self._ki.shape != (n,) or self._i_limit.shape != (n,):
            raise ValueError(f"ki and i_limit must have shape ({n},)")
        if np.any(self._ki < 0) or np.any(self._i_limit < 0):
            raise ValueError("ki and i_limit must be nonnegative")
        self._clock = clock
        self._i_tau = np.zeros(n)
        self._i_t: float | None = None

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
        self, state: JointState, q_des: np.ndarray | None = None
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
        tau = self._dyn.gravity(q) + self._payload_tau(q) + self._integral(anchor, q)
        if self._gravity_comp:
            tau = tau - self._dyn.gravity(q)  # plant adds gravity itself
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
        mass = max(0.0, float(mass_kg))
        com = (None if mass == 0.0 or com_offset is None
               else np.array(com_offset, dtype=float, copy=True).ravel())
        self._payload = (mass, com)

    # ponytail: fixed integration window; make it configurable if a plant needs
    # a different transient/settle boundary.
    I_WINDOW_RAD = 0.02

    def reset_integral(self) -> None:
        """Drop the accumulated integral torque (disarm, law change)."""
        self._i_tau = np.zeros_like(self._i_tau)
        self._i_t = None

    def _integral(self, q_des: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Advance and return the clamped integral torque.

        dt comes from the executor's own clock, so building several commands in
        one tick (hold anchor + hold) integrates once, not once per call. Only
        errors inside I_WINDOW_RAD integrate: a larger error is a transient the
        PD is already fighting, and winding up on it overshoots at the end.
        A joint with no spring (kp == 0: Float, pose hold) carries none.
        """
        now = self._clock()
        dt = 0.0 if self._i_t is None else min(max(now - self._i_t, 0.0), 0.05)
        self._i_t = now
        err = np.asarray(q_des, dtype=float) - np.asarray(q, dtype=float)
        live = (self._kp > 0) & (np.abs(err) < self.I_WINDOW_RAD)
        self._i_tau = np.where(live, self._i_tau + self._ki * err * dt, self._i_tau)
        self._i_tau = np.clip(self._i_tau, -self._i_limit, self._i_limit)
        self._i_tau[self._kp <= 0] = 0.0
        return self._i_tau.copy()

    def _payload_tau(self, q: np.ndarray) -> np.ndarray:
        mass, com = self._payload
        if mass <= 0.0 or not self._payload_frame:
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

    def step(self, pt: TrajectoryPoint, state: JointState) -> JointServoCommand:
        # qdd comes off the SAME polynomial as q and qd (JointTrajectory is a
        # cubic Hermite). It used to be a 1 ms forward difference of qd, which
        # both lagged half a step and straddled knots -- and differenced a qd
        # that was not the derivative of the q being commanded anyway.
        tau_ff = self._dyn.rnea(state.position, pt.velocity, pt.acceleration)
        tau_ff = tau_ff + self._payload_tau(np.asarray(state.position, dtype=float))
        tau_ff = tau_ff + self._integral(pt.position, state.position)
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
    def completion_tolerances(self) -> tuple[float, float]:
        """Joint position and velocity limits used by default completion."""
        return self._done_pos_tol, self._done_vel_tol


def _check_integral() -> None:
    """The integral term: accumulates, clamps, windows, respects kp=0, resets."""

    class _NoDynamics:
        def gravity(self, q):
            return np.zeros(len(q))

    t = [0.0]
    ex = JointTrajectoryExecutor(
        "arm", ["j1", "j2"], _NoDynamics(), np.array([100.0, 0.0]),
        np.array([10.0, 1.0]), np.array([50.0, 50.0]),
        ki=np.array([200.0, 200.0]), i_limit=np.array([1.0, 1.0]), clock=lambda: t[0],
    )
    state = JointState(position=np.zeros(2), velocity=np.zeros(2))
    target = np.array([0.005, 0.005])
    assert np.allclose(ex.hold_command(state, q_des=target).tau_ff, 0.0)  # first call: no dt
    t[0] = 0.01
    tau = ex.hold_command(state, q_des=target).tau_ff
    assert np.isclose(tau[0], 200 * 0.005 * 0.01) and tau[1] == 0.0, tau  # kp=0 joint: none
    assert np.allclose(ex.hold_command(state, q_des=target).tau_ff, tau)  # same tick: once
    for k in range(200):
        t[0] = 0.02 + 0.01 * k
        tau = ex.hold_command(state, q_des=target).tau_ff
    assert np.isclose(tau[0], 1.0), tau  # clamped at i_limit
    t[0] += 0.01
    far = ex.hold_command(state, q_des=np.array([0.5, 0.0])).tau_ff
    assert np.isclose(far[0], 1.0), far  # outside the window: held, not grown
    ex.reset_integral()
    t[0] += 0.01
    assert np.allclose(ex.hold_command(state, q_des=target).tau_ff, 0.0)
    off = JointTrajectoryExecutor(
        "arm", ["j1"], _NoDynamics(), np.array([100.0]), np.array([10.0]), np.array([50.0]),
        clock=lambda: t[0],
    )
    off.hold_command(state_1 := JointState(np.zeros(1), np.zeros(1)), q_des=np.array([0.01]))
    t[0] += 1.0
    assert off.hold_command(state_1, q_des=np.array([0.01])).tau_ff[0] == 0.0  # default off
    print("trajectory_executor: integral OK")


if __name__ == "__main__":
    _check_integral()
