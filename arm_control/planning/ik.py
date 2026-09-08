"""Damped least-squares inverse kinematics using Pinocchio."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover
    raise ImportError("PinocchioIK requires the `pinocchio` Python package.") from exc


class PinocchioIK:
    """Damped least-squares inverse kinematics solver backed by Pinocchio."""

    def __init__(
        self,
        urdf_path: str | Path,
        ee_frame: str,
        joint_names: Sequence[str],
        *,
        max_iters: int = 200,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        damping: float = 1e-4,
        step_scale: float = 1.0,
        restarts: int = 16,
    ) -> None:
        self._model = pin.buildModelFromUrdf(str(urdf_path))
        self._data = pin.Data(self._model)
        self._ee_frame = ee_frame
        self._frame_id = self._model.getFrameId(ee_frame)
        if self._frame_id >= len(self._model.frames):
            raise ValueError(f"ee_frame {ee_frame!r} not found in URDF")
        self._joint_names = list(joint_names)

        # Map the user-facing joint names to their q-indices in the model.
        # Joints not in this list stay fixed at zero (e.g. gripper joints).
        self._q_indices: list[int] = []
        self._v_indices: list[int] = []
        for name in self._joint_names:
            if not self._model.existJointName(name):
                raise ValueError(f"joint {name!r} not found in URDF")
            jid = self._model.getJointId(name)
            joint = self._model.joints[jid]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"joint {name!r} has nq={joint.nq}, nv={joint.nv}; "
                    "only single-DOF joints are supported"
                )
            self._q_indices.append(joint.idx_q)
            self._v_indices.append(joint.idx_v)
        self._q_indices_arr = np.array(self._q_indices, dtype=int)
        self._v_indices_arr = np.array(self._v_indices, dtype=int)
        # Joint limits for the controlled joints; iterates are projected onto
        # them so solutions are physically commandable (URDF limits; infinite
        # limits make the clip a no-op).
        self._q_lower_hard = self._model.lowerPositionLimit[self._q_indices_arr].copy()
        self._q_upper_hard = self._model.upperPositionLimit[self._q_indices_arr].copy()
        # Solve limits keep solutions strictly inside the hard stops: a goal
        # ON a stop is unreachable in practice (sim limit springs and real
        # stops both settle short of it, so trajectory completion tolerances
        # are never met).
        margin = np.minimum(0.02, 0.1 * (self._q_upper_hard - self._q_lower_hard))
        self._q_lower = self._q_lower_hard + margin
        self._q_upper = self._q_upper_hard - margin

        self._max_iters = int(max_iters)
        self._tol_pos = float(tol_pos)
        self._tol_rot = float(tol_rot)
        self._damping = float(damping)
        self._step_scale = float(step_scale)
        self._restarts = max(1, int(restarts))

    @property
    def n_joints(self) -> int:
        return len(self._joint_names)

    @property
    def hard_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) actual URDF joint limits."""
        return self._q_lower_hard.copy(), self._q_upper_hard.copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6 x n end-effector Jacobian in the LOCAL frame, controlled columns only.

        Same call the solver makes each iteration, exposed because a jog needs
        it for a reason the solver does not: how close this pose is to losing a
        Cartesian direction entirely.
        """
        q = np.asarray(q, dtype=float).ravel()
        if q.shape != (self.n_joints,):
            raise ValueError(f"q must have shape ({self.n_joints},)")
        q_full = pin.neutral(self._model)
        q_full[self._q_indices_arr] = q
        pin.forwardKinematics(self._model, self._data, q_full)
        pin.updateFramePlacements(self._model, self._data)
        J_full = pin.computeFrameJacobian(
            self._model, self._data, q_full, self._frame_id, pin.LOCAL
        )
        return np.asarray(J_full[:, self._v_indices_arr], dtype=float)

    def sigma_min(self, q: np.ndarray, *, rows: str = "pos") -> float:
        """Smallest singular value of the Jacobian: distance to a singularity.

        At a singularity some Cartesian direction costs unbounded joint rate,
        which is exactly the failure a velocity jog walks into -- the operator
        asks for 1 cm/s and the wrist tries to slew. sigma_min collapsing toward
        zero is that condition, and it is cheap enough to evaluate per tick.

        ``rows="pos"`` (default) uses the three TRANSLATIONAL rows. That is the
        honest measure for a translation jog: the full 6xn matrix mixes metres
        with radians, so its singular values depend on the unit choice and a
        threshold tuned on one arm means nothing on another. ``rows="all"`` is
        there for an orientation jog, where the mixing is unavoidable and the
        threshold has to be read as arm-specific.

        Computed here rather than read from a vendor API on purpose. libfranka
        exposes a Jacobian, but this package drives more than one arm, and on
        the FR3 the motion path lives on the RT machine while this host is
        read-only -- Pinocchio gives the same matrix for every arm we support.
        """
        J = self.jacobian(q)
        if rows == "pos":
            J = J[:3, :]
        elif rows != "all":
            raise ValueError(f"rows must be 'pos' or 'all', got {rows!r}")
        return float(np.linalg.svd(J, compute_uv=False)[-1])

    def fk(self, q: np.ndarray) -> np.ndarray:
        """4x4 EE pose at ``q`` in the same frame IK targets live in.

        The dock verify compares this (from MEASURED joints — the only
        mm-capable sensing the bench also has) against the seat target.
        """
        q = np.asarray(q, dtype=float).ravel()
        if q.shape != (self.n_joints,):
            raise ValueError(f"q must have shape ({self.n_joints},)")
        q_full = pin.neutral(self._model)
        q_full[self._q_indices_arr] = q
        pin.forwardKinematics(self._model, self._data, q_full)
        pin.updateFramePlacements(self._model, self._data)
        M = self._data.oMf[self._frame_id]
        T = np.eye(4)
        T[:3, :3] = M.rotation
        T[:3, 3] = M.translation
        return T

    def solve(
        self,
        target_T: np.ndarray,
        q0: np.ndarray,
        validate=None,
    ) -> np.ndarray | None:
        """Solve IK for ``target_T`` starting from ``q0``, with restarts.

        Damped least-squares from a single seed is fragile: a seed on a joint
        bound (a parked arm often is) wedges against the per-step clip, and
        even interior seeds miss basins (measured ~50% on random reachable
        poses). Attempts run in order — ``q0`` first (preserves the
        near-current-pose solution when it converges), then mid-range, then
        deterministic interior samples. Returns the first success or ``None``.

        ``validate``: optional ``callable(q) -> bool``; a converged solution
        failing it is discarded and the next restart tried. Callers use this
        to reject self-colliding IK branches so the motion planner is never
        handed an invalid goal.
        """
        target_T = np.asarray(target_T, dtype=float)
        if target_T.shape != (4, 4):
            raise ValueError("target_T must be 4x4")
        q0 = np.asarray(q0, dtype=float)
        if q0.shape != (self.n_joints,):
            raise ValueError(f"q0 must have shape ({self.n_joints},)")

        for seed in self._seeds(q0, self._restarts):
            sol = self._solve_from(target_T, seed)
            if sol is not None and (validate is None or validate(sol)):
                return sol
        return None

    def solve_candidates(
        self,
        target_T: np.ndarray,
        q0: np.ndarray,
        validate=None,
        *,
        attempts: int = 64,
    ) -> list[np.ndarray]:
        """Return unique valid solutions from a bounded deterministic seed search.

        Uses the same seed order as ``solve`` but keeps searching after success.
        Solutions within 1e-6 per joint are duplicates; bounded joints never wrap.
        """
        if isinstance(attempts, bool) or not isinstance(attempts, (int, np.integer)) \
                or not 1 <= attempts <= 128:
            raise ValueError('attempts must be an integer in [1, 128]')
        target_T = np.asarray(target_T, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        if target_T.shape != (4, 4) or not np.isfinite(target_T).all():
            raise ValueError('target_T must be finite and 4x4')
        if q0.shape != (self.n_joints,) or not np.isfinite(q0).all():
            raise ValueError(f'q0 must be finite with shape ({self.n_joints},)')
        span = self._q_upper - self._q_lower
        if not (np.isfinite(self._q_lower).all() and np.isfinite(self._q_upper).all()
                and np.isfinite(span).all() and (span > 0).all()):
            raise ValueError('candidate search requires finite positive joint spans')
        candidates = []
        for seed in self._seeds(q0, attempts):
            sol = self._solve_from(target_T, seed)
            if sol is None:
                continue
            sol = np.asarray(sol, dtype=float)
            if sol.shape != q0.shape or not np.isfinite(sol).all() \
                    or (sol < self._q_lower).any() or (sol > self._q_upper).any():
                continue
            if validate is not None and not validate(sol):
                continue
            if not any(np.allclose(sol, q, atol=1e-6, rtol=0) for q in candidates):
                candidates.append(sol.copy())
        return candidates

    def _seeds(self, q0: np.ndarray, attempts: int) -> list[np.ndarray]:
        span = self._q_upper - self._q_lower
        inner_lo = self._q_lower + 0.1 * span
        inner_hi = self._q_upper - 0.1 * span
        rng = np.random.default_rng(0)  # fresh per call: reproducible seeds
        home = np.clip(np.zeros(self.n_joints), self._q_lower, self._q_upper)
        # q0 first (nearest solution when it works), then the home/rest pose —
        # some collision-free branches are only reachable from its basin, and
        # a parked arm's sagged q0 alone can miss them — then mid-range and
        # deterministic interior samples.
        seeds = [q0, home, 0.5 * (self._q_lower + self._q_upper)]
        seeds += [
            rng.uniform(inner_lo, inner_hi) for _ in range(attempts - 3)
        ]
        return seeds[:attempts]

    def _solve_from(self, target_T: np.ndarray, q0: np.ndarray) -> np.ndarray | None:
        # Build a full model-sized configuration; only update controlled joints.
        q_full = pin.neutral(self._model)
        q_full[self._q_indices_arr] = np.clip(q0, self._q_lower, self._q_upper)

        target_SE3 = pin.SE3(target_T[:3, :3], target_T[:3, 3])
        for _ in range(self._max_iters):
            pin.forwardKinematics(self._model, self._data, q_full)
            pin.updateFramePlacements(self._model, self._data)
            current = self._data.oMf[self._frame_id]
            err_SE3 = target_SE3.actInv(current)
            err = pin.log6(err_SE3).vector  # 6-vec [v; w]
            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            if pos_err < self._tol_pos and rot_err < self._tol_rot:
                return q_full[self._q_indices_arr].copy()
            J_full = pin.computeFrameJacobian(
                self._model, self._data, q_full, self._frame_id, pin.LOCAL
            )
            # Restrict Jacobian to the controlled joint columns.
            J = J_full[:, self._v_indices_arr]
            JJt = J @ J.T + (self._damping ** 2) * np.eye(6)
            dq_sub = -self._step_scale * J.T @ np.linalg.solve(JJt, err)
            dq_full = np.zeros(self._model.nv)
            dq_full[self._v_indices_arr] = dq_sub
            q_full = pin.integrate(self._model, q_full, dq_full)
            q_full[self._q_indices_arr] = np.clip(
                q_full[self._q_indices_arr], self._q_lower, self._q_upper
            )
        return None


def _self_check() -> None:
    """Check restart semantics without relying on a robot's convergence basins."""
    ik = object.__new__(PinocchioIK)
    ik._joint_names = ['a', 'b']
    ik._q_lower = np.full(2, -1.)
    ik._q_upper = np.full(2, 1.)
    ik._restarts = 16
    calls = []

    def solve_from(target, seed):
        calls.append(seed.copy())
        return seed.copy()

    ik._solve_from = solve_from
    target, start = np.eye(4), np.array([.2, .3])
    assert np.array_equal(ik.solve(target, start), start) and len(calls) == 1
    calls.clear()
    assert ik.solve(target, start, validate=lambda q: False) is None
    legacy_seeds = np.array(calls)
    assert len(calls) == 16, 'legacy restart budget changed'
    assert callable(getattr(ik, 'solve_candidates', None)), 'missing candidate enumeration'
    calls.clear()
    candidates = ik.solve_candidates(target, start)
    assert len(calls) == 64, 'candidate attempts must bound seed solves'
    assert np.array_equal(calls[:16], legacy_seeds), 'legacy seed order changed'
    assert len(candidates) == 63, 'identical home and midpoint were not deduplicated'
    assert np.array_equal(candidates, ik.solve_candidates(target, start))
    for attempts in (1, 2, 3, 128):
        calls.clear()
        ik.solve_candidates(target, start, attempts=attempts)
        assert len(calls) == attempts, 'candidate budget boundary changed'
    filtered = ik.solve_candidates(target, start, validate=lambda q: q[0] > 0)
    assert filtered and all(q[0] > 0 for q in filtered), 'invalid branch escaped'
    assert ik.solve_candidates(target, start, validate=lambda q: False) == []
    ik._solve_from = lambda target, seed: np.array([np.nan, 0.])
    assert ik.solve_candidates(target, start, attempts=1) == []
    ik._solve_from = lambda target, seed: np.array([2., 0.])
    assert ik.solve_candidates(target, start, attempts=1) == []
    for attempts in (True, 0, -1, 129, 1.5, '64'):
        try:
            ik.solve_candidates(target, start, attempts=attempts)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid attempts: {attempts!r}')
    for bad_target, bad_start in ((np.eye(3), start), (target, np.zeros(3)),
                                   (target * np.nan, start), (target, start * np.nan)):
        try:
            ik.solve_candidates(bad_target, bad_start)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted malformed or nonfinite candidate input')
    print('ik self-check OK: legacy 16, bounded deterministic candidates, filtering')


if __name__ == '__main__':
    _self_check()
