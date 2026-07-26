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
            rng.uniform(inner_lo, inner_hi) for _ in range(self._restarts - 3)
        ]
        for seed in seeds[: self._restarts]:
            sol = self._solve_from(target_T, seed)
            if sol is not None and (validate is None or validate(sol)):
                return sol
        return None

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
