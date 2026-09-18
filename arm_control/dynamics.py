"""Shared RNEA / gravity-comp wrapper around Pinocchio.

Operates over a subset of model joints (the ones in ``joint_names``), holding
non-listed joints at neutral, zero velocity, zero acceleration — every
consumer (trajectory executor, impedance float, orchestrator) loads Pinocchio
the same way and shares the same subset semantics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover
    raise ImportError("PinocchioDynamics requires the `pinocchio` package.") from exc


class PinocchioDynamics:
    """Compute RNEA torques over a named joint subset of a Pinocchio model."""

    def __init__(self, urdf_path: str | Path, joint_names: Sequence[str]) -> None:
        self._model = pin.buildModelFromUrdf(str(urdf_path))
        self._data = pin.Data(self._model)
        self._joint_names = list(joint_names)

        q_idx: list[int] = []
        v_idx: list[int] = []
        for name in self._joint_names:
            if not self._model.existJointName(name):
                raise ValueError(f"joint {name!r} not found in URDF")
            jid = self._model.getJointId(name)
            joint = self._model.joints[jid]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"joint {name!r} has nq={joint.nq} nv={joint.nv}; "
                    "PinocchioDynamics only supports single-DoF joints"
                )
            q_idx.append(int(joint.idx_q))
            v_idx.append(int(joint.idx_v))
        self._q_indices = np.array(q_idx, dtype=int)
        self._v_indices = np.array(v_idx, dtype=int)

    @property
    def n_joints(self) -> int:
        return len(self._joint_names)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) FACTORY position limits, straight off the URDF.

        The robot description is the one place these digits live -- the FR3's
        are the vendor's to a ten-thousandth of a radian -- so a deployment
        must READ them rather than restate them. Read off the model this class
        already built for RNEA: parsing the same URDF a second time would be a
        second copy with its own way of going stale.

        Pinocchio reports +/-inf for a continuous joint; callers get that
        unchanged, since a fabricated bound would be worse than an honest
        infinity.
        """
        return (self._model.lowerPositionLimit[self._q_indices].copy(),
                self._model.upperPositionLimit[self._q_indices].copy())

    @property
    def velocity_limits(self) -> np.ndarray:
        """FACTORY joint velocity limits (URDF ``limit velocity``), per joint."""
        return self._model.velocityLimit[self._v_indices].copy()

    @property
    def effort_limits(self) -> np.ndarray:
        """FACTORY joint torque limits (URDF ``limit effort``), per joint."""
        return self._model.effortLimit[self._v_indices].copy()

    def _embed(self, q_sub: np.ndarray, qd_sub: np.ndarray, qdd_sub: np.ndarray):
        q = pin.neutral(self._model)
        qd = np.zeros(self._model.nv)
        qdd = np.zeros(self._model.nv)
        q[self._q_indices] = q_sub
        qd[self._v_indices] = qd_sub
        qdd[self._v_indices] = qdd_sub
        return q, qd, qdd

    def rnea(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        n = self.n_joints
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)
        if q.shape != (n,):
            raise ValueError(f"q must have shape ({n},), got {q.shape}")
        if qd.shape != (n,) or qdd.shape != (n,):
            raise ValueError(f"qd/qdd must have shape ({n},)")
        q_full, qd_full, qdd_full = self._embed(q, qd, qdd)
        tau_full = pin.rnea(self._model, self._data, q_full, qd_full, qdd_full)
        return np.asarray(tau_full[self._v_indices], dtype=float)

    def gravity(self, q: np.ndarray) -> np.ndarray:
        n = self.n_joints
        return self.rnea(q, np.zeros(n), np.zeros(n))

    def payload_gravity(
        self,
        q: np.ndarray,
        mass_kg: float,
        frame_name: str,
        com_offset: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Extra joint torques to carry a point payload rigid on ``frame_name``.

        The URDF knows only the bare arm; after a grasp the held module's
        weight is unmodeled and pure PD carries it with steady-state sag
        (~0.02 rad at the shadow-run gains — measured). This is the missing
        feedforward: tau = J_p(q)^T · (m g ẑ) at the payload's CoM point p.

        ``com_offset`` is the CoM in the frame's LOCAL coordinates. A module
        gripped ~12 cm out of the flange torques the wrist; hanging its
        weight at the frame origin leaves that moment on the PD (measured
        2 mm EE sag at the dock posture — enough to fail a mm seat verify).
        """
        n = self.n_joints
        q = np.asarray(q, dtype=float)
        if q.shape != (n,):
            raise ValueError(f"q must have shape ({n},), got {q.shape}")
        q_full, _, _ = self._embed(q, np.zeros(n), np.zeros(n))
        if not self._model.existFrame(frame_name):
            raise ValueError(f"frame {frame_name!r} not found in URDF")
        frame_id = self._model.getFrameId(frame_name)
        J = pin.computeFrameJacobian(
            self._model, self._data, q_full, frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_lin = J[:3]
        if com_offset is not None:
            # Point jacobian at p = frame ∘ offset: v_p = v_o + ω × r, so
            # J_p = J_lin − [r]× J_ang with r the world-frame offset vector.
            oMf = pin.updateFramePlacement(self._model, self._data, frame_id)
            r = oMf.rotation @ np.asarray(com_offset, dtype=float).ravel()
            J_lin = J_lin - pin.skew(r) @ J[3:]
        tau_full = J_lin.T @ np.array([0.0, 0.0, 9.81 * float(mass_kg)])
        return np.asarray(tau_full[self._v_indices], dtype=float)
