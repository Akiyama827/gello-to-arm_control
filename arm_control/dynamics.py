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

    def __init__(self, urdf_path: str | Path, joint_names: Sequence[str], *, gravity=None) -> None:
        self._model = pin.buildModelFromUrdf(str(urdf_path))
        if gravity is not None:
            gravity = np.asarray(gravity, dtype=float)
            if gravity.shape != (3,) or not np.isfinite(gravity).all():
                raise ValueError("gravity must be three finite values in the model base frame")
            self._model.gravity.linear = gravity
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

    def frame_pose(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        """``base_T_frame`` (4x4) at the arm joints ``q``, on this model."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_joints,):
            raise ValueError(f"q must have shape ({self.n_joints},), got {q.shape}")
        if not self._model.existFrame(frame_name):
            raise ValueError(f"frame {frame_name!r} not found in URDF")
        q_full, _, _ = self._embed(q, np.zeros(self.n_joints), np.zeros(self.n_joints))
        pin.framesForwardKinematics(self._model, self._data, q_full)
        return self._data.oMf[self._model.getFrameId(frame_name)].homogeneous.copy()

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
        feedforward: tau = J_p(q)^T · (-m gravity) at the payload's CoM point p,
        with gravity expressed in the model's base frame.

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
        tau_full = J_lin.T @ (-np.asarray(self._model.gravity.linear) * float(mass_kg))
        return np.asarray(tau_full[self._v_indices], dtype=float)


_HINGE, _SLIDE = 3, 2  # mujoco.mjtJoint.mjJNT_HINGE / mjJNT_SLIDE (no MuJoCo import for a number)


class BodyListDynamics:
    """RNEA over a revisioned rigid-body list: a model shipped as numbers (no URDF, no meshes, no pickle).

    ``description``: ``joints`` (the commanded order), ``bodies`` (parent index, pose, mass, CoM, inertia, an optional scalar
    ``joint`` {name, kind, axis, pos}), ``gravity``, ``inactive_joints`` (command slots the tree does not contain: held at
    model zero) and optional ``unmeasured_joints`` (tree joints no motor reports: model zero). Joint ``kind`` is MuJoCo's
    ``mjtJoint`` value (hinge 3, slide 2), the composer the model comes from. A load model's body list (the base, with the
    modules grafted onto it) is one of these; the supervisor installs it with load_model/1 (``ArmClient.set_load_model``).
    """

    def __init__(self, description):
        bodies = description['bodies']
        self.joints = list(description['joints'])
        unmeasured = description.get('unmeasured_joints', [])
        if (not isinstance(unmeasured, list) or not all(isinstance(n, str) for n in unmeasured)
                or len(set(unmeasured)) != len(unmeasured) or set(unmeasured) & set(self.joints)):
            raise ValueError('invalid unmeasured joint declarations')
        if not 0 < len(self.joints) <= 64 or len(set(self.joints)) != len(self.joints):
            raise ValueError('invalid dynamics joint order')
        if not 0 < len(bodies) <= 256:
            raise ValueError('invalid dynamics tree size')
        self.model = pin.Model()
        self.model.gravity.linear = self._array(description['gravity'], (3,))
        links = [(0, pin.SE3.Identity())]
        for bid, body in enumerate(bodies, 1):
            parent = body['parent']
            if not isinstance(parent, int) or not 0 <= parent < bid:
                raise ValueError('invalid dynamics parent order')
            parent_joint, parent_pose = links[parent]
            pose = parent_pose * pin.SE3(self._rot(body['quat']), self._array(body['pos'], (3,)))
            joint = body['joint']
            if joint is not None:
                axis = self._array(joint['axis'], (3,))
                if abs(np.linalg.norm(axis) - 1.0) > 1e-6:
                    raise ValueError('invalid joint axis')
                offset = self._array(joint['pos'], (3,))
                if joint['kind'] == _HINGE:
                    jm = pin.JointModelRevoluteUnaligned(axis)
                elif joint['kind'] == _SLIDE:
                    jm = pin.JointModelPrismaticUnaligned(axis)
                else:
                    raise ValueError('non-scalar dynamics joint')
                if (joint['name'] not in self.joints and joint['name'] not in description.get('unmeasured_joints', [])) or self.model.existJointName(joint['name']):
                    raise ValueError('unexpected or duplicate dynamics joint')
                parent_joint = self.model.addJoint(parent_joint, jm,
                    pose * pin.SE3(np.eye(3), offset), joint['name'])
                pose = pin.SE3(np.eye(3), -offset)
            mass = float(body['mass'])
            inertia = self._array(body['inertia'], (3,))
            if not np.isfinite(mass) or mass < 0 or np.any(inertia < 0):
                raise ValueError('invalid body inertia')
            rotation = self._rot(body['inertia_quat'])
            self.model.appendBodyToJoint(parent_joint,
                pin.Inertia(mass, self._array(body['com'], (3,)),
                            rotation @ np.diag(inertia) @ rotation.T), pose)
            links.append((parent_joint, pose))
        self.data = self.model.createData()
        inactive = description['inactive_joints']
        if (not isinstance(inactive, list) or len(set(inactive)) != len(inactive)
                or not set(inactive) <= set(self.joints)):
            raise ValueError('invalid inactive inventory slots')
        represented = set(self.model.names[1:])
        if represented != (set(self.joints) - set(inactive)) | set(description.get('unmeasured_joints', [])):
            raise ValueError('model does not represent every required base and attached joint')
        n = len(self.joints)
        limits = description.get('joint_limits')
        self._limits = (np.full(n, -np.inf), np.full(n, np.inf)) if limits is None else (
            np.asarray(limits['lower'], dtype=float), np.asarray(limits['upper'], dtype=float))
        if (any(v.shape != (n,) or np.isnan(v).any() for v in self._limits)
                or np.any(self._limits[0] > self._limits[1])):
            raise ValueError('invalid joint limits')
        self.slots, self.indices = [], []
        for slot, name in enumerate(self.joints):
            if self.model.existJointName(name):
                self.slots.append(slot)
                self.indices.append(self.model.joints[self.model.getJointId(name)].idx_v)

    @staticmethod
    def _array(value, shape):
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('nonfinite or malformed numeric dynamics')
        return array

    @classmethod
    def _rot(cls, value):
        quat = cls._array(value, (4,))
        if abs(np.linalg.norm(quat) - 1.0) > 1e-6:
            raise ValueError('invalid dynamics quaternion')
        w, x, y, z = quat  # MuJoCo and the scene contract store w,x,y,z
        return pin.Quaternion(w, x, y, z).matrix()

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) per commanded joint, from the description's optional ``joint_limits`` {lower, upper} (as the model's
        source states them, e.g. MuJoCo's ``jnt_range``); +/-inf where none is stated -- an honest infinity, never a made-up bound.
        Read only when this model is an executor's CONSTRUCTION model (a body-list arm, ``arm.body_list``)."""
        return self._limits

    def rnea(self, q, qd, qdd):
        vectors = [self._array(v, (len(self.joints),)) for v in (q, qd, qdd)]
        # The two-axis hardware stream contains no module-motor feedback.
        # Unmeasured internal joints use declared model zero, never fabricated
        # measured slots or motor authority. Hardware release remains blocked.
        full = []
        for vector in vectors:
            embedded = np.zeros(self.model.nv)
            embedded[self.indices] = vector[self.slots]
            full.append(embedded)
        output = np.zeros(len(self.joints))
        output[self.slots] = pin.rnea(self.model, self.data, *full)[self.indices]
        if not np.isfinite(output).all():
            raise ValueError('nonfinite RNEA output')
        return output

    def gravity(self, q):
        return self.rnea(q, np.zeros(len(q)), np.zeros(len(q)))
