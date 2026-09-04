"""Static frame transforms — ONE root for the whole workcell.

The ChArUco plate bolted flat to the table **is** the world frame (it is the
only calibration artifact; the caller's hand-eye calibration measures every
extrinsic against it). Everything perception publishes — ``object_poses``,
``scene_cloud``, the workspace crop — is expressed in that world frame, and
each arm converts to its own base with its own ``world_T_arm``.

That split is what makes a second arm possible: "the base frame" is ambiguous
once two arms watch the same table, but the board is not. Perception has no
idea which arm will act on what it sees.

This module is deliberately NOT a transform tree (no TF2): the workcell has a
handful of STATIC transforms and no timestamped interpolation problem. All it
owns is the config-spec -> 4x4 conversion that four call sites used to
open-code, plus the two arm-mount helpers.

Frame naming follows the codebase convention ``a_T_b`` = the pose of ``b``
expressed in ``a``, so ``a_T_b @ b_T_c == a_T_c`` reads left to right.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "T_from_spec",
    "invert",
    "transform_points",
    "pose_xyzquat_to_T",
    "T_to_pose_xyzquat",
    "world_T_arm",
    "arm_T_world",
]


def _rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    import pinocchio as pin

    values = np.asarray(rpy, dtype=float).ravel()
    if values.shape != (3,):
        raise ValueError("rpy must be a 3-vector")
    return np.asarray(pin.rpy.rpyToMatrix(*values))


def _quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    import pinocchio as pin

    values = np.asarray(quat, dtype=float).ravel()
    if values.shape != (4,):
        raise ValueError("quat must be a 4-vector [qw,qx,qy,qz]")
    qw, qx, qy, qz = values
    return np.asarray(pin.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix())


def T_from_spec(spec: Any) -> np.ndarray:
    """Config value -> 4x4 homogeneous transform.

    Accepts every shape the YAML configs use, so callers never branch:

    - a 4x4 row-major nested list / 16-vector (what ``calibrate_hand_eye.py``
      prints for ``world_T_cam`` and ``ee_T_cam``)
    - a 7-vector ``[x,y,z,qw,qx,qy,qz]`` (the pose convention on the wire)
    - a mapping with ``origin``/``xyz``/``pos`` plus ``rpy`` or ``quat``
      (how arm mounts and scene entries are authored)
    - ``None`` -> identity
    """
    if spec is None:
        return np.eye(4)
    if isinstance(spec, Mapping):
        T = np.eye(4)
        for key in ("origin", "xyz", "pos", "position"):
            if key in spec:
                T[:3, 3] = np.asarray(spec[key], dtype=float).ravel()
                break
        if "rpy" in spec:
            T[:3, :3] = _rpy_to_matrix(spec["rpy"])
        elif "quat" in spec:
            T[:3, :3] = _quat_to_matrix(spec["quat"])
        return T
    arr = np.asarray(spec, dtype=float)
    if arr.shape == (4, 4):
        return arr.copy()
    if arr.size == 16:
        return arr.reshape(4, 4).copy()
    if arr.size == 7:
        T = np.eye(4)
        T[:3, 3] = arr.ravel()[:3]
        T[:3, :3] = _quat_to_matrix(arr.ravel()[3:])
        return T
    raise ValueError(
        f"cannot read a transform from a value of shape {arr.shape} — expected a "
        "4x4 matrix, a 7-vector [x,y,z,qw,qx,qy,qz], or a mapping with "
        "origin + rpy/quat"
    )


def pose_xyzquat_to_T(pose: Sequence[float] | np.ndarray) -> np.ndarray:
    """``[x,y,z,qw,qx,qy,qz]`` (the wire pose convention) -> 4x4.

    The strict single-shape twin of :func:`T_from_spec` for code paths where a
    non-7-vector is a bug, not an alternate spelling.
    """
    values = np.asarray(pose, dtype=float).ravel()
    if values.shape != (7,):
        raise ValueError("pose must be a 7-vector [x,y,z,qw,qx,qy,qz]")
    T = np.eye(4)
    T[:3, 3] = values[:3]
    T[:3, :3] = _quat_to_matrix(values[3:])
    return T


def T_to_pose_xyzquat(T: np.ndarray) -> list[float]:
    """Inverse of :func:`pose_xyzquat_to_T` (normalized, qw first)."""
    import pinocchio as pin

    T = np.asarray(T, dtype=float)
    q = pin.Quaternion(T[:3, :3])
    q.normalize()
    return [float(v) for v in (*T[:3, 3], q.w, q.x, q.y, q.z)]


def invert(T: np.ndarray) -> np.ndarray:
    """Rigid-transform inverse (transpose + rotated translation, not solve)."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to an Nx3 point array."""
    T = np.asarray(T, dtype=float)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be Nx3")
    return pts @ T[:3, :3].T + T[:3, 3]


def world_T_arm(cfg: Any) -> np.ndarray:
    """Pose of this config's arm base in the world (board) frame.

    Reads ``arm.world_origin`` / ``arm.world_rpy``. Identity when neither is
    set (the bench case where the board sits at the arm base) — which is why
    these two keys, unlike the rest of the ``arm:`` block, tolerate absence.
    """
    get = cfg.get if hasattr(cfg, "get") else dict(cfg).get
    arm = dict(get("arm") or {})
    origin = arm.get("world_origin")
    rpy = arm.get("world_rpy")
    return T_from_spec(
        {
            "origin": np.zeros(3) if origin is None else origin,
            "rpy": np.zeros(3) if rpy is None else rpy,
        }
    )


def arm_T_world(cfg: Any) -> np.ndarray:
    """Inverse of :func:`world_T_arm` — world poses into this arm's base frame."""
    return invert(world_T_arm(cfg))


def _demo() -> None:
    """Self-check: spec parsing, inverse, and the round trip through an arm mount."""
    # Every spec shape lands on the same transform.
    rot_z_180 = np.array(
        [[-1.0, 0.0, 0.0, 0.35], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]]
    )
    from_map = T_from_spec({"origin": [0.35, 0.0, 0.0], "rpy": [0.0, 0.0, np.pi]})
    from_mat = T_from_spec(rot_z_180.tolist())
    from_vec = T_from_spec([0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # qw=0, qz=1
    assert np.allclose(from_map, rot_z_180, atol=1e-9), from_map
    assert np.allclose(from_mat, rot_z_180, atol=1e-9), from_mat
    assert np.allclose(from_vec, rot_z_180, atol=1e-9), from_vec
    assert np.allclose(T_from_spec(None), np.eye(4))

    # invert really inverts, and matches the dense solve.
    assert np.allclose(invert(rot_z_180) @ rot_z_180, np.eye(4), atol=1e-12)
    assert np.allclose(invert(rot_z_180), np.linalg.inv(rot_z_180), atol=1e-12)

    # The pick-place scene's arm mount: a module at the world origin must land
    # at arm-base [0.35, 0, 0] (arm faces -x from 0.35 m out).
    cfg = {"arm": {"world_origin": [0.35, 0.0, 0.0], "world_rpy": [0, 0, np.pi]}}
    assert np.allclose(world_T_arm(cfg), rot_z_180, atol=1e-9)
    module_world = np.zeros((1, 3))
    module_arm = transform_points(arm_T_world(cfg), module_world)
    assert np.allclose(module_arm, [[0.35, 0.0, 0.0]], atol=1e-9), module_arm

    # A config with no mount keys is the identity (bench: board at the arm base).
    assert np.allclose(world_T_arm({}), np.eye(4))
    print("frames: ok")


if __name__ == "__main__":
    _demo()
