"""Point cloud and object-pose wire contracts."""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from ._arrow import _pack, _unpack, pack_json_message, unpack_json_message


def pack_points(points: np.ndarray) -> pa.Array:
    """Nx3 point cloud as a flat float64 array (low-rate preview payloads)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be Nx3")
    return _pack(pts)


def unpack_points(arrow: pa.Array) -> np.ndarray:
    flat = _unpack(arrow)
    if flat.size % 3:
        raise ValueError("point payload length not divisible by 3")
    return flat.reshape(-1, 3)


# --------------------------------------------------------------------------- #
# object_poses — the pose-estimator -> consumer contract. Hand-rolled on both
# ends until 2026-07-26; once perception lives in its own repo this codec is
# the ONLY place the schema exists, so drift between publisher and consumer
# becomes an import error instead of a silent field mismatch.
#
# ``object_id`` is OPAQUE here: this package never interprets it. What the
# tracked bodies ARE -- modules, fixtures, bricks -- is the caller's domain,
# and naming them in the wire format is what made this codec look like an
# assembly-task message rather than the pose contract it is.
# --------------------------------------------------------------------------- #


def pack_object_poses(
    *,
    camera: str,
    frame: str,
    stamp: float,
    objects: list[dict],
    tags: list[dict] | None = None,
) -> pa.Array:
    """``objects``: dicts with ``object_id`` and ``pose_xyzquat`` (7 floats,
    [x y z qw qx qy qz]) plus free-form extras (``fitness``…). ``frame`` names
    the frame the poses are expressed in — consumers assert on it."""
    for m in objects:
        pose = m.get("pose_xyzquat")
        if pose is None or len(pose) != 7:
            raise ValueError(f"object entry needs a 7-float pose_xyzquat: {m}")
    return pack_json_message(
        "object_poses",
        {
            "camera": camera,
            "frame": frame,
            "stamp": float(stamp),
            "objects": objects,
            "tags": list(tags or []),
        },
    )


def unpack_object_poses(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="object_poses")
