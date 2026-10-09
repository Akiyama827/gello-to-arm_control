"""Point cloud and object-pose wire contracts."""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from ._arrow import _pack, _unpack, unpack_json_message


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


def unpack_object_poses(payload: pa.Array) -> dict:
    """``frame`` (world), ``objects`` and ``stamp``: the capture time, Unix seconds on this PC, which a live consumer's
    freshness rests on (the planner falls back to arrival time without it)."""
    return unpack_json_message(payload, expected_schema="object_poses")
