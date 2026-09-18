"""Validated Cartesian pose-hold settings shared by Arrow and RT encoders.

The plant captures its measured pose on a new id. Units: kc N/m, Nm/rad;
dc Ns/m, Nms/rad; nullspace kp Nm/rad and kd Nms/rad. These ceilings are
protocol bounds, not a claim that every allowed setting is safe on a robot.
"""
from __future__ import annotations

import math

POSE_HOLD_SIZE = 15


def pose_hold_values(spec: dict) -> list[float]:
    keys = {"id", "kc", "dc", "nullspace_kp", "nullspace_kd"}
    if set(spec) != keys:
        raise ValueError(f"pose hold requires exactly {sorted(keys)}")
    ident = float(spec["id"])
    if not math.isfinite(ident) or not (0 < ident < 2**32) or ident != int(ident):
        raise ValueError("pose hold id must be a nonzero uint32")
    kc, dc = list(spec["kc"]), list(spec["dc"])
    if len(kc) != 6 or len(dc) != 6:
        raise ValueError("Cartesian stiffness/damping require six axes")
    gains = [float(x) for x in (*kc, *dc, spec["nullspace_kp"], spec["nullspace_kd"])]
    caps = [1000]*3 + [100]*3 + [200]*3 + [50]*3 + [20, 10]
    if any(not math.isfinite(x) or not 0 <= x <= cap for x, cap in zip(gains, caps)):
        raise ValueError("pose hold gains must be finite, nonnegative and within protocol bounds")
    return [ident, *gains]


def unpack_pose_hold_values(values) -> dict:
    values = list(values)
    if len(values) != POSE_HOLD_SIZE:
        raise ValueError("pose hold block must contain 15 values")
    spec = dict(id=values[0], kc=values[1:7], dc=values[7:13],
                nullspace_kp=values[13], nullspace_kd=values[14])
    checked = pose_hold_values(spec)
    return dict(id=int(checked[0]), kc=checked[1:7], dc=checked[7:13],
                nullspace_kp=checked[13], nullspace_kd=checked[14])
