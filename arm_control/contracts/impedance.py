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


CARTESIAN_SIZE = 21


def cartesian_values(spec: dict) -> list[float]:
    """The v8 Cartesian-impedance tail (operator 2026-10-09): ``task_R`` (3x3, the task axes as COLUMNS in the robot BASE
    frame), ``kc`` and ``dc`` (task-frame diagonals [x, y, z, rx, ry, rz]). No target pose: the RT box computes it as
    FK(q_des) with the model it measures with. ``frame`` must say ``base`` -- a world-frame spec is refused here, never
    sent. Bounds are pose hold's (one envelope for every Cartesian law on the RT box), mirrored by ``valid_cartesian_spec``."""
    keys = {"frame", "task_R", "kc", "dc"}
    if set(spec) != keys:
        raise ValueError(f"cartesian requires exactly {sorted(keys)}")
    if spec["frame"] != "base":
        raise ValueError("the cartesian tail is in the robot BASE frame (convert world-frame specs before sending)")
    rows = [list(map(float, row)) for row in spec["task_R"]]
    kc, dc = [float(x) for x in spec["kc"]], [float(x) for x in spec["dc"]]
    if len(rows) != 3 or any(len(row) != 3 for row in rows) or len(kc) != 6 or len(dc) != 6:
        raise ValueError("cartesian requires a 3x3 task_R and six-axis kc/dc")
    flat = [v for row in rows for v in row]
    if not all(math.isfinite(v) for v in (*flat, *kc, *dc)):
        raise ValueError("cartesian values must be finite")
    for i in range(3):
        for j in range(3):
            if abs(sum(rows[i][k] * rows[j][k] for k in range(3)) - (i == j)) > 1e-6:
                raise ValueError("task_R must be orthonormal")
    r = rows
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    if abs(det - 1.0) > 1e-6:
        raise ValueError("task_R must be a rotation (det +1)")
    caps = [1000] * 3 + [100] * 3 + [200] * 3 + [50] * 3
    if any(not 0 <= x <= cap for x, cap in zip((*kc, *dc), caps)):
        raise ValueError("cartesian gains must be nonnegative and within the pose-hold envelope")
    return [*flat, *kc, *dc]


if __name__ == "__main__":
    # The refusals protocol_selfcheck applies to the same tail (valid_cartesian_spec), index by index.
    good = dict(frame="base", task_R=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], kc=[1000, 200, 200, 20, 5, 5],
                dc=[130, 40, 40, .5, .25, .25])
    base = cartesian_values(good)
    assert len(base) == CARTESIAN_SIZE and base[:9] == [0, -1, 0, 1, 0, 0, 0, 0, 1]
    bad = {0: 2.0, 8: -1.0, 9: 1000.1, 12: 100.1, 15: 200.1, 18: 50.1, 10: -1.0, 20: -.1, 4: math.nan, 16: math.inf}
    for index, value in bad.items():
        values = list(base)
        values[index] = value
        spec = dict(frame="base", task_R=[values[0:3], values[3:6], values[6:9]], kc=values[9:15], dc=values[15:21])
        try:
            cartesian_values(spec)
        except ValueError:
            continue
        raise AssertionError(f"accepted a bad value at {index}")
    for wrong in (dict(good, frame="world"), {k: v for k, v in good.items() if k != "dc"}, dict(good, extra=1)):
        try:
            cartesian_values(wrong)
        except ValueError:
            continue
        raise AssertionError(f"accepted {sorted(wrong)}")
    print("PASS cartesian_values: rotation, envelope, finiteness and frame refusals match valid_cartesian_spec")
