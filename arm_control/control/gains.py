"""Hardware-side gain contract for the DM MIT wire format.

The DM MIT control frame packs kp/kd/torque into fixed-width unsigned fields
with per-motor-type ranges (see ``DmMotorLimits``).  ``_float_to_uint`` clips
out-of-range values silently, so the Isaac baseline validated at kp=600/kd=60
would encode as kp=500/kd=5 with no warning.  This module makes that a hard
error instead: gains outside the DM encode range are rejected, never clipped.

Mirrors the Isaac-side ``validate_servo_gains`` intent (reject, do not clip),
but checks against the wire-format encode range rather than the configured
implicit-actuator gains.  Pure function, no I/O.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arm_control.plants.dm.backend import DmMotorLimits


def validate_hardware_gains(
    kp: float, kd: float, torque: float, limits: "DmMotorLimits"
) -> None:
    """Reject a per-motor kp/kd/torque triple that the DM frame cannot encode.

    Per-motor scalar check against ``limits`` (a ``DmMotorLimits``), matching the
    per-motor signature of ``pack_mit_control_frame`` — the encode seam is
    per-motor, so validation is too.  Raises ``ValueError`` naming the offending
    field, its value, and the encode range on the first violation; passes
    silently (returns ``None``) when all three are in range.
    """
    for name, value, lo, hi in (
        ("kp", kp, limits.kp_min, limits.kp_max),
        ("kd", kd, limits.kd_min, limits.kd_max),
        ("torque", torque, limits.torque_min, limits.torque_max),
    ):
        # `not (lo <= value <= hi)` is True for NaN (all comparisons False), so a
        # NaN gain raises this clear error instead of int(nan) deep in the packer.
        if not (lo <= value <= hi):
            raise ValueError(
                f"{name}={value:g} is out of DM encode range [{lo:g}, {hi:g}]; "
                "gains must fit the wire format, not be silently clipped"
            )


def validate_torque_limits(
    limits, n: int | None = None, *, where: str = "torque_limits"
) -> np.ndarray:
    """Return ``limits`` as a frozen float vector, or raise.

    One check behind every torque clamp in the stack. A NaN limit is worse than
    a wrong one: ``abs(tau) > nan`` is False, so the clamp silently never fires,
    and ``np.clip(tau, -lim, lim)`` with a negative ``lim`` inverts the interval
    and returns the limit itself. Both must be configuration errors, not runtime
    surprises. Pass ``n`` to also pin the joint count.
    """
    values = np.asarray(limits, dtype=float).copy()
    if values.ndim != 1 or not values.size:
        raise ValueError(f"{where} must be a non-empty joint vector")
    if n is not None and values.size != n:
        raise ValueError(f"{where} length {values.size} != {n}")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{where} must be finite and positive, got {values.tolist()}")
    values.setflags(write=False)
    return values


def _self_check() -> None:
    """The NaN/negative limits that used to slip past the clamp must now raise."""
    ok = validate_torque_limits([9.0, 3.0], 2)
    assert ok.tolist() == [9.0, 3.0] and not ok.flags.writeable
    for bad in ([float("nan"), 3.0], [-9.0, 3.0], [0.0, 3.0], [], [[9.0, 3.0]]):
        try:
            validate_torque_limits(bad, 2)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad torque limits: {bad}")
    try:
        validate_torque_limits([9.0, 3.0], 3)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a length mismatch")
    # The failure mode this guards: an unvalidated NaN limit clamps NOTHING.
    tau, lim = np.array([1e6]), np.array([float("nan")])
    assert not (np.abs(tau) > lim)[0], "premise: NaN comparison is False, so no clamp"
    print("gains self-check OK")


if __name__ == "__main__":
    _self_check()
