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
