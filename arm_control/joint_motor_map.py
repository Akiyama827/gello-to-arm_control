"""Hand width <-> finger joint value: both fingers move symmetrically, 0 = closed, each opens by half the width."""
from __future__ import annotations


def finger_from_width(width_m: float) -> float:
    """One finger joint's value for an opening ``width_m`` between the fingers."""
    return width_m / 2.0


def width_from_finger(finger_m: float) -> float:
    """The opening between the fingers for one finger joint's value; the inverse of :func:`finger_from_width`."""
    return 2.0 * finger_m


__all__ = ["finger_from_width", "width_from_finger"]
