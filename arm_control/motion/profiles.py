"""Closed-form motion profiles with no planner or optimizer dependency."""
from __future__ import annotations

import numpy as np

from .types import JointTrajectory


def trapezoidal_trajectory(
    start: np.ndarray, goal: np.ndarray, max_vel: np.ndarray, max_acc: np.ndarray,
) -> JointTrajectory:
    """Rest-to-rest straight motion under joint velocity/acceleration limits.

    All moving joints share scalar progress along the line. Its bang-coast-bang
    speed profile is triangular when the move cannot reach the speed cap.
    Three or four Hermite knots represent the quadratic/linear pieces exactly.
    Work is O(number of joints), independent of distance or sample rate; this
    is suitable for a command handler that must not run an iterative solver.
    Acceleration can jump at phase boundaries; there is no jerk constraint.
    A zero-length move is rejected so the caller can retain its existing hold.
    """
    start, goal, max_vel, max_acc = (
        np.asarray(x, dtype=float) for x in (start, goal, max_vel, max_acc)
    )
    if start.ndim != 1 or not start.size or any(
        x.shape != start.shape for x in (goal, max_vel, max_acc)
    ):
        raise ValueError("profile inputs must be equal, nonempty 1-D vectors")
    if not all(np.isfinite(x).all() for x in (start, goal, max_vel, max_acc)):
        raise ValueError("profile inputs must be finite")
    if np.any(max_vel <= 0) or np.any(max_acc <= 0):
        raise ValueError("profile velocity and acceleration limits must be positive")
    delta = goal - start
    moving = np.abs(delta) > 0
    if not np.any(moving):
        raise ValueError("zero-length motion needs a hold, not a trajectory")
    distance = np.abs(delta[moving])
    speed = float(np.min(max_vel[moving] / distance))
    acceleration = float(np.min(max_acc[moving] / distance))
    peak = min(speed, np.sqrt(acceleration))
    ramp = peak / acceleration
    cruise = max(0., (1. - peak * ramp) / peak)
    # Avoid indistinguishable knots at the triangular/trapezoidal boundary.
    if cruise <= 32 * np.finfo(float).eps * ramp:
        ramp = 1. / np.sqrt(acceleration)
        peak = np.sqrt(acceleration)
        times = np.array([0., ramp, 2 * ramp])
        progress = np.array([0., .5, 1.])
        rates = np.array([0., peak, 0.])
    else:
        times = np.array([0., ramp, ramp + cruise, 2 * ramp + cruise])
        progress = np.array([0., .5 * peak * ramp, 1. - .5 * peak * ramp, 1.])
        rates = np.array([0., peak, peak, 0.])
    positions = start + progress[:, None] * delta
    positions[[0, -1]] = (start, goal)
    return JointTrajectory(times, positions, rates[:, None] * delta)
