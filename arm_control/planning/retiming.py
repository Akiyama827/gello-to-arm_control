"""Local parabolic corner blends and TOPPRA constrained path timing.

TOPPRA: Pham & Pham, IEEE T-RO 2018, doi:10.1109/TRO.2018.2819195.
The geometric path is C1 (linear segments joined by quadratic Bezier blends).
Both terms of qdd = q_s * sdd + q_ss * sd**2 enter its acceleration constraints.
The returned Hermite curve is independently bounded, as it is what we execute.
"""
from __future__ import annotations

import numpy as np

from arm_control.motion import JointTrajectory


# Euclidean joint-space deviation, not a Cartesian clearance allowance.
BLEND_TOLERANCE_RAD = 0.002
INTERPOLATION_TOLERANCE_RAD = 1e-6
COLLISION_STEP_RAD = 0.001
RETIMING_STEPS_RAD = (0.02, 0.01, 0.005, 0.0025)


class _BlendedPath:
    """TOPPRA geometric-path interface; exact line/quadratic derivatives."""

    def __init__(self, waypoints, tolerance):
        delta = np.diff(waypoints, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        directions = delta / lengths[:, None]
        segments, spans = [], []
        previous = waypoints[0]

        def line(end):
            distance = np.linalg.norm(end - previous)
            if distance > 1e-12:
                segments.append((previous, end - previous, np.zeros_like(end)))
                spans.append(distance)

        for i in range(1, len(waypoints) - 1):
            incoming, outgoing = directions[i - 1:i + 1]
            # Bezier controls all lie within tolerance of the corner, so the
            # entire blend does too. Quarter-leg trimming prevents overlap.
            trim = min(tolerance, .25 * lengths[i - 1], .25 * lengths[i])
            start = waypoints[i] - trim * incoming
            end = waypoints[i] + trim * outgoing
            line(start)
            segments.append((start, 2 * trim * incoming, trim * (outgoing - incoming)))
            spans.append(2 * trim)
            previous = end
        line(waypoints[-1])
        self.coefficients = np.asarray(segments)
        self.knots = np.r_[0., np.cumsum(spans)]
        self.dof = waypoints.shape[1]
        self.path_interval = self.knots[[0, -1]]

    def __call__(self, s, order=0):
        s = np.asarray(s, dtype=float)
        index = np.clip(np.searchsorted(self.knots, s, side="right") - 1,
                        0, len(self.knots) - 2)
        h = (self.knots[index + 1] - self.knots[index])[..., None]
        u = ((s - self.knots[index])[..., None]) / h
        c = self.coefficients[index]
        if order == 0:
            return c[..., 0, :] + u * (c[..., 1, :] + u * c[..., 2, :])
        if order == 1:
            return (c[..., 1, :] + 2 * u * c[..., 2, :]) / h
        if order == 2:
            return 2 * c[..., 2, :] / h**2
        raise ValueError("path derivative order must be 0, 1 or 2")


def bound_trajectory(trajectory, max_vel, max_acc):
    """Bound the executed cubic by uniform time scaling, preserving geometry."""
    bounds = trajectory.bounds()
    if not all(np.isfinite(v).all() for v in bounds.values()):
        raise ValueError("nonfinite trajectory extrema")
    scale = max(1., float(np.max(bounds["qd_abs_max"] / max_vel)),
                float(np.sqrt(np.max(bounds["qdd_abs_max"] / max_acc))))
    if scale > 1.:
        scale *= 1. + 1e-10
        trajectory = JointTrajectory(
            trajectory.times * scale, trajectory.positions, trajectory.velocities / scale)
    final = trajectory.bounds()
    if (np.any(final["qd_abs_max"] > max_vel + 1e-8)
            or np.any(final["qdd_abs_max"] > max_acc + 1e-8)):
        raise ValueError("executed cubic exceeds operating limits after timing")
    return trajectory


def _time_path(waypoints, vmax, amax, ds, tolerance):
    # Planning-only dependency: controller and RT processes do not import it.
    try:
        import toppra as ta
        from toppra.algorithm import TOPPRA
        from toppra.constraint import JointAccelerationConstraint, JointVelocityConstraint
    except ImportError as exc:
        raise ValueError("TOPPRA retiming requires arm_control[planning]") from exc

    path = _BlendedPath(waypoints, tolerance)
    # Include every geometric boundary and several interior points per blend.
    grid = np.unique(np.concatenate([
        np.linspace(a, b, max(4, int(np.ceil((b - a) / ds))) + 1)
        for a, b in zip(path.knots[:-1], path.knots[1:])]))
    algorithm = TOPPRA(
        [JointVelocityConstraint(vmax), JointAccelerationConstraint(amax)],
        path, gridpoints=grid, solver_wrapper="seidel",
        parametrizer="ParametrizeConstAccel")
    _, speed, _ = algorithm.compute_parameterization(0., 0.)
    if speed is None or not np.isfinite(speed).all() or np.any(speed < 0):
        raise ValueError("TOPPRA could not find a finite rest-to-rest timing")
    pairs = speed[:-1] + speed[1:]
    if np.any(pairs <= 0):
        raise ValueError("TOPPRA returned an interval with no forward progress")
    times = np.r_[0., np.cumsum(2 * np.diff(grid) / pairs)]
    reference = ta.ParametrizeConstAccel(path, grid, speed)
    q, v = reference(times), reference(times, 1)
    h = np.diff(times)
    midpoint = .5 * (times[:-1] + times[1:])
    hermite_midpoint = .5 * (q[:-1] + q[1:]) + h[:, None] / 8 * (v[:-1] - v[1:])
    error = np.linalg.norm(reference(midpoint) - hermite_midpoint, axis=1)
    # Each interval is quadratic geometry composed with quadratic s(t): a
    # quartic. Its Hermite error is c4*(t-a)^2*(t-b)^2, whose norm peaks at
    # the midpoint and shrinks by 16 on bisection. This is an error bound,
    # not a heuristic sample of an arbitrary function.
    divisions = np.maximum(1, np.ceil((error / INTERPOLATION_TOLERANCE_RAD)**.25).astype(int))
    samples = np.concatenate([
        np.linspace(a, b, int(n) + 1)[:-1]
        for a, b, n in zip(times[:-1], times[1:], divisions)] + [times[-1:]])
    positions, velocities = reference(samples), reference(samples, 1)
    positions[[0, -1]] = waypoints[[0, -1]]
    velocities[[0, -1]] = 0.
    return bound_trajectory(JointTrajectory(samples, positions, velocities), vmax, amax)


def time_parameterize_blended(waypoints, max_vel, max_acc, *, ds=0.02,
                              blend_tolerance_rad=BLEND_TOLERANCE_RAD):
    """Retiming with local blends, per-joint v/a caps and zero endpoint speed.

    No heuristic acceleration taper or implicit jerk limit. Smaller blend
    tolerance preserves tighter geometry at the cost of slower corner motion.
    Exact reversals require a stop; other corners are traversed continuously.
    """
    wps = np.asarray(waypoints, dtype=float)
    vmax, amax = (np.asarray(x, dtype=float).ravel() for x in (max_vel, max_acc))
    if wps.ndim != 2 or len(wps) < 2 or wps.shape[1] == 0 or not np.isfinite(wps).all():
        raise ValueError("waypoints must be finite (N>=2, n_dof)")
    for value in (vmax, amax):
        if value.shape != (wps.shape[1],) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("max_vel/max_acc must be finite positive joint vectors")
    if not np.isfinite(ds) or ds <= 0 or not np.isfinite(blend_tolerance_rad) or blend_tolerance_rad <= 0:
        raise ValueError("ds and blend_tolerance_rad must be finite and positive")
    wps = wps[np.r_[True, np.linalg.norm(np.diff(wps, axis=0), axis=1) > 1e-12]]
    if len(wps) < 2:
        raise ValueError("waypoints describe zero motion")
    # A true cusp cannot have nonzero speed. Solve each side rest-to-rest.
    direction = np.diff(wps, axis=0)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    cusps = np.flatnonzero(np.linalg.norm(direction[:-1] + direction[1:], axis=1) < 1e-8) + 1
    boundaries = np.r_[0, cusps, len(wps) - 1]
    parts = [_time_path(wps[a:b + 1], vmax, amax, ds, blend_tolerance_rad)
             for a, b in zip(boundaries[:-1], boundaries[1:])]
    times, positions, velocities, offset = [], [], [], 0.
    for i, part in enumerate(parts):
        start = int(i > 0)
        times.append(part.times[start:] + offset)
        positions.append(part.positions[start:])
        velocities.append(part.velocities[start:])
        offset += part.duration_sec
    return bound_trajectory(JointTrajectory(np.concatenate(times), np.vstack(positions),
                                            np.vstack(velocities)), vmax, amax)


def collision_checked_retiming(waypoints, max_vel, max_acc, in_collision):
    """Retry smaller blends and finer timing; fail closed on sampled contact."""
    for ds in RETIMING_STEPS_RAD:
        trajectory = time_parameterize_blended(
            waypoints, max_vel, max_acc, ds=ds,
            blend_tolerance_rad=BLEND_TOLERANCE_RAD * ds / RETIMING_STEPS_RAD[0])
        if not any(in_collision(q) for q in trajectory.densify(COLLISION_STEP_RAD)):
            return trajectory
    raise ValueError("blended trajectory collides; no checked blend in the retry ladder")


def _self_check():
    for angle in (0, 15, 90, 180):
        w = np.zeros((3, 2))
        w[1:, 0] = .5
        w[2] += .5 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        tr = time_parameterize_blended(w, np.full(2, .4), np.full(2, .8))
        assert np.max(tr.bounds()["qd_abs_max"]) <= .4 + 1e-8
        assert np.max(tr.bounds()["qdd_abs_max"]) <= .8 + 1e-8
    print("TOPPRA retiming self-check OK")


if __name__ == "__main__":
    _self_check()
