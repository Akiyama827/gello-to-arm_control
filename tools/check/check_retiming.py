"""Deterministic TOPPRA/corner/consumer checks, no hardware or pytest."""
from __future__ import annotations

from unittest.mock import patch
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from arm_control.motion import JointTrajectory
from arm_control.planning.retiming import (
    _BlendedPath, BLEND_TOLERANCE_RAD, COLLISION_STEP_RAD,
    collision_checked_retiming, time_parameterize_blended,
)


def check_curve(tr, vmax, amax):
    assert all(np.isfinite(x).all() for x in (tr.times, tr.positions, tr.velocities))
    assert np.all(np.diff(tr.times) > 0) and np.max(np.abs(tr.velocities[[0, -1]])) == 0
    bounds = tr.bounds()
    assert np.all(bounds['qd_abs_max'] <= vmax + 1e-8), bounds
    assert np.all(bounds['qdd_abs_max'] <= amax + 1e-8), bounds
    # Independent scipy evaluator at both sides of every knot and midpoints.
    spline = CubicHermiteSpline(tr.times, tr.positions, tr.velocities)
    times = np.r_[tr.times, np.nextafter(tr.times[1:], tr.times[:-1]),
                  .5 * (tr.times[:-1] + tr.times[1:])]
    assert np.all(np.abs(spline(times, 1)) <= vmax + 1e-8)
    assert np.all(np.abs(spline(times, 2)) <= amax + 1e-8)


def corner(angle):
    w = np.zeros((3, 7))
    w[1:, 0] = .5
    w[2, :2] += .5 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    return w


def main():
    vmax, amax = np.full(7, .4), np.full(7, .8)
    for delta in (1e-4, .0017, .02, .5, 2.):
        w = np.zeros((2, 7))
        w[-1, 0] = delta
        tr = time_parameterize_blended(w, vmax, amax)
        check_curve(tr, vmax, amax)
        assert delta / .4 <= tr.duration_sec < 4 * np.sqrt(delta / .8) + delta / .4 + 1
    for angle in (0, 15, 45, 90, 135, 179, 180):
        w = corner(angle)
        tr = time_parameterize_blended(w, vmax, amax)
        check_curve(tr, vmax, amax)
        assert np.array_equal(tr.positions[[0, -1]], w[[0, -1]])
        if angle == 15:
            assert tr.duration_sec < 4., 'corner merely globally slowed'
            assert tr.bounds()['q_min'][1] >= -1.1e-6, 'old backward excursion returned'
            print('15-degree corner:', tr.duration_sec, tr.bounds()['qd_abs_max'].max(),
                  tr.bounds()['qdd_abs_max'].max())
        if angle == 180:
            index = np.argmin(np.linalg.norm(tr.positions - w[1], axis=1))
            assert np.max(np.abs(tr.velocities[index])) == 0., 'cusp did not stop'
    w = corner(15)
    tr = time_parameterize_blended(np.insert(w, 1, w[0], axis=0), vmax, amax)
    check_curve(tr, vmax, amax)
    # Geometry has one derivative from both sides; stationary incoming joint
    # stays still until the actual blend, instead of borrowing an outgoing v.
    path = _BlendedPath(w, BLEND_TOLERANCE_RAD)
    for s in path.knots[1:-1]:
        assert np.allclose(path(float(s) - 1e-10, 1), path(float(s) + 1e-10, 1), atol=1e-6)
    assert np.max(np.abs(path(np.linspace(0., .49, 20))[:, 1])) == 0.
    assert path(.5, 1)[1] > 0., 'blend is not moving through the corner'
    for bad in (np.full((2, 7), np.nan), np.zeros((2, 7)), np.zeros((2, 0))):
        try:
            time_parameterize_blended(bad, vmax, amax)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid waypoints admitted')
    for key in ('ds', 'blend_tolerance_rad'):
        for value in (0, -1, np.nan):
            try:
                time_parameterize_blended(w, vmax, amax, **{key: value})
            except ValueError:
                pass
            else:
                raise AssertionError((key, value))
    rng = np.random.default_rng(20260910)
    for _ in range(40):
        w = np.cumsum(rng.uniform(-.4, .4, (4, 7)), axis=0)
        v, a = rng.uniform(.2, .6, 7), rng.uniform(.4, 1.2, 7)
        check_curve(time_parameterize_blended(w, v, a), v, a)
    # A forbidden blend region is avoided by a smaller blend, not by dropping
    # the collision check. Corner at (0.5,0) with vertical outgoing segment.
    w = corner(90)
    calls = []
    from arm_control.planning import retiming
    original = retiming.time_parameterize_blended

    def spy(*args, **kw):
        calls.append(kw['blend_tolerance_rad'])
        return original(*args, **kw)

    def blocked(q):
        return q[0] < .4998 and q[1] > .0002

    with patch.object(retiming, 'time_parameterize_blended', spy):
        tr = collision_checked_retiming(w, vmax, amax, blocked)
    assert len(calls) > 1 and calls[-1] < calls[0], calls
    assert not any(blocked(q) for q in tr.densify(COLLISION_STEP_RAD))
    try:
        collision_checked_retiming(w, vmax, amax, lambda q: True)
    except ValueError:
        pass
    else:
        raise AssertionError('all-colliding ladder returned a trajectory')
    # The standalone console must validate its FINAL curve, not only q0/q1.
    from arm_control.ui.arm_console import plan_trajectory
    from types import SimpleNamespace
    world = SimpleNamespace(lower=np.full(7, -3.), upper=np.full(7, 3.),
                            in_collision=lambda q: False)
    ompl = SimpleNamespace(plan=lambda start, end: corner(15))
    with patch('arm_control.planning.retiming.collision_checked_retiming',
               side_effect=ValueError('final curve collision')):
        try:
            plan_trajectory(world, ompl, corner(15)[0], corner(15)[-1], vmax, amax)
        except ValueError as exc:
            assert 'final curve collision' in str(exc)
        else:
            raise AssertionError('console bypassed final curve validation')
    # ... and plans a FRESH route when one's finished curve collides; no route is the operator's "no path", not a collision.
    from arm_control.planning.retiming import RouteCollides
    wall = SimpleNamespace(lower=np.full(2, -2.), upper=np.full(2, 2.), in_collision=lambda q: bool(.4 < q[0] < .6 and q[1] < .5))
    s, g = np.zeros(2), np.array([1., 0.])

    def routes(*seq):
        calls = []
        return calls, SimpleNamespace(plan=lambda a, b: (calls.append(1), seq[min(len(calls), len(seq)) - 1])[1])

    calls, ompl = routes(np.vstack([s, g]), np.vstack([s, [0, 1], [1, 1], g]))
    flown = plan_trajectory(wall, ompl, s, g, np.ones(2), np.full(2, 2.))
    assert len(calls) == 2 and not any(wall.in_collision(q) for q in flown.densify(COLLISION_STEP_RAD)), calls
    calls, ompl = routes(None)
    try:
        plan_trajectory(wall, ompl, s, g, np.ones(2), np.full(2, 2.))
    except RouteCollides:
        raise AssertionError('no route reported as a collision')
    except ValueError as exc:
        assert str(exc) == 'no collision-free path found (try a different target)', exc
    else:
        raise AssertionError('the console returned a plan without a route')
    # Exact extrema must still catch the old endpoint-position/velocity bug.
    old = JointTrajectory([0., .05], [[0.], [0.]], [[0.], [.1]])
    assert np.isclose(old.bounds()['qdd_abs_max'][0], 8.)
    # Recorded replay retains its geometry but must also fit the newly shared
    # acceleration gate, including its measured-pose join.
    from arm_control.control.replay_adapter import build_replay
    t = np.linspace(0, 3, 301)
    q = np.tile((.2 * np.sin(8 * t))[:, None], (1, 7))
    times, positions, velocities, _ = build_replay(
        t, q, np.zeros(7), {'vel_cap': .4}, max_acc=.8)
    check_curve(JointTrajectory(times, positions, velocities), vmax, amax)
    print('TOPPRA: small moves, corners, cusps, finite inputs, 40 random paths, collision retries, console seam and re-plan: PASS')


if __name__ == '__main__':
    main()
