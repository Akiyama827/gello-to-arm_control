"""Closed-form ramp checks against an independent SciPy polynomial oracle."""
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from arm_control.motion.profiles import trapezoidal_trajectory


def main():
    cases = 0
    # Simulation and hardware tilt caps; triangular, cruise, reverse, near-zero.
    for vmax, amax in ((.6, 1.5), (np.pi / 60, np.pi / 30)):
        for distance in (1e-8, .001, .2618, .8, vmax**2 / amax):
            for sign in (-1, 1):
                start = np.array([.1, -.2, .3])
                delta = sign * distance * np.array([1., -.5, 0.])
                limits_v = np.full(3, vmax)
                limits_a = np.full(3, amax)
                tr = trapezoidal_trajectory(start, start + delta, limits_v, limits_a)
                assert len(tr.times) in (3, 4)
                assert np.array_equal(tr.positions[[0, -1]], [start, start + delta])
                assert np.max(np.abs(tr.velocities[[0, -1]])) == 0
                expected = (2 * np.sqrt(distance / amax) if distance <= vmax**2 / amax
                            else distance / vmax + vmax / amax)
                assert np.isclose(tr.duration_sec, expected, rtol=1e-8)
                spline = CubicHermiteSpline(tr.times, tr.positions, tr.velocities)
                ts = np.r_[tr.times, np.nextafter(tr.times[1:], tr.times[:-1]),
                           .5 * (tr.times[:-1] + tr.times[1:])]
                assert np.all(np.abs(spline(ts, 1)) <= limits_v + 1e-7)
                assert np.all(np.abs(spline(ts, 2)) <= limits_a + 1e-7)
                assert np.allclose(spline(ts)[:, 2], start[2], atol=1e-14)
                bounds = tr.bounds()
                assert np.all(bounds['q_min'] >= np.minimum(start, start + delta) - 1e-12)
                assert np.all(bounds['q_max'] <= np.maximum(start, start + delta) + 1e-12)
                assert np.all(bounds['qd_abs_max'] <= limits_v + 1e-7)
                assert np.all(bounds['qdd_abs_max'] <= limits_a + 1e-7)
                cases += 1
    # Different joints govern speed and acceleration; preserve a straight path.
    tr = trapezoidal_trajectory(np.zeros(2), np.array([1., 2.]),
                               np.array([.1, 1.]), np.array([1., .2]))
    assert np.isclose(tr.duration_sec, 11.)
    assert np.allclose(tr.positions[:, 1], 2 * tr.positions[:, 0])
    for args in (([0], [0], [1], [1]), ([0], [1], [0], [1]),
                 ([0], [1], [1], [-1]), ([0], [np.nan], [1], [1]),
                 ([0], [1, 2], [1], [1]), ([], [], [], [])):
        try:
            trapezoidal_trajectory(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid profile accepted: {args}')
    print(f'motion profiles: PASS ({cases} tilt cases, coordinated limits, invalid inputs)')


if __name__ == '__main__':
    main()
