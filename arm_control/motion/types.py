"""Engine-neutral joint state, command, and sampled trajectory values."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass
class JointTrajectory:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float).copy()
        self.positions = np.asarray(self.positions, dtype=float).copy()
        if self.times.ndim != 1:
            raise ValueError("times must be a 1-D array")
        if self.positions.ndim != 2:
            raise ValueError("positions must be a 2-D array")
        if len(self.times) != len(self.positions):
            raise ValueError("times length must match positions rows")
        if len(self.times) < 2:
            raise ValueError("trajectory must contain at least two points")
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if self.velocities is not None:
            self.velocities = np.asarray(self.velocities, dtype=float).copy()
            if self.velocities.shape != self.positions.shape:
                raise ValueError("velocities shape must match positions shape")

    @property
    def duration_sec(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def num_joints(self) -> int:
        return int(self.positions.shape[1])

    def sample_at(self, time_sec: float) -> TrajectoryPoint:
        """One reference state off ONE polynomial: q, dq/dt, d2q/dt2.

        CUBIC HERMITE when the waypoints carry velocities, because the three
        signals have to be the same curve. The previous version interpolated
        position and velocity LINEARLY and INDEPENDENTLY, so qd was not the
        derivative of q -- measured on a real retimed leg (0.4 rad/s cap):
        max |dq/dt - qd_provided| = 0.127 rad/s, 32% of the velocity cap. The
        position loop, the velocity loop and the RNEA feedforward were each
        tracking a different trajectory. The Hermite makes these derivatives
        consistent; the retimer and execution policy separately bound its
        extrema. Legal waypoint velocities alone do not bound a cubic.

        This is the ONLY reference evaluator in the stack -- sim and hardware
        both come through here, and the RT server holds the last command
        rather than interpolating -- so there is nothing to keep in sync.
        """
        t = float(time_sec)
        if t <= self.times[0]:
            return self._point(0)
        if t >= self.times[-1]:
            return self._point(len(self.times) - 1)

        upper = int(np.searchsorted(self.times, t, side="right"))
        lower = upper - 1
        h = self.times[upper] - self.times[lower]
        s = (t - self.times[lower]) / h
        q0, q1 = self.positions[lower], self.positions[upper]
        if self.velocities is None:
            # No velocity waypoints: the chord IS the trajectory. Linear, and
            # already self-consistent -- the chord slope is its derivative.
            return TrajectoryPoint(
                time=t,
                position=(1.0 - s) * q0 + s * q1,
                velocity=(q1 - q0) / h,
                acceleration=np.zeros(self.num_joints, dtype=float),
            )
        v0, v1 = self.velocities[lower], self.velocities[upper]
        s2, s3 = s * s, s * s * s
        return TrajectoryPoint(
            time=t,
            position=((2 * s3 - 3 * s2 + 1) * q0 + (s3 - 2 * s2 + s) * h * v0
                      + (-2 * s3 + 3 * s2) * q1 + (s3 - s2) * h * v1),
            velocity=((6 * s2 - 6 * s) * q0 / h + (3 * s2 - 4 * s + 1) * v0
                      + (-6 * s2 + 6 * s) * q1 / h + (3 * s2 - 2 * s) * v1),
            acceleration=((12 * s - 6) * q0 / (h * h) + (6 * s - 4) * v0 / h
                          + (-12 * s + 6) * q1 / (h * h) + (6 * s - 2) * v1 / h),
        )

    def bounds(self) -> dict[str, np.ndarray]:
        """EXACT per-joint extrema of the flown curve, not of its waypoints.

        Between two knots this is a cubic, so every bound is closed-form and
        there is nothing to sample: position extrema are the endpoints plus
        the real roots of the quadratic dq/dt, velocity extrema the endpoints
        plus the single root of the linear d2q/dt2, and acceleration is
        monotonic so its extrema ARE the endpoints.

        Waypoint-and-chord checking cannot see any of this: a cubic can hold
        an interior velocity extremum with both endpoint velocities legal.
        Returns {'q_min','q_max','qd_abs_max','qdd_abs_max'}, each length
        num_joints.
        """
        n = self.num_joints
        q_min = np.full(n, np.inf)
        q_max = np.full(n, -np.inf)
        qd_abs = np.zeros(n)
        qdd_abs = np.zeros(n)
        for i in range(len(self.times) - 1):
            h = self.times[i + 1] - self.times[i]
            # Same basis as sample_at, in normalised s so the algebra is the
            # textbook one; the caller never sees s.
            q0, q1 = self.positions[i], self.positions[i + 1]
            if self.velocities is None:
                v0 = v1 = (q1 - q0) / h
            else:
                v0, v1 = self.velocities[i], self.velocities[i + 1]
            # q(s) = c3 s^3 + c2 s^2 + c1 s + c0
            c0 = q0
            c1 = h * v0
            c2 = -3 * q0 - 2 * h * v0 + 3 * q1 - h * v1
            c3 = 2 * q0 + h * v0 - 2 * q1 + h * v1

            def q_at(s):
                return ((c3 * s + c2) * s + c1) * s + c0

            cand_q = [q_at(0.0), q_at(1.0)]
            # dq/ds = 3c3 s^2 + 2c2 s + c1 -- per joint, so solve columnwise.
            a, b, c = 3 * c3, 2 * c2, c1
            with np.errstate(invalid="ignore", divide="ignore"):
                disc = b * b - 4 * a * c
                root = np.sqrt(np.where(disc > 0, disc, 0.0))
                for sign in (1.0, -1.0):
                    quad = np.where(np.abs(a) > 1e-300, (-b + sign * root) / (2 * a), np.nan)
                    lin = np.where(np.abs(b) > 1e-300, -c / b, np.nan)  # degenerate: linear
                    s_ = np.where(np.abs(a) > 1e-300, quad, lin)
                    s_ = np.where((disc > 0) | (np.abs(a) <= 1e-300), s_, np.nan)
                    s_ = np.where((s_ > 0.0) & (s_ < 1.0), s_, np.nan)
                    cand_q.append(np.where(np.isnan(s_), q_at(0.0), q_at(np.nan_to_num(s_))))
            stack = np.vstack(cand_q)
            q_min = np.minimum(q_min, stack.min(axis=0))
            q_max = np.maximum(q_max, stack.max(axis=0))

            # dq/dt is quadratic in s; its extremum sits where d2q/ds2 = 0.
            def qd_at(s):
                return ((3 * c3 * s + 2 * c2) * s + c1) / h

            cand_v = [np.abs(qd_at(0.0)), np.abs(qd_at(1.0))]
            with np.errstate(invalid="ignore", divide="ignore"):
                s_v = np.where(np.abs(c3) > 1e-300, -c2 / (3 * c3), np.nan)
            s_v = np.where((s_v > 0.0) & (s_v < 1.0), s_v, np.nan)
            cand_v.append(np.where(np.isnan(s_v), 0.0, np.abs(qd_at(np.nan_to_num(s_v)))))
            qd_abs = np.maximum(qd_abs, np.vstack(cand_v).max(axis=0))

            # d2q/dt2 is LINEAR in s, so the endpoints are the extrema.
            qdd_abs = np.maximum(
                qdd_abs,
                np.maximum(np.abs(2 * c2) / (h * h), np.abs(6 * c3 + 2 * c2) / (h * h)),
            )
        return {"q_min": q_min, "q_max": q_max,
                "qd_abs_max": qd_abs, "qdd_abs_max": qdd_abs}

    def densify(self, max_step_rad: float) -> np.ndarray:
        """Positions along the FLOWN curve, no joint moving more than a step.

        For collision checking the thing that will actually be executed. The
        planner's own waypoints are the chord; the executor flies the cubic
        between them, and the two differ (measured on retimed FR3 legs: up to
        4.5 mrad, ~2 mm at the flange -- enough to matter against a
        connector's lead-in).

        The step bound is honest rather than nominal: dt is chosen from the
        curve's EXACT peak speed (see bounds()), so |dq| <= |qd|max * dt holds
        for every joint everywhere, including inside a segment that bulges.
        """
        if not max_step_rad > 0:
            raise ValueError("max_step_rad must be positive")
        peak = float(np.max(self.bounds()["qd_abs_max"]))
        duration = self.duration_sec
        if peak <= 0.0 or duration <= 0.0:
            return np.vstack([self.positions[0], self.positions[-1]])
        steps = int(np.ceil(duration * peak / max_step_rad))
        times = np.linspace(self.times[0], self.times[-1], max(steps, 1) + 1)
        return np.vstack([self.sample_at(t).position for t in times])

    def _point(self, index: int) -> TrajectoryPoint:
        velocity = self.velocities[index] if self.velocities is not None else np.zeros(self.num_joints, dtype=float)
        return TrajectoryPoint(
            time=float(self.times[index]),
            position=self.positions[index].copy(),
            velocity=np.asarray(velocity, dtype=float).copy(),
            # Off the ends the trajectory is over (or has not started); the
            # executor holds, and a held reference has no acceleration.
            acceleration=np.zeros(self.num_joints, dtype=float),
        )


@dataclass(frozen=True)
class JointState:
    position: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class JointServoCommand:
    q_des: np.ndarray
    qd_des: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


def _self_check() -> None:
    """The reference must be ONE curve: qd == dq/dt and qdd == d2q/dt2.

    This is the regression that mattered — independent linear interpolation of
    q and qd passed every shape check while handing the position loop, the
    velocity loop and the RNEA feedforward three different trajectories.

    Tested by EXACT reproduction rather than finite differences: sample the
    knots off a known cubic and the Hermite interpolant must BE that cubic
    everywhere, so q/qd/qdd are checked against closed forms at machine
    precision with no differencing tolerance to argue about. The independent
    linear version fails this on the first interior sample -- it returns the
    chord between two knots, not the curve through them.
    """
    rng = np.random.default_rng(0)
    a, b, c, d = (rng.normal(0, 0.5, 7) for _ in range(4))
    cubic = lambda t: a + b * t + c * t**2 + d * t**3          # noqa: E731
    d_cubic = lambda t: b + 2 * c * t + 3 * d * t**2           # noqa: E731
    dd_cubic = lambda t: 2 * c + 6 * d * t                     # noqa: E731

    times = np.cumsum(np.concatenate([[0.0], rng.uniform(0.02, 0.5, 12)]))
    traj = JointTrajectory(times,
                           np.array([cubic(t) for t in times]),
                           np.array([d_cubic(t) for t in times]))
    # Open interval: the two ENDS are deliberately clamped (see _point) and a
    # synthetic cubic does not end at rest the way a retimed leg does.
    for t in np.linspace(0.0, traj.duration_sec, 999)[1:-1]:
        pt = traj.sample_at(t)
        assert np.allclose(pt.position, cubic(t), atol=1e-12), f"q at {t}"
        assert np.allclose(pt.velocity, d_cubic(t), atol=1e-10), f"qd at {t}"
        assert np.allclose(pt.acceleration, dd_cubic(t), atol=1e-8), f"qdd at {t}"

    # Waypoints are INTERPOLATED, not approximated: the curve passes through
    # every planned q and qd, or the plan reviewed is not the plan that runs.
    for i, t in enumerate(times):
        pt = traj.sample_at(t)
        assert np.allclose(pt.position, traj.positions[i], atol=1e-12), f"knot {i} q"
        assert np.allclose(pt.velocity, traj.velocities[i], atol=1e-12), f"knot {i} qd"

    # bounds() must never UNDER-report: an admission check built on it has to
    # fail safe. Compared against a dense resampling of the same curve, with
    # both sides of every knot included -- acceleration STEPS at a knot, so a
    # uniform grid misses the one-sided limit and reads low by orders of
    # magnitude (measured: 163 rad/s^2 low). That blind spot is exactly the
    # one waypoint-and-chord checking has, which is why bounds() is analytic.
    b = traj.bounds()
    grid = np.unique(np.concatenate([
        times, np.nextafter(times[1:], times[:-1]), np.nextafter(times[:-1], times[1:]),
        np.linspace(times[0], times[-1], 20001)]))
    qs = np.array([traj.sample_at(t).position for t in grid])
    vs = np.array([traj.sample_at(t).velocity for t in grid])
    as_ = np.array([traj.sample_at(t).acceleration for t in grid])
    assert np.all(b["q_max"] >= qs.max(axis=0) - 1e-9), "q_max under-reports"
    assert np.all(b["q_min"] <= qs.min(axis=0) + 1e-9), "q_min under-reports"
    assert np.all(b["qd_abs_max"] >= np.abs(vs).max(axis=0) - 1e-9), "qd under-reports"
    assert np.all(b["qdd_abs_max"] >= np.abs(as_).max(axis=0) - 1e-9), "qdd under-reports"
    # ...and must be TIGHT, or every admission check inherits false refusals.
    assert np.all(b["qd_abs_max"] <= np.abs(vs).max(axis=0) + 1e-3), "qd too loose"

    # An interior extremum the endpoints cannot see: both ends at rest, so a
    # waypoint check reads 0 rad/s while the curve actually moves.
    bulge = JointTrajectory(np.array([0.0, 1.0]), np.array([[0.0], [0.0]]),
                            np.array([[0.6], [-0.6]]))
    assert abs(bulge.bounds()["qd_abs_max"][0] - 0.6) < 1e-12, bulge.bounds()
    assert bulge.bounds()["q_max"][0] > 0.086, "interior position bulge missed"

    # densify(): no joint may step further than asked, anywhere.
    step = 0.01
    dense = traj.densify(step)
    assert np.abs(np.diff(dense, axis=0)).max() <= step + 1e-9, "densify step exceeded"
    assert np.allclose(dense[0], traj.sample_at(times[0]).position)
    assert np.allclose(dense[-1], traj.sample_at(times[-1]).position)

    # Endpoint clamping still holds the last sample, and a velocity-free
    # trajectory still falls back to the chord.
    assert np.allclose(traj.sample_at(-5.0).position, traj.positions[0])
    assert np.allclose(traj.sample_at(1e6).position, traj.positions[-1])
    plain = JointTrajectory(np.array([0.0, 2.0]), np.array([[0.0], [4.0]]))
    mid = plain.sample_at(1.0)
    assert np.allclose(mid.position, 2.0) and np.allclose(mid.velocity, 2.0)

    print("trajectory self-check OK (cubic reproduced exactly; knots "
          "interpolated; ends clamped; bounds exact and safe; densify bounded)")


if __name__ == "__main__":
    _self_check()
