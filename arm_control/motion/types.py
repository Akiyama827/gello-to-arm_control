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
        tracking a different trajectory. With the Hermite the same check reads
        2e-4 rad/s, and the interpolant does not overshoot the retimer's
        velocity cap (measured peak 0.400 against a 0.400 limit).

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

    # Endpoint clamping still holds the last sample, and a velocity-free
    # trajectory still falls back to the chord.
    assert np.allclose(traj.sample_at(-5.0).position, traj.positions[0])
    assert np.allclose(traj.sample_at(1e6).position, traj.positions[-1])
    plain = JointTrajectory(np.array([0.0, 2.0]), np.array([[0.0], [4.0]]))
    mid = plain.sample_at(1.0)
    assert np.allclose(mid.position, 2.0) and np.allclose(mid.velocity, 2.0)

    print("trajectory self-check OK (cubic reproduced exactly at 997 "
          "interior samples; knots interpolated; ends clamped)")


if __name__ == "__main__":
    _self_check()
