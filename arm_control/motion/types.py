"""Engine-neutral joint state, command, and sampled trajectory values."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    position: np.ndarray
    velocity: np.ndarray


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
        t = float(time_sec)
        if t <= self.times[0]:
            return self._point(0)
        if t >= self.times[-1]:
            return self._point(len(self.times) - 1)

        upper = int(np.searchsorted(self.times, t, side="right"))
        lower = upper - 1
        t0 = self.times[lower]
        t1 = self.times[upper]
        alpha = (t - t0) / (t1 - t0)
        position = (1.0 - alpha) * self.positions[lower] + alpha * self.positions[upper]
        if self.velocities is not None:
            velocity = (1.0 - alpha) * self.velocities[lower] + alpha * self.velocities[upper]
        else:
            velocity = (self.positions[upper] - self.positions[lower]) / (t1 - t0)
        return TrajectoryPoint(time=t, position=position, velocity=velocity)

    def _point(self, index: int) -> TrajectoryPoint:
        velocity = self.velocities[index] if self.velocities is not None else np.zeros(self.num_joints, dtype=float)
        return TrajectoryPoint(
            time=float(self.times[index]),
            position=self.positions[index].copy(),
            velocity=np.asarray(velocity, dtype=float).copy(),
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
