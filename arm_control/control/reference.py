"""The released reference: one planned trajectory, its time, its sampling (CTRL-030).

``TrajectoryReference`` is the reference provider for planned joint legs and
the single owner of what used to be spread over the controller and the
executor: the trajectory being executed, its start on the controller's clock,
and evaluation of both. Evaluation is ``JointTrajectory.sample_at`` (the one
cubic Hermite in the stack); nothing here updates state, so samples may be
taken in any order and repeated with identical results. The executor receives
a sampled point and never sees trajectory time.

A negotiated native reference delegates phase and sampling to its selected
feedback source instead. It holds no PC start time in that mode; the planned
trajectory remains available for endpoint and Cartesian knot association.
Neither the SDK nor planner gains an interpolation implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from arm_control.motion import JointTrajectory, TrajectoryPoint


@dataclass(frozen=True)
class TrajectoryReference:
    trajectory: JointTrajectory
    t_start: float | None
    #: Precomputed world EE poses, one per knot (FK of the knots). The
    #: controller owns no kinematics, so they ride with the plan.
    cartesian_poses: np.ndarray | None = None
    # Native execution supplies the already selected RT sample. It owns phase;
    # a PC clock and PC interpolation cannot also own this released reference.
    sample_source: Callable[[], TrajectoryPoint] | None = None

    terminal: bool = True

    def __post_init__(self) -> None:
        if (self.t_start is None) == (self.sample_source is None):
            raise ValueError("reference needs exactly one local clock or selected sample source")

    @property
    def duration_sec(self) -> float:
        return self.trajectory.duration_sec

    def elapsed(self, t_now: float) -> float:
        """Seconds into the trajectory on the controller's clock."""
        if self.sample_source is not None:
            return float(self.sample_source().time)
        return float(t_now) - self.t_start

    def sample(self, t_now: float) -> TrajectoryPoint:
        """q, qd, qdd off one polynomial; clamped to the ends outside it."""
        if self.sample_source is not None:
            return self.sample_source()
        return self.trajectory.sample_at(self.elapsed(t_now))

    @property
    def end_phase(self) -> float:
        return float(self.trajectory.times[-1])

    @property
    def end(self) -> TrajectoryPoint:
        return self.trajectory.sample_at(self.end_phase)

    def cartesian_pose(self, t_now: float):
        """The precomputed pose of the knot segment being commanded, or None."""
        poses = self.cartesian_poses
        if poses is None:
            return None
        times = self.trajectory.times
        tau = np.clip(self.elapsed(t_now), times[0], times[-1])
        idx = int(np.searchsorted(times, tau, side="right")) - 1
        return poses[max(0, min(idx, len(poses) - 1))]


def _self_check() -> None:
    traj = JointTrajectory([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]),
                           np.array([[0.0], [1.5], [0.0]]))
    poses = np.arange(3 * 7, dtype=float).reshape(3, 7)
    ref = TrajectoryReference(traj, 10.0, poses)
    times = np.linspace(9.5, 11.5, 41)
    forward = [ref.sample(t) for t in times]
    shuffled = np.random.default_rng(0).permutation(len(times))
    for i in shuffled:  # any order, repeated: identical values, no hidden state
        again = ref.sample(times[i])
        for f in ("position", "velocity", "acceleration"):
            assert np.array_equal(getattr(again, f), getattr(forward[i], f))
    assert np.array_equal(ref.sample(9.0).position, [0.0])        # start retained
    assert np.array_equal(ref.sample(99.0).position, [1.0])       # endpoint retained
    assert np.array_equal(ref.sample(99.0).velocity, [0.0])
    assert np.array_equal(ref.end.position, [1.0])
    assert ref.cartesian_pose(10.0)[0] == 0 and ref.cartesian_pose(10.6)[0] == 7
    assert ref.cartesian_pose(50.0)[0] == 14 and ref.cartesian_pose(0.0)[0] == 0
    assert TrajectoryReference(traj, 0.0).cartesian_pose(0.3) is None
    point = traj.sample_at(.6)
    native = TrajectoryReference(traj, None, poses, lambda: point)
    for clock in (-1e6, 0., 1e6):
        assert native.sample(clock) is point and native.elapsed(clock) == .6
        assert native.cartesian_pose(clock)[0] == 7
    for start, source in ((None, None), (0., lambda: point)):
        try:
            TrajectoryReference(traj, start, sample_source=source)
        except ValueError:
            pass
        else:
            raise AssertionError('ambiguous reference clock accepted')
    print("control.reference: pure sampling, endpoint retention, knot poses OK")


if __name__ == "__main__":
    _self_check()
