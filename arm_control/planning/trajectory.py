"""Joint trajectory representation and sampling helpers."""
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


def time_parameterize_blended(
    waypoints: np.ndarray,
    max_vel: np.ndarray,
    max_acc: np.ndarray,
    *,
    ds: float = 0.02,
    corner_slowdown_floor: float = 0.2,
    soft_speed_frac: float = 0.5,
    soft_acc_floor: float = 0.15,
) -> JointTrajectory:
    """Continuous-velocity retiming over the WHOLE waypoint path.

    The old per-segment trapezoidal retimer (deleted 2026-07-22) brought
    every joint to a FULL STOP at every waypoint — an OMPL path with a
    handful of waypoints then crawled at a fraction of ``max_vel``
    (measured: ~0.06 rad/s effective against a 1.0 rad/s limit). This
    parameterizer densifies the path in joint space and runs the classic
    numerical forward/backward velocity passes, so speed is continuous along
    the path and the arm stops only at the two ends.

    Corner handling: the per-sample velocity cap is scaled by the cosine of
    the local direction change (floored at ``corner_slowdown_floor``) — a
    cheap stand-in for full time-optimal blending. Direction (and therefore
    commanded joint velocity) still flips discretely at sharp corners; the
    1 kHz plant PD low-passes the resulting feed-forward step. Upgrade path
    if bench tracking demands it: real parabolic blends / TOTG.

    Soft launch/landing: below ``soft_speed_frac`` of the local velocity cap,
    the allowed acceleration tapers linearly with speed (floored at
    ``soft_acc_floor``) — the trapezoid's full-decel-to-standstill jerk
    impulse at the ends becomes an exponential-style landing, so the arm
    settles during the taper instead of arriving hot (bench 2026-07-23:
    end-of-motion ringing at kp=40/kd=2.5). ``soft_acc_floor=1`` disables.
    """
    wps = np.asarray(waypoints, dtype=float)
    if wps.ndim != 2 or wps.shape[0] < 2:
        raise ValueError("waypoints must be (N>=2, n_dof)")
    vmax = np.asarray(max_vel, dtype=float).ravel()
    amax = np.asarray(max_acc, dtype=float).ravel()
    if vmax.shape != (wps.shape[1],) or amax.shape != (wps.shape[1],):
        raise ValueError("max_vel/max_acc shape must match n_dof")
    if np.any(vmax <= 0) or np.any(amax <= 0):
        raise ValueError("max_vel/max_acc must be > 0")

    # Drop zero-motion waypoints, densify to <= ds joint-space arc length.
    deltas = np.diff(wps, axis=0)
    keep = np.linalg.norm(deltas, axis=1) > 1e-12
    if not np.any(keep):
        raise ValueError("waypoints describe zero motion")
    pts: list[np.ndarray] = [wps[0]]
    for i in np.flatnonzero(keep):
        seg = wps[i + 1] - wps[i]
        # >= 2, never 1: a single interval has no INTERIOR sample, and both
        # endpoint caps are pinned to zero below — the forward/backward passes
        # then leave v == 0 everywhere and the time integration falls back on
        # its 1e-9 divide-by-zero floor, so a 1.7 mrad move retimed to 40 DAYS
        # (measured; the executor's done() waits on plan time and hung there
        # forever). Any move shorter than `ds` hit this.
        n_sub = max(2, int(np.ceil(np.linalg.norm(seg) / float(ds))))
        for k in range(1, n_sub + 1):
            pts.append(wps[i] + seg * (k / n_sub))
    path = np.vstack(pts)
    n = path.shape[0]

    step = np.diff(path, axis=0)
    step_len = np.linalg.norm(step, axis=1)
    tangent = step / step_len[:, None]  # per-interval unit direction

    # Per-interval caps from the binding joint, corner-scaled.
    with np.errstate(divide="ignore"):
        v_int = np.min(np.where(np.abs(tangent) > 0, vmax / np.abs(tangent), np.inf), axis=1)
        a_int = np.min(np.where(np.abs(tangent) > 0, amax / np.abs(tangent), np.inf), axis=1)
    cos_turn = np.ones(n)  # per-SAMPLE corner factor
    if n > 2:
        dots = np.einsum("ij,ij->i", tangent[:-1], tangent[1:])
        cos_turn[1:-1] = np.clip(dots, corner_slowdown_floor, 1.0)
    v_cap = np.empty(n)
    v_cap[0] = v_cap[-1] = 0.0
    v_cap[1:-1] = np.minimum(v_int[:-1], v_int[1:]) * cos_turn[1:-1]

    # ponytail: linear low-speed acc taper, not true jerk limits — Ruckig/TOTG
    # if the bench outgrows it.
    def _a_soft(a: float, v_now: float, v_ref: float) -> float:
        scale = max(soft_acc_floor, min(1.0, v_now / max(v_ref, 1e-9)))
        return a * scale

    v = v_cap.copy()
    for i in range(n - 1):  # forward: acceleration limit, soft launch
        a = _a_soft(a_int[i], v[i], soft_speed_frac * v_int[i])
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * a * step_len[i]))
    for i in range(n - 2, -1, -1):  # backward: deceleration limit, soft landing
        a = _a_soft(a_int[i], v[i + 1], soft_speed_frac * v_int[i])
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a * step_len[i]))

    times = np.empty(n)
    times[0] = 0.0
    for i in range(n - 1):
        pair = max(v[i] + v[i + 1], 1e-9)
        times[i + 1] = times[i] + 2.0 * step_len[i] / pair
    velocities = np.zeros_like(path)
    velocities[:-1] = tangent * v[:-1, None]
    velocities[-1] = 0.0
    return JointTrajectory(times=times, positions=path, velocities=velocities)




def _self_check() -> None:
    """A move must be retimed to a sane duration at EVERY scale."""
    vmax = np.full(7, 1.0)
    amax = np.full(7, 2.0)
    for delta in (1e-4, 1.7e-3, 0.02, 0.5, 2.0):
        wps = np.zeros((2, 7))
        wps[1, 0] = delta
        traj = time_parameterize_blended(wps, vmax, amax)
        # Bracket: never faster than the velocity limit allows, never slower
        # than a full-stop-at-both-ends triangular profile at the acc floor.
        floor = delta / vmax[0]
        ceiling = 4.0 * np.sqrt(delta / (0.15 * amax[0])) + 1.0
        assert floor <= traj.duration_sec <= ceiling, (
            f"{delta} rad retimed to {traj.duration_sec}s "
            f"(expected {floor:.3f}..{ceiling:.3f})"
        )
        print(f"  {delta:8.4f} rad -> {traj.duration_sec:7.3f} s")
    print("trajectory self-check OK")


if __name__ == "__main__":
    _self_check()
