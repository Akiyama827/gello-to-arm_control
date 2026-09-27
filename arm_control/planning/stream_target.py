"""Explicit target_stop_hermite_grid/1 composition strategy.

Produces a timed suffix; it owns neither execution nor renewal. The caller
supplies a receiver-issued cut (or measured start), stream envelope and trusted
curve admission. No packet-arrival interval is a trajectory duration.
"""
from dataclasses import dataclass
import math

import numpy as np

from arm_control.motion import JointTrajectory


@dataclass(frozen=True)
class TargetStopGrid:
    duration_grid_s: tuple[float, ...]
    progress_fractions: tuple[float, ...]
    strategy_id: str = 'target_stop_hermite_grid/1'

    def __post_init__(self):
        if self.strategy_id != 'target_stop_hermite_grid/1':
            raise ValueError('unknown target strategy')
        for name in ('duration_grid_s', 'progress_fractions'):
            values = tuple(getattr(self, name))
            if not 1 <= len(values) <= 16 or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError(f'{name} needs 1..16 positive finite entries')
            object.__setattr__(self, name, values)
        if any(a >= b for a, b in zip(self.duration_grid_s, self.duration_grid_s[1:])):
            raise ValueError('duration grid must strictly increase')
        if self.progress_fractions[0] != 1 or any(a <= b for a, b in zip(self.progress_fractions, self.progress_fractions[1:])):
            raise ValueError('progress fractions must strictly decrease from one')

    def toward(self, target, *, cut_phase, q, qd, observed_phase, stream_policy,
               execution_policy, kp, kd, check):
        """First admissible (curve, progress fraction, duration), or ValueError.

        check(curve) returns a refusal string; scene admission remains trusted
        composition work, never replaced by a UI's preflight result.
        """
        q, qd, target = (np.asarray(value, dtype=float) for value in (q, qd, target))
        if q.ndim != 1 or qd.shape != q.shape or target.shape != q.shape:
            raise ValueError('target and cut vectors must have the same joint shape')
        if not all(np.isfinite(value).all() for value in (q, qd, target)):
            raise ValueError('target and cut must be finite')
        if self.duration_grid_s[-1] > stream_policy.max_future_s:
            raise ValueError('strategy duration exceeds stream policy future budget')
        for alpha in self.progress_fractions:
            for duration in self.duration_grid_s:
                if cut_phase+duration-observed_phase > stream_policy.max_future_s:
                    continue
                curve = JointTrajectory([cut_phase, cut_phase+duration],
                                        [q, q+alpha*(target-q)], [qd, np.zeros_like(q)])
                plan = dict(times=curve.times, positions=curve.positions,
                            velocities=curve.velocities, kp=kp, kd=kd)
                if not execution_policy.plan_error(plan, q, len(q)) and not check(curve):
                    return curve, alpha, duration
        raise ValueError('no target suffix satisfies stream envelope and scene admission')
