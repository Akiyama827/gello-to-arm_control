"""Neutral trusted-composition seam, not a planner or an admission certificate.

RobotRuntime captures snapshot() on its loop: it must be bounded and nonblocking.
check(curve, snapshot) runs on the single bounded non-RT admission worker and
returns an empty string on success or a refusal reason. Project composition owns
scene/model/load truth; producer messages cannot populate these facts.
"""
from dataclasses import dataclass
import math
from typing import Protocol

from arm_control.motion import JointTrajectory


@dataclass(frozen=True)
class StreamPolicy:
    stream_policy_id: str
    min_commit_lead_s: float
    max_candidate_age_s: float
    max_future_s: float
    start_velocity_tolerance_rad_s: tuple[float, ...]

    def __post_init__(self):
        if not self.stream_policy_id:
            raise ValueError('stream policy requires an identity')
        for name in ('min_commit_lead_s', 'max_candidate_age_s', 'max_future_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.min_commit_lead_s + self.max_candidate_age_s >= self.max_future_s:
            raise ValueError('commit lead + candidate age must fit future budget')
        values = tuple(self.start_velocity_tolerance_rad_s)
        if not values or not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('start velocity tolerance must be finite/nonnegative')
        object.__setattr__(self, 'start_velocity_tolerance_rad_s', values)


@dataclass(frozen=True)
class StreamFacts:
    model_token: str
    scene_token: str
    policy: StreamPolicy
    load_revision: str
    transient_mass_kg: float | None

    @property
    def empty_load(self):
        return (bool(self.load_revision) and self.transient_mass_kg is not None
                and math.isfinite(self.transient_mass_kg) and self.transient_mass_kg == 0.)


class StreamAdmission(Protocol):
    def snapshot(self) -> StreamFacts: ...

    def check(self, curve: JointTrajectory, facts: StreamFacts) -> str: ...
