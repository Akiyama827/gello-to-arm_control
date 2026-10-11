"""Negotiated stream/1 values; execution v1 and RT wire remain unchanged.

A stream is one execution, with revision attempts that never own authority.
Only the supervisor issues splice challenges. This module is device-free.
"""
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from arm_control.execution.contract import ExecutionStatus, Phase, Stamp, Start, TrajectoryRequest
from arm_control.motion import JointTrajectory

STREAM_VERSION = 1


def _text(name, value):
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f'{name} must be a nonempty string of at most 256 characters')


def _sequence(name, value):
    if type(value) is not int or not 0 < value < 2**64:
        raise ValueError(f'{name} must be a positive uint64')


def _curve(value):
    if not isinstance(value, JointTrajectory) or value.velocities is None:
        raise ValueError('stream requires explicit Hermite velocities')
    if not 2 <= len(value.times) <= 128 or not 1 <= value.num_joints <= 16:
        raise ValueError('stream capacity is 2..128 knots x 1..16 joints')
    result = JointTrajectory(value.times, value.positions, value.velocities)
    for array in (result.times, result.positions, result.velocities):
        if not np.isfinite(array).all():
            raise ValueError('stream curve must be finite')
        array.setflags(write=False)
    return result


class SelectionState(str, Enum):
    NOT_REQUESTED = 'not_requested'
    PENDING = 'pending'
    SELECTED = 'selected'
    REFUSED = 'refused'


@dataclass(frozen=True)
class StreamOpen:
    request_id: str
    open_seq: int
    instance_id: str
    joint_names: tuple[str, ...]
    model_token: str
    scene_token: str
    stream_policy_id: str
    initial: JointTrajectory
    start: Start = Start.ON_ADMIT
    admit_by: Stamp | None = None

    def __post_init__(self):
        _sequence('open_seq', self.open_seq)
        for name in ('model_token', 'scene_token', 'stream_policy_id'):
            _text(name, getattr(self, name))
        object.__setattr__(self, 'initial', _curve(self.initial))
        object.__setattr__(self, 'joint_names', tuple(self.joint_names))
        self.as_request()  # share existing execution identity/deadline validation
        if self.initial.times[0] != 0.:
            raise ValueError('initial stream phase must start at zero')

    def as_request(self):
        return TrajectoryRequest(self.request_id, self.instance_id, self.joint_names,
                                 self.initial, model_token=self.model_token,
                                 start=self.start, admit_by=self.admit_by)


@dataclass(frozen=True)
class StreamRevision:
    request_id: str
    execution_id: str
    proposal_seq: int
    expected_revision: int
    anchor_id: str
    scene_token: str
    suffix: JointTrajectory
    terminal: bool = False
    admit_by: Stamp | None = None

    def __post_init__(self):
        for name in ('request_id', 'execution_id', 'anchor_id', 'scene_token'):
            _text(name, getattr(self, name))
        _sequence('proposal_seq', self.proposal_seq)
        _sequence('expected_revision', self.expected_revision)
        if type(self.terminal) is not bool:
            raise ValueError('terminal must be bool')
        object.__setattr__(self, 'suffix', _curve(self.suffix))
        if self.terminal and np.any(self.suffix.velocities[-1] != 0.):
            raise ValueError('terminal revision must have zero endpoint velocity')
        # Same deadline convention as planned requests, without adding a validator.
        TrajectoryRequest(self.request_id, 'stream',
                          tuple(f'j{i}' for i in range(self.suffix.num_joints)),
                          self.suffix, admit_by=self.admit_by)


@dataclass(frozen=True)
class SpliceAnchor:
    anchor_id: str
    expected_revision: int
    cut_phase: float
    q: tuple[float, ...]
    qd: tuple[float, ...]
    issued_at: Stamp
    expires_at: Stamp

    def __post_init__(self):
        _text('anchor_id', self.anchor_id)
        _sequence('expected_revision', self.expected_revision)
        for name in ('q', 'qd'):
            value = tuple(float(x) for x in getattr(self, name))
            if not 1 <= len(value) <= 16 or not all(math.isfinite(x) for x in value):
                raise ValueError(f'{name} must be a finite joint vector')
            object.__setattr__(self, name, value)
        if len(self.q) != len(self.qd) or not math.isfinite(self.cut_phase) or self.cut_phase < 0:
            raise ValueError('invalid splice knot')
        if (self.issued_at.clock, self.issued_at.domain) != (self.expires_at.clock, self.expires_at.domain):
            raise ValueError('anchor expiry must use its issuance clock')
        if self.expires_at.ns <= self.issued_at.ns:
            raise ValueError('anchor must have a positive lifetime')


@dataclass(frozen=True)
class RevisionStatus:
    execution_id: str
    request_id: str
    expected_revision: int | None
    revision: int | None
    phase: Phase
    selection_state: SelectionState
    at: Stamp
    reason: str = ''


@dataclass(frozen=True)
class StreamView:
    execution: ExecutionStatus
    model_token: str
    scene_token: str
    stream_policy_id: str
    selected_reference_id: int | None = None
    selected_revision: int | None = None
    selected_phase: float | None = None
    protected_until: float | None = None
    selected_end_phase: float | None = None
    selected_terminal: bool | None = None
    pending_request_id: str | None = None
    pending_revision: int | None = None
    splice: SpliceAnchor | None = None
