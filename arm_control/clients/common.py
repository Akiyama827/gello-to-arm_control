"""Shared pieces: stream context, report, suffix construction from the supervisor's splice anchor."""
from dataclasses import dataclass, field
import itertools

import numpy as np

from arm_control.execution.contract import Phase
from arm_control.execution.stream import SelectionState, StreamOpen, StreamRevision
from arm_control.motion import JointTrajectory
from arm_control.sdk import Refused, SupervisorError

_OPEN_SEQ = itertools.count(1)  # open and proposal numbers are marks per supervisor SESSION, across streams:
_PROPOSAL_SEQ = itertools.count(1)
_REQUEST_ID = itertools.count(1)  # request ids are keyed per supervisor session too: unique across client instances  # strictly increasing per process satisfies that for any number of streams
MAX_KNOTS = 127  # stream/1 capacity is 128 knots including the splice anchor


class NeedsOperator(RuntimeError):
    """A hardware state a client must never repair itself (re-arm, home, open the Hand, discard calibration)."""


@dataclass(frozen=True)
class StreamContext:
    """What the caller (project code) knows about the robot and scene; the client invents none of it."""
    instance_id: str
    joints: tuple
    model_token: str
    scene_token: str
    stream_policy_id: str


@dataclass(frozen=True)
class ActionSpec:
    """What a policy's numbers mean, stated once and recorded with every run (CTRL-052): joints (named, in order), the units
    the policy emits (``rad`` or ``normalized``), composition (``absolute`` positions or ``delta`` on the observation's
    ``base_q``), the longest horizon one chunk may cover, the control profile the run expects, and, for ``normalized``
    actions, the per-joint map to radians.

    Normalization is ``rad = offset + scale * value`` for an absolute action and ``rad = scale * value`` for a delta (an
    offset belongs to a position, never to a difference). Nothing is guessed: a normalized spec needs both vectors, finite,
    one entry per joint, ``scale > 0``. ``control_profile`` is the supervisor's preset id the run expects (``None``: not
    asserted); a run that finds another selected refuses to start rather than switch it."""
    joints: tuple
    composition: str = 'absolute'
    units: str = 'rad'
    horizon_s: float | None = None
    control_profile: str | None = None
    offset: tuple | None = None
    scale: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, 'joints', tuple(self.joints))
        if not self.joints:
            raise ValueError('an action spec names its joints')
        if self.composition not in ('absolute', 'delta'):
            raise ValueError(f'unknown composition {self.composition!r}')
        if self.units not in ('rad', 'normalized'):
            raise ValueError(f'unknown units {self.units!r} (rad or normalized)')
        if self.horizon_s is not None and not self.horizon_s > 0:
            raise ValueError('horizon_s must be positive')
        n = len(self.joints)
        if self.units == 'rad':
            if self.offset is not None or self.scale is not None:
                raise ValueError('a rad spec has no normalization map')
            return
        if self.offset is None or self.scale is None:
            raise ValueError('normalized actions need both offset and scale, one entry per joint: nothing is assumed')
        offset, scale = (np.asarray(v, dtype=float).ravel() for v in (self.offset, self.scale))
        if offset.shape != (n,) or scale.shape != (n,) or not (np.isfinite(offset).all() and np.isfinite(scale).all()):
            raise ValueError('offset and scale must be finite, one entry per joint')
        if not (scale > 0).all():
            raise ValueError('scale must be positive for every joint')
        object.__setattr__(self, 'offset', tuple(offset.tolist()))
        object.__setattr__(self, 'scale', tuple(scale.tolist()))

    def to_radians(self, values, composition: str):
        """(T, n) policy values -> radians for this spec; ``composition`` is the chunk's and must be the spec's."""
        if composition != self.composition:
            raise ValueError(f'chunk composition {composition!r} differs from the spec ({self.composition!r})')
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.joints):
            raise ValueError('values must be (T, joints): a shape is never broadcast')
        if self.units == 'rad':
            return values
        scaled = np.asarray(self.scale) * values
        return scaled if composition == 'delta' else np.asarray(self.offset) + scaled

    def record(self) -> dict:
        """The spec as a plain dict, for execution reports and datasets."""
        return dict(joints=list(self.joints), composition=self.composition, units=self.units, horizon_s=self.horizon_s,
                    control_profile=self.control_profile,
                    offset=None if self.offset is None else list(self.offset),
                    scale=None if self.scale is None else list(self.scale))


@dataclass
class Report:
    """Three separate facts per high-level action. ``requested``: sent to the supervisor (nothing is claimed).
    ``admitted``: the supervisor's trusted admission accepted it (asynchronous: the reply to the request only says
    REQUESTED). ``selected``: the RT selected that revision, i.e. it governs motion. ``selected_revision`` is the stream's
    current one. An action can be admitted and never selected, or refused after it was requested."""
    requested: list = field(default_factory=list)
    admitted: list = field(default_factory=list)
    selected: list = field(default_factory=list)
    unknown: list = field(default_factory=list)   # the fate WAS unknown at some point (no reply / unreadable status); the id
    # may later also appear in ``selected`` or ``refused`` once it is learned; listed once (``mark_unknown``)
    refused: list = field(default_factory=list)   # (id, reason)
    replaced: list = field(default_factory=list)  # accepted by the client, superseded before it was sent
    late: list = field(default_factory=list)      # dropped as too old or for a closed session
    selected_revision: int | None = None
    notes: list = field(default_factory=list)
    spec: dict | None = None                      # the ActionSpec the run declared (``ActionSpec.record()``), if any
    profile_at_start: str | None = None           # the supervisor's selected preset id when the stream opened

    def mark_unknown(self, item_id) -> None:
        if item_id not in self.unknown:
            self.unknown.append(item_id)


class StreamSession:
    """One stream execution plus an ordered revision counter (proposal_seq/expected_revision)."""

    def __init__(self, arm, ctx: StreamContext, *, hold_s: float):
        if hold_s <= 0:
            raise ValueError('hold_s must be positive')
        self.arm, self.ctx = arm, ctx
        q = np.asarray(arm.state().q, dtype=float)
        curve = JointTrajectory([0., hold_s / 2, hold_s], [q, q, q], np.zeros((3, len(q))))
        open_seq = next(_OPEN_SEQ)
        view = arm.open_stream(StreamOpen(f'open-{open_seq}', open_seq, ctx.instance_id, tuple(ctx.joints),
                                          ctx.model_token, ctx.scene_token, ctx.stream_policy_id, curve))
        self.execution_id = view.execution.execution_id
        self.closed, self.last_error = False, None

    def view(self):
        return self.arm.stream_state(self.execution_id)

    def revise(self, request_id: str, q_knots, dt: float, *, terminal: bool = False):
        """Splice ``q_knots`` (n x joints, absolute) from the supervisor's anchor, ``dt`` seconds apart."""
        q_knots = np.atleast_2d(np.asarray(q_knots, dtype=float))
        if not 1 <= len(q_knots) <= MAX_KNOTS or not np.isfinite(q_knots).all() or not dt > 0:
            raise ValueError(f'chunk needs 1..{MAX_KNOTS} finite knots and dt > 0')
        anchor = self.arm.stream_anchor(self.execution_id)
        q0 = np.asarray(anchor.q)
        knots = np.vstack([q0, q_knots])
        times = anchor.cut_phase + dt * np.arange(len(knots))
        vel = np.zeros_like(knots)
        vel[0] = np.asarray(anchor.qd)
        if len(knots) > 2:
            vel[1:-1] = (knots[2:] - knots[:-2]) / (2 * dt)
        revision = StreamRevision(request_id, self.execution_id, next(_PROPOSAL_SEQ), anchor.expected_revision,
                                  anchor.anchor_id, self.ctx.scene_token, JointTrajectory(times, knots, vel),
                                  terminal=terminal)
        return self.arm.revise_stream(revision)

    def close(self, reason: str = 'client closed') -> bool:
        """Cancel the stream. True only when the cancel was confirmed; otherwise the session stays open so the caller
        can retry, and the failure is in ``last_error``."""
        if self.closed:
            return True
        try:
            self.arm.cancel(self.execution_id, reason)
        except (SupervisorError, TimeoutError, OSError) as exc:
            self.last_error = repr(exc)
            return False
        self.closed = True
        return True


def request_id(prefix: str) -> str:
    return f'{prefix}.{next(_REQUEST_ID)}'


def classify(status) -> str:
    """'refused', 'selected', 'ended', 'admitted' or 'pending' for one revision status. 'ended': admitted, then
    expired/cancelled/superseded without the RT selecting it."""
    if status.phase is Phase.REJECTED or status.selection_state is SelectionState.REFUSED:
        return 'refused'
    if status.selection_state is SelectionState.SELECTED:
        return 'selected'
    if status.phase.terminal:
        return 'ended'
    if status.phase is not Phase.REQUESTED:
        return 'admitted'
    return 'pending'


def resolved(status) -> bool:
    """A revision attempt that no longer needs polling."""
    return classify(status) in ('refused', 'selected', 'ended')


def note_outcome(report, action_id, status) -> str:
    """Fold one polled revision status into ``report`` (each fact once); returns its classification."""
    kind = classify(status)
    if kind == 'ended':
        if all(a != action_id for a, _ in report.refused):
            report.refused.append((action_id, f'ended without selection: {status.phase.value} {status.reason}'.strip()))
        return kind
    if kind == 'refused':
        if all(a != action_id for a, _ in report.refused):
            report.refused.append((action_id, status.reason or 'revision refused'))
        return kind
    if kind in ('admitted', 'selected') and action_id not in report.admitted:
        report.admitted.append(action_id)
    if kind == 'selected' and action_id not in report.selected:
        report.selected.append(action_id)
    return kind


__all__ = ['MAX_KNOTS', 'NeedsOperator', 'StreamContext', 'Report', 'StreamSession', 'classify', 'note_outcome',
           'request_id', 'resolved', 'Refused']
