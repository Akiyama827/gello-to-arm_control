"""Arm-identity-preserving aggregate state, plan, and lifecycle values.

The canonical multi-arm state is a GROUP of per-arm states, never a merged
joint vector. ``np.zeros(14)`` cannot represent 7+6 DOF, one faulted arm, one
stationary arm, different grippers or different update rates; a mapping keyed
by arm id can. These values aggregate the existing per-arm primitives
(``JointState``, ``JointTrajectory``, ``JointServoCommand``) around them --
they do not replace them.

Every field that can be unknown is ``None`` rather than a fabricated value:
a missing velocity is NOT zero, and a missing measurement is NOT the spawn
pose. Preserving the measured/unknown distinction is a safety property, not
a style choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

#: The two public bimanual slots. The generic plant and coordinator work by
#: arm/actor id; these names exist so the public layer and example graphs can
#: be written concretely without hardcoding them deeper down.

class PairLifecycle(str, Enum):
    """Explicit pair state machine. Never inferred from logs.

    IDLE      no pair is active
    STAGING   both member plans sent; waiting for both staging ACKs
    READY     both arms ACKed staging; pair may be released
    EXECUTING both member plans released; waiting for both member results
    COMPLETE  every member finished successfully (terminal)
    CANCELLED a cancel propagated through both controllers (terminal)
    FAULTED   a member faulted or failed; the partner was cancelled (terminal)
    REJECTED  the pair was refused before executing: a member rejected
              staging, or an execute precondition was unmet (terminal)

    A terminal state accepts a new ``stage_pair``; every non-terminal state
    refuses one. ``READY`` and only ``READY`` accepts ``execute_pair``.
    """

    IDLE = "idle"
    STAGING = "staging"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAULTED = "faulted"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def active(self) -> bool:
        """A pair is in flight: staging decisions, cancels and faults apply."""
        return self in (
            PairLifecycle.STAGING,
            PairLifecycle.READY,
            PairLifecycle.EXECUTING,
        )


_TERMINAL = frozenset(
    {
        PairLifecycle.COMPLETE,
        PairLifecycle.CANCELLED,
        PairLifecycle.FAULTED,
        PairLifecycle.REJECTED,
    }
)


@dataclass(frozen=True)
class ArmControlState:
    """One arm's observable control state. NEVER merged across arms.

    ``position``/``velocity`` are the latest MEASURED joint values the
    coordinator has been shown, or ``None`` when nothing fresh is known --
    ``velocity=None`` means unknown, not zero and not settled. Timestamps use
    the coordinator's own clock (a monotonic reading, not a wall date).
    """

    arm_id: str
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    armed: bool | None = None
    ready: bool | None = None
    executing: bool | None = None
    running_plan_id: str | None = None
    mode_law: str | None = None
    stopped: bool | None = None
    frozen: bool | None = None
    fault_reason: str | None = None
    session_id: str | None = None
    model_token: str | None = None
    observed_at: float | None = None
    state_age_sec: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not self.arm_id:
            raise ValueError("ArmControlState.arm_id must be a non-empty string")
        for name in ("position", "velocity"):
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.array(value, dtype=float, copy=True).ravel()
            if not np.isfinite(arr).all():
                raise ValueError(f"ArmControlState.{name} must be finite or None")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if (
            self.position is not None
            and self.velocity is not None
            and self.position.shape != self.velocity.shape
        ):
            raise ValueError(
                "ArmControlState position and velocity must agree when both are known"
            )

    def with_observation(self, *, now: float) -> ArmControlState:
        """Refresh the freshness stamp against a coordinator clock reading."""
        if self.observed_at is None:
            return replace(self, state_age_sec=None)
        return replace(self, state_age_sec=max(0.0, float(now) - self.observed_at))

    def to_payload(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "position": None if self.position is None else self.position.tolist(),
            "velocity": None if self.velocity is None else self.velocity.tolist(),
            "armed": self.armed,
            "ready": self.ready,
            "executing": self.executing,
            "running_plan_id": self.running_plan_id,
            "mode_law": self.mode_law,
            "stopped": self.stopped,
            "frozen": self.frozen,
            "fault_reason": self.fault_reason,
            "session_id": self.session_id,
            "model_token": self.model_token,
            "observed_at": self.observed_at,
            "state_age_sec": self.state_age_sec,
        }


@dataclass(frozen=True)
class ArmPlan:
    """One arm's member plan inside a pair.

    ``plan_id`` is the id the member controller will ACK and report results
    under; it must be unique within a pair and may be staged exactly ONCE --
    the coordinator keeps consumed ids and refuses reuse, so a late reply
    from a superseded attempt can never match a live pair's member id.
    ``payload`` is opaque to coordination -- the wire payload or decoded plan
    body the underlying endpoint accepts -- and is handed through verbatim.
    """

    arm_id: str
    plan_id: str
    payload: Any
    gated: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not self.arm_id:
            raise ValueError("ArmPlan.arm_id must be a non-empty string")
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise ValueError("ArmPlan.plan_id must be a non-empty string")
        if not self.gated:
            # A pair is staged, then released by one coordinated execute. An
            # ungated member would start moving the moment it loaded, before
            # its partner was even staged -- silently undoing the barrier.
            raise ValueError(
                f"ArmPlan {self.plan_id!r} must be gated: a coordinated pair is "
                "loaded first and released together by execute_pair"
            )


@dataclass(frozen=True)
class BimanualPlan:
    """A coordinated plan for the two public bimanual arms."""

    pair_id: str
    left: ArmPlan
    right: ArmPlan

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("BimanualPlan.pair_id must be a non-empty string")
        if self.left.arm_id == self.right.arm_id:
            raise ValueError(
                "BimanualPlan needs two distinct member arm ids, "
                f"got {self.left.arm_id!r} and {self.right.arm_id!r}"
            )
        if len({self.left.plan_id, self.right.plan_id}) != 2:
            raise ValueError(
                "pair member plan ids must be distinct: "
                f"{self.left.plan_id!r} staged on both arms"
            )

    def members(self) -> dict[str, ArmPlan]:
        """The pair as an arm-id-keyed mapping (the coordinator's view)."""
        return {self.left.arm_id: self.left, self.right.arm_id: self.right}


@dataclass(frozen=True)
class PairMemberState:
    """One member's progress through the pair lifecycle."""

    arm_id: str
    plan_id: str | None = None
    stage_acked: bool | None = None
    stage_reason: str = ""
    result_ok: bool | None = None
    result_reason: str = ""
    released_at: float | None = None
    cancel_confirmed: bool | None = None
    cancel_error: str = ""

    def to_payload(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "plan_id": self.plan_id,
            "stage_acked": self.stage_acked,
            "stage_reason": self.stage_reason,
            "result_ok": self.result_ok,
            "result_reason": self.result_reason,
            "released_at": self.released_at,
            "cancel_confirmed": self.cancel_confirmed,
            "cancel_error": self.cancel_error,
        }


@dataclass(frozen=True)
class PairState:
    """The pair's lifecycle record: one id, two member plans, two ACKs,
    two results -- in that order, with the timing the coordinator actually
    observed."""

    pair_id: str | None = None
    lifecycle: PairLifecycle = PairLifecycle.IDLE
    members: tuple[PairMemberState, ...] = ()
    staged_at: float | None = None
    ready_at: float | None = None
    stage_skew_sec: float | None = None
    release_skew_sec: float | None = None
    faulted_arm: str | None = None
    reason: str = ""
    completed_at: float | None = None
    cancellation_pending: tuple[str, ...] = ()

    def member(self, arm_id: str) -> PairMemberState | None:
        return next((m for m in self.members if m.arm_id == arm_id), None)

    def ok(self) -> bool | None:
        """Terminal outcome: True only when every member succeeded."""
        if self.lifecycle is PairLifecycle.COMPLETE:
            return all(m.result_ok for m in self.members)
        if self.lifecycle in (
            PairLifecycle.CANCELLED,
            PairLifecycle.FAULTED,
            PairLifecycle.REJECTED,
        ):
            return False
        return None

    def to_payload(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "lifecycle": self.lifecycle.value,
            "members": [m.to_payload() for m in self.members],
            "staged_at": self.staged_at,
            "ready_at": self.ready_at,
            "stage_skew_sec": self.stage_skew_sec,
            "release_skew_sec": self.release_skew_sec,
            "faulted_arm": self.faulted_arm,
            "reason": self.reason,
            "completed_at": self.completed_at,
            "cancellation_pending": list(self.cancellation_pending),
            "ok": self.ok(),
        }


@dataclass(frozen=True)
class BimanualState:
    """The public bimanual state: two per-arm states plus the pair record.

    ``ready`` is TRUE only when both arms report ready -- never a default.
    """

    left: ArmControlState
    right: ArmControlState
    pair: PairState = field(default_factory=PairState)

    def __post_init__(self) -> None:
        for member in (self.left, self.right):
            if not isinstance(member, ArmControlState):
                raise ValueError("BimanualState members must be ArmControlState values")
        if not isinstance(self.pair, PairState):
            raise ValueError("BimanualState.pair must be a PairState value")
        if self.left.arm_id == self.right.arm_id:
            raise ValueError(
                "BimanualState needs two distinct arm ids, "
                f"got {self.left.arm_id!r} and {self.right.arm_id!r}"
            )

    def arms(self) -> dict[str, ArmControlState]:
        """Both arm states as an arm-id-keyed mapping."""
        return {self.left.arm_id: self.left, self.right.arm_id: self.right}

    @property
    def ready(self) -> bool:
        """Both arms report ready. Unknown is not ready."""
        return bool(self.left.ready is True and self.right.ready is True)

    def to_payload(self) -> dict:
        return {
            "left": self.left.to_payload(),
            "right": self.right.to_payload(),
            "pair": self.pair.to_payload(),
            "ready": self.ready,
        }
