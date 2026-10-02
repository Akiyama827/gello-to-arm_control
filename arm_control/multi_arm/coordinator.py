"""Two-controller staging barrier with fresh-state admission and observed cancels.

The coordinator never produces motor commands. Release timestamps describe send
ordering, not measured physical start times. Every member retains its controller.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from arm_control.contracts.motion import unpack_plan
from arm_control.motion import JointTrajectory
from arm_control.multi_arm.synchronization import release_skew_sec, stage_skew_sec
from arm_control.multi_arm.types import (
    ArmControlState,
    BimanualPlan,
    BimanualState,
    PairLifecycle,
    PairMemberState,
    PairState,
)


@dataclass
class _PairRun:
    plan: BimanualPlan
    staged_at: float
    sessions: dict[str, str]
    models: dict[str, str | None]
    lifecycle: PairLifecycle = PairLifecycle.STAGING
    stage_acks: dict[str, tuple[bool, str]] = field(default_factory=dict)
    ack_times: dict[str, float] = field(default_factory=dict)
    results: dict[str, tuple[bool, str]] = field(default_factory=dict)
    releases: dict[str, float] = field(default_factory=dict)
    pending_cancels: set[str] = field(default_factory=set)
    cancel_errors: dict[str, str] = field(default_factory=dict)
    cancellation_requested: bool = False
    ready_at: float | None = None
    completed_at: float | None = None
    faulted_arm: str | None = None
    reason: str = ""

    def state(self) -> PairState:
        return PairState(
            pair_id=self.plan.pair_id,
            lifecycle=self.lifecycle,
            members=tuple(
                PairMemberState(
                    arm_id=a,
                    plan_id=p.plan_id,
                    stage_acked=self.stage_acks[a][0] if a in self.stage_acks else None,
                    stage_reason=self.stage_acks.get(a, (None, ""))[1],
                    result_ok=self.results[a][0] if a in self.results else None,
                    result_reason=self.results.get(a, (None, ""))[1],
                    released_at=self.releases.get(a),
                    cancel_confirmed=(
                        a not in self.pending_cancels
                        if self.cancellation_requested
                        else None
                    ),
                    cancel_error=self.cancel_errors.get(a, ""),
                )
                for a, p in self.plan.members().items()
            ),
            staged_at=self.staged_at,
            ready_at=self.ready_at,
            stage_skew_sec=stage_skew_sec(self.ack_times),
            release_skew_sec=release_skew_sec(self.releases),
            completed_at=self.completed_at,
            faulted_arm=self.faulted_arm,
            reason=self.reason,
            cancellation_pending=tuple(sorted(self.pending_cancels)),
        )


class BimanualCoordinator:
    """Exactly two arm endpoints; one pair owns the group until cleared.

    ``admission`` revalidates consumer scene/attachment provenance at staging and
    release. All API calls and event delivery must use one owning event loop.
    State stamps must share the coordinator's monotonic clock domain.
    """

    def __init__(
        self,
        arms: Mapping,
        *,
        clock=None,
        staging_timeout_s=2.0,
        state_timeout_s=1.0,
        admission: Callable | None = None,
        release_context: Callable | None = None,
    ):
        self.arms = dict(arms)
        if len(self.arms) != 2:
            raise ValueError("a bimanual coordinator requires exactly two arms")
        for arm_id, handle in self.arms.items():
            if not isinstance(arm_id, str) or not arm_id or handle.arm_id != arm_id:
                raise ValueError("arm handle identity must match its registry key")
        for value in (staging_timeout_s, state_timeout_s):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("pair timeouts must be finite and positive")
        self._clock = clock or time.monotonic
        self.staging_timeout_s = float(staging_timeout_s)
        self.state_timeout_s = float(state_timeout_s)
        self.admission = admission
        # In-process controllers can share a reference epoch. Transport-backed
        # handles retain their existing barrier semantics without this hook.
        self._release_context = release_context or nullcontext
        self._run: _PairRun | None = None
        self._last: dict[str, PairState] = {}
        self._consumed_pair_ids: set[str] = set()
        self._consumed_plan_ids: set[str] = set()
        self._listeners = []

    def on_pair_result(self, listener):
        self._listeners.append(listener)

    def _states(self) -> dict[str, ArmControlState]:
        # States first, then `now`: a sample received while the arms were being read is not 'from the future'.
        # (Each arm's age is computed against a clock read taken after that arm's own state.)
        states = self.arm_states()
        now = self._clock()
        for a, s in states.items():
            if not isinstance(s.session_id, str) or not s.session_id:
                raise ValueError(f"{a}: controller session unknown")
            if (
                s.observed_at is None
                or not math.isfinite(s.observed_at)
                or s.observed_at > now
                or s.state_age_sec is None
                or not math.isfinite(s.state_age_sec)
                or s.state_age_sec > self.state_timeout_s
            ):
                raise ValueError(f"{a}: measured state missing or stale")
            if s.position is None or s.velocity is None:
                raise ValueError(f"{a}: measured position/velocity unknown")
            if s.armed is not True or s.ready is not True:
                raise ValueError(f"{a}: not armed and ready")
            if any(type(v) is not bool for v in (s.executing, s.stopped, s.frozen)):
                raise ValueError(f"{a}: controller execution state unknown")
            if s.stopped or s.frozen or s.fault_reason or s.mode_law != "joint":
                raise ValueError(
                    f"{a}: controller faulted, frozen or outside joint mode"
                )
        return states

    @staticmethod
    def _validate_member(member, state):
        body = unpack_plan(member.payload)
        if body.get("plan_id") != member.plan_id or body.get("gated") is not True:
            raise ValueError(f"{member.arm_id}: member identity/gated payload mismatch")
        if body.get("model_token") != state.model_token:
            raise ValueError(f"{member.arm_id}: plan model token changed")
        q, dq, times = body["positions"], body["velocities"], body["times"]
        if (
            len(times) < 2
            or times[0] != 0
            or not np.all(np.diff(times) > 0)
            or not all(np.isfinite(x).all() for x in (q, dq, times))
            or q.shape != dq.shape
            or q.shape != (len(times), len(state.position))
        ):
            raise ValueError(f"{member.arm_id}: malformed trajectory")
        JointTrajectory(times=times, positions=q, velocities=dq)

    def stage_pair(self, plan: BimanualPlan):
        if self._run and (self.lifecycle.active or self._run.pending_cancels):
            raise ValueError("previous pair is active or cancellation is unconfirmed")
        members = plan.members()
        if set(members) != set(self.arms):
            raise ValueError("pair members differ from coordinated arms")
        if plan.pair_id in self._consumed_pair_ids:
            raise ValueError("pair id already consumed")
        if any(p.plan_id in self._consumed_plan_ids for p in members.values()):
            raise ValueError("member plan id already consumed")
        states = self._states()
        for a, p in members.items():
            if states[a].executing or states[a].running_plan_id:
                raise ValueError(f"{a}: another plan owns the controller")
            self._validate_member(p, states[a])
        if self.admission:
            self.admission(plan, states)
        self._consumed_pair_ids.add(plan.pair_id)
        self._consumed_plan_ids.update(p.plan_id for p in members.values())
        self._run = _PairRun(
            plan,
            self._clock(),
            {a: s.session_id for a, s in states.items()},
            {a: s.model_token for a, s in states.items()},
        )
        for a, p in members.items():
            try:
                self.arms[a].stage(p)
            except Exception as exc:
                self._abort(PairLifecycle.REJECTED, f"{a}: staging failed: {exc}", a)
                break
            if not self.lifecycle.active:  # synchronous negative ACK
                break

    def execute_pair(self, pair_id):
        run = self._require(pair_id)
        if run.lifecycle is PairLifecycle.EXECUTING:
            return
        if run.lifecycle is not PairLifecycle.READY:
            raise ValueError(f"pair {pair_id}: {run.lifecycle.value}, not ready")
        try:
            self._check_run()
            states = self._states()
            for a, p in run.plan.members().items():
                if states[a].running_plan_id != p.plan_id or states[a].executing:
                    raise ValueError(f"{a}: staged plan is no longer loaded")
                self._validate_member(p, states[a])
            if self.admission:
                self.admission(run.plan, states)
        except Exception as exc:
            self._abort(PairLifecycle.REJECTED, str(exc))
            return
        run.lifecycle = PairLifecycle.EXECUTING
        a = None
        try:
            with self._release_context():
                for a, p in run.plan.members().items():
                    self.arms[a].execute(p.plan_id)
                    run.releases[a] = self._clock()
                    # In-process refusals must stop the release of the partner.
                    self._poll(a)
                    if run.lifecycle is not PairLifecycle.EXECUTING:
                        break
        except Exception as exc:
            self._abort(PairLifecycle.FAULTED, f"{a}: release failed: {exc}", a)

    def cancel_pair(self, reason="operator cancellation"):
        if self._run and self.lifecycle.active:
            self._abort(PairLifecycle.CANCELLED, reason)
        elif self._run and self._run.pending_cancels:
            self._send_cancels()

    def _send_cancels(self):
        run = self._run
        for a in tuple(run.pending_cancels):
            try:
                self.arms[a].cancel(run.reason, plan_id=run.plan.members()[a].plan_id)
                run.cancel_errors.pop(a, None)
            except Exception as exc:
                run.cancel_errors[a] = str(exc)

    def _abort(self, lifecycle, reason, arm=None):
        run = self._run
        if run is None or not run.lifecycle.active:
            return
        run.lifecycle, run.reason, run.faulted_arm = lifecycle, str(reason), arm
        run.completed_at = self._clock()
        run.cancellation_requested = True
        run.pending_cancels.update(self.arms)
        self._send_cancels()  # always attempt both, even if one send raises
        self._finish()

    def _finish(self):
        state = self._run.state()
        self._last[state.pair_id] = state
        for callback in tuple(self._listeners):
            callback(state)

    def on_member_event(self, arm_id, event):
        if arm_id not in self.arms:
            raise KeyError(arm_id)
        run = self._run
        if run is None:
            return
        runtime = event.get("runtime_state")
        kind = event.get("kind")
        if kind in {"plan_staged", "plan_cleared"} and not runtime:
            return  # an unbound acknowledgement cannot satisfy this session
        if runtime and runtime.get("session_id") != run.sessions[arm_id]:
            if run.lifecycle.active:
                self._abort(
                    PairLifecycle.FAULTED, f"{arm_id}: controller restarted", arm_id
                )
            return
        if kind == "fault":
            self._abort(
                PairLifecycle.FAULTED, event.get("reason", "member fault"), arm_id
            )
            return
        if event.get("plan_id") != run.plan.members()[arm_id].plan_id:
            return
        if kind == "plan_cleared" and run.cancellation_requested:
            if event.get("ok") is True:
                run.pending_cancels.discard(arm_id)
                run.cancel_errors.pop(arm_id, None)
            else:
                run.cancel_errors[arm_id] = event.get("reason", "clear refused")
        elif kind == "plan_staged" and run.lifecycle is PairLifecycle.STAGING:
            if arm_id in run.stage_acks:
                return
            ok = event.get("ok") is True
            run.stage_acks[arm_id] = (ok, str(event.get("reason", "")))
            run.ack_times[arm_id] = self._clock()
            if not ok:
                self._abort(
                    PairLifecycle.REJECTED, f"{arm_id}: {event.get('reason')}", arm_id
                )
            elif len(run.stage_acks) == 2:
                run.lifecycle = PairLifecycle.READY
                run.ready_at = self._clock()
        elif kind == "leg_result":
            if run.lifecycle.active and event.get("ok") is not True:
                run.results[arm_id] = (False, str(event.get("reason", "")))
                self._abort(
                    PairLifecycle.FAULTED, event.get("reason", "member failed"), arm_id
                )
            elif run.lifecycle is PairLifecycle.EXECUTING and arm_id not in run.results:
                run.results[arm_id] = (True, str(event.get("reason", "")))
                if len(run.results) == 2:
                    run.lifecycle = PairLifecycle.COMPLETE
                    run.completed_at = self._clock()
                    self._finish()
        self._last[run.plan.pair_id] = run.state()

    def _check_run(self):
        run = self._run
        states = self._states()
        for a, s in states.items():
            if s.session_id != run.sessions[a]:
                raise ValueError(f"{a}: controller restarted")
            if s.model_token != run.models[a]:
                raise ValueError(f"{a}: model changed")

    def check_timeouts(self):
        run = self._run
        if run is None or not run.lifecycle.active:
            return
        try:
            self._check_run()
        except ValueError as exc:
            self._abort(PairLifecycle.FAULTED, str(exc))
            return
        if (
            run.lifecycle is PairLifecycle.STAGING
            and self._clock() - run.staged_at > self.staging_timeout_s
        ):
            self._abort(PairLifecycle.REJECTED, "staging ACK timeout")

    def _poll(self, a):
        poll = getattr(self.arms[a], "poll_events", None)
        if poll:
            for event in poll():
                self.on_member_event(a, event)

    def pump(self):
        for a in self.arms:
            self._poll(a)
        self.check_timeouts()

    def _require(self, pair_id):
        if self._run is None or self._run.plan.pair_id != pair_id:
            raise ValueError(f"no active pair {pair_id!r}")
        return self._run

    @property
    def active_pair_id(self):
        return self._run.plan.pair_id if self._run else None

    @property
    def lifecycle(self):
        return self._run.lifecycle if self._run else PairLifecycle.IDLE

    def pair_state(self, pair_id=None):
        if self._run and (pair_id is None or pair_id == self.active_pair_id):
            return self._run.state()
        if pair_id in self._last:
            return self._last[pair_id]
        return PairState() if pair_id is None else self._last[pair_id]

    def arm_states(self, *, now=None):
        return {
            a: h.state().with_observation(now=self._clock() if now is None else now)
            for a, h in self.arms.items()
        }

    def state(self):
        values = list(self.arm_states().values())
        return BimanualState(left=values[0], right=values[1], pair=self.pair_state())
