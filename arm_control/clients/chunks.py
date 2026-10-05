"""An action-chunk policy client (variable inference delay): bounded, replace-the-tail, never retroactive.

A policy answers a past observation after an unknown delay. The rules here:
* a chunk names its joints, units (rad), composition (absolute, or delta on the observation's ``base_q``) and the
  supervisor-clock ``observed_at`` of the observation it used; none of that is guessed or silently reordered;
* a response older than ``max_age_s``, older than the observation of what the RT already selected (or an in-flight chunk, or one whose fate is unknown), or from before the last
  ``cancel()``/``start()`` is DROPPED (reported as late); a closed session is never restarted by a late response;
* at most one revision is in flight and one chunk waits; a newer chunk replaces the waiting one (reported as replaced);
* every chunk is spliced from the supervisor's anchor, so the committed prefix is never rewritten;
* the supervisor's admission has the final word on limits: a refusal is reported with its reason, nothing is clipped.
"""
from dataclasses import dataclass

import numpy as np

from arm_control.clients.common import (
    ActionSpec, NeedsOperator, Report, StreamContext, StreamSession, note_outcome, request_id)
from arm_control.sdk import Refused, SupervisorError


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    joints: tuple
    values: object                  # (T, n) rad: absolute positions, or deltas on base_q
    dt_s: float
    observed_at: object             # arm_control.execution.contract.Stamp (supervisor clock)
    generation: int                 # ActionChunkClient.generation when the observation was taken
    mode: str = 'absolute'
    base_q: tuple | None = None
    units: str = 'rad'


class ActionChunkClient:
    def __init__(self, arm, ctx: StreamContext, *, max_age_s: float, hold_s: float, spec: ActionSpec | None = None):
        if not max_age_s > 0:
            raise ValueError('max_age_s must be positive (it is the caller\'s deadline, there is no default)')
        if spec is not None and spec.joints != tuple(ctx.joints):
            raise ValueError('the action spec names different joints (or another order) than the stream context')
        self.arm, self.ctx, self.max_age_s, self.hold_s, self._spec = arm, ctx, max_age_s, hold_s, spec
        self.generation, self.session, self.report = 0, None, Report()
        self.report.spec = None if spec is None else spec.record()
        self.inflight, self.waiting, self._sent_obs_ns = None, None, -1
        self.cancel_pending = False

    @property
    def spec(self) -> ActionSpec | None:
        """The declared spec (read-only: ``Report.spec`` records it once, so it cannot be swapped under a recorded run)."""
        return self._spec

    def start(self) -> int:
        """Open a stream; chunks produced before this call (an older generation) are late forever."""
        # The profile is checked BEFORE anything is cancelled or opened: a refusal leaves a running stream running. It is a
        # separate call from the stream open, so the owning session could switch preset between the two (only the owner can);
        # a switch after the open ends the stream (the supervisor cancels live executions on a preset change).
        self._check_profile()
        if not self.cancel('restart'):
            raise NeedsOperator('previous stream not confirmed cancelled; not opening a second one')
        self.generation += 1
        self.session = StreamSession(self.arm, self.ctx, hold_s=self.hold_s)
        self.inflight = self.waiting = None
        self._sent_obs_ns = -1
        return self.generation

    def _check_profile(self) -> None:
        """The control profile the spec expects must be the one selected: a run never switches it, and never assumes it."""
        expected = None if self._spec is None else self._spec.control_profile
        profile = None
        if self.arm.extensions.get('controller') == 1:
            try:
                profile = self.arm.controller_state().preset_id
            except (SupervisorError, TimeoutError, OSError) as exc:
                if expected is not None:  # asserted but unverifiable: refuse; not asserted: record best effort, never abort
                    raise NeedsOperator(f'the run expects control profile {expected!r} but reading the profile failed: '
                                        f'{exc!r}') from exc
                self.report.notes.append(f'control profile not recorded: {exc!r}')
        self.report.profile_at_start = profile
        if expected is None:
            return
        if self.arm.extensions.get('controller') != 1:
            raise NeedsOperator(f'the run expects control profile {expected!r} but the supervisor cannot report its profile')
        if profile != expected:
            raise NeedsOperator(f'control profile {self.report.profile_at_start!r} is selected, the run expects {expected!r}: '
                                'select it explicitly; a client never switches it')

    def cancel(self, reason: str = 'cancelled') -> bool:
        """True when the stream cancel was confirmed (or there was none). Otherwise the session is kept so ``cancel`` can
        be retried, a note says so, and ``start()`` refuses to open a second stream."""
        self.generation += 1  # a response to anything requested before now can no longer land
        self.inflight, self.waiting = None, None
        if self.session is not None and not self.session.close(reason):
            self.cancel_pending = True  # no chunk may move the arm on a stream the operator tried to stop
            self.report.notes.append(f'CANCEL NOT CONFIRMED: {self.session.last_error}; retry cancel()')
            return False
        self.session, self.cancel_pending = None, False
        return True

    def _knots(self, chunk: Chunk):
        values = np.asarray(chunk.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.ctx.joints):
            raise ValueError('chunk must be (T, joints)')
        if self._spec is not None:
            values = self._spec.to_radians(values, chunk.mode)  # the declared units and normalization; composition must match
        if chunk.mode == 'delta':
            if chunk.base_q is None:
                raise ValueError('a delta chunk composes on base_q, the joint position at observation time')
            base = np.asarray(chunk.base_q, dtype=float)
            if base.shape != (len(self.ctx.joints),):
                raise ValueError('base_q must have one entry per joint (it is never broadcast)')
            values = base + values
            if not np.isfinite(values).all():
                raise ValueError('the composed chunk is not finite')
            return values
        if chunk.mode == 'absolute':
            if not np.isfinite(values).all():
                raise ValueError('the chunk is not finite')
            return values
        raise ValueError(f'unknown composition {chunk.mode!r}')

    def _check(self, chunk: Chunk):
        """None if the chunk may be sent, else ('late'|'refused', reason)."""
        if self.cancel_pending:
            return 'late', 'a cancel is outstanding (not confirmed): the stream is being stopped'
        if self.session is None or self.session.closed:
            return 'late', 'no open session (a late response does not restart one)'
        if chunk.generation != self.generation:
            return 'late', 'requested before the last start/cancel'
        if tuple(chunk.joints) != tuple(self.ctx.joints):
            return 'refused', 'joint names or order differ from the stream context'
        if not np.isfinite(chunk.dt_s) or chunk.dt_s <= 0:
            return 'refused', 'dt_s must be positive and finite'
        expected_units = 'rad' if self._spec is None else self._spec.units
        if chunk.units != expected_units:
            return 'refused', f'units {chunk.units!r}: this run declared {expected_units!r}'
        if self._spec is not None and self._spec.horizon_s is not None:
            horizon = len(chunk.values) * chunk.dt_s
            if horizon > self._spec.horizon_s + 1e-9:
                return 'refused', f'chunk horizon {horizon:.3f} s exceeds the declared {self._spec.horizon_s} s'
        try:
            self._knots(chunk)  # shape, composition and base_q are checked when submitted, not when finally sent
        except ValueError as exc:
            return 'refused', str(exc)
        try:
            now = self.arm.now()
            if (now.clock, now.domain) != (chunk.observed_at.clock, chunk.observed_at.domain):
                return 'refused', 'observation stamp is not on the supervisor clock'
            age = (now.ns - chunk.observed_at.ns) / 1e9
        except (AttributeError, SupervisorError, TimeoutError, OSError) as exc:
            return 'refused', f'cannot age the observation: {exc!r}'
        if age > self.max_age_s:
            return 'late', f'observation {age:.3f} s old, deadline {self.max_age_s} s'
        newest = max(self._sent_obs_ns, self.inflight[2] if self.inflight else -1)
        if chunk.observed_at.ns < newest:
            return 'late', 'older than the observation of what was already sent'
        return None

    def _send(self, chunk: Chunk) -> None:
        rid = request_id(chunk.chunk_id)
        try:
            knots = self._knots(chunk)
            status = self.session.revise(rid, knots, chunk.dt_s)
        except (Refused, ValueError) as exc:  # the supervisor answered, or nothing was sent: a true refusal
            self.report.refused.append((chunk.chunk_id, str(exc)))
            return
        except (SupervisorError, TimeoutError, OSError) as exc:
            # no usable reply: the revision may have been admitted and selected (a lost reply). Keep it in flight so
            # the ordering rule holds and pump() finds out; the fate is 'unknown' until it does.
            self.report.requested.append(chunk.chunk_id)
            self.report.mark_unknown(chunk.chunk_id)
            self.report.notes.append(f'{chunk.chunk_id}: no reply to the revision ({exc!r}); fate unknown')
            self.inflight = (chunk.chunk_id, rid, chunk.observed_at.ns)
            return
        self.report.requested.append(chunk.chunk_id)
        if note_outcome(self.report, chunk.chunk_id, status) == 'refused':
            return
        self.inflight = (chunk.chunk_id, rid, chunk.observed_at.ns)

    def submit(self, chunk: Chunk) -> str:
        """'sent', 'waiting', 'late' or 'refused'. Never blocks on the robot."""
        verdict = self._check(chunk)
        if verdict is not None:
            (self.report.late if verdict[0] == 'late' else self.report.refused).append((chunk.chunk_id, verdict[1]))
            return verdict[0]
        if self.inflight is not None:
            if self.waiting is not None:
                self.report.replaced.append(self.waiting.chunk_id)
            self.waiting = chunk
            return 'waiting'
        self._send(chunk)
        return 'sent' if self.inflight is not None else 'refused'

    def pump(self) -> Report:
        """Resolve the in-flight revision, then send the waiting chunk if it is still valid."""
        if self.session is not None and not self.session.closed:
            try:
                view = self.session.view()
            except (SupervisorError, TimeoutError, OSError) as exc:
                self.report.notes.append(f'stream unreadable: {exc!r}')
                return self.report
            self.report.selected_revision = view.selected_revision
            if view.execution.terminal:
                self.report.notes.append(f'stream ended: {view.execution.phase.value}')
                self.session.closed, self.inflight, self.waiting = True, None, None  # not restarted from here
                return self.report
            if self.inflight is not None:
                chunk_id, rid, observed_ns = self.inflight
                try:
                    kind = note_outcome(self.report, chunk_id, self.arm.stream_revision(self.session.execution_id, rid))
                except TimeoutError:
                    return self.report  # try again on the next pump
                except (SupervisorError, OSError) as exc:  # e.g. history pruned: the fate is unknown, do not stay stuck
                    self.report.notes.append(f'{chunk_id}: revision status unavailable ({exc!r}); fate unknown')
                    self.report.mark_unknown(chunk_id)
                    kind = 'selected'  # conservative: it may have been selected, so nothing older may follow it

                if kind not in ('refused', 'selected', 'ended'):
                    return self.report
                if kind == 'selected':
                    self._sent_obs_ns = max(self._sent_obs_ns, observed_ns)  # only what the RT selected is "sent"
                self.inflight = None
            if self.inflight is None and self.waiting is not None:
                chunk, self.waiting = self.waiting, None
                verdict = self._check(chunk)
                if verdict is None:
                    self._send(chunk)
                else:
                    (self.report.late if verdict[0] == 'late' else self.report.refused).append((chunk.chunk_id, verdict[1]))
        return self.report
