"""Bounded native-horizon transport; no interpolation, dynamics or task policy.

The caller retains PC admission and approves initial selection. Once selected,
this owner uploads unchanged windows of that same admitted trajectory, retaining
whole existing segments at every revision. One worker and one candidate; TCP
framing/backpressure remains RtBackend's responsibility.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import logging
import os
import threading
import time

import numpy as np

from arm_control.motion import JointTrajectory, TrajectoryPoint
from . import protocol as p
from .client import RtLinkError, ReferenceRefused, ReferenceCancelledBeforeDispatch


class _NotDispatched(RtLinkError):
    """Local proof that COMMIT never entered the transport."""


class SelectionUnresolved(RtLinkError):
    """A stream candidate's COMMIT outcome stayed unknown past SELECTION_UNRESOLVED_S."""


# CTRL-033 (operator decision): how long a candidate may stay 'pending' before the
# horizon reports it. Until then a pending candidate suppresses every command.
SELECTION_UNRESOLVED_S = 0.25

# CTRL-034 B1 (V1, operator decision): the declared PC envelope bounds. The RT's
# effective ceiling is min(--hold-ms, valid_for_us) and --hold-ms defaults to
# 100 ms, so this is 100 ms unless a unit lowers hold_ms.
VALID_FOR_S = 0.1
PHASE_WINDOW_S = 0.1
# The fixed send rate of the dedicated sender. The sender may repeat a decision,
# but it NEVER re-stamps one: the envelope always carries the decision's own
# sample time and phase, so its reported age grows while the decision is reused.
ENVELOPE_PERIOD_S = 0.005
SENDER_GAP_S = 0.020
_LOG = logging.getLogger(__name__)


class ReferenceHorizon:
    def __init__(self, link):
        self.link = link
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='reference-upload')
        self._job = None
        self._cancel_job = None
        self._interrupt = threading.Event()
        self._envelope_lock = threading.Lock()
        self._envelope_decision = None
        self._envelope_last = None
        self._envelope_stop = threading.Event()
        self._envelope_thread = threading.Thread(
            target=self._envelope_loop, name='reference-envelope', daemon=True)
        self._next_id = 0
        self.id = 0
        self.key = None
        self.trajectory = None
        self._window = (0, 0)
        self._revision = 0
        self._upload = None
        self._cut = 0.
        self._stage_ready = False
        self.selection_requested = False
        self._commit_undispatched = False
        self.dropped = {}  # key -> how the RT's drop was confirmed, or 'unconfirmed'
        self._cancel_key = None
        self.error = ''
        self.point = None
        self.feedback = None
        self.observed = None
        self.observed_point = None
        self.received_at = 0.
        self._session = 0
        self._epoch = 0
        self.stream = False
        self.terminal = True
        self.protected_until = 0.
        self.candidate_selection = 'not_requested'
        self._pending_since = None
        self.candidate_error = ''
        self.candidate_result = None
        self._candidate_curve = None
        self._candidate_terminal = True
        self._latest_phase = 0.
        self._selection_rx = None
        self._selection_phase = 0.
        self._first_tracking_at = None
        self.selection_latencies = deque(maxlen=128)
        self._diagnostic_lock = threading.Lock()
        self._timeline = deque(maxlen=256)
        self._first_events = deque(maxlen=128)
        self._latest_first_event = {}
        self._wake_lateness = deque(maxlen=4096)
        self._sender_gaps = deque(maxlen=256)
        self._sender_gaps_pending = deque(maxlen=256)
        self._schedstat_fd = None
        self._schedstat_status = 'unavailable'
        self._schedstat_error = None
        self._schedstat_warning_emitted = False
        self._schedstat_first_timeslices = None
        self._schedstat_last_timeslices = None
        self._schedstat_stagnant_loops = 0
        self._envelope_thread.start()

    def _schedstat_sample(self):
        """Sender-thread CPU, runqueue wait and timeslices from its proc fd."""
        if self._schedstat_fd is None:
            return None
        try:
            fields = os.pread(self._schedstat_fd, 128, 0).split()
            if len(fields) != 3:
                raise ValueError(f'expected 3 schedstat fields, got {len(fields)}')
            sample = tuple(int(field) for field in fields)
            if any(value < 0 for value in sample):
                raise ValueError('negative schedstat field')
            return sample
        except (OSError, ValueError, IndexError) as exc:
            self._schedstat_error = str(exc)
            self._schedstat_status = 'unavailable'
            os.close(self._schedstat_fd)
            self._schedstat_fd = None
            return None

    def _runqueue_wait_ns(self):
        sample = self._schedstat_sample()
        # A zero wait counter alone cannot establish that the kernel collects it.
        # Use it only after the timeslice counter has demonstrated liveness.
        return sample[1] if sample is not None and self._schedstat_status == 'ok' else None

    def _check_schedstat_liveness(self, sample):
        if sample is None:
            return
        timeslices = sample[2]
        if self._schedstat_first_timeslices is None:
            self._schedstat_first_timeslices = timeslices
            self._schedstat_stagnant_loops = 1
        elif timeslices > self._schedstat_last_timeslices:
            self._schedstat_status = 'ok'
            self._schedstat_stagnant_loops = 0
        else:
            self._schedstat_stagnant_loops += 1
            if self._schedstat_stagnant_loops >= 20:
                self._schedstat_status = 'not_collected'
                self._schedstat_error = 'timeslice count did not increase in 20 sender loops'
                os.close(self._schedstat_fd)
                self._schedstat_fd = None
        self._schedstat_last_timeslices = timeslices

    def _record(self, kind, reference_id=None, revision=None, *, at=None, **details):
        at = time.monotonic() if at is None else at
        event = dict(kind=kind, pc_monotonic_s=at, recorded_wall_time_ns=time.time_ns(),
                     reference_id=self.id if reference_id is None else reference_id,
                     revision=self._revision if revision is None else revision,
                     **details)
        with self._diagnostic_lock:
            self._timeline.append(event)

    def _record_first(self, kind, reference_id, revision, *, at=None, **details):
        key = (kind, reference_id, revision)
        with self._diagnostic_lock:
            if self._latest_first_event.get(kind) == key:
                return
            if key in self._first_events:
                return
            self._first_events.append(key)
            self._latest_first_event[kind] = key
        self._record(kind, reference_id, revision, at=at, **details)

    def diagnostics(self):
        """Bounded snapshot; percentile sorting is never done on the sender."""
        with self._diagnostic_lock:
            events = sorted(self._timeline, key=lambda event: event['pc_monotonic_s'])
            ages = sorted(self._wake_lateness)
            gaps = list(self._sender_gaps)
        percentile = ages[min(len(ages)-1, int(.999 * (len(ages)-1)))] if ages else 0.
        return dict(timeline=events, schedstat_status=self._schedstat_status,
                    schedstat_timeslices_start=self._schedstat_first_timeslices,
                    schedstat_timeslices_end=self._schedstat_last_timeslices,
                    sender_wake_lateness_ms=dict(
            samples=len(ages), p99_9=percentile*1000,
            max=(ages[-1]*1000 if ages else 0.)), sender_gaps=gaps)

    def dump_diagnostics(self, reason):
        _LOG.warning('reference diagnostics %s', json.dumps(
            dict(reason=reason, **self.diagnostics()), separators=(',', ':')))

    def _sender_interval(self, kind, duration, at, cpu_start, cpu_end,
                         runqueue_start, runqueue_end):
        if kind == 'wake_late':
            with self._diagnostic_lock:
                self._wake_lateness.append(max(0., duration))
        if duration > SENDER_GAP_S:
            cpu_delta = max(0., cpu_end-cpu_start)
            active = cpu_delta >= duration * .5
            runqueue_delta = (None if runqueue_start is None or runqueue_end is None
                              else max(0, runqueue_end-runqueue_start))
            # process_time covers all process threads, including native ones.
            # A measured lock wait plus active CPU is the specific Python-lock
            # signal; other active intervals do not identify a GIL owner.
            if kind == 'decision_lock_wait':
                cause = 'python_lock_contention' if active else 'lock_wait_cpu_idle'
            elif kind == 'wake_late':
                if runqueue_delta is not None:
                    cause = ('os_runqueue_delay' if runqueue_delta >= duration*5e8
                             else 'python_lock_or_same_process' if active
                             else 'process_cpu_idle')
                else:
                    cause = 'process_cpu_active' if active else 'os_scheduler_delay'
            elif kind in ('state_lock_wait', 'udp_send_duration'):
                cause = kind  # an operation blocked; do not call it OS starvation
            else:
                cause = 'process_cpu_active' if active else 'process_cpu_idle'
            event = dict(kind=kind, duration_ms=duration*1000,
                         pc_monotonic_s=at, recorded_wall_time_ns=time.time_ns(),
                         cpu_start_s=cpu_start, cpu_end_s=cpu_end,
                         cpu_delta_ms=cpu_delta*1000,
                         runqueue_wait_start_ns=runqueue_start,
                         runqueue_wait_end_ns=runqueue_end,
                         runqueue_wait_delta_ns=runqueue_delta, classification=cause)
            with self._diagnostic_lock:
                self._sender_gaps.append(event)
                self._sender_gaps_pending.append(event)

    def _drain_sender_gaps(self):
        """Called by the supervisor, never the sender thread."""
        with self._diagnostic_lock:
            pending = [self._sender_gaps_pending.popleft()
                       for _ in range(min(8, len(self._sender_gaps_pending)))]
        for event in pending:
            _LOG.warning('reference sender gap %s', json.dumps(event, separators=(',', ':')))

    @property
    def session_id(self):
        return self._session

    @property
    def arm_epoch(self):
        return self._epoch

    @property
    def ready(self):
        return self._stage_ready and self._cancel_job is None

    @property
    def busy(self):
        if self.stream:
            return (self._job is not None or self._cancel_job is not None or self._stage_ready
                    or self.candidate_selection == 'pending')
        return (self._job is not None or self._cancel_job is not None or self._stage_ready
                or (self.selection_requested and not self.feedback
                    and not self._interrupt.is_set() and not self.error))

    @property
    def selected(self):
        return self.feedback is not None and self.feedback['status'] in (
            p.REFERENCE_RUNNING, p.REFERENCE_END)

    @property
    def revision(self):
        return self.feedback['revision'] if self.feedback is not None else 0

    def stage(self, key, trajectory: JointTrajectory, *, selection_deadline_ns=None,
              stream=False, terminal=True):
        if self.busy:
            raise RtLinkError('reference transfer already pending')
        if trajectory.times[0] != 0 or trajectory.velocities is None:
            raise ValueError('native execution requires zero-origin Hermite knots')
        self._next_id += 1
        self.stream, self.terminal = bool(stream), bool(terminal)
        self.protected_until = 0.
        self.candidate_selection, self.candidate_error = 'not_requested', ''
        self.candidate_result = None
        self._candidate_curve = None
        self._latest_phase = 0.
        # A transport caller cannot mutate future windows after admission.
        self.id, self.key = self._next_id, key
        self.trajectory = JointTrajectory(trajectory.times, trajectory.positions, trajectory.velocities)
        for values in (self.trajectory.times, self.trajectory.positions, self.trajectory.velocities):
            values.setflags(write=False)
        self.selection_requested = False
        self._selection_deadline_ns = selection_deadline_ns
        self._session = self.link.session_id
        self._epoch = int(self.link.motor_health()['arm_epoch'])
        self._interrupt = threading.Event()
        self.error, self.point, self.feedback = '', None, None
        self._selection_rx = self._first_tracking_at = None
        self._selection_phase = 0.
        self._window, self._revision = (0, 0), 0
        self._stage_window(0)
        return self.id

    def _stage_window(self, first):
        traj = self.trajectory
        end = min(first + p.MAX_REFERENCE_KNOTS, len(traj.times))
        old_end = self._window[1]
        cut = 0. if self._revision == 0 else float(traj.times[old_end - 2])
        window = JointTrajectory(traj.times[first:end], traj.positions[first:end],
                                 traj.velocities[first:end])
        revision = self._revision + 1
        cancelled = self._interrupt
        identity = self.id
        session, epoch = self._session, self._epoch

        def upload():
            if cancelled.is_set():
                raise RtLinkError('reference transfer cancelled')
            return self.link.upload_reference(
                window, reference_id=identity, revision=revision,
                expected_revision=revision-1, terminal=self.terminal and end == len(traj.times),
                session_id=session, arm_epoch=epoch)

        self._window, self._revision, self._cut = (first, end), revision, cut
        self._stage_ready = False
        self._job = ('stage', self._worker.submit(upload))

    def splice(self, suffix, *, cut, expected_revision):
        """Copy the selected whole prefix, retaining the segment covering phase.

        Pure bounded composition; the trusted admission worker checks this exact
        result before stage_suffix. No sampler or coefficient reconstruction.
        """
        if not self.stream or not self.selected or expected_revision != self.revision:
            raise ValueError('splice requires the selected stream revision')
        old = self.trajectory
        indices = np.flatnonzero(old.times == cut)
        if len(indices) != 1 or cut < self.protected_until:
            raise ValueError('cut is not an eligible selected knot')
        last = int(indices[0])
        if (suffix.times[0] != cut or not np.array_equal(suffix.positions[0], old.positions[last])
                or not np.array_equal(suffix.velocities[0], old.velocities[last])):
            raise ValueError('suffix must preserve exact cut time/q/qd')
        first = max(0, int(np.searchsorted(old.times, self.point.time, side='right')) - 1)
        if first >= last:
            raise ValueError('splice missed its whole-segment prefix')
        curve = JointTrajectory(np.r_[old.times[first:last+1], suffix.times[1:]],
                                np.vstack([old.positions[first:last+1], suffix.positions[1:]]),
                                np.vstack([old.velocities[first:last+1], suffix.velocities[1:]]))
        if len(curve.times) > p.MAX_REFERENCE_KNOTS:
            raise ValueError('combined reference exceeds 128 knots')
        for arr in (curve.times, curve.positions, curve.velocities):
            arr.setflags(write=False)
        return curve

    def stage_suffix(self, curve, *, cut, expected_revision, terminal,
                     latest_phase, selection_deadline_ns):
        """Upload the already admitted bounded splice through this same worker."""
        if self.busy or not self.stream or not self.selected or self.terminal:
            raise RtLinkError('stream candidate slot is unavailable')
        if expected_revision != self.revision:
            raise ValueError('selected revision changed before upload')
        self._candidate_curve, self._candidate_terminal = curve, terminal
        self._revision, self._cut = expected_revision+1, cut
        self._latest_phase, self._selection_deadline_ns = latest_phase, selection_deadline_ns
        self.candidate_selection, self.candidate_error = 'not_requested', ''
        self.candidate_result = None
        identity, revision, session, epoch = self.id, self._revision, self._session, self._epoch
        self._job = ('stage', self._worker.submit(
            self.link.upload_reference, curve, reference_id=identity, revision=revision,
            expected_revision=expected_revision, terminal=terminal,
            session_id=session, arm_epoch=epoch))

    def abort_candidate(self):
        """Only an unselected attempt; never retire the selected stream."""
        if self.candidate_selection == 'pending':
            raise RtLinkError('selection pending; reconcile RT feedback before abort')
        self.link.interrupt_reference_upload()
        # The single worker serializes ABORT after any outstanding upload.
        self._job = ('abort', self._worker.submit(
            self.link.abort_reference_upload, session_id=self._session, arm_epoch=self._epoch))
        self._stage_ready = False
        self._candidate_curve = None

    def approve(self):
        """Initial caller admission, or continuation of its immutable admitted plan."""
        if not self.ready:
            raise RtLinkError('reference candidate is not admitted')
        cancelled = self._interrupt
        identity, revision, upload, cut = self.id, self._revision, self._upload, self._cut
        session, epoch = self._session, self._epoch
        deadline = self._selection_deadline_ns if revision == 1 or self.stream else None
        latest = self._latest_phase if self.stream else cut
        observed_phase = None if self.point is None else self.point.time

        def commit():
            try:
                return dispatch()
            except ReferenceCancelledBeforeDispatch:
                self._commit_undispatched = True  # read by retire() after this job
                raise

        def dispatch():
            if cancelled.is_set():
                raise ReferenceCancelledBeforeDispatch('reference selection cancelled')
            budget = 0 if deadline is None else deadline-time.monotonic_ns()
            if deadline is not None and budget <= 0:
                raise _NotDispatched('selection deadline expired before COMMIT')
            phase_deadline = latest
            if revision > 1 and deadline is not None:
                # v7 reserves selection_budget_ns for initial selection. For a
                # revision its existing latest_phase is the deadline. Add only
                # the remaining PC duration to the captured RT phase: ignoring
                # feedback/worker age makes this earlier, never later. No host
                # timestamp subtraction or predicted motion sample is involved.
                if observed_phase is None:
                    raise _NotDispatched('no selected phase for revision deadline')
                phase_deadline = min(latest, observed_phase + budget / 1e9)
                budget = 0
            # RT anchors the duration at SEAL, which precedes receipt of its
            # ACK and this PC budget calculation. This can refuse early but
            # cannot grant extra network/queue time or compare host clocks.
            return self.link.commit_reference(
                upload, reference_id=identity, revision=revision,
                expected_revision=revision-1, protected_until=cut, latest_phase=phase_deadline,
                session_id=session, arm_epoch=epoch, selection_budget_ns=budget,
                _trace=lambda kind, at, value: self._record(
                    kind, identity, revision, at=at, value=value))

        self._stage_ready = False
        self.selection_requested = True
        self._commit_undispatched = False
        self.candidate_selection = 'pending'
        self._pending_since = time.monotonic()
        self._job = ('commit', self._worker.submit(commit))

    def _record_drop(self, key, how):
        """Remember how the RT's drop of ``key`` was confirmed. A proof is never downgraded
        to 'unconfirmed'; the map is bounded (the runtime pops what it consumes)."""
        if key is None or (how == 'unconfirmed' and self.dropped.get(key) not in (None, 'unconfirmed')):
            return
        self.dropped[key] = how
        for stale in list(self.dropped)[:-16]:
            del self.dropped[stale]

    def _note_drop(self, how):
        """A first-revision COMMIT that provably never selected: nothing to drop on the RT."""
        if self._revision == 1:
            self._record_drop(self.key, how)

    def poll(self, state, received_at):
        """Once per PC control tick; takes one immutable feedback sample."""
        if self._schedstat_error is not None and not self._schedstat_warning_emitted:
            self._schedstat_warning_emitted = True
            _LOG.warning('reference sender schedstat %s; CPU-time fallback: %s',
                         self._schedstat_status, self._schedstat_error)
        self._drain_sender_gaps()
        if self._job is not None and self._job[1].done():
            kind, job = self._job
            self._job = None
            try:
                result = job.result()
            except Exception as exc:
                if self.stream:
                    self.candidate_error = str(exc)
                    if kind == 'commit' and isinstance(exc, (ReferenceRefused, _NotDispatched)):
                        self.candidate_selection = 'refused'
                        self.candidate_result = exc.result if isinstance(exc, ReferenceRefused) else p.REF_LATE
                        self._note_drop('refused')
                    elif kind == 'commit' and isinstance(exc, ReferenceCancelledBeforeDispatch):
                        # COMMIT never left this host: proof of non-selection, so
                        # the slot does not wait for the queued ABORT/CANCEL.
                        self.candidate_selection = 'not_requested'
                        self._note_drop('undispatched')
                    # A timeout after COMMIT never proves nonselection. Continue
                    # observing the old selected reference, without freeing slot.
                    if kind != 'commit' and kind != 'abort':
                        self.candidate_selection = 'not_requested'
                if kind == 'commit' and isinstance(exc, ReferenceRefused) and self._revision == 1:
                    self.selection_requested = False  # RT proved it did not select
                if not self._interrupt.is_set() and not self.stream:
                    self.error = str(exc)
            else:
                if kind == 'stage' and not self._interrupt.is_set():
                    self._upload, self._stage_ready = result, True
        if self._cancel_job is not None and self._cancel_job.done():
            try:
                self._cancel_job.result()
            except RtLinkError as exc:
                self.error = str(exc)
                self._record_drop(self._cancel_key, 'unconfirmed')
            else:
                self._record_drop(self._cancel_key, 'ack')
                # The one worker runs CANCEL after any COMMIT, so its ack settles
                # a selection the PC never got to observe (RUNNING retired within
                # ms of COMMIT). Without this the slot stays pending, busy blocks
                # every command, and the RT's command stream is lost.
                if self.candidate_selection == 'pending':
                    self.candidate_selection = 'not_requested'
            self._cancel_job = None
            self._stage_ready = False
        ref = None if state is None else state.reference
        bound = (ref is not None and state.session_id == self._session
                 and state.arm_epoch == self._epoch)
        self.observed = dict(ref) if bound else None
        self.observed_point = None
        if bound and ref['status'] in (p.REFERENCE_RUNNING, p.REFERENCE_END):
            self.observed_point = TrajectoryPoint(
                time=ref['phase'], position=np.array(ref['q_ref']),
                velocity=np.array(ref['qd_ref']), acceleration=np.array(ref['qdd_ref']))
        valid = bound and ref['reference_id'] == self.id
        self.feedback = dict(ref) if valid else None
        self.point = self.observed_point if valid else None
        self.received_at = received_at
        if valid and self.selected and self._selection_rx is None:
            self._selection_rx = received_at
            self._selection_phase = float(ref['phase'])
            self._record_first('first_selected_pc_state', self.id, ref['revision'],
                               at=received_at, phase=float(ref['phase']))
        if (self.stream and valid and ref['revision'] == self._revision
                and ref['status'] in (p.REFERENCE_RUNNING, p.REFERENCE_END, p.REFERENCE_UNDERFLOW)):
            self.candidate_selection = 'selected'
            if self._candidate_curve is not None:
                self.trajectory = self._candidate_curve
                self._candidate_curve = None
                self.terminal = self._candidate_terminal
                self.protected_until = self._cut
        if valid and ref['status'] == p.REFERENCE_UNDERFLOW:
            self.error = 'REFERENCE_UNDERFLOW'
        if (valid and ref['status'] == p.REFERENCE_CANCELLED
                and not self._interrupt.is_set() and not self.error):
            # CTRL-034 decision (a): RT RETIRED this reference because its
            # dynamics envelope expired (the wire status is the ordinary
            # CANCELLED; the cause is in the RT fault journal). Report the
            # segment as failed — a local cancel sets _interrupt and never
            # reaches here — and do NOT plan a continuation implicitly.
            self.error = 'REFERENCE_EXPIRED'
            self.dump_diagnostics('REFERENCE_EXPIRED')
        # Evidence recorded in this same poll wins: an expired reference is reported
        # as REFERENCE_EXPIRED (never healed by an implicit hold), not as unresolved.
        if (self.stream and self.candidate_selection == 'pending' and not self.error
                and self._pending_since is not None
                and time.monotonic() - self._pending_since > SELECTION_UNRESOLVED_S):
            # The slot leaves 'pending' here, so it can never again suppress commands.
            # The RT may or may not have selected: the caller cancels and reports it.
            self.candidate_selection = 'unresolved'
            self.candidate_error = 'selection unresolved'
            raise SelectionUnresolved(
                f'selection unresolved: COMMIT pending for over {SELECTION_UNRESOLVED_S * 1000:.0f} ms')
        if self.stream or self._interrupt.is_set() or self.error or not self.selected:
            return
        # Replacements were already admitted as part of the full PC plan. No
        # new scene policy or coefficient reconstruction runs here.
        if self.ready and self._revision > 1:
            self.approve()
        elif not self.busy and ref['revision'] == self._revision:
            first, end = self._window
            if end < len(self.trajectory.times):
                midpoint = first + (end-first)//2
                if ref['phase'] >= self.trajectory.times[midpoint]:
                    start = int(np.searchsorted(self.trajectory.times, ref['phase'], side='right'))-1
                    if start >= end-2:
                        self.error = 'reference refill missed protected prefix'
                    else:
                        self._stage_window(max(first, start))

    # -- fixed-rate envelope stream (CTRL-034 B1) -----------------------------
    def publish(self, command, *, reference_id, revision, phase, sample_received_at,
                tracking=True):
        """Publish one PC dynamics decision: the command, the reference phase it
        was computed at, and the RT state sample it was computed from.

        The decision is atomic and immutable once published. The sender may
        repeat it at the fixed rate, but it never re-stamps it: every envelope
        carries this sample time and this phase, so its age keeps growing until
        the caller publishes again or the ceiling makes the stream stop.
        """
        at = time.monotonic()
        with self._envelope_lock:
            self._envelope_decision = (dict(command), int(reference_id), int(revision),
                                       float(phase), float(sample_received_at))
            if (tracking and reference_id == self.id and self._selection_rx is not None
                    and self._first_tracking_at is None):
                self._first_tracking_at = time.monotonic()
                # Initial reference phase is RT time since SELECTED. It is a
                # duration, so adding it needs no synchronized PC/RT clocks.
                self.selection_latencies.append(
                    (self.id, self._first_tracking_at - self._selection_rx
                     + self._selection_phase))
        self._record_first('first_publish', reference_id, revision, at=at,
                           sample_received_at=sample_received_at, phase=phase,
                           tracking=tracking)

    def unpublish(self):
        with self._envelope_lock:
            self._envelope_decision = None

    def last_seq(self):
        """Sequence of the most recent envelope the sender put on the wire.

        The caller has just published a decision, so this is the previous
        send of the same running stream (one sender period old at most). It is
        the plant-acceptance evidence the supervisor's SENT/APPLIED receipts
        need; the sender's asynchronous first send cannot be awaited.
        """
        with self._envelope_lock:
            if self._envelope_last is None or self._envelope_decision is None:
                return None
            return (self._envelope_last[1] if self._envelope_last[0]
                    == self._envelope_decision[1] else None)

    def _envelope_loop(self):
        try:
            self._schedstat_fd = os.open('/proc/thread-self/schedstat', os.O_RDONLY)
            self._schedstat_status = 'not_collected'  # pending liveness proof
        except OSError as exc:
            self._schedstat_error = str(exc)
        due = time.monotonic()
        previous = None
        previous_cpu = None
        previous_runqueue = None
        wait_cpu = None
        wait_runqueue = None
        while not self._envelope_stop.is_set():
            started = time.monotonic()
            started_cpu = time.process_time()
            started_stat = self._schedstat_sample()
            self._check_schedstat_liveness(started_stat)
            started_runqueue = (started_stat[1] if started_stat is not None
                                and self._schedstat_status == 'ok' else None)
            if wait_cpu is not None:
                self._sender_interval('wake_late', max(0., started-due), started,
                                      wait_cpu, started_cpu, wait_runqueue,
                                      started_runqueue)
            if previous is not None and self._envelope_decision is not None:
                self._sender_interval('period_gap', started-previous, started,
                                      previous_cpu, started_cpu, previous_runqueue,
                                      started_runqueue)
            previous = started
            previous_cpu = started_cpu
            previous_runqueue = started_runqueue
            self._send_envelope()
            due = started + ENVELOPE_PERIOD_S
            remaining = due - time.monotonic()
            if remaining > 0:
                wait_cpu = time.process_time()
                wait_runqueue = self._runqueue_wait_ns()
                self._envelope_stop.wait(remaining)
            else:
                # A slow send/lock call overran the period; this was not a
                # late wake-up from sleep. Its interval has its own cause.
                wait_cpu = None
        if self._schedstat_fd is not None:
            os.close(self._schedstat_fd)
            self._schedstat_fd = None

    def _send_envelope(self):
        waiting = time.monotonic()
        waiting_cpu = time.process_time()
        waiting_runqueue = self._runqueue_wait_ns()
        with self._envelope_lock:
            acquired = time.monotonic()
            acquired_cpu = time.process_time()
            decision = self._envelope_decision
        acquired_runqueue = self._runqueue_wait_ns()
        self._sender_interval('decision_lock_wait', acquired-waiting, acquired,
                              waiting_cpu, acquired_cpu, waiting_runqueue,
                              acquired_runqueue)
        if decision is None:
            return
        command, reference_id, revision, phase, sample_received_at = decision
        if time.monotonic() - sample_received_at > VALID_FOR_S:
            # The decision itself is older than the ceiling: send NOTHING, so the
            # RT's freshness rule — not a re-stamped heartbeat — owns the hold.
            return
        if self.link.session_id != self._session:
            return
        reading = time.monotonic()
        reading_cpu = time.process_time()
        reading_runqueue = self._runqueue_wait_ns()
        state, _received_at = self.link.latest_state()
        self._sender_interval('state_lock_wait', time.monotonic()-reading,
                              time.monotonic(), reading_cpu, time.process_time(),
                              reading_runqueue, self._runqueue_wait_ns())
        ref = None if state is None else state.reference
        if (state is None or ref is None or not state.armed
                or state.session_id != self._session or state.arm_epoch != self._epoch):
            return
        if ref['reference_id'] != reference_id:
            return
        if ref['status'] not in (p.REFERENCE_RUNNING, p.REFERENCE_END):
            return
        if self._interrupt.is_set() or self.error:
            # Candidate cancellation/errors fence its envelopes, but cannot
            # silence an older selected terminal endpoint in the same session.
            if reference_id == self.id or ref['status'] != p.REFERENCE_END:
                return
        # B2: keep the decision's OWN revision even after RT switches — the RT's
        # revision-(r-1) prefix rule admits it until the caller republishes.
        try:
            sending = time.monotonic()
            sending_cpu = time.process_time()
            sending_runqueue = self._runqueue_wait_ns()
            seq = self.link.apply_reference_envelope(
                command, reference_id=reference_id, revision=revision,
                phase=phase, sample_received_at=sample_received_at,
                valid_for_s=VALID_FOR_S, phase_window_s=PHASE_WINDOW_S,
                session_id=self._session, arm_epoch=self._epoch)
        except (RtLinkError, ValueError, OSError):
            self._sender_interval('udp_send_duration', time.monotonic()-sending,
                                  time.monotonic(), sending_cpu, time.process_time(),
                                  sending_runqueue, self._runqueue_wait_ns())
            return
        self._sender_interval('udp_send_duration', time.monotonic()-sending,
                              time.monotonic(), sending_cpu, time.process_time(),
                              sending_runqueue, self._runqueue_wait_ns())
        if seq is not None:
            self._record_first('first_envelope_sent', reference_id, revision,
                               at=time.monotonic(), sequence=seq,
                               sample_received_at=sample_received_at, phase=phase)
        with self._envelope_lock:
            self._envelope_last = (reference_id, seq)

    def cancel(self):
        if not self.id or self._interrupt.is_set():
            return
        self._interrupt.set()
        self.link.interrupt_reference_upload()
        identity, revision = self.id, self._revision
        selected_revision = self.revision
        session, epoch = self._session, self._epoch
        unselected = not self.selection_requested

        def retire():
            # The one worker runs this after any COMMIT job, so the flag is final.
            if unselected or self._commit_undispatched:
                # The RT never saw COMMIT: only a sealed upload can exist.
                try:
                    self.link.abort_reference_upload(session_id=session, arm_epoch=epoch)
                except ReferenceRefused as exc:
                    if exc.result != p.REF_RETIRED:
                        raise
                if not unselected and selected_revision:
                    # An undispatched replacement leaves the selected revision running.
                    try:
                        self.link.cancel_reference(identity, selected_revision,
                                                   session_id=session, arm_epoch=epoch)
                    except ReferenceRefused as exc:
                        if exc.result not in (p.REF_REVISION, p.REF_RETIRED):
                            raise  # reported as an error; the slot is already settled
                return
            try:
                self.link.cancel_reference(identity, revision, session_id=session, arm_epoch=epoch)
            except ReferenceRefused as exc:
                if exc.result != p.REF_REVISION:
                    raise
                # BEGIN of a replacement may not have arrived; the selected
                # older revision still needs explicit retirement.
                if selected_revision and selected_revision != revision:
                    self.link.cancel_reference(identity, selected_revision,
                                               session_id=session, arm_epoch=epoch)

        self._stage_ready = False
        self._cancel_key = self.key
        self._cancel_job = self._worker.submit(retire)

    def close(self):
        self._interrupt.set()
        self._envelope_stop.set()
        with self._envelope_lock:
            self._envelope_decision = None
        self.link.interrupt_reference_upload()
        self._worker.shutdown(wait=False, cancel_futures=True)
