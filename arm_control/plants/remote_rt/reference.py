"""Bounded native-horizon transport; no interpolation, dynamics or task policy.

The caller retains PC admission and approves initial selection. Once selected,
this owner uploads unchanged windows of that same admitted trajectory, retaining
whole existing segments at every revision. One worker and one candidate; TCP
framing/backpressure remains RtBackend's responsibility.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np

from arm_control.motion import JointTrajectory, TrajectoryPoint
from . import protocol as p
from .client import RtLinkError, ReferenceRefused


class _NotDispatched(RtLinkError):
    """Local proof that COMMIT never entered the transport."""


# CTRL-034 B1: the fixed rate of the dedicated envelope sender. The sender owns
# the v7 stream so an upload, a plan or a TCP round-trip in the caller's loop
# cannot stretch the gap between envelopes; the RT's freshness ceiling is 50 ms.
ENVELOPE_PERIOD_S = 0.005


class ReferenceHorizon:
    def __init__(self, link):
        self.link = link
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='reference-upload')
        self._job = None
        self._cancel_job = None
        self._interrupt = threading.Event()
        self._envelope_lock = threading.Lock()
        self._envelope_decision = None
        self._envelope_identity = None
        self._envelope_last = None
        self._envelope_stop = threading.Event()
        self._envelope_thread = threading.Thread(
            target=self._envelope_loop, name='reference-envelope', daemon=True)
        self._envelope_thread.start()
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
        self.candidate_error = ''
        self.candidate_result = None
        self._candidate_curve = None
        self._candidate_terminal = True
        self._latest_phase = 0.

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
            if cancelled.is_set():
                raise RtLinkError('reference selection cancelled')
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
                session_id=session, arm_epoch=epoch, selection_budget_ns=budget)

        self._stage_ready = False
        self.selection_requested = True
        self.candidate_selection = 'pending'
        self._job = ('commit', self._worker.submit(commit))

    def poll(self, state, received_at):
        """Once per PC control tick; takes one immutable feedback sample."""
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
    def publish(self, command, *, reference_id, revision):
        """Publish the latest PC dynamics decision for the fixed-rate sender.

        The identity is the RT selection the decision was computed against (the
        RT's observed id/revision, which may still be an endpoint while a new
        candidate uploads). The caller publishes only while the controller is
        actually tracking that selection; ``unpublish`` on every other command
        path is what keeps the stream a decision, never a blind heartbeat.
        """
        with self._envelope_lock:
            self._envelope_decision = dict(command)
            self._envelope_identity = (int(reference_id), int(revision),
                                       self._session, self._epoch)

    def unpublish(self):
        with self._envelope_lock:
            self._envelope_decision = None

    def last_seq(self):
        """Sequence of the most recent envelope sent for the current decision."""
        with self._envelope_lock:
            if self._envelope_last is None or self._envelope_identity is None:
                return None
            seq, identity = self._envelope_last
            return seq if identity == self._envelope_identity else None

    def _envelope_loop(self):
        # One fresh RT state sample per envelope: `phase` and `sample_received_at`
        # both come from THIS read, so `age_us` measures the sender-to-RT
        # transport, not the caller's tick or an upload round-trip.
        while not self._envelope_stop.is_set():
            started = time.monotonic()
            self._send_envelope()
            remaining = ENVELOPE_PERIOD_S - (time.monotonic() - started)
            if remaining > 0:
                self._envelope_stop.wait(remaining)

    def _send_envelope(self):
        with self._envelope_lock:
            decision, identity = self._envelope_decision, self._envelope_identity
        if decision is None or identity is None:
            return
        if self._interrupt.is_set() or self.error or self.link.session_id != self._session:
            return
        state, received_at = self.link.latest_state()
        ref = None if state is None else state.reference
        if (state is None or ref is None or not state.armed
                or state.session_id != self._session or state.arm_epoch != self._epoch):
            return
        if (ref['reference_id'] != identity[0] or ref['revision'] != identity[1]
                or ref['status'] not in (p.REFERENCE_RUNNING, p.REFERENCE_END)):
            return
        try:
            seq = self.link.apply_reference_envelope(
                decision, reference_id=ref['reference_id'], revision=ref['revision'],
                phase=ref['phase'], sample_received_at=received_at,
                valid_for_s=.05, phase_window_s=.05,
                session_id=self._session, arm_epoch=self._epoch)
        except (RtLinkError, ValueError, OSError):
            return
        with self._envelope_lock:
            self._envelope_last = (seq, identity)

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
            if unselected:
                try:
                    self.link.abort_reference_upload(session_id=session, arm_epoch=epoch)
                except ReferenceRefused as exc:
                    if exc.result != p.REF_RETIRED:
                        raise
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
        self._cancel_job = self._worker.submit(retire)

    def close(self):
        self._interrupt.set()
        self._envelope_stop.set()
        with self._envelope_lock:
            self._envelope_decision = None
        self.link.interrupt_reference_upload()
        self._worker.shutdown(wait=False, cancel_futures=True)
