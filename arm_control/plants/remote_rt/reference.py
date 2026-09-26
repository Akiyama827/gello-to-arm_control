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


class ReferenceHorizon:
    def __init__(self, link):
        self.link = link
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='reference-upload')
        self._job = None
        self._cancel_job = None
        self._interrupt = threading.Event()
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

    def stage(self, key, trajectory: JointTrajectory, *, selection_deadline_ns=None):
        if self.busy:
            raise RtLinkError('reference transfer already pending')
        if trajectory.times[0] != 0 or trajectory.velocities is None:
            raise ValueError('native execution requires zero-origin Hermite knots')
        self._next_id += 1
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
                expected_revision=revision-1, terminal=end == len(traj.times),
                session_id=session, arm_epoch=epoch)

        self._window, self._revision, self._cut = (first, end), revision, cut
        self._stage_ready = False
        self._job = ('stage', self._worker.submit(upload))

    def approve(self):
        """Initial caller admission, or continuation of its immutable admitted plan."""
        if not self.ready:
            raise RtLinkError('reference candidate is not admitted')
        cancelled = self._interrupt
        identity, revision, upload, cut = self.id, self._revision, self._upload, self._cut
        session, epoch = self._session, self._epoch
        deadline = self._selection_deadline_ns if revision == 1 else None

        def commit():
            if cancelled.is_set():
                raise RtLinkError('reference selection cancelled')
            budget = 0 if deadline is None else deadline-time.monotonic_ns()
            if deadline is not None and budget <= 0:
                raise RtLinkError('selection deadline expired before COMMIT')
            # RT anchors the duration at SEAL, which precedes receipt of its
            # ACK and this PC budget calculation. This can refuse early but
            # cannot grant extra network/queue time or compare host clocks.
            return self.link.commit_reference(
                upload, reference_id=identity, revision=revision,
                expected_revision=revision-1, protected_until=cut, latest_phase=cut,
                session_id=session, arm_epoch=epoch, selection_budget_ns=budget)

        self._stage_ready = False
        self.selection_requested = True
        self._job = ('commit', self._worker.submit(commit))

    def poll(self, state, received_at):
        """Once per PC control tick; takes one immutable feedback sample."""
        if self._job is not None and self._job[1].done():
            kind, job = self._job
            self._job = None
            try:
                result = job.result()
            except Exception as exc:
                if kind == 'commit' and isinstance(exc, ReferenceRefused) and self._revision == 1:
                    self.selection_requested = False  # RT proved it did not select
                if not self._interrupt.is_set():
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
        if valid and ref['status'] == p.REFERENCE_UNDERFLOW:
            self.error = 'REFERENCE_UNDERFLOW'
        if self._interrupt.is_set() or self.error or not self.selected:
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
        self.link.interrupt_reference_upload()
        self._worker.shutdown(wait=False, cancel_futures=True)
