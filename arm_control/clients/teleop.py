"""Deadman teleoperation: the operator's setpoint moves the arm only while the deadman is held and fresh.

Each setpoint is spliced from the supervisor's anchor at a bounded joint speed, and its horizon never exceeds
``deadman_timeout_s``: a shorter step is taken toward the target (recorded in ``report.notes``; the loop repeats the
command). So if the calling loop itself stalls, at most one timeout's worth of travel is already committed, after which
the executor holds the last reference. When the loop is alive, a released deadman or a heartbeat older than the timeout
CANCELS the stream (the executor holds); ``tick()`` must be called at the loop rate. If the cancel is not confirmed the
session stays open and ``command``/``tick`` retry and say so ('stop_failed'). The module defines no timeout (a required
argument), never re-arms, and the supervisor's admission still checks every revision. A cubic Hermite step that ends at rest
peaks at about 1.5x its average speed.
"""
import time

import numpy as np

from arm_control.clients.common import Report, StreamContext, StreamSession, note_outcome, request_id
from arm_control.sdk import SupervisorError


class DeadmanTeleop:
    def __init__(self, arm, ctx: StreamContext, *, deadman_timeout_s: float, max_speed_rad_s: float,
                 min_step_s: float, hold_s: float, clock=time.monotonic):
        if not (deadman_timeout_s > 0 and max_speed_rad_s > 0 and min_step_s > 0):
            raise ValueError('deadman_timeout_s, max_speed_rad_s and min_step_s must be positive')
        if min_step_s > deadman_timeout_s:
            raise ValueError('min_step_s cannot exceed the deadman timeout (a step would outlast it)')
        self.arm, self.ctx, self.clock = arm, ctx, clock
        self.timeout, self.speed, self.min_step, self.hold_s = deadman_timeout_s, max_speed_rad_s, min_step_s, hold_s
        self.session, self.last_beat, self.report, self.pending = None, None, Report(), None

    def start(self) -> None:
        self.session = StreamSession(self.arm, self.ctx, hold_s=self.hold_s)
        self.last_beat, self.pending = self.clock(), None

    def heartbeat(self) -> None:
        self.last_beat = self.clock()

    def expired(self) -> bool:
        return self.session is None or self.clock() - self.last_beat > self.timeout

    def _settle(self) -> bool:
        """Fold the previous revision's fate into the report; True if it is still unresolved."""
        if self.pending is None:
            return False
        try:
            kind = note_outcome(self.report, self.pending[0],
                                self.arm.stream_revision(self.session.execution_id, self.pending[1]))
        except (SupervisorError, TimeoutError, OSError):
            return True
        if kind in ('refused', 'selected'):
            self.pending = None
        return self.pending is not None

    def command(self, q_target, *, held: bool) -> str:
        """'requested' (sent; see ``report`` for admitted/selected), 'busy' (the previous revision is unresolved: nothing
        sent), 'released' / 'expired' (stopped), 'stop_failed' (stop NOT confirmed, retry), or 'refused'."""
        if self.session is None or self.session.closed:
            return 'expired'
        if not held:
            return self._stop('deadman released')
        if self.expired():
            return self._stop('deadman heartbeat expired')
        self.heartbeat()
        if self._settle():
            return 'busy'
        try:
            q_now = np.asarray(self.arm.state().q, dtype=float)
            delta = np.asarray(q_target, dtype=float) - q_now
            travel = float(np.max(np.abs(delta)))
            dt = max(travel / self.speed, self.min_step)
            if dt > self.timeout:  # never commit more than one deadman timeout of motion
                delta *= self.speed * self.timeout / travel
                dt = self.timeout
                self.report.notes.append('target limited to one deadman timeout of travel')
            rid = request_id('teleop')
            status = self.session.revise(rid, (q_now + delta)[None, :], dt)
        except (ValueError, SupervisorError, TimeoutError, OSError) as exc:
            self.report.refused.append(('teleop', str(exc)))
            return 'refused'
        self.report.requested.append(rid)
        if note_outcome(self.report, rid, status) == 'refused':
            return 'refused'
        self.pending = (rid, rid)
        return 'requested'

    def tick(self) -> str | None:
        """Call at the loop rate even without input: a missing heartbeat stops the arm."""
        if self.session is not None and not self.session.closed and self.expired():
            return self._stop('deadman heartbeat expired')
        return None

    def _stop(self, reason: str) -> str:
        if not self.session.close(reason):
            self.report.notes.append(f'STOP NOT CONFIRMED ({reason}): {self.session.last_error}; will retry')
            return 'stop_failed'
        self.report.notes.append(reason)
        return 'released' if 'released' in reason else 'expired'
