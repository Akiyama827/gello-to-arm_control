"""Deadman teleoperation: the operator's setpoint moves the arm only while the deadman is held and fresh.

Every setpoint is spliced from the supervisor's anchor at a bounded joint speed. When the deadman is released or its
last heartbeat is older than ``deadman_timeout_s`` (a required argument: this module defines no timeout) the stream is
CANCELLED, which makes the executor hold. It never re-arms; a stalled GUI therefore stops the arm, not extends the last
setpoint. The supervisor's admission still checks every revision.
"""
import time

import numpy as np

from arm_control.clients.common import Report, StreamContext, StreamSession
from arm_control.execution.contract import Phase


class DeadmanTeleop:
    def __init__(self, arm, ctx: StreamContext, *, deadman_timeout_s: float, max_speed_rad_s: float,
                 min_step_s: float, hold_s: float, clock=time.monotonic):
        if not (deadman_timeout_s > 0 and max_speed_rad_s > 0 and min_step_s > 0):
            raise ValueError('deadman_timeout_s, max_speed_rad_s and min_step_s must be positive')
        self.arm, self.ctx, self.clock = arm, ctx, clock
        self.timeout, self.speed, self.min_step, self.hold_s = deadman_timeout_s, max_speed_rad_s, min_step_s, hold_s
        self.session, self.last_beat, self.report, self._n = None, None, Report(), 0

    def start(self) -> None:
        self.session = StreamSession(self.arm, self.ctx, hold_s=self.hold_s)
        self.last_beat = self.clock()

    def heartbeat(self) -> None:
        self.last_beat = self.clock()

    def expired(self) -> bool:
        return self.session is None or self.clock() - self.last_beat > self.timeout

    def command(self, q_target, *, held: bool) -> str:
        """'moved', 'released' (deadman not held: stopped), 'expired' (stale heartbeat: stopped), or 'refused'."""
        if self.session is None or self.session.closed:
            return 'expired'
        if not held:
            return self._stop('deadman released')
        if self.expired():
            return self._stop('deadman heartbeat expired')
        self.heartbeat()
        q_now = np.asarray(self.arm.state().q, dtype=float)
        target = np.asarray(q_target, dtype=float)
        dt = max(float(np.max(np.abs(target - q_now))) / self.speed, self.min_step)
        self._n += 1
        rid = f'teleop-{self._n}'
        try:
            status = self.session.revise(rid, target[None, :], dt)
        except ValueError as exc:
            self.report.refused.append((rid, str(exc)))
            return 'refused'
        if status.phase is Phase.REJECTED:
            self.report.refused.append((rid, status.reason))
            return 'refused'
        self.report.admitted.append(rid)
        return 'moved'

    def tick(self) -> str | None:
        """Call at the GUI/loop rate even without input: a missing heartbeat stops the arm."""
        if self.session is not None and not self.session.closed and self.expired():
            return self._stop('deadman heartbeat expired')
        return None

    def _stop(self, reason: str) -> str:
        self.session.close(reason)
        self.report.notes.append(reason)
        return 'released' if 'released' in reason else 'expired'
