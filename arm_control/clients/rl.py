"""An RL-style step-window client. The servo runs on its own; ``step`` only proposes and then observes.

``step(q_target, window_s)`` splices one absolute target reached over the window, waits for the window and reports the
ACTUAL elapsed time and the outcome. Task success and task termination are the project's: ``terminated`` is always
False here, ``truncated`` means infrastructure ended the step (refused, expired, faulted). ``reset()`` checks the robot
is in a state the policy may continue from and otherwise raises :class:`NeedsOperator`: it never re-arms, homes, opens
the Hand, discards calibration or restarts a cancelled stream.
"""
from dataclasses import dataclass
import time

import numpy as np

from arm_control.clients.common import NeedsOperator, Report, StreamContext, StreamSession
from arm_control.execution.contract import Lifecycle, Phase
from arm_control.sdk import Refused


@dataclass(frozen=True)
class StepResult:
    elapsed_s: float
    outcome: str            # completed | intervention | refused | expired | faulted
    terminated: bool        # always False: task success is project-owned
    truncated: bool         # infrastructure ended the step
    selected_revision: int | None
    reason: str = ''


class StepWindow:
    def __init__(self, arm, ctx: StreamContext, *, hold_s: float, poll_s: float = 0.01, clock=time.monotonic):
        self.arm, self.ctx, self.hold_s, self.poll_s, self.clock = arm, ctx, hold_s, poll_s, clock
        self.session, self.report, self._n = None, Report(), 0

    def reset(self) -> Report:
        """Open the stream the episode runs on. Refuses (NeedsOperator) unless the robot is armed and not faulted."""
        authority = self.arm.authority()
        if authority.lifecycle is not Lifecycle.ARMED or authority.fault is not None:
            raise NeedsOperator(f'robot is {authority.lifecycle.value}: re-arm/recovery is an operator action')
        if self.session is not None:
            self.session.close('episode reset')
        self.session = StreamSession(self.arm, self.ctx, hold_s=self.hold_s)
        return self.report

    def step(self, q_target, window_s: float, *, intervention=lambda: False) -> StepResult:
        if self.session is None or self.session.closed:
            raise NeedsOperator('no open episode stream: call reset() (the client never restarts one on its own)')
        self._n += 1
        rid = f'step-{self._n}'
        start = self.clock()
        ended = self.session.view().execution
        if ended.terminal:  # infrastructure ended the episode stream since the last step
            self.session.closed = True
            return StepResult(self.clock() - start, 'faulted' if ended.phase is Phase.FAULTED else 'expired',
                              False, True, None, ended.reason)
        try:
            status = self.session.revise(rid, np.asarray(q_target, dtype=float)[None, :], window_s)
        except Refused as exc:
            self.report.refused.append((rid, str(exc)))
            return StepResult(self.clock() - start, 'refused', False, True, None, str(exc))
        if status.phase is Phase.REJECTED:
            self.report.refused.append((rid, status.reason))
            return StepResult(self.clock() - start, 'refused', False, True, None, status.reason)
        self.report.admitted.append(rid)
        outcome, reason = 'completed', ''
        while self.clock() - start < window_s:
            if intervention():
                outcome = 'intervention'
                break
            view = self.session.view()
            if view.execution.terminal:
                outcome = 'faulted' if view.execution.phase is Phase.FAULTED else 'expired'
                reason = view.execution.reason
                self.session.closed = True
                break
            time.sleep(self.poll_s)
        view = self.session.view() if not self.session.closed else None
        selected = None if view is None else view.selected_revision
        elapsed = self.clock() - start
        return StepResult(elapsed, outcome, False, outcome in ('refused', 'expired', 'faulted'), selected, reason)
