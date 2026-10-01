"""An RL-style step-window client. The servo runs on its own; ``step`` only proposes and then observes.

``step(q_target, window_s)`` splices one absolute target reached over the window, waits for the window and reports the
ACTUAL elapsed time and the outcome. ``completed`` means the RT selected the revision and the window elapsed; a
revision the supervisor refused (at once or after admission) is ``refused`` and one still unselected at the end is
``not_selected``, both truncations. Task success and task termination are the project's: ``terminated`` is always False.
An ``intervention`` returns early but does NOT stop or hold the arm (it keeps moving toward the target): the caller
decides to cancel. ``reset()`` checks the robot is in a state the policy may continue from and otherwise raises
:class:`NeedsOperator`: it never re-arms, homes, opens the Hand, discards calibration or restarts a cancelled stream.
"""
from dataclasses import dataclass
import time

import numpy as np

from arm_control.clients.common import NeedsOperator, Report, StreamContext, StreamSession, note_outcome, request_id
from arm_control.execution.contract import Lifecycle, Phase
from arm_control.sdk import SupervisorError


@dataclass(frozen=True)
class StepResult:
    elapsed_s: float
    outcome: str            # completed | intervention | refused | not_selected | expired | faulted
    terminated: bool        # always False: task success is project-owned
    truncated: bool         # infrastructure ended or refused the step
    selected_revision: int | None
    reason: str = ''


class StepWindow:
    def __init__(self, arm, ctx: StreamContext, *, hold_s: float, poll_s: float = 0.01, clock=time.monotonic):
        self.arm, self.ctx, self.hold_s, self.poll_s, self.clock = arm, ctx, hold_s, poll_s, clock
        self.session, self.report = None, Report()

    def reset(self) -> Report:
        """Open the stream the episode runs on. Refuses (NeedsOperator) unless the robot is armed and not faulted."""
        authority = self.arm.authority()
        if authority.lifecycle is not Lifecycle.ARMED or authority.fault is not None:
            raise NeedsOperator(f'robot is {authority.lifecycle.value}: re-arm/recovery is an operator action')
        if self.session is not None and not self.session.close('episode reset'):
            raise NeedsOperator(f'previous episode stream not confirmed cancelled: {self.session.last_error}')
        self.session = StreamSession(self.arm, self.ctx, hold_s=self.hold_s)
        return self.report

    def _result(self, start, outcome, *, reason='', selected=None):
        return StepResult(self.clock() - start, outcome, False, outcome not in ('completed', 'intervention'),
                          selected, reason)

    def step(self, q_target, window_s: float, *, intervention=lambda: False) -> StepResult:
        if self.session is None or self.session.closed:
            raise NeedsOperator('no open episode stream: call reset() (the client never restarts one on its own)')
        rid = request_id('step')
        start = self.clock()
        ended = self.session.view().execution
        if ended.terminal:  # infrastructure ended the episode stream since the last step
            self.session.closed = True
            return self._result(start, 'faulted' if ended.phase is Phase.FAULTED else 'expired', reason=ended.reason)
        try:
            status = self.session.revise(rid, np.asarray(q_target, dtype=float)[None, :], window_s)
        except SupervisorError as exc:
            self.report.refused.append((rid, str(exc)))
            return self._result(start, 'refused', reason=str(exc))
        self.report.requested.append(rid)
        kind = note_outcome(self.report, rid, status)
        outcome, reason = None, ''
        while outcome is None and self.clock() - start < window_s:
            if kind == 'refused':
                outcome, reason = 'refused', status.reason
            elif intervention():
                outcome = 'intervention'
            else:
                view = self.session.view()
                if view.execution.terminal:
                    outcome = 'faulted' if view.execution.phase is Phase.FAULTED else 'expired'
                    reason, self.session.closed = view.execution.reason, True
                else:
                    try:
                        status = self.arm.stream_revision(self.session.execution_id, rid)
                        kind = note_outcome(self.report, rid, status)
                    except (SupervisorError, TimeoutError, OSError):
                        pass
                    time.sleep(self.poll_s)
        if outcome is None:
            status = self.arm.stream_revision(self.session.execution_id, rid)
            kind = note_outcome(self.report, rid, status)
            outcome = 'completed' if kind == 'selected' else 'refused' if kind == 'refused' else 'not_selected'
            reason = status.reason
        selected = None if self.session.closed else self.session.view().selected_revision
        return self._result(start, outcome, reason=reason, selected=selected)
