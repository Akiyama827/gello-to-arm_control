"""CTRL-040 group execution through per-arm supervisors (the negotiated ``group/1`` SDK opt-in).

``SdkGroupCoordinator`` is the supervisor-backed counterpart of :class:`~arm_control.multi_arm.group.GroupCoordinator`: it
holds one claimed ``ArmClient`` per member (each arm's supervisor owns that arm's single RT session, so the PC cannot open a
second one) and drives JOIN, stage, COMMIT_GROUP and CANCEL through them. The start record is the same
:class:`~arm_control.multi_arm.group.GroupStartRecord` built from the RT's own selection stamps; the PC barrier skew stays a
separate, differently labelled number.

Differences from the direct-link coordinator, stated plainly: references are uploaded by each supervisor AFTER release (the
supervisor stages, then COMMITs), so members reach COMMIT_GROUP at slightly different times and the RT's rendezvous budget
must cover that spread (a member that commits more than ``budget_s`` before its peer is refused ``REF_LATE``); the budget and
H defaults are the operator-approved ones; nothing here is a qualification (no hardware, no FR3-host measurement, and no unit
file may pass ``--group-*`` yet).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

from arm_control.execution.contract import Phase, Start
from arm_control.multi_arm.group import (
    GROUP_BUDGET_S, GROUP_HORIZON_S, GroupStartError, GroupStartRecord, MemberStart)
from arm_control.plants.remote_rt import protocol as rtp
from arm_control.sdk import Refused, SupervisorError


class SdkGroupCoordinator:
    def __init__(self, arms: dict, *, budget_s: float = GROUP_BUDGET_S, horizon_s: float = GROUP_HORIZON_S):
        if not 2 <= len(arms) <= 4:
            raise ValueError('a group needs 2 to 4 members')
        if not (0 < budget_s <= rtp.GROUP_MAX_BUDGET_NS / 1e9):
            raise ValueError('budget must be positive and within the wire bound')
        if not horizon_s > 0:
            raise ValueError('H must be positive')
        self.arms, self.budget_s, self.horizon_s = dict(arms), budget_s, horizon_s
        self.names = list(self.arms)
        self.last_executions: dict = {}  # execution id per member of the latest start (also after a failed one)

    def _validate(self, requests: dict) -> None:
        if set(requests) != set(self.arms):
            raise ValueError('one request per member')
        for name, request in requests.items():
            if request.start is not Start.ON_RELEASE:
                raise ValueError(f'{name}: a group member is released by the group, not on admission')
            if self.arms[name].extensions.get('group') != 1:
                raise GroupStartError(f'group refused: {name} supervisor does not advertise group/1', {name: 'no group'})
            span = float(request.trajectory.times[-1] - request.trajectory.times[0])
            if span < self.horizon_s:
                raise ValueError(f'{name}: reference covers {span:.3f} s, less than H = {self.horizon_s:.3f} s')

    @staticmethod
    def _domain(domains: dict) -> str:
        attested = {n: d for n, d in domains.items() if d is not None}
        if attested and len(attested) != len(domains):
            raise GroupStartError('group refused: RT clock domain is attested on some members only', domains)
        if len(set(domains.values())) != 1:
            raise GroupStartError('group refused: members are not on one RT clock domain', domains)
        d = next(iter(domains.values()))
        return 'unattested' if d is None else d

    def start(self, group_id: int, requests: dict) -> GroupStartRecord:
        """JOIN, stage, COMMIT_GROUP together; on ANY failure the group is cancelled on every member."""
        self._validate(requests)
        count = len(self.names)
        executions: dict = {}
        self.last_executions = executions
        try:
            return self._start(group_id, requests, count, executions)
        except BaseException as original:
            errors = self.cancel(group_id, executions)
            if errors:
                original.cancel_errors = errors  # a member may still be joined
            raise

    def _start(self, group_id, requests, count, executions) -> GroupStartRecord:
        domains = {}
        try:
            for index, name in enumerate(self.names):
                domains[name] = self.arms[name].group_join(group_id, index, count).get('clock_domain')
        except (Refused, SupervisorError) as exc:
            raise GroupStartError(f'group {group_id} could not be joined: {exc}') from exc
        domain = self._domain(domains)
        try:
            for name in self.names:
                status = self.arms[name].execute(requests[name])
                executions[name] = status.execution_id
                if status.phase is not Phase.ADMITTED:
                    raise GroupStartError(f'group {group_id}: {name} not admitted: {status.phase.value} {status.reason}',
                                          {name: status.reason})
        except (Refused, SupervisorError) as exc:
            raise GroupStartError(f'group {group_id} could not be staged: {exc}') from exc
        budget_ns = int(self.budget_s * 1e9)

        def release(item):
            index, name = item
            sent = time.monotonic()
            try:
                self.arms[name].group_release(executions[name], group_id, budget_ns)
            except (Refused, SupervisorError, TimeoutError) as exc:
                return name, exc, sent, time.monotonic()
            return name, None, sent, time.monotonic()

        with ThreadPoolExecutor(max_workers=count) as pool:
            released = list(pool.map(release, enumerate(self.names)))
        report = {}
        for name, error, _, _ in released:
            if error is not None:
                report[name] = f'{type(error).__name__}: {error}'
        if report:
            raise GroupStartError(f'group {group_id} could not be released', report)
        # Selection is observed asynchronously: every member's RT reply (stamps) or refusal, within the budget.
        deadline = time.monotonic() + self.budget_s + 2.0
        replies: dict = {}
        while len(replies) < count and time.monotonic() < deadline:
            for name in self.names:
                if name in replies:
                    continue
                reply = self.arms[name].group_report(executions[name])
                if reply:
                    replies[name] = reply
                    continue
                status = self.arms[name].status(executions[name])
                if status.terminal:  # rejected/expired before the RT answered
                    replies[name] = dict(error=f'{status.phase.value}: {status.reason}')
            time.sleep(0.005)
        bad = {n: r.get('error', 'no reply') for n, r in replies.items() if 'selected_ns' not in r}
        bad.update({n: 'no reply within the budget' for n in self.names if n not in replies})
        if bad:
            raise GroupStartError(f'group {group_id} did not start', bad)
        members = {}
        for (name, _, sent, got) in released:
            r = replies[name]
            if not (r['selected_ns'] > 0 and r['deadline_ns'] > 0):
                raise GroupStartError(f'group {group_id}: {name} reported no selection stamp', {name: str(r)})
            members[name] = MemberStart(name, int(r['reference_id']), int(r['selected_ns']), int(r['deadline_ns']),
                                        sent, got)
        return GroupStartRecord(group_id, domain, members)

    def cancel(self, group_id: int, executions: dict | None = None) -> dict:
        """Best effort on every member (a failing member does not stop the others); returns the errors."""
        errors = {}
        for name in self.names:
            arm = self.arms[name]
            try:
                if executions and name in executions:
                    arm.cancel(executions[name], 'group cancelled')
                arm.group_cancel(group_id)
            except Exception as exc:  # noqa: BLE001 - a failing member must not stop the others
                errors[name] = f'{type(exc).__name__}: {exc}'
        return errors

    def events(self) -> dict:
        return {name: self.arms[name].group_events() for name in self.names}
