"""PC-side driver for RT-scheduled group execution (CTRL-040): prepare, commit, skew report.

A group start is bounded coordination, not an atomic physical event. This module drives the RT servers'
group operations (`RtBackend.group_join/group_commit/group_cancel`, protocol v7 additive) for members that
share ONE RT host: JOIN every member, upload every reference, COMMIT all members at once with one shared
budget, and report each member's start from the RT's own selection stamps (the RT host clock), never from
PC send times. The PC barrier skew is kept only as a separate, differently labelled number.

Defaults are the values the operator approved on 2026-10-01 (``Control/docs/ctrl040-constants-proposal.md``):
``BUDGET`` 150 ms and ``H`` 350 ms are PC-side; ``PEER_MAX_AGE`` 30 ms and ``MIN_LEAD`` 10 ms are RT-server
flags (``--group-peer-max-age-ms``, ``--group-min-lead-ms``) that the servers of a group must be started with.
No unit file may pass ``--group-*`` until the FR3-host measurement and the CTRL-042 decision.

Through supervisors this is ``supervised_group.SdkGroupCoordinator`` (group/1, native references only). NOT wired into
the dual console: its pair legs run each arm's streamed path, started at one PC stamp (scheduled_release/1, bounded by
``MIN_LEAD``..``BUDGET``); an RT-side group start for the console is a separate, unapproved step. What this module refuses to do: start a group whose members are on different or half-attested RT clock
domains (a group whose members are ALL unattested, i.e. older servers, is accepted and labelled 'unattested';
the shared record already confines a group to one host), whose references are shorter than ``H`` or exceed the wire
limits, or that has fewer than two members.

Lifecycle: a started group stays joined on the RT until ``cancel(group_id)``. Call it when the group is finished
(both references ended: a DONE member only leaves, nothing is retired) or to abort (a running member is retired and
its peers follow). A later ``start()`` with a new id is refused at JOIN while an old group is still joined.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time

from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.client import ReferenceRefused, RtLinkError

# Approved 2026-10-01 (PC-side).
GROUP_BUDGET_S = 0.150
GROUP_HORIZON_S = 0.350
# Approved 2026-10-01 (RT-server flags; here so benches and deployment tooling name one source).
GROUP_PEER_MAX_AGE_MS = 30
GROUP_MIN_LEAD_MS = 10


class GroupStartError(RtLinkError):
    """The group did not start. ``members`` maps each member to what the RT said (or the error text)."""

    def __init__(self, message: str, members: dict | None = None):
        super().__init__(message)
        self.members = members or {}


@dataclass(frozen=True)
class MemberStart:
    """One member's start. ``rt_selected_ns`` is on the RT host clock (``clock_domain``)."""

    member: str
    reference_id: int
    rt_selected_ns: int
    rt_deadline_ns: int
    pc_commit_sent: float      # PC monotonic: when this member's COMMIT_GROUP was sent
    pc_reply_received: float   # PC monotonic: when its reply arrived


@dataclass(frozen=True)
class GroupStartRecord:
    group_id: int
    clock_domain: str
    members: dict[str, MemberStart] = field(default_factory=dict)

    @property
    def rt_skew_sec(self) -> float | None:
        """Spread of the members' RT selection stamps: the measured start skew (one RT host clock)."""
        stamps = [m.rt_selected_ns for m in self.members.values()]
        return (max(stamps) - min(stamps)) / 1e9 if len(stamps) >= 2 else None

    @property
    def pc_send_skew_sec(self) -> float | None:
        """Spread of the PC's COMMIT send times: PC barrier skew, NOT execution skew."""
        sent = [m.pc_commit_sent for m in self.members.values()]
        return max(sent) - min(sent) if len(sent) >= 2 else None

    @property
    def deadline_agreed(self) -> bool:
        return len({m.rt_deadline_ns for m in self.members.values()}) == 1

    def to_payload(self) -> dict:
        return {
            "group_id": self.group_id,
            "clock_domain": self.clock_domain,
            "rt_selected_ns": {k: m.rt_selected_ns for k, m in self.members.items()},
            "rt_deadline_ns": {k: m.rt_deadline_ns for k, m in self.members.items()},
            "rt_skew_sec": self.rt_skew_sec,
            "pc_send_skew_sec": self.pc_send_skew_sec,
            "deadline_agreed": self.deadline_agreed,
        }


class GroupCoordinator:
    """Drives one group of RT links. Members are named; every link must be a group-capable RtBackend."""

    def __init__(self, links: dict, *, budget_s: float = GROUP_BUDGET_S,
                 horizon_s: float = GROUP_HORIZON_S):
        if not 2 <= len(links) <= 4:
            raise ValueError("a group needs 2 to 4 members")
        if not (0 < budget_s <= rtp.GROUP_MAX_BUDGET_NS / 1e9):
            raise ValueError("budget must be positive and within the wire bound")
        if not horizon_s > 0:
            raise ValueError("H must be positive")
        self.links, self.budget_s, self.horizon_s = dict(links), budget_s, horizon_s
        self.names = list(self.links)

    # -- checks that need no RT contact ---------------------------------------------------------
    def _domain(self) -> str:
        """One attested RT clock domain, or 'unattested' (older servers) - never a mixture or a mismatch:
        RT selection stamps are only comparable on one clock."""
        domains = {n: getattr(link, "clock_domain", None) for n, link in self.links.items()}
        attested = {n: d for n, d in domains.items() if d is not None}
        if attested and len(attested) != len(domains):
            raise GroupStartError("group refused: RT clock domain is attested on some members only", domains)
        if len(set(domains.values())) != 1:
            raise GroupStartError("group refused: members are not on one RT clock domain", domains)
        d = next(iter(domains.values()))
        return "unattested" if d is None else ":".join(d)

    def _validate(self, trajectories: dict, reference_ids: dict) -> str:
        if set(trajectories) != set(self.links) or set(reference_ids) != set(self.links):
            raise ValueError("one trajectory and one reference id per member")
        for name, link in self.links.items():
            if not getattr(link, "group_capable", False):
                raise GroupStartError(f"group refused: {name} has no group record", {name: "no group"})
        for name, trajectory in trajectories.items():
            knots = len(trajectory.times)
            if not 2 <= knots <= rtp.MAX_REFERENCE_KNOTS:
                raise ValueError(f"{name}: reference needs 2..{rtp.MAX_REFERENCE_KNOTS} knots, has {knots}")
            joints = getattr(self.links[name], "n", None)
            if joints is not None and trajectory.positions.shape[1] != joints:
                raise ValueError(f"{name}: reference has {trajectory.positions.shape[1]} joints, the arm {joints}")
            span = float(trajectory.times[-1] - trajectory.times[0])
            if span < self.horizon_s:
                raise ValueError(f"{name}: reference covers {span:.3f} s, less than H = {self.horizon_s:.3f} s")
        return self._domain()

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self, group_id: int, trajectories: dict, reference_ids: dict) -> GroupStartRecord:
        """JOIN, upload, COMMIT together. Returns the start record; on ANY failure the group is cancelled on
        every member; GroupStartError carries what each member reported (selected members included)."""
        domain = self._validate(trajectories, reference_ids)
        count = len(self.names)
        try:
            return self._start(group_id, trajectories, reference_ids, domain, count)
        except BaseException as original:
            # Whatever went wrong (not only a link error), no member is left joined or running.
            errors = self.cancel(group_id)
            if errors:  # the caller must be able to tell that a member may still be joined
                original.cancel_errors = errors
            raise

    def _start(self, group_id, trajectories, reference_ids, domain, count) -> GroupStartRecord:
        try:
            for index, name in enumerate(self.names):
                self.links[name].group_join(group_id, index, count)
            uploads = {name: self.links[name].upload_reference(trajectories[name],
                                                                reference_id=reference_ids[name])
                       for name in self.names}
        except RtLinkError as exc:
            raise GroupStartError(f"group {group_id} could not be prepared: {exc}") from exc

        budget_ns = int(self.budget_s * 1e9)

        def commit(index_name):
            index, name = index_name
            sent = time.monotonic()
            try:
                reply = self.links[name].group_commit(
                    uploads[name], reference_id=reference_ids[name], group_id=group_id,
                    member_index=index, budget_ns=budget_ns)
            except (ReferenceRefused, RtLinkError) as exc:
                return name, exc, sent, time.monotonic()
            return name, reply, sent, time.monotonic()

        with ThreadPoolExecutor(max_workers=count) as pool:
            outcomes = list(pool.map(commit, enumerate(self.names)))
        # Every member's outcome, including a member that WAS selected before a peer failed.
        report = {}
        for n, r, _, _ in outcomes:
            if isinstance(r, ReferenceRefused):
                report[n] = (r.result, r.reply)
            elif isinstance(r, RtLinkError):
                report[n] = str(r)
            else:
                report[n] = ("selected", {k: r[k] for k in ("selected_ns", "deadline_ns", "state")})
        if any(not isinstance(r, dict) for _, r, _, _ in outcomes):
            raise GroupStartError(f"group {group_id} did not start", report)
        members = {}
        for n, r, sent, got in outcomes:
            if not (r["selected_ns"] > 0 and r["deadline_ns"] > 0):
                raise GroupStartError(f"group {group_id}: {n} reported no selection stamp", report)
            members[n] = MemberStart(n, reference_ids[n], r["selected_ns"], r["deadline_ns"], sent, got)
        return GroupStartRecord(group_id, domain, members)

    def cancel(self, group_id: int) -> dict:
        """Best effort on every member (a failing member does not stop the others); returns the errors."""
        errors = {}
        count = len(self.names)
        for index, name in enumerate(self.names):
            try:
                self.links[name].group_cancel(group_id, index, count)
            except Exception as exc:  # noqa: BLE001 - a failing member must not stop the others
                errors[name] = f"{type(exc).__name__}: {exc}"
        return errors

    def events(self) -> dict:
        """Unsolicited RT reports (a peer fault/cancel/loss retired that member), drained per member."""
        out = {}
        for name, link in self.links.items():
            items = []
            while link.group_events:
                items.append(link.group_events.popleft())
            out[name] = items
        return out
