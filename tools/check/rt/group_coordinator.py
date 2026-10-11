"""GroupCoordinator (CTRL-040 PC side) against two fake RT servers started with the APPROVED constants.

Approved 2026-10-01: PEER_MAX_AGE 30 ms and MIN_LEAD 10 ms (server flags), BUDGET 150 ms and H 350 ms (PC).
Cases: a normal start with the per-member skew report built from the RT selection stamps (and kept apart from
the PC send skew); a peer that never commits (REF_LATE after about BUDGET); a budget below MIN_LEAD (refused
at commit, before any wait); a reference shorter than H (refused before any RT contact); members on different
RT clock domains (refused: stamps are not comparable, the card's "host boot mismatch" case); peer process
loss detected within the approved PEER_MAX_AGE plus a tick (measured, and checked against hold_ms); the
event reaching the survivor's client. Fake backends only: nothing here says anything about FCI timing or the
FR3 host.

    PYTHONPATH=. python tools/check/rt/group_coordinator.py path/to/arm_rt_server
"""
from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for path in (_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tempfile
import time

from arm_control.multi_arm.group import (
    GROUP_BUDGET_S, GROUP_HORIZON_S, GROUP_MIN_LEAD_MS, GROUP_PEER_MAX_AGE_MS,
    GroupCoordinator, GroupStartError,
)
from arm_control.plants.remote_rt import protocol as p
from group_execution import curve, event_of, pair, running, wait_for

APPROVED = dict(peer_max_age_ms=GROUP_PEER_MAX_AGE_MS, min_lead_ms=GROUP_MIN_LEAD_MS)


def approved_pair(binary, tmp, extra=None):
    kwargs = {"all": dict(APPROVED)}
    kwargs.update(extra or {})
    return pair(binary, tmp, member_kwargs=kwargs)


def ref_id(m):
    s = m.state()
    return s.reference['reference_id'] if s is not None and s.reference else 0


def coordinator(members, **kw):
    return GroupCoordinator({f"arm{m.index}": m.link for m in members}, **kw)


def happy(binary, tmp, skews):
    with approved_pair(binary, tmp) as (a, b):
        c = coordinator((a, b))
        record = c.start(1, {"arm0": curve(1.0), "arm1": curve(1.0)}, {"arm0": 1, "arm1": 1})
        assert record.deadline_agreed and record.clock_domain != "unattested", record
        assert record.rt_skew_sec is not None and record.rt_skew_sec < GROUP_BUDGET_S  # RT stamps, host clock
        assert record.pc_send_skew_sec is not None  # a different number, labelled as PC barrier skew
        payload = record.to_payload()
        assert set(payload["rt_selected_ns"]) == {"arm0", "arm1"} and payload["rt_skew_sec"] == record.rt_skew_sec
        wait_for(lambda: running(a) and running(b))
        skews.append(record.rt_skew_sec * 1e3)
        assert c.cancel(1) == {}
        wait_for(lambda: not running(a) and not running(b))


def peer_never_commits(binary, tmp):
    with approved_pair(binary, tmp) as (a, b):
        # Only one member is driven: the other joined but never commits.
        from arm_control.plants.remote_rt.client import ReferenceRefused
        for m in (a, b):
            m.link.group_join(2, m.index, m.count)
        ua = a.upload()
        started = time.monotonic()
        try:
            a.link.group_commit(ua, reference_id=a.ref_id, group_id=2, member_index=0,
                                budget_ns=int(GROUP_BUDGET_S * 1e9))
        except ReferenceRefused as exc:
            assert exc.result == p.REF_LATE
            waited = time.monotonic() - started
            assert GROUP_BUDGET_S - .03 <= waited < GROUP_BUDGET_S + .5, waited
        else:
            raise AssertionError("commit without a committing peer accepted")


def budget_below_min_lead(binary, tmp):
    with approved_pair(binary, tmp) as (a, b):
        c = coordinator((a, b), budget_s=GROUP_MIN_LEAD_MS / 1e3 / 2)  # 5 ms < MIN_LEAD 10 ms
        started = time.monotonic()
        try:
            c.start(3, {"arm0": curve(1.0), "arm1": curve(1.0)}, {"arm0": 1, "arm1": 1})
        except GroupStartError as exc:
            results = [v[0] for v in exc.members.values()]
            # the member whose commit came second is refused REF_LATE; the other sees its peer cancel
            assert p.REF_LATE in results and set(results) <= {p.REF_LATE, p.REF_GROUP_PEER}, exc.members
            assert time.monotonic() - started < GROUP_BUDGET_S + 1.0
        else:
            raise AssertionError("a budget below MIN_LEAD was accepted")
        assert not running(a) and not running(b)


def failure_paths_leave_nothing_joined(binary, tmp):
    """An error that is not a link error must still cancel the group (round-1 review): a reference the RT
    refuses only at upload (too many knots on the second member) is caught before any RT contact; a fault
    injected after both joined (an exception in the commit path) leaves no member joined or running."""
    import numpy as np
    from arm_control.motion import JointTrajectory
    with approved_pair(binary, tmp) as (a, b):
        c = coordinator((a, b))
        many = JointTrajectory(np.linspace(0, 1, 200), np.zeros((200, 3)), np.zeros((200, 3)))
        try:
            c.start(7, {"arm0": curve(1.0), "arm1": many}, {"arm0": 1, "arm1": 1})
        except ValueError:
            pass
        else:
            raise AssertionError("a 200-knot reference was accepted")
        assert ref_id(a) == 0 and ref_id(b) == 0  # refused before any RT contact
        # After both joined: an unexpected exception (not an RtLinkError) in the upload path.
        real = b.link.upload_reference

        def boom(*args, **kwargs):
            raise KeyError('injected')
        b.link.upload_reference = boom
        try:
            c.start(8, {"arm0": curve(1.0), "arm1": curve(1.0)}, {"arm0": 1, "arm1": 1})
        except KeyError:
            pass
        else:
            raise AssertionError("the injected error vanished")
        b.link.upload_reference = real
        # Nothing stayed joined: a NEW group starts at once, and its outcome lists every member.
        record = c.start(9, {"arm0": curve(1.0), "arm1": curve(1.0)}, {"arm0": 2, "arm1": 2})  # ids never reused
        assert record.deadline_agreed
        c.cancel(9)
        wait_for(lambda: not running(a) and not running(b))
        # An unexpected exception AFTER a member was selected (a proxy performs the real group COMMIT for
        # arm0, so it is running, then raises): both members end stopped and free for a new group.
        real_commit = a.link.group_commit

        def commit_then_boom(*args, **kwargs):
            real_commit(*args, **kwargs)
            raise KeyError('after selection')
        a.link.group_commit = commit_then_boom
        try:
            c.start(10, {"arm0": curve(2.0), "arm1": curve(2.0)}, {"arm0": 3, "arm1": 3})
        except (KeyError, GroupStartError):
            pass
        else:
            raise AssertionError("the injected error vanished")
        a.link.group_commit = real_commit
        # Nothing runs any more; give the RT a moment to finish retiring (a start right after a cancel can be
        # refused by BEGIN while the RT still holds the old reference: the known CTRL-034 item 4a).
        wait_for(lambda: not running(a) and not running(b))
        time.sleep(.3)
        record = c.start(11, {"arm0": curve(1.0), "arm1": curve(1.0)}, {"arm0": 4, "arm1": 4})
        assert record.deadline_agreed
        c.cancel(11)


def short_reference_and_clock_domains(binary, tmp):
    with approved_pair(binary, tmp) as (a, b):
        c = coordinator((a, b))
        try:  # shorter than H: refused before any RT contact
            c.start(4, {"arm0": curve(GROUP_HORIZON_S - .1), "arm1": curve(1.0)}, {"arm0": 1, "arm1": 1})
        except ValueError as exc:
            assert "H" in str(exc)
        else:
            raise AssertionError("a reference shorter than H was accepted")
        assert ref_id(a) == 0 and ref_id(b) == 0
    other_host = tmp / "machine-id-other"
    other_host.write_text("fedcba9876543210fedcba9876543210\n")
    with approved_pair(binary, tmp, {1: {"machine_id_file": other_host}}) as (a, b):
        assert a.link.clock_domain != b.link.clock_domain
        c = coordinator((a, b))
        try:
            c.start(5, {"arm0": curve(1.0), "arm1": curve(1.0)}, {"arm0": 1, "arm1": 1})
        except GroupStartError as exc:
            assert "one RT clock domain" in str(exc), exc
        else:
            raise AssertionError("members on different clock domains were accepted")
        assert ref_id(a) == 0  # refused before any RT contact


def peer_loss_within_approved_age(binary, tmp, latencies):
    with approved_pair(binary, tmp) as (a, b):
        c = coordinator((a, b))
        c.start(6, {"arm0": curve(3.0), "arm1": curve(3.0)}, {"arm0": 1, "arm1": 1})
        wait_for(lambda: running(a) and running(b))
        killed = time.monotonic()
        b.proc.kill()  # by PID
        b.proc.wait(3)
        # Enforcement is the survivor's servo tick retiring its reference: seen on its state stream.
        wait_for(lambda: not running(a), 2.0)
        enforced = time.monotonic() - killed
        e = event_of(a)  # the report to the PC (control thread, polls at 1 ms while in a group)
        assert e["peer_verdict"] == p.PEER_STALE, e
        reported = e["received_at"] - killed
        # PEER_MAX_AGE 30 ms + one tick + state-stream period (4 ms) must sit well inside hold_ms (100 ms).
        assert enforced < 0.080, enforced
        assert reported < 0.060, reported  # with the 100 ms idle control poll this was ~95-100 ms
        latencies.append((enforced * 1e3, reported * 1e3))
        assert not a.state().faulted


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "rt/build/arm_rt_server")
    skews, latencies = [], []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "machine-id").write_text("0123456789abcdef0123456789abcdef\n")
        (tmp / "boot-id").write_text("11111111-2222-3333-4444-555555555555\n")
        for _ in range(3):
            happy(binary, tmp, skews)
        peer_never_commits(binary, tmp)
        budget_below_min_lead(binary, tmp)
        short_reference_and_clock_domains(binary, tmp)
        failure_paths_leave_nothing_joined(binary, tmp)
        for _ in range(3):
            peer_loss_within_approved_age(binary, tmp, latencies)
    print(f"RT selection skew (ms, RT host clock, fake servers): {[round(s, 3) for s in skews]}")
    print("peer loss, kill to (enforced on the survivor's state stream, reported to its client), ms: "
          f"{[(round(a, 1), round(b, 1)) for a, b in latencies]}")
    print("PASS: GroupCoordinator with the approved constants: start record from RT stamps, peer that never "
          "commits, budget below MIN_LEAD, reference shorter than H, mixed clock domains, peer loss within "
          "hold_ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
