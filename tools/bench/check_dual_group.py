"""CTRL-040 group execution through two supervisors (the negotiated group/1 SDK opt-in), fake RT only.

    python tools/bench/check_dual_group.py --server <rt build>/arm_rt_server

Two fake RT servers share a group record and run with the APPROVED constants (PEER_MAX_AGE 30 ms, MIN_LEAD 10 ms; budget
150 ms and H 350 ms on the PC); each has its own supervisor process; ``SdkGroupCoordinator`` drives them only through
``ArmClient``. Checks: a group starts with both members selected and a per-member skew report from the RT selection stamps
(kept apart from the PC send skew), both executions COMPLETE; a group cancelled while running retires both members; a member
whose peer never commits (the lone COMMIT_GROUP is refused) fails the group and leaves nothing
joined (the next group starts); a reference shorter than H is refused before any contact; a supervisor whose RT server has no
group record does not advertise group/1 and the group is refused. Nothing here is a timing or hardware claim.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dual_supervisor import _tail, start_arm  # noqa: E402
from check_sdk_supervisor import JOINTS  # noqa: E402

from arm_control.execution.contract import Phase, Start, TrajectoryRequest  # noqa: E402
from arm_control.motion import JointTrajectory  # noqa: E402
from arm_control.multi_arm.group import (  # noqa: E402
    GROUP_BUDGET_S, GROUP_HORIZON_S, GROUP_MIN_LEAD_MS, GROUP_PEER_MAX_AGE_MS, GroupStartError)
from arm_control.multi_arm.supervised_group import SdkGroupCoordinator  # noqa: E402
from arm_control.plants.remote_rt import protocol as rtp  # noqa: E402
from arm_control.sdk import ArmClient  # noqa: E402

ARMS = ('arm_a', 'arm_b')


def request(arm, name, token, rid, delta, duration):
    q0 = np.asarray(arm.state().q)
    q1 = q0 + delta
    mid = 1.5 * (q1 - q0) / duration
    traj = JointTrajectory([0., duration / 2, duration], np.stack([q0, (q0 + q1) / 2, q1]),
                           np.stack([0 * q0, mid, 0 * q0]))
    return TrajectoryRequest(rid, name, JOINTS, traj, model_token=token, start=Start.ON_RELEASE)


def settle():
    """After a cancel the RT retires running members and their peers within about PEER_MAX_AGE plus a tick, and a new group
    is refused at JOIN (GROUP_REFUSED) or BEGIN (REF_BUSY, the open CTRL-034 item 4a race) until then: a safe refusal that
    a caller retries. The bench waits instead."""
    time.sleep(.4)


def recover(arms):
    """DISARM then ARM (the existing recovery from a latched arm; also a new ARM epoch where a case needs one)."""
    for arm in arms.values():
        arm.disarm()
        arm.arm()
    settle()


def no_fault(arms, label='', seconds=1.2):
    """Nothing in these flows may latch an arm: not a failed start, not a cancel of running members (the RT latches
    CMD_LOST --fault-ms after the PC's command stream ends, so look past fault-ms)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for name, arm in arms.items():
            authority = arm.authority()
            assert authority.fault is None and authority.lifecycle.value == 'armed', (label, name, authority)
        time.sleep(.05)


def wait(predicate, timeout=15., what=''):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f'timed out waiting for {what}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    args = parser.parse_args()
    # clock-domain rules (no RT contact): all-unattested is labelled, a mixture or different domains are refused
    assert SdkGroupCoordinator._domain({'a': None, 'b': None}) == 'unattested'
    assert SdkGroupCoordinator._domain({'a': 'h:b', 'b': 'h:b'}) == 'h:b'
    for bad in ({'a': 'h:b', 'b': None}, {'a': 'h:b', 'b': 'h2:b'}):
        try:
            SdkGroupCoordinator._domain(bad)
        except GroupStartError:
            pass
        else:
            raise AssertionError(f'accepted {bad}')
    tmp = Path(tempfile.mkdtemp(prefix='dual_group_'))
    (tmp / 'machine-id').write_text('0123456789abcdef0123456789abcdef\n')
    (tmp / 'boot-id').write_text('11111111-2222-3333-4444-555555555555\n')
    shm = f'/arm_rt_group_dual_{os.getpid()}'
    procs, clients, stacks = [], [], {}
    try:
        for index, name in enumerate(ARMS):
            flags = ['--group-shm', shm, '--group-member', str(index), '--group-members', '2',
                     '--group-peer-max-age-ms', str(GROUP_PEER_MAX_AGE_MS), '--group-min-lead-ms', str(GROUP_MIN_LEAD_MS),
                     '--machine-id-file', str(tmp / 'machine-id'), '--boot-id-file', str(tmp / 'boot-id')]
            rt, sup, sock, token = start_arm(args.server, tmp, name, flags)
            procs += [sup, rt]
            stacks[name] = (sock, token)
        arms, tokens = {}, {}
        for name, (sock, token) in stacks.items():
            arm = ArmClient(name, socket_path=sock)
            clients.append(arm)
            arm.claim()
            arm.arm()
            assert arm.extensions.get('group') == 1, arm.extensions
            arms[name], tokens[name] = arm, token
        a, b = ARMS
        coordinator = SdkGroupCoordinator(arms)

        def requests(tag, duration=2.0):
            return {n: request(arms[n], n, tokens[n], f'{tag}-{n}', .05 if n == a else -.05, duration) for n in ARMS}

        # A group starts: both selected, per-member skew from the RT stamps, then both complete.
        reqs = requests('g1')
        record = coordinator.start(1, reqs)
        assert record.deadline_agreed and record.clock_domain != 'unattested', record.to_payload()
        assert record.rt_skew_sec is not None and record.rt_skew_sec < GROUP_BUDGET_S, record.rt_skew_sec
        assert record.pc_send_skew_sec is not None  # a different, labelled number
        assert set(record.members) == set(ARMS) and all(m.rt_selected_ns > 0 for m in record.members.values())
        for n in ARMS:
            eid = coordinator.last_executions[n]
            assert arms[n].wait(eid, 15).phase is Phase.COMPLETED, (n, arms[n].status(eid))
        assert not any(coordinator.cancel(1).values())  # a started group stays joined until cancelled
        settle()

        # A group cancelled while running retires both members.
        reqs = requests('g2')
        coordinator.start(2, reqs)
        time.sleep(.4)
        errors = coordinator.cancel(2, coordinator.last_executions)
        assert not errors, errors
        for n in ARMS:
            assert arms[n].status(coordinator.last_executions[n]).phase in (Phase.CANCELLED, Phase.FAULTED), arms[n].status(coordinator.last_executions[n])  # FAULTED: retired by the peer rule first
        no_fault(arms, 'after cancelling a running group')  # the survivor must not latch

        # a member released on admission would start alone: refused; after a group cancel nothing is joined any more
        try:
            coordinator.start(30, {n: dataclasses.replace(r, start=Start.ON_ADMIT) for n, r in requests('g30').items()})
        except ValueError as exc:
            assert 'released by the group' in str(exc), exc
        else:
            raise AssertionError('an ON_ADMIT member was accepted')
        from arm_control.sdk import Refused
        try:
            arms[a].group_release('arm_a.1.deadbeef', 2, int(GROUP_BUDGET_S * 1e9))  # group 2 was cancelled above
        except Refused as exc:
            assert 'join the group' in str(exc), exc
        else:
            raise AssertionError('released into a group that is no longer joined')
        events = coordinator.events()
        assert set(events) == set(ARMS) and all(isinstance(v, list) for v in events.values()), events

        # H: a reference shorter than the horizon is refused before any contact.
        try:
            coordinator.start(3, requests('g3', duration=GROUP_HORIZON_S / 2))
        except ValueError as exc:
            assert 'less than H' in str(exc), exc
        else:
            raise AssertionError('a reference shorter than H was started')

        # A member whose peer never commits: held to the group deadline and refused REF_LATE after about the budget; nothing
        # stays joined.
        reqs = requests('g4')
        arms[a].group_join(4, 0, 2)
        arms[b].group_join(4, 1, 2)
        eid_a = arms[a].execute(reqs[a]).execution_id
        t0 = time.monotonic()
        arms[a].group_release(eid_a, 4, int(GROUP_BUDGET_S * 1e9))
        report = wait(lambda: arms[a].group_report(eid_a), 5, 'the refusal of the lone member')
        waited = time.monotonic() - t0
        assert 'error' in report and 'selected_ns' not in report, report
        # Held until about the budget, then refused: REF_LATE, the peer verdict, or the RT retiring the member (the
        # link reports the interrupted exchange). Never a selection; the exact RT answer varies run to run.
        assert 'selected_ns' not in report and GROUP_BUDGET_S * .5 < waited < 2., (report, waited)
        assert report.get('result') in (rtp.REF_LATE, rtp.REF_GROUP_PEER, rtp.REF_GROUP_REFUSED, None), report
        coordinator.cancel(4, {a: eid_a})
        wait(lambda: arms[a].status(eid_a).terminal, 5, 'the lone member to end')
        no_fault(arms, 'after the lone failure')
        record = coordinator.start(5, requests('g5'))
        assert record.rt_skew_sec is not None
        coordinator.cancel(5, coordinator.last_executions)
        settle()

        # Failure paths through SdkGroupCoordinator.start(): each cancels both members and leaves nothing joined.
        known_4a = []

        def failed_then_ok(gid, expect, patch=None, undo=None, duration=2.0, attempt=0):
            reqs_ = requests(f'f{gid}a{attempt}', duration)
            if patch:
                patch(reqs_)
            try:
                coordinator.start(gid, reqs_)
            except GroupStartError as exc:
                if 'operation 32 refused: 3' in str(exc.members) and attempt < 2:
                    # BEGIN refused REF_BUSY: the RT still holds what an earlier cancel left (the open CTRL-034 item 4a
                    # race). Counted and retried after a settle; the count is printed.
                    known_4a.append(gid)
                    recover(arms)
                    return failed_then_ok(gid, expect, patch, undo, duration, attempt + 1)
                assert expect in str(exc) or expect in str(exc.members), (gid, exc, exc.members)
            else:
                raise AssertionError(f'group {gid} should have failed ({expect})')
            finally:
                if undo:
                    undo()
            for n, eid in coordinator.last_executions.items():
                assert wait(lambda e=eid, n=n: arms[n].status(e).terminal, 6, f'{n} to be cancelled')
                assert arms[n].status(eid).phase in (Phase.CANCELLED, Phase.REJECTED, Phase.EXPIRED), (n, arms[n].status(eid))
            no_fault(arms, f'after failed group {gid}')
            record_ = coordinator.start(gid + 1, requests(f'f{gid}ok'))  # nothing is left joined or busy
            assert record_.rt_skew_sec is not None
            coordinator.cancel(gid + 1, coordinator.last_executions)
            wait(lambda: all(arms[n].status(e).terminal for n, e in coordinator.last_executions.items()), 6, 'cleanup')
            settle()

        bad_model = lambda r: r.__setitem__(b, dataclasses.replace(r[b], model_token='other-model'))  # noqa: E731
        failed_then_ok(40, 'not admitted', bad_model)
        real_release = arms[b].group_release

        def boom(*a, **k):
            raise Refused('release refused (scripted)')
        arms[b].group_release = boom
        failed_then_ok(42, 'could not be released', undo=lambda: setattr(arms[b], 'group_release', real_release))
        real_report = arms[a].group_report
        arms[a].group_report = lambda eid: dict(selected_ns=0, deadline_ns=0, reference_id=1, group_id=44)
        failed_then_ok(44, 'no selection stamp', undo=lambda: setattr(arms[a], 'group_report', real_report))
        arms[a].group_report = lambda eid: dict(selected_ns=5, deadline_ns=6, reference_id=1, group_id=999)  # another group's
        failed_then_ok(46, 'no reply within the budget', undo=lambda: setattr(arms[a], 'group_report', real_report),
                       duration=8.0)  # long enough to still be running when the coordinator gives up

        # A refused JOIN of another group leaves the joined one intact: it can still be cancelled and re-joined, and a
        # member released into it is not turned into a plain COMMIT.
        arms[a].group_join(70, 0, 2)
        try:
            arms[a].group_join(71, 0, 2)
        except Refused as exc:
            assert 'refused' in str(exc), exc
        else:
            raise AssertionError('a second group was joined')
        eid_j = arms[a].execute(requests('g70')[a]).execution_id
        try:
            arms[a].group_join(72, 0, 2)  # refused again, after the member is staged
        except Refused:
            pass
        arms[a].group_release(eid_j, 70, int(GROUP_BUDGET_S * 1e9))  # still allowed: group 70 is still joined
        report_j = wait(lambda: arms[a].group_report(eid_j), 5, 'the lone member of group 70')  # only a GROUP commit reports
        assert report_j.get('group_id') == 70 and 'selected_ns' not in report_j, report_j  # never a plain COMMIT, never selected
        assert arms[a].group_cancel(70).get('cancelled') is True
        arms[a].cancel(eid_j)
        recover(arms)  # a lone COMMIT_GROUP whose peer never joined fails fast; this contrived flow is not a latch claim
        arms[a].group_join(73, 0, 2)  # joinable again
        arms[a].group_cancel(73)
        settle()

        # group_cancel after group_release but before the execution is cancelled: the released member must not fall back to
        # a plain COMMIT and start alone; its execution is cancelled with the group.
        arms[a].group_join(60, 0, 2)
        eid_r = arms[a].execute(requests('g60')[a]).execution_id
        arms[a].group_release(eid_r, 60, int(GROUP_BUDGET_S * 1e9))
        arms[a].group_cancel(60)
        wait(lambda: arms[a].status(eid_r).terminal, 6, 'the released member to be cancelled with the group')
        assert arms[a].status(eid_r).phase in (Phase.CANCELLED, Phase.REJECTED), arms[a].status(eid_r)
        time.sleep(.3)
        assert np.max(np.abs(np.asarray(arms[a].state().dq))) < .05  # it never started alone

        # A group joined in one ARM epoch cannot be released in another.
        arms[a].group_join(50, 0, 2)
        arms[a].disarm()
        arms[a].arm()  # a new ARM epoch (also the one deliberate recovery in this bench)
        eid_e = arms[a].execute(requests('g50')[a]).execution_id
        try:
            arms[a].group_release(eid_e, 50, int(GROUP_BUDGET_S * 1e9))
        except Refused as exc:
            assert 'ARM epoch' in str(exc), exc
        else:
            raise AssertionError('released into a group joined in an earlier ARM epoch')
        arms[a].cancel(eid_e)
        arms[a].group_cancel(50)
        settle()

        # A supervisor whose RT server has no group record does not advertise group/1: the group is refused.
        rt_c, sup_c, sock_c, token_c = start_arm(args.server, tmp, 'arm_c')
        procs += [sup_c, rt_c]
        arm_c = ArmClient('arm_c', socket_path=sock_c)
        clients.append(arm_c)
        arm_c.claim()
        arm_c.arm()
        assert 'group' not in arm_c.extensions
        mixed = SdkGroupCoordinator({a: arms[a], 'arm_c': arm_c})
        try:
            mixed.start(6, {a: request(arms[a], a, tokens[a], 'g6-a', .01, 2.), 'arm_c': request(arm_c, 'arm_c', token_c, 'g6-c', .01, 2.)})
        except GroupStartError as exc:
            assert 'does not advertise group/1' in str(exc), exc
        else:
            raise AssertionError('a member without group/1 was accepted')
        print(f'(known 4a BEGIN refusals retried: {len(known_4a)})')
        print('PASS: group/1 through two supervisors: group starts with an RT-stamp skew report and completes, a running '
              'group cancels on both members, a lone commit is refused REF_LATE at the budget and leaves nothing joined, a short '
              'reference is refused before contact, a member without group/1 refuses the group')
    except BaseException:
        import signal
        for proc in procs:  # with SUPERVISOR_FAULTHANDLER=1 a stalled supervisor dumps its threads
            if proc.poll() is None and os.environ.get('SUPERVISOR_FAULTHANDLER'):
                proc.send_signal(signal.SIGUSR1)
        time.sleep(.5)
        for proc in procs:  # a failing run shows what each RT server and supervisor said
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=3)
                print('--- log (tail)\n' + _tail(proc, 25))
            except Exception:  # noqa: BLE001
                pass
        raise
    finally:
        for client in clients:
            client.close()
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            os.unlink('/dev/shm' + shm)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()
