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
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dual_supervisor import start_arm  # noqa: E402
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

        # A group cancelled while running retires both members.
        reqs = requests('g2')
        coordinator.start(2, reqs)
        time.sleep(.4)
        errors = coordinator.cancel(2, coordinator.last_executions)
        assert not errors, errors
        for n in ARMS:
            assert arms[n].status(coordinator.last_executions[n]).phase is Phase.CANCELLED

        # a member released on admission would start alone: refused; after a group cancel nothing is joined any more
        import dataclasses
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

        # A member that never commits: its peer is refused (NOT_ADMITTED here: through a supervisor a reference is uploaded
        # only after release, so the peer has staged nothing) and the group fails; nothing stays joined.
        reqs = requests('g4')
        arms[a].group_join(4, 0, 2)
        arms[b].group_join(4, 1, 2)
        eid_a = arms[a].execute(reqs[a]).execution_id
        t0 = time.monotonic()
        arms[a].group_release(eid_a, 4, int(GROUP_BUDGET_S * 1e9))
        report = wait(lambda: arms[a].group_report(eid_a), 5, 'the refusal of the lone member')
        assert 'error' in report and 'selected_ns' not in report, report
        assert time.monotonic() - t0 < 2., time.monotonic() - t0
        assert report['result'] in (rtp.REF_NOT_ADMITTED, rtp.REF_LATE), report  # the peer never staged/committed
        coordinator.cancel(4, {a: eid_a})
        wait(lambda: arms[a].status(eid_a).terminal, 5, 'the lone member to end')
        record = coordinator.start(5, requests('g5'))
        assert record.rt_skew_sec is not None
        coordinator.cancel(5, coordinator.last_executions)

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
        print('PASS: group/1 through two supervisors: group starts with an RT-stamp skew report and completes, a running '
              'group cancels on both members, a lone commit is refused and leaves nothing joined, a short '
              'reference is refused before contact, a member without group/1 refuses the group')
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
