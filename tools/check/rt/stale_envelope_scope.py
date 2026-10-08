"""Fake-RT regression: a prior session's native envelope cannot own a new ARM.

Each repetition selects the same (reference_id, revision) in two sessions on
one server. Session A sends one accepted envelope, then disarms. Session B
commits after that packet is stale and sends none: B must stay RUNNING until
its own 100 ms first-envelope deadline, then retire. No physical backend is
permitted. The default 50 repetitions are the offline collision check.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as p
from arm_control.plants.remote_rt.client import RtBackend, RtConfig


def unused_port(kind):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def selected_state(rt, reference_id, *, timeout=0.35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state, received = rt.latest_state()
        if (state is not None and state.session_id == rt.session_id
                and state.arm_epoch == rt._arm_epoch and state.reference
                and state.reference['reference_id'] == reference_id
                and state.reference['revision'] == 1):
            return state, received, time.monotonic()
        time.sleep(.001)
    raise AssertionError(f'no selected state for reference {reference_id}')


def open_fake(config):
    rt = RtBackend(config, ['j0', 'j1', 'j2'])
    rt.open()
    try:
        if rt.backend_name != 'fake' or rt.hold_ms != 100:
            raise AssertionError('bench refuses non-fake or non-100-ms server')
        if not rt.negotiate_reference('stale-envelope-scope-fixture'):
            raise AssertionError('server lacks negotiated native reference')
        rt.enable_all()
        return rt
    except BaseException:
        rt.close()
        raise


def curve():
    return JointTrajectory(np.array([0., 2.]), np.zeros((2, 3)), np.zeros((2, 3)))


def commit_async(rt, reference_id, worker):
    upload = rt.upload_reference(curve(), reference_id=reference_id, terminal=True)
    return worker.submit(rt.commit_reference, upload, reference_id=reference_id)


def repetition(config, reference_id):
    command = dict(position=np.zeros(3), velocity=np.zeros(3),
                   torque=np.zeros(3), kp=np.ones(3) * 2, kd=np.ones(3) * .1)
    a = open_fake(config)
    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            reply = commit_async(a, reference_id, worker)
            state, received, _ = selected_state(a, reference_id)
            assert state.reference['status'] == p.REFERENCE_RUNNING, state.reference
            seq = a.apply_reference_envelope(
                command, reference_id=reference_id, revision=1,
                phase=state.reference['phase'], sample_received_at=received,
                valid_for_s=.1, phase_window_s=.1)
            assert seq is not None
            deadline = time.monotonic() + .1
            while time.monotonic() < deadline:
                state, _ = a.latest_state()
                if (state and state.session_id == a.session_id
                        and state.arm_epoch == a._arm_epoch
                        and state.last_cmd_seq == seq):
                    break
                time.sleep(.001)
            else:
                raise AssertionError('A envelope never reached RT state/APPLIED')
            assert reply.result(timeout=2)['result'] == p.REF_OK
    finally:
        a.close()  # confirms DISARM before a second control session exists

    # The mailbox is persistent but udp_rx writes nothing while disarmed.
    # Age A's packet beyond the 100 ms ceiling before B commits.
    time.sleep(.12)
    b = open_fake(config)
    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            upload = b.upload_reference(curve(), reference_id=reference_id, terminal=True)
            committed = time.monotonic()
            reply = worker.submit(b.commit_reference, upload, reference_id=reference_id)
            state, _, first = selected_state(b, reference_id)
            if state.reference['status'] != p.REFERENCE_RUNNING:
                raise AssertionError(
                    f'B first state {state.reference["status"]} at '
                    f'{(first-committed)*1000:.1f} ms; stale A envelope reused')
            # No B envelope exists: only B's own admission clock may retire it.
            while time.monotonic() - committed < .04:
                state, _ = b.latest_state()
                assert state.reference['status'] == p.REFERENCE_RUNNING, state.reference
                time.sleep(.001)
            deadline = committed + .16
            while time.monotonic() < deadline:
                state, _ = b.latest_state()
                if state.reference['status'] == p.REFERENCE_CANCELLED:
                    retired = time.monotonic()
                    break
                time.sleep(.001)
            else:
                raise AssertionError('B did not retire at its own first-envelope limit')
            assert .075 <= retired - committed <= .16, retired - committed
            assert reply.result(timeout=2)['result'] == p.REF_OK
            return (first - committed) * 1000, (retired - committed) * 1000
    finally:
        b.close()


@contextmanager
def target(args):
    if args.remote:
        if not (args.udp_port and args.tcp_port and args.instance_id) or args.binary:
            raise ValueError('--remote requires ports and instance ID, without a binary')
        yield RtConfig(host=args.remote, udp_port=args.udp_port,
                       tcp_port=args.tcp_port, instance_id=args.instance_id)
        return
    if not args.binary:
        raise ValueError('provide a local fake binary or --remote HOST')
    udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    with tempfile.TemporaryFile(mode='w+') as log:
        proc = subprocess.Popen([
            str(Path(args.binary).resolve()), '--backend', 'fake', '--n', '3',
            '--instance-id', 'bench_fake_scope', '--joint-names', 'j0,j1,j2',
            '--bind', '127.0.0.1', '--udp-port', str(udp), '--tcp-port', str(tcp),
            '--hold-ms', '100', '--fault-ms', '1000', '--enable-reference',
        ], stdout=log, stderr=log)
        try:
            time.sleep(.2)
            yield RtConfig(host='127.0.0.1', udp_port=udp, tcp_port=tcp,
                           instance_id='bench_fake_scope')
        finally:
            proc.terminate()
            proc.wait(3)
            log.seek(0)
            print(log.read()[-3000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', nargs='?')
    parser.add_argument('--remote')
    parser.add_argument('--udp-port', type=int)
    parser.add_argument('--tcp-port', type=int)
    parser.add_argument('--instance-id')
    parser.add_argument('--repetitions', type=int, default=50)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    with target(args) as config:
        for n in range(1, args.repetitions + 1):
            first_ms, retired_ms = repetition(config, n)
            print(f'PASS collision {n}/{args.repetitions}: '
                  f'B first RUNNING {first_ms:.1f} ms, '
                  f'own-limit retirement {retired_ms:.1f} ms', flush=True)


if __name__ == '__main__':
    main()
