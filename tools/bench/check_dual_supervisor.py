"""Two supervisors, two fake RT servers, one pair coordinator over SdkArmHandle (dual-arm migration, independent mode).

    python tools/bench/check_dual_supervisor.py --server <rt build>/arm_rt_server

Each arm has its own fake RT server and its own supervisor process; the bimanual coordinator drives them only through
``ArmClient`` handles. Checks: a pair stages, releases and completes (both executions COMPLETED); a pair cancelled while
running is CANCELLED on both arms with the cancels confirmed; a member the supervisor path cannot carry (Cartesian-impedance
plan) rejects the pair and the partner's staged plan is cleared; a plan for another model is refused before anything is
staged. Fake RT only: nothing here is a timing or hardware claim (no group start yet, so release skew is the PC's).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sdk_supervisor import JOINTS, free_port, tcp_listening, write_config  # noqa: E402

from arm_control.contracts.motion import pack_plan  # noqa: E402
from arm_control.multi_arm.coordinator import BimanualCoordinator  # noqa: E402
from arm_control.multi_arm.supervised import SdkArmHandle  # noqa: E402
from arm_control.multi_arm.types import ArmPlan, BimanualPlan, PairLifecycle  # noqa: E402
from arm_control.execution.contract import Phase  # noqa: E402
from arm_control.sdk import ArmClient  # noqa: E402

ARMS = ('arm_a', 'arm_b')


def start_arm(server, tmp: Path, name: str):
    udp, tcp = free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM)
    rt = subprocess.Popen([server, '--backend', 'fake', '--n', '3', '--instance-id', name, '--joint-names',
                           ','.join(JOINTS), '--udp-port', str(udp), '--tcp-port', str(tcp), '--hold-ms', '100',
                           '--fault-ms', '600', '--enable-reference'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 10
    while not tcp_listening(tcp):
        assert rt.poll() is None, rt.stdout.read()
        assert time.monotonic() < deadline, f'{name}: arm_rt_server did not listen'
        time.sleep(0.02)
    workdir = tmp / name
    workdir.mkdir()
    cfg = write_config(workdir, udp, tcp)
    cfg.write_text(cfg.read_text().replace('sdk_arm', name))
    gravity = dict(instance_id=name, revision=1, vector=[0., 0., -9.81], units='m/s^2', frame='base',
                   source=dict(type='nominal_assumption', evidence='analytical fixture'),
                   reviewed_against=dict(bundle_id='fixture', revision=1, installation_epoch='fixture',
                                         model_sha256=hashlib.sha256((workdir / 'three.urdf').read_bytes()).hexdigest(),
                                         base_sha256=hashlib.sha256(b'identity base fixture').hexdigest()))
    digest = hashlib.sha256(json.dumps(gravity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    token = f'analytical-fixture:g{digest}'
    with cfg.open('a') as f:
        f.write('model: ' + json.dumps(dict(token=token, bundle_id='fixture', revision=1, gravity=gravity)) + '\n')
    sock = workdir / f'{name}.sock'
    sup = subprocess.Popen([sys.executable, '-m', 'arm_control.supervisor.service', '--config', str(cfg),
                            '--socket', str(sock)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           env=dict(os.environ))
    deadline = time.monotonic() + 30
    while True:
        assert sup.poll() is None, sup.stdout.read()
        assert time.monotonic() < deadline, f'{name}: supervisor did not come up'
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(sock))
            break
        except OSError:
            time.sleep(0.05)
        finally:
            probe.close()
    return rt, sup, sock, token


def plan_for(arm: ArmClient, pair: str, name: str, token: str, delta: float, duration: float = 1.5, **extra):
    q0 = np.asarray(arm.state().q)
    q1 = q0 + delta
    mid = 1.5 * (q1 - q0) / duration
    payload = pack_plan(plan_id=f'{pair}:{name}', phase='paired_move', gated=True,
                        times=[0., duration / 2, duration], positions=np.stack([q0, (q0 + q1) / 2, q1]),
                        velocities=np.stack([0 * q0, mid, 0 * q0]), kp=[100.] * 3, kd=[20.] * 3,
                        model_token=token, **extra)
    return ArmPlan(name, f'{pair}:{name}', payload)


def pump_until(coordinator, predicate, timeout=15.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        coordinator.pump()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f'timed out in {coordinator.lifecycle}: {coordinator.state().pair.to_payload()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='dual_supervisor_'))
    procs, clients = [], []
    try:
        stacks = {}
        for name in ARMS:
            rt, sup, sock, token = start_arm(args.server, tmp, name)
            procs += [sup, rt]
            stacks[name] = (sock, token)
        handles = {}
        for name, (sock, token) in stacks.items():
            arm = ArmClient(name, socket_path=sock)
            clients.append(arm)
            arm.claim()
            arm.arm()
            handles[name] = SdkArmHandle(arm, name, model_token=token)
        arms = {name: h._arm for name, h in handles.items()}
        seen = []
        for h in handles.values():
            real_poll = h.poll_events
            h.poll_events = (lambda real=real_poll: [seen.append(e) or e for e in real()])
        coordinator = BimanualCoordinator(handles, staging_timeout_s=5., state_timeout_s=2.)
        a, b = ARMS
        tokens = {name: stacks[name][1] for name in ARMS}

        def stage(pair, **extra_for):
            plans = [plan_for(arms[n], pair, n, tokens[n], .05 if n == a else -.05, **extra_for.get(n, {}))
                     for n in ARMS]
            coordinator.stage_pair(BimanualPlan(pair, *plans))

        # A pair stages, releases and completes: both executions COMPLETED.
        stage('p1')
        pump_until(coordinator, lambda: coordinator.lifecycle is PairLifecycle.READY)
        coordinator.execute_pair('p1')
        pump_until(coordinator, lambda: coordinator.lifecycle.terminal)
        state = coordinator.pair_state('p1')
        assert coordinator.lifecycle is PairLifecycle.COMPLETE and state.ok(), state.to_payload()
        for n in ARMS:
            eid = handles[n]._plans[f'p1:{n}']['execution']
            assert arms[n].status(eid).phase is Phase.COMPLETED, (n, arms[n].status(eid))
        assert state.release_skew_sec is not None  # the PC's release skew, labelled as such (no RT group start here)

        # A pair cancelled while running: CANCELLED on both arms, cancels confirmed.
        stage('p2')
        pump_until(coordinator, lambda: coordinator.lifecycle is PairLifecycle.READY)
        coordinator.execute_pair('p2')
        time.sleep(.4)
        coordinator.cancel_pair('operator')
        pump_until(coordinator, lambda: not coordinator.pair_state('p2').cancellation_pending)
        assert coordinator.lifecycle is PairLifecycle.CANCELLED
        for n in ARMS:
            eid = handles[n]._plans[f'p2:{n}']['execution']
            assert arms[n].status(eid).phase is Phase.CANCELLED, (n, arms[n].status(eid))

        assert not [e for e in seen if e['kind'] == 'leg_result' and e['plan_id'].startswith('p2')], seen

        # A member the supervisor path cannot carry rejects the pair; the partner's staged plan is cleared.
        cart = dict(cartesian=dict(task_R=np.eye(3), kc=[1.] * 6, dc=[1.] * 6))
        stage('p3', **{b: cart})
        pump_until(coordinator, lambda: coordinator.lifecycle.terminal and not coordinator.pair_state('p3').cancellation_pending)
        assert coordinator.lifecycle is PairLifecycle.REJECTED and 'Cartesian' in coordinator.pair_state('p3').reason, \
            coordinator.pair_state('p3').to_payload()
        eid = handles[a]._plans['p3:' + a]['execution']
        assert arms[a].status(eid).phase is Phase.CANCELLED, arms[a].status(eid)
        assert 'p3:' + b not in handles[b]._plans

        # A plan for another model is refused before anything is staged.
        bad = plan_for(arms[b], 'p4', b, 'some-other-model', -.05)
        good = plan_for(arms[a], 'p4', a, tokens[a], .05)
        try:
            coordinator.stage_pair(BimanualPlan('p4', good, bad))
        except ValueError as exc:
            assert 'model token' in str(exc), exc
        else:
            raise AssertionError('a plan for another model was staged')
        assert 'p4:' + a not in handles[a]._plans
        # A leg that fails mid-flight faults the pair and the partner is cancelled: the result is not always ok.
        for n in ARMS:
            arms[n].arm() if arms[n].authority().lifecycle.value != 'armed' else None
        stage('p5')
        pump_until(coordinator, lambda: coordinator.lifecycle is PairLifecycle.READY)
        coordinator.execute_pair('p5')
        time.sleep(.3)
        arms[b].disarm()  # the supervisor cancels b's execution: its leg_result must be a failure
        pump_until(coordinator, lambda: coordinator.lifecycle.terminal and not coordinator.pair_state('p5').cancellation_pending)
        assert coordinator.lifecycle is PairLifecycle.FAULTED and coordinator.pair_state('p5').faulted_arm == b, \
            coordinator.pair_state('p5').to_payload()
        failed = [e for e in seen if e['kind'] == 'leg_result' and e['plan_id'] == 'p5:' + b]
        assert failed and failed[0]['ok'] is False, seen
        print('PASS: two supervisors, one pair coordinator over SdkArmHandle: pair completes, mid-flight cancel confirmed '
              'on both arms, an uncarriable member rejects the pair and clears its partner, a foreign model is refused')
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


if __name__ == '__main__':
    main()
