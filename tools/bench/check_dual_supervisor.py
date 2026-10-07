"""Two supervisors, two fake RT servers, one pair coordinator over SdkArmHandle (dual-arm migration, independent mode).

    python tools/bench/check_dual_supervisor.py --server <rt build>/arm_rt_server

Each arm has its own fake RT server and its own supervisor process; the bimanual coordinator drives them only through
``ArmClient`` handles. Checks: a pair stages, releases and completes (both executions COMPLETED); a pair cancelled while
running is CANCELLED on both arms with the cancels confirmed; a member the supervisor path cannot carry (Cartesian-impedance
plan) rejects the pair and the partner's staged plan is cleared; a plan for another model is refused before anything is
staged (by the coordinator, and by the supervisor when the handle is called directly); a failing leg (disarm mid-flight)
faults the pair and cancels the partner. Unit cases cover conversion fidelity, observed_at, pruning, a lost staging reply and
repeated fault events. Fake RT only: nothing here is a timing or hardware claim (no group start yet, so release skew is the PC's).
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
from arm_control.sdk import ArmClient, Refused, SupervisorError  # noqa: E402
from arm_control.execution.contract import (  # noqa: E402
    Authority, Clock, Fault, Lifecycle, RobotIdentity, RobotState, Stamp, Start, Validity)
from arm_control.multi_arm.supervised import plan_body, plan_request  # noqa: E402

ARMS = ('arm_a', 'arm_b')


def _log_text(proc):
    # pread, not seek+read: the child shares this file description, so moving its offset could garble what it writes next.
    log = getattr(proc, '_log', None)
    if log is None:
        return ''
    try:
        return os.pread(log.fileno(), os.fstat(log.fileno()).st_size, 0).decode(errors='replace')
    except (OSError, ValueError) as exc:
        return f'<log unavailable: {exc}>'


def _tail(proc, lines=40):
    return ''.join(_log_text(proc).splitlines(True)[-lines:])


# The lines that name an expiry or a stall, wherever they sit in a long log (loop-gap lines can push them out of the tail).
KEY_LINES = ('expired', 'sender gap', 'reference diagnostics', 'fault', 'FAULT')


def start_arm(server, tmp: Path, name: str, extra_args=()):
    udp, tcp = free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM)
    rt = subprocess.Popen([server, '--backend', 'fake', '--n', '3', '--instance-id', name, '--joint-names',
                           ','.join(JOINTS), '--udp-port', str(udp), '--tcp-port', str(tcp), '--hold-ms', '100',
                           '--fault-ms', '600', '--enable-reference', *extra_args],
                          stdout=(rt_log := tempfile.TemporaryFile(mode='w+')), stderr=subprocess.STDOUT, text=True)
    rt._log = rt_log  # a file, not a pipe: nobody drains it, and a full pipe would block the child
    deadline = time.monotonic() + 10
    while not tcp_listening(tcp):
        assert rt.poll() is None, _tail(rt)
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
    entry = [sys.executable, '-m', 'arm_control.supervisor.service']
    if os.environ.get('SUPERVISOR_FAULTHANDLER'):  # SIGUSR1 dumps every thread's stack into the supervisor's log
        entry = [sys.executable, '-c', 'import faulthandler, signal, runpy, sys; '
                 'faulthandler.register(signal.SIGUSR1, all_threads=True); '
                 'sys.argv[0] = "arm_control.supervisor.service"; '
                 'runpy.run_module("arm_control.supervisor.service", run_name="__main__")']
    sup = subprocess.Popen([*entry, '--config', str(cfg),
                            '--socket', str(sock)], stdout=(sup_log := tempfile.TemporaryFile(mode='w+')),
                           stderr=subprocess.STDOUT, text=True,
                           env=dict(os.environ))
    sup._log = sup_log
    deadline = time.monotonic() + 30
    while True:
        assert sup.poll() is None, _tail(sup)
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


class FakeArm:
    """Just enough ArmClient for the handle's logic: scripted statuses, errors and a controllable clock."""
    extensions = {}
    session = 'fake-session'
    identity = RobotIdentity('fake', ('j1', 'j2', 'j3'), 'fake')

    def __init__(self):
        self.statuses, self.fault, self.errors, self.executions = {}, None, {}, 0
        self.age_s, self.disarmed, self.by_request, self.first_now_delay = .001, False, {}, 0.
        self.hold_on_error = False

    def now(self):
        if self.first_now_delay:
            delay, self.first_now_delay = self.first_now_delay, 0.
            time.sleep(delay)
        return Stamp(time.monotonic_ns(), Clock.PC, 'pc:fake')

    def state(self):
        now = time.monotonic_ns()
        return RobotState('fake', Validity.VALID, q=np.zeros(3), dq=np.zeros(3),
                          source=Stamp(now - int(self.age_s * 1e9), Clock.RT_HOST, 'rt:fake'),
                          received=Stamp(now - int(self.age_s * 1e9), Clock.PC, 'pc:fake'))

    def authority(self):
        if self.fault:
            return Authority(Lifecycle.FAULTED, 's', 1, fault=Fault('plant', self.fault, 'rt', 1))
        return Authority(Lifecycle.CLAIMED if self.disarmed else Lifecycle.ARMED, 's', 1)

    def status(self, execution_id):
        if 'status' in self.errors:
            raise self.errors['status']
        if execution_id not in self.statuses:
            raise Refused(f'unknown execution {execution_id!r}')
        return self.statuses[execution_id]

    def execute(self, request):
        error = self.errors.get('execute')
        if request.request_id in self.by_request and not (error and not self.hold_on_error):
            return self.statuses[self.by_request[request.request_id]]  # idempotent by request id, as the supervisor is
        if error and not self.hold_on_error:
            raise error
        self.executions += 1
        self.last_request = request
        eid = f'e{self.executions}'
        status = type('S', (), dict(execution_id=eid, phase=Phase.ADMITTED, terminal=False, reason=''))()
        self.statuses[eid], self.by_request[request.request_id] = status, eid
        if error:  # the supervisor took it but the reply was lost
            raise error
        return status

    def release(self, execution_id):
        return self.statuses[execution_id]

    def cancel(self, execution_id, reason='cancelled', timeout_s=2.):
        if 'cancel' in self.errors:
            raise self.errors['cancel']
        status = type('S', (), dict(execution_id=execution_id, phase=Phase.CANCELLED, terminal=True, reason=reason))()
        self.statuses[execution_id] = status
        return status


def plan_for_fake(plan_id):
    return ArmPlan('fake', f'{plan_id}:fake', pack_plan(
        plan_id=f'{plan_id}:fake', phase='x', gated=True, times=[0., 1.], positions=np.zeros((2, 3)),
        velocities=np.zeros((2, 3)), kp=[100.] * 3, kd=[20.] * 3, model_token='tok'))


def unit_cases():
    """Handle logic that a short real run cannot reach: conversion fidelity, observed_at against the coordinator clock,
    pruning of forgotten executions, fault events that repeat, and a staging reply that was lost."""
    plan = plan_for_fake('u1')
    body = plan_body(plan.payload)
    request, dropped = plan_request(body, 'fake', ('j1', 'j2', 'j3'))
    assert request.start is Start.ON_RELEASE and request.model_token == 'tok' and request.request_id == 'u1:fake'
    assert np.allclose(request.gains.kp, 100.) and np.allclose(request.gains.kd, 20.) and dropped == []
    with_completion = plan_body(pack_plan(plan_id='u2', phase='x', gated=True, times=[0., 1.], positions=np.zeros((2, 3)),
                                          velocities=np.zeros((2, 3)), kp=[1.] * 3, kd=[1.] * 3, model_token='tok',
                                          cartesian_poses=np.zeros((2, 7)),
                                          completion=dict(vel_tol=.02, goal_q=[0., 0., 0.], residual_max=.01)))
    request, dropped = plan_request(with_completion, 'fake', ('j1', 'j2', 'j3'))
    assert request.completion.vel_tol == .02 and request.completion.residual_max == .01 and dropped == ['cartesian_poses']
    for name, bad in (('empty kp and kd', dict(with_completion, kp=np.array([]), kd=np.array([]))),
                      ('empty kp', dict(with_completion, kp=np.array([]))), ('ungated', dict(with_completion, gated=False)),
                      ('unknown field', dict(with_completion, extra=1)),
                      ('cartesian', dict(with_completion, cartesian=dict(task_R=np.eye(3))))):
        try:
            plan_request(bad, 'fake', ('j1', 'j2', 'j3'))
        except ValueError:
            pass
        else:
            raise AssertionError(f'not refused: {name}')

    arm = FakeArm()
    handle = SdkArmHandle(arm, 'fake', model_token='tok')
    # observed_at is the sample's reception time on the coordinator's clock: never after a clock read taken once the
    # state has been read (the coordinator now reads `now` after the states), and the age is never negative.
    for _ in range(200):
        state = handle.state()
        after = time.monotonic()
        assert state.observed_at is not None and state.observed_at <= after, (state.observed_at, after)
        assert state.state_age_sec is not None and state.state_age_sec >= 0.
    # a faulted or disarmed robot is not ready; a fault freezes and carries its reason; a dropped field is recorded
    arm.disarmed = True
    assert handle.state().armed is False and handle.state().ready is False
    arm.disarmed = False
    arm.fault = 'plant fault'
    faulted = handle.state()
    assert faulted.frozen is True and faulted.fault_reason and 'plant fault' in faulted.fault_reason, faulted
    arm.fault = None
    handle.poll_events()
    handle.stage(ArmPlan('fake', 'u2', pack_plan(
        plan_id='u2', phase='x', gated=True, times=[0., 1.], positions=np.zeros((2, 3)), velocities=np.zeros((2, 3)),
        kp=[1.] * 3, kd=[1.] * 3, model_token='tok', cartesian_poses=np.zeros((2, 7)))))
    assert handle.dropped == {'u2': ['cartesian_poses']}, handle.dropped
    handle.cancel('x', plan_id='u2')
    handle.poll_events()
    # a slow first supervisor call does not bias the clock offset for the handle's life
    slow = FakeArm()
    slow.first_now_delay = .3
    slow_handle = SdkArmHandle(slow, 'fake', model_token='tok')
    observed = slow_handle.state().observed_at
    assert abs(time.monotonic() - observed) < .05, (time.monotonic() - observed)  # observed_at tracks the real clock
    # a released plan whose execution the supervisor forgot is a failed leg, not silence; a changed fault reason repeats
    fresh = SdkArmHandle(arm, 'fake', model_token='tok')
    fresh.stage(plan_for_fake('f1'))
    fresh.poll_events()
    fresh.execute('f1:fake')
    arm.statuses.pop(arm.by_request['f1:fake'])
    lost = [e for e in fresh.poll_events() if e['kind'] == 'leg_result']
    assert lost and lost[0]['ok'] is False and 'outcome unknown' in lost[0]['reason'], lost
    arm.fault = 'first reason'
    assert len([e for e in fresh.poll_events() if e['kind'] == 'fault']) == 1
    arm.fault = 'second reason'  # a changed reason without a recovery in between is a new event
    assert len([e for e in fresh.poll_events() if e['kind'] == 'fault']) == 1
    arm.fault = None
    fresh.poll_events()
    # at most 64 executions are remembered for lookups
    for i in range(70):
        fresh.stage(plan_for_fake(f'c{i}'))
        fresh.cancel('x', plan_id=f'c{i}:fake')
    fresh.poll_events()
    assert len(fresh._executions) <= 64, len(fresh._executions)
    # an execution the supervisor forgot is not 'loaded' (state stays usable), and a pruned plan costs nothing later
    handle.stage(plan)
    handle.execute('u1:fake')
    arm.statuses.clear()
    assert handle.state().running_plan_id is None and not handle._plans
    # a staging reply that was lost: not staged, and the cancel is NOT confirmed as cleared
    arm.errors['execute'] = SupervisorError('connection closed before the reply')
    handle.stage(plan_for_fake('u3'))
    assert handle.poll_events()[-1]['reason'].startswith('staging outcome unknown')
    arm.errors.clear()
    handle.cancel('x', plan_id='u3:fake')  # resubmits the same request (idempotent), then cancels it: truthful
    cleared = [e for e in handle.poll_events() if e['kind'] == 'plan_cleared']
    assert cleared and cleared[0]['ok'] is True, cleared
    assert arm.statuses[arm.by_request['u3:fake']].phase is Phase.CANCELLED
    # the supervisor took the plan but the reply was lost: the cancel finds it by request id and cancels it
    arm.errors['execute'], arm.hold_on_error = SupervisorError('connection closed before the reply'), True
    handle.stage(plan_for_fake('u3b'))
    assert arm.by_request['u3b:fake'] and not arm.statuses[arm.by_request['u3b:fake']].terminal
    arm.errors.clear()
    arm.hold_on_error = False
    held_before = arm.executions
    handle.cancel('x', plan_id='u3b:fake')
    assert arm.executions == held_before, 'resubmitting created a second execution'
    assert arm.statuses[arm.by_request['u3b:fake']].phase is Phase.CANCELLED
    assert [e for e in handle.poll_events() if e['kind'] == 'plan_cleared'][-1]['ok'] is True
    # a cancel that cannot reach the supervisor is NOT confirmed, and a refusal (not an unknown outcome) is cleared
    arm.errors['execute'] = SupervisorError('closed')
    handle.stage(plan_for_fake('u3c'))
    handle.poll_events()
    handle.cancel('x', plan_id='u3c:fake')
    assert [e for e in handle.poll_events() if e['kind'] == 'plan_cleared'][-1]['ok'] is False
    arm.errors['execute'] = Refused('not now')
    handle.stage(plan_for_fake('u3d'))
    handle.poll_events()
    handle.cancel('x', plan_id='u3d:fake')
    assert [e for e in handle.poll_events() if e['kind'] == 'plan_cleared'][-1]['ok'] is True
    arm.errors.clear()
    # a cancel the supervisor did not confirm is reported as not cleared; a confirmed one as cleared
    handle.stage(plan_for_fake('u4'))
    handle.poll_events()
    arm.errors['cancel'] = TimeoutError('link stall')
    handle.cancel('x', plan_id='u4:fake')
    assert handle.poll_events()[-1]['ok'] is False
    arm.errors.clear()
    handle.cancel('x', plan_id='u4:fake')
    assert handle.poll_events()[-1]['ok'] is True
    # a fault is an event each time it happens, not once per handle
    arm.fault = 'overload'
    assert [e['kind'] for e in handle.poll_events()] == ['fault']
    assert handle.poll_events() == []
    arm.fault = None
    handle.poll_events()
    arm.fault = 'overload again'
    assert [e['kind'] for e in handle.poll_events()] == ['fault']
    arm.fault = None
    # an unreachable supervisor is a state with a reason, not an exception out of the coordinator's pump
    arm.errors['status'] = SupervisorError('closed')
    handle.stage(plan_for_fake('u5'))
    arm.errors['execute'] = None
    arm.errors.pop('execute')
    state = handle.state()
    assert state.fault_reason is None or 'unreachable' in state.fault_reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    args = parser.parse_args()
    unit_cases()
    tmp = Path(tempfile.mkdtemp(prefix='dual_supervisor_'))
    procs, clients = [], []
    try:
        stacks = {}
        for name in ARMS:
            rt, sup, sock, token = start_arm(args.server, tmp, name)
            rt._name, sup._name = f'{name} arm_rt_server', f'{name} supervisor'
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
            eid = handles[n].execution_of(f'p1:{n}')
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
            eid = handles[n].execution_of(f'p2:{n}')
            assert arms[n].status(eid).phase is Phase.CANCELLED, (n, arms[n].status(eid))

        assert not [e for e in seen if e['kind'] == 'leg_result' and e['plan_id'].startswith('p2')], seen

        # A member the supervisor path cannot carry rejects the pair; the partner's staged plan is cleared.
        cart = dict(cartesian=dict(task_R=np.eye(3), kc=[1.] * 6, dc=[1.] * 6))
        stage('p3', **{b: cart})
        pump_until(coordinator, lambda: coordinator.lifecycle.terminal and not coordinator.pair_state('p3').cancellation_pending)
        assert coordinator.lifecycle is PairLifecycle.REJECTED and 'Cartesian' in coordinator.pair_state('p3').reason, \
            coordinator.pair_state('p3').to_payload()
        eid = handles[a].execution_of('p3:' + a)
        assert arms[a].status(eid).phase is Phase.CANCELLED, arms[a].status(eid)
        assert handles[b].execution_of('p3:' + b) is None

        # A plan for another model is refused before anything is staged.
        bad = plan_for(arms[b], 'p4', b, 'some-other-model', -.05)
        good = plan_for(arms[a], 'p4', a, tokens[a], .05)
        try:
            coordinator.stage_pair(BimanualPlan('p4', good, bad))
        except ValueError as exc:
            assert 'model token' in str(exc), exc
        else:
            raise AssertionError('a plan for another model was staged')
        assert handles[a].execution_of('p4:' + a) is None
        # Many consecutive pairs (the stale-state fault the reviewer saw after ~25 pairs): short moves, all complete.
        for i in range(30):
            plans = [plan_for(arms[n], f'm{i}', n, tokens[n], .004 if n == a else -.004, duration=.2) for n in ARMS]
            coordinator.stage_pair(BimanualPlan(f'm{i}', *plans))
            pump_until(coordinator, lambda: coordinator.lifecycle is PairLifecycle.READY)
            coordinator.execute_pair(f'm{i}')
            pump_until(coordinator, lambda: coordinator.lifecycle.terminal)
            assert coordinator.lifecycle is PairLifecycle.COMPLETE, (i, coordinator.pair_state(f'm{i}').to_payload())
        assert len(handles[a]._plans) <= 1, 'ended plans were not pruned'

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
        # Supervisor-side model enforcement, past the coordinator's own check: the handle stages a plan for another model.
        other = plan_for(arms[a], 'p6', a, 'some-other-model', .05)
        handles[a].stage(other)
        refusal = [e for e in handles[a].poll_events() if e['kind'] == 'plan_staged'][-1]
        assert refusal['ok'] is False and 'model' in refusal['reason'], refusal
        print('PASS: unit cases (conversion, observed_at, pruning, lost staging reply, fault events); two supervisors, one pair '
              'coordinator over SdkArmHandle: pair completes, mid-flight cancel confirmed on both arms, an uncarriable '
              'member rejects the pair and clears its partner, a foreign model is refused by the coordinator and by the '
              'supervisor, a failing leg faults the pair')
    except BaseException:
        # An intermittent fault (REFERENCE_EXPIRED once in 51 runs, 2026-10-07) is only diagnosable from the children's own
        # records: the supervisor's reference diagnostics (sender gaps, scheduling) and the RT server's fault journal detail.
        for proc in procs:
            name, text = getattr(proc, '_name', proc.args[0]), _log_text(proc)
            keys = [line for line in text.splitlines() if any(k in line for k in KEY_LINES)]
            print(f'--- {name}: key lines ---', *keys[-40:], f'--- {name} (last 120 lines) ---',
                  ''.join(text.splitlines(True)[-120:]), sep='\n', file=sys.stderr)
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


if __name__ == '__main__':
    main()
