"""SupervisedHardware, the dual console's runtime over two arm supervisors, against two fake RT servers.

    python tools/check/check_supervised_hardware.py --server <rt build>/arm_rt_server

The runtime surface the console uses, driven through the real stack (fake RT -> supervisor process -> ArmClient ->
SdkArmHandle): construction claims nothing; connect() claims and waits for fresh measured feedback and fresh Hands;
ARM/DISARM do not block the caller, a pending request holds readiness back, a refused one is the arm's fault until the next
succeeds; a pair plan runs through ``BimanualCoordinator`` over ``runtime.handles``; stop() and close() leave both arms
disarmed; shared-object coupling is refused. Hands are a local double and the scene a stub: the console-level checks stay in
check_dual_hardware (direct links) until a supervisor deployment exists for the real pair. Nothing here is a hardware claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dual_supervisor import ARMS, plan_for, start_arm  # noqa: E402
from check_sdk_supervisor import JOINTS  # noqa: E402

from arm_control.execution.contract import Lifecycle  # noqa: E402
from arm_control.multi_arm import diagnostics
from arm_control.multi_arm.coordinator import BimanualCoordinator  # noqa: E402
from arm_control.multi_arm.supervised_hardware import SupervisedHardware  # noqa: E402
from arm_control.multi_arm.types import BimanualPlan, PairLifecycle  # noqa: E402
from arm_control.supervisor.runtime import Refused as ServiceRefused, RobotRuntime  # noqa: E402
from arm_control.supervisor.service import dispatch  # noqa: E402
from arm_control.sdk import ArmClient, Refused, SupervisorError  # noqa: E402


class FakeHand:
    def __init__(self, *_):
        self.measured, self.cancels, self.closed = True, 0, False

    def start(self):
        pass

    def snapshot(self):
        return dict(width=.08, measured=self.measured, available=True, active_stop=True, busy=False)

    def cancel_move(self):
        self.cancels += 1

    def pop_move_results(self):
        return []

    def close(self):
        self.closed = True


class StubWorld:
    model = data = None

    def __init__(self):
        self.held = None

    def set_held_positions(self, held):
        self.held = held

    def _fk(self, q):
        self.fk = np.asarray(q)


def wait(predicate, timeout=10., what=''):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError(f'timed out waiting for {what}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='supervised_hw_'))
    procs, runtimes = [], []
    try:
        socks, tokens = {}, {}
        for name in ARMS:
            rt, sup, sock, token = start_arm(args.server, tmp, name)
            procs += [sup, rt]
            socks[name], tokens[name] = sock, token
        robots = {a: dict(arm=dict(joints=list(JOINTS), gripper_joints=[]), rt=dict(instance_id=a, host='x', hand_port=1))
                  for a in ARMS}
        config = SimpleNamespace(scene=SimpleNamespace(actors=[SimpleNamespace(name=a) for a in ARMS]), robots=robots,
                                 model_tokens=tokens, shared_object=None)
        hands = {}

        def hand(host, port):
            hands[len(hands)] = FakeHand()
            return hands[len(hands) - 1]

        def make():
            rt = SupervisedHardware(config, arm_factory=lambda a, cfg: ArmClient(a, socket_path=socks[a]),
                                    hand_factory=hand, world_factory=lambda c: StubWorld())
            runtimes.append(rt)
            return rt

        probe = {a: ArmClient(a, socket_path=socks[a]) for a in ARMS}  # a second, independent view of each supervisor
        runtime = make()
        # Construction claims nothing (the lifecycle stays CONNECTED, not CLAIMED).
        assert all(p.authority().lifecycle is Lifecycle.CONNECTED for p in probe.values()), \
            [p.authority() for p in probe.values()]
        assert runtime.mode == 'independent' and runtime.backend_name == 'supervised' and not runtime.fresh
        assert runtime.slices == {'arm_a': slice(0, 3), 'arm_b': slice(3, 6)} and runtime.object_pose() is None

        runtime.connect()
        assert runtime.fresh and runtime.feedback_ready and len(runtime.hands) == 2
        assert all(p.authority().lifecycle is Lifecycle.CLAIMED for p in probe.values())  # connecting claims, arms nothing
        for a in ARMS:
            try:
                probe[a].claim()
            except Refused:
                pass
            else:
                raise AssertionError('another client claimed an arm this runtime owns')
        joints = runtime.joint_state()
        assert joints['position'].shape == joints['velocity'].shape == (6,) and np.isfinite(joints['position']).all()
        np.testing.assert_allclose(joints['position'][:3], probe['arm_a'].state().q, atol=1e-6)
        try:
            runtime.set_shared_grasps(True)
        except ValueError as exc:
            assert 'direct hardware runtime' in str(exc), exc
        else:
            raise AssertionError('shared coupling admitted')
        assert set(runtime.telemetry()) == set(ARMS) and runtime.diagnostic_sample()['arms']['arm_a']['measurement_valid']

        # ARM does not block the caller; a pending request holds readiness back; arming again while pending is refused.
        a, b = ARMS
        handle_a = runtime.handles[a]
        entered, release = threading.Event(), threading.Event()
        real_arm = handle_a._inner._arm.arm

        def slow_arm(*args, **kw):
            entered.set()
            assert release.wait(5)
            return real_arm(*args, **kw)

        with patch.object(handle_a._inner._arm, 'arm', slow_arm):
            began = time.monotonic()
            handle_a.set_armed(True)
            assert time.monotonic() - began < .5, 'set_armed blocked the caller'
            assert entered.wait(2)
            assert handle_a.state().ready is False
            try:
                handle_a.set_armed(True)
            except ValueError as exc:
                assert 'acknowledgement' in str(exc), exc
            else:
                raise AssertionError('armed again while pending')
            release.set()
            wait(lambda: handle_a.state().ready, what='arm_a ready after the held ARM')
        assert probe[a].authority().lifecycle is Lifecycle.ARMED and handle_a.state().fault_reason is None

        # A pending DISARM on an arm that is still armed: not ready until the stop is confirmed, then re-armable.
        entered.clear()
        release.clear()
        real_disarm = handle_a._inner._arm.disarm

        def slow_disarm(*args, **kw):
            entered.set()
            assert release.wait(5)
            return real_disarm(*args, **kw)

        with patch.object(handle_a._inner._arm, 'disarm', slow_disarm):
            handle_a.set_armed(False)
            assert entered.wait(2) and probe[a].authority().lifecycle is Lifecycle.ARMED
            assert handle_a.state().ready is False, 'ready while a DISARM is pending'
            release.set()
            wait(lambda: probe[a].authority().lifecycle is not Lifecycle.ARMED, what='arm_a disarmed')
        wait(lambda: not handle_a._pending or (handle_a.state() and not handle_a._pending), what='the DISARM reaped')
        handle_a.set_armed(True)
        wait(lambda: handle_a.state().ready, what='arm_a re-armed')

        # A refused ARM is the arm's fault until the next request succeeds.
        handle_b = runtime.handles[b]
        with patch.object(handle_b._inner._arm, 'arm', side_effect=Refused('refused for the test')):
            handle_b.set_armed(True)
            wait(lambda: handle_b.state().fault_reason, what='the refusal as a fault')
        assert 'refused for the test' in handle_b.state().fault_reason and not runtime.fresh
        handle_b.set_armed(True)
        wait(lambda: handle_b.state().ready and not handle_b.state().fault_reason, what='arm_b recovered')
        assert runtime.fresh

        # An invalid measurement is not fresh (the last good pose is kept for display) and recovers with the next valid one.
        good = runtime.joint_state()['position'].copy()
        real_state = handle_b._inner.state
        with patch.object(handle_b._inner, 'state', lambda: replace(real_state(), position=None, velocity=None)):
            runtime.step()
            assert not runtime.fresh and not runtime.diagnostic_sample()['arms'][b]['measurement_valid']
            np.testing.assert_allclose(runtime.joint_state()['position'], good)
        runtime.step()
        assert runtime.fresh and runtime.diagnostic_sample()['arms'][b]['measurement_valid']

        # A pair plan runs through the coordinator over the runtime's handles.
        runtime.step()
        before = runtime.joint_state()['position'].copy()
        coordinator = BimanualCoordinator(runtime.handles, staging_timeout_s=5., state_timeout_s=2.)
        plans = [plan_for(runtime.clients[n], 'hw1', n, tokens[n], .05 if n == a else -.05) for n in ARMS]
        samples = []

        def record():
            """What the console records each tick: the runtime's diagnostic sample plus the pair identity it joins on."""
            runtime.step()
            coordinator.pump()
            sample = runtime.diagnostic_sample()
            members = coordinator.pair_state('hw1').members
            sample.update(pair_id='hw1', lifecycle=coordinator.lifecycle.value,
                          pair_plans={m.arm_id: m.plan_id for m in members})
            samples.append(sample)

        coordinator.stage_pair(BimanualPlan('hw1', *plans))
        end = time.monotonic() + 15
        while coordinator.lifecycle is not PairLifecycle.READY and time.monotonic() < end:
            record()
            time.sleep(.01)
        coordinator.execute_pair('hw1')
        while not coordinator.lifecycle.terminal and time.monotonic() < end:
            record()
            time.sleep(.01)
        assert coordinator.lifecycle.terminal, coordinator.lifecycle
        # CTRL-051 through the supervisors: the lineage of the executed pair joins on identities in the supervisors' own
        # controller ticks (trace/1), exactly as it does for the direct runtime; ticks that fell off a ring are counted.
        lineage = diagnostics.reconstruct_lineage(samples, pair_id='hw1')
        assert lineage['complete'], (lineage['missing'], lineage['foreign_arms'], lineage['declared_plans'])
        assert set(lineage['links']) == set(diagnostics.LINE_LINKS) and not lineage['foreign_arms'], lineage
        assert all(a_['trace_seq'] > 0 and a_['trace_errors'] == 0 for a_ in samples[-1]['arms'].values()), samples[-1]['arms']
        assert all(isinstance(a_['trace_dropped'], int) for a_ in samples[-1]['arms'].values())
        assert coordinator.lifecycle is PairLifecycle.COMPLETE and coordinator.pair_state('hw1').ok()
        wait(lambda: (runtime.step() or True) and abs(runtime.joint_state()['position'][0] - before[0] - .05) < .01,
             what='the measured motion')
        assert abs(runtime.joint_state()['position'][3] - before[3] + .05) < .01

        # trace/1 drop detection: a reader that comes back after the 64-tick ring turned over is told how many ticks it missed,
        # and the runtime counts them; a prompt reader sees none.
        first = runtime.clients[a].trace_samples(0)
        assert first['dropped'] == 0 and first['samples'] and first['capacity'] == 64 and first['errors'] == 0, first['dropped']
        seqs = [t['seq'] for t in first['samples']]
        assert seqs == sorted(set(seqs)) and seqs[-1] == first['latest_seq'], seqs
        prompt = runtime.clients[a].trace_samples(first['latest_seq'])
        assert all(t['seq'] > first['latest_seq'] for t in prompt['samples']) and prompt['dropped'] == 0, prompt['dropped']
        runtime.step()
        gap_before = runtime._trace_dropped[a]
        time.sleep(1.0)  # 100 ticks at 100 Hz: the 64-tick ring turns over
        late = runtime.clients[a].trace_samples(first['latest_seq'])
        assert late['dropped'] > 0 and late['dropped'] == late['samples'][0]['seq'] - first['latest_seq'] - 1, late['dropped']
        assert len(late['samples']) <= 64
        assert all(t['seq'] > first['latest_seq'] for t in late['samples']), 'an old sample was returned again'
        runtime.step()
        assert runtime._trace_dropped[a] - gap_before > 0, 'the runtime did not count the ticks it missed'
        assert runtime.diagnostic_sample()['arms'][a]['trace_dropped'] == runtime._trace_dropped[a]
        before_prompt = runtime._trace_dropped[a]
        for _ in range(5):
            runtime.step()
            time.sleep(.01)
        assert runtime._trace_dropped[a] == before_prompt, 'a prompt reader was told it missed ticks'

        # A supervisor that stops answering the trace read: the held sample is marked stale (never presented as current), and a
        # working read clears the mark.
        real_trace = runtime.clients[a].trace_samples

        def broken(*args, **kw):
            raise SupervisorError('trace read failed')
        runtime.clients[a].trace_samples = broken
        try:
            runtime.step()
            sample_a = runtime.diagnostic_sample()['arms'][a]
            assert sample_a['trace_stale'] and sample_a['seq'] > 0, 'the held sample was dropped or presented as current'
            assert not runtime.diagnostic_sample()['arms'][b]['trace_stale']
        finally:
            runtime.clients[a].trace_samples = real_trace
        runtime.step()
        assert not runtime.diagnostic_sample()['arms'][a]['trace_stale']
        # a supervisor without trace/1 has no trace (nothing invented, nothing stale); the SDK and the service refuse the op
        bare = object.__new__(SupervisedHardware)
        bare.clients = {a: SimpleNamespace(extensions={})}
        bare._trace_seq, bare._trace_dropped, bare._trace_errors, bare._trace_latest, bare._trace_ok = (
            {a: 0}, {a: 0}, {a: 0}, {}, {a: True})
        bare._read_trace(a)
        assert not bare._trace_latest and bare._trace_ok[a] and bare._trace_seq[a] == 0
        gate = object.__new__(ArmClient)
        gate.extensions = {}
        for call in (lambda: gate.trace_samples(0), lambda: dispatch(SimpleNamespace(extensions={}), 's', 'trace_samples', {})):
            try:
                call()
            except (Refused, ServiceRefused):
                pass
            else:
                raise AssertionError('trace_samples without the extension')
        # a record that cannot be built is counted and never raises into the loop, and does not advance the sequence
        stub = SimpleNamespace(_last_packet=(object(), 0.), link=SimpleNamespace(), controller=SimpleNamespace(),
                               _native=None, _executions={}, _authority=SimpleNamespace(lifecycle=Lifecycle.ARMED),
                               _trace_seq=7, _trace=[], _trace_errors=0)
        RobotRuntime._record_trace(stub, {})
        assert stub._trace_errors == 1 and stub._trace_seq == 7 and not stub._trace, 'a failed record was not counted'

        # A per-arm DISARM (the console's disarm button) cancels that arm's Hand; stop() disarms both and cancels the Hands.
        arm_a_hand = hands[0].cancels
        handle_a.set_armed(False)
        assert hands[0].cancels > arm_a_hand
        wait(lambda: probe[a].authority().lifecycle is not Lifecycle.ARMED, what='arm_a disarmed by the per-arm request')
        wait(lambda: not handle_a._pending or (handle_a.state() and not handle_a._pending), what='the DISARM reaped')
        handle_a.set_armed(True)
        wait(lambda: handle_a.state().ready, what='arm_a re-armed')
        cancels = sum(h.cancels for h in hands.values())
        runtime.stop()
        wait(lambda: all(p.authority().lifecycle is not Lifecycle.ARMED for p in probe.values()), what='both disarmed')
        assert sum(h.cancels for h in hands.values()) > cancels

        # close(): disarmed, clients and Hands closed; a closed runtime refuses to arm.
        runtime.close()
        assert all(h.closed for h in hands.values())
        try:
            runtime.handles[a].set_armed(True)
        except ValueError as exc:
            assert 'closed' in str(exc), exc
        else:
            raise AssertionError('armed after close')
        assert all(p.authority().lifecycle is not Lifecycle.ARMED for p in probe.values())

        # close() while an ARM is in flight: the queued DISARM still runs, so the arm is not left armed without an owner.
        hands.clear()
        third = make()
        third.connect()
        slow_enter = threading.Event()
        real_third = third.handles[a]._inner._arm.arm

        def slow_third(*args, **kw):
            slow_enter.set()
            time.sleep(.5)
            return real_third(*args, **kw)

        with patch.object(third.handles[a]._inner._arm, 'arm', slow_third):
            third.handles[a].set_armed(True)
            assert slow_enter.wait(2)
            third.close()
        assert probe[a].authority().lifecycle is not Lifecycle.ARMED, probe[a].authority()
        # stop() followed by close() likewise
        hands.clear()
        fourth = make()
        fourth.connect()
        for h in fourth.handles.values():
            h.set_armed(True)
        wait(lambda: all(h.state().ready for h in fourth.handles.values()), what='fourth armed')
        fourth.stop()
        fourth.close()
        assert all(p.authority().lifecycle is not Lifecycle.ARMED for p in probe.values())
        # stop() while an ARM is in flight queues its DISARM behind it; close() must let that DISARM run
        hands.clear()
        fifth = make()
        fifth.connect()
        enter5 = threading.Event()
        real5 = fifth.handles[a]._inner._arm.arm

        def slow5(*args, **kw):
            enter5.set()
            time.sleep(.5)
            return real5(*args, **kw)

        with patch.object(fifth.handles[a]._inner._arm, 'arm', slow5):
            fifth.handles[a].set_armed(True)
            assert enter5.wait(2)
            fifth.stop()
            fifth.close()
        assert all(p.authority().lifecycle is not Lifecycle.ARMED for p in probe.values())

        # Hands that are not measured keep the runtime from being fresh, and a failed connect closes everything.
        hands.clear()
        second = make()
        orig = FakeHand.snapshot
        with patch.object(FakeHand, 'snapshot', lambda self: dict(orig(self), measured=False)):
            try:
                second.connect()
            except ValueError as exc:
                assert 'fresh measured feedback' in str(exc), exc
            else:
                raise AssertionError('connected without measured Hands')
        assert second._closed and all(h.closed for h in hands.values())
        for p in probe.values():
            p.close()
    finally:
        for rt in runtimes:
            rt.close()
        for p in procs:
            p.terminate()
    print('PASS: SupervisedHardware over two supervisors: claim on connect only, non-blocking ARM with pending/fault state, '
          'a pair plan through the coordinator, stop/close disarm, shared coupling refused')


if __name__ == '__main__':
    main()
