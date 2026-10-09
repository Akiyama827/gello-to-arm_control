"""The Dora controller contract over a supervisor (DoraArmBridge), against the supervisor over a simulated plant. No Dora runtime.

    python tools/check/check_dora_arm.py

``emit`` captures what the bridge would send on its Dora outputs; the inputs are the same Arrow messages the planner and console send
(``pack_plan``, ``pack_control_update``). Checked: the startup claim and capability/authority announcements, the ARM edge, the
``ready`` event with the first measured pose, ``motor_state`` shape and values, a gated plan (staged, executed, ``leg_result``), an
UNGATED plan (released when admitted), a targeted cancel (``plan_cleared``), named-preset gains (accepted, ``mode`` event) and
non-preset gains (refused), ``jog`` as a supervisor setpoint (the arm follows while re-sent, holds when it stops; refused with the
reason past a joint limit, beyond abort_pos_err_rad, malformed, from a non-owner, while a plan is staged or runs, DISARMed; an
execute of a plan a jog superseded fails its leg; an owner that leaves takes its setpoint with it), the unsupported ``gripper`` (refused with a
notice, rate limited), ``stop`` (cancel + DISARM),
malformed input (never raises), and configuration drift (the bridge refuses to start against a supervisor running other gains).
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sdk_supervisor import write_config  # noqa: E402

from arm_control.contracts._arrow import pack_json_message, unpack_json_message  # noqa: E402
from arm_control.contracts.motion import pack_control_update, pack_jog, pack_plan, unpack_controller_event  # noqa: E402
from arm_control.contracts.motor import unpack_motor_state  # noqa: E402
from arm_control.multi_arm.supervised import AsyncAuthorityHandle, SdkArmHandle  # noqa: E402
from arm_control.sdk import ArmClient, Refused  # noqa: E402
from arm_control.supervisor.dora_arm import DoraArmBridge, default_gains_and_presets, release_after_loss  # noqa: E402
from arm_control.config import load_robot_config  # noqa: E402

KP, KD = np.full(3, 100.), np.full(3, 5.)  # kd 20 destabilises this 0.02 kg m^2 fixture at the sim's 4 ms step; 5 is its default
FIRM = dict(kp=[150., 150., 150.], kd=[25., 25., 25.])


def wait(predicate, timeout=10., what=''):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError(f'timed out waiting for {what}')


class Capture:
    def __init__(self):
        self.out = []

    def __call__(self, topic, value):
        self.out.append((topic, value))

    def events(self, kind=None):
        found = [unpack_controller_event(v) for t, v in self.out if t == 'controller_event']
        return [e for e in found if kind is None or e['kind'] == kind]

    def last(self, topic):
        return [v for t, v in self.out if t == topic][-1]


def plan_message(plan_id, q0, q1, duration, *, gated, token=None):
    mid = 1.5 * (q1 - q0) / duration
    return pack_plan(plan_id=plan_id, phase='move', gated=gated, times=[0., duration / 2, duration],
                     positions=np.stack([q0, (q0 + q1) / 2, q1]), velocities=np.stack([0 * q0, mid, 0 * q0]),
                     kp=KP.tolist(), kd=KD.tolist(), model_token=token)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='dora_arm_'))
    cfg_path = write_config(tmp, 0, 0)
    text = cfg_path.read_text().replace("  ee_frame: l3\n", "  ee_frame: l3\n  gripper_joints: []\n").replace("  native_reference: true\n", "")
    text += (f"sim_model_path: {tmp / 'three.urdf'}\nsim_gravcomp: true\nsim_timestep: 0.001\n"
             "sim_servo: {state_hz: 250, hold_ms: 100, fault_ms: 600, hold_kp: 50, hold_kd: 5}\n"
             "control_mode:\n  jog: {timeout_s: 0.3}\n  gain_presets:\n    firm: {kp: [150, 150, 150], kd: [25, 25, 25]}\n")
    cfg_path.write_text(text)
    sock = tmp / 'sdk_arm.sock'
    log = open(tmp / 'supervisor.log', 'w')
    sup = subprocess.Popen([sys.executable, '-m', 'arm_control.supervisor.service', '--config', str(cfg_path), '--socket', str(sock),
                            '--plant', 'sim'], stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
    arm = handle = None
    ok = False
    try:
        end = time.monotonic() + 30
        while True:
            assert sup.poll() is None, (tmp / 'supervisor.log').read_text()[-2000:]
            assert time.monotonic() < end, 'the supervisor did not come up'
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(sock))
                break
            except OSError:
                time.sleep(.05)
            finally:
                probe.close()
        arm = ArmClient('sdk_arm', socket_path=sock)
        handle = AsyncAuthorityHandle(SdkArmHandle(arm, 'arm', model_token=None))
        kp, kd, presets = default_gains_and_presets(load_robot_config(cfg_path))
        assert set(presets) == {'firm'} and np.allclose(kp, 100.) and np.allclose(kd, 5.), (presets, kp, kd)  # kd 20 is clamped to the 5.0 default ceiling
        cap = Capture()
        bridge = DoraArmBridge(arm, handle, presets, cap, arm_id='arm')
        # configuration drift: a bridge resolved against other defaults refuses to start
        try:
            bridge.check_config(kp * 2, kd)
        except RuntimeError as exc:
            assert 'drift' in str(exc), exc
        else:
            raise AssertionError('a drifted configuration was accepted')
        bridge.check_config(kp, kd)
        bridge.start()
        topics = [t for t, _ in cap.out]
        assert topics == ['model_revision', 'arm'], topics
        assert unpack_motor_state  # imported for the decode below

        def pump(until, what, timeout=10.):
            return wait(lambda: (bridge.pump(), until())[1], timeout, what)

        pump(lambda: any(t == 'motor_state' for t, _ in cap.out), 'the first motor_state')
        state = unpack_motor_state(cap.last('motor_state'), 3)
        assert state['position'].shape == (3,) and np.isfinite(state['position']).all()
        # ARM: the edge is announced and the ready event carries the first fresh pose
        bridge.on_control(pack_control_update(arm=True))
        pump(lambda: cap.events('ready'), 'the ready event')
        edges = [unpack_json_message(v)['armed'] for t, v in cap.out if t == 'arm']
        assert edges == [False, True], edges  # announced once at start, once on the ARM edge
        health_after_arm = [unpack_json_message(v) for t, v in cap.out if t == 'motor_health']
        assert health_after_arm[-1]['armed'] is True and health_after_arm[0]['armed'] is False, health_after_arm  # on change, not on the period
        assert len(health_after_arm) <= 6, 'motor_health is flooding'
        ready = cap.events('ready')[0]
        assert ready['q'].shape == (3,) and ready['runtime_state']['armed'] is True and ready['runtime_state']['n_arm'] == 3, ready
        q0 = np.asarray(arm.state().q)
        # a GATED plan: staged, then executed on `execute`, then leg_result ok
        bridge.on_plan(plan_message('g1', q0, q0 + [.1, -.1, .05], 1.0, gated=True))
        pump(lambda: [e for e in cap.events('plan_staged') if e['plan_id'] == 'g1'], 'plan_staged g1')
        assert [e for e in cap.events('plan_staged') if e['plan_id'] == 'g1'][0]['ok'] is True
        assert not [e for e in cap.events('leg_result') if e['plan_id'] == 'g1'], 'a gated plan ran before execute'
        bridge.on_control(pack_control_update(execute='g1'))
        pump(lambda: [e for e in cap.events('leg_result') if e['plan_id'] == 'g1'], 'leg_result g1', 15.)
        assert [e for e in cap.events('leg_result') if e['plan_id'] == 'g1'][0]['ok'] is True, [e for e in cap.events('leg_result')]
        # an UNGATED plan is released when the supervisor admits it
        q1 = np.asarray(arm.state().q)
        bridge.on_plan(plan_message('u1', q1, q1 - [.1, -.1, .05], 1.0, gated=False))
        pump(lambda: [e for e in cap.events('leg_result') if e['plan_id'] == 'u1'], 'leg_result u1', 15.)
        assert [e for e in cap.events('leg_result') if e['plan_id'] == 'u1'][0]['ok'] is True, [(e['plan_id'], e['ok'], e['reason']) for e in cap.events('leg_result')]
        # a new plan REPLACES a staged-but-unreleased one (as on the in-process controller); no plan_cleared is reported for it
        qr = np.asarray(arm.state().q)
        bridge.on_plan(plan_message('r1', qr, qr + [.05, 0, 0], 1.0, gated=True))
        pump(lambda: [e for e in cap.events('plan_staged') if e['plan_id'] == 'r1'], 'plan_staged r1')
        bridge.on_plan(plan_message('r2', qr, qr + [.06, 0, 0], 1.0, gated=True))
        pump(lambda: [e for e in cap.events('plan_staged') if e['plan_id'] == 'r2'], 'plan_staged r2')
        assert [e for e in cap.events('plan_staged') if e['plan_id'] == 'r2'][0]['ok'] is True, cap.events('plan_staged')
        assert not [e for e in cap.events('plan_cleared') if e['plan_id'] == 'r1'], 'the replacement was reported as a clear'
        bridge.on_control(pack_control_update(execute='r1'))  # the replaced plan is gone
        bridge.on_control(pack_control_update(execute='r2'))
        pump(lambda: [e for e in cap.events('leg_result') if e['plan_id'] == 'r2'], 'leg_result r2', 15.)
        assert [e for e in cap.events('leg_result') if e['plan_id'] == 'r2'][0]['ok'] is True
        # plan + execute with NO pump between, then another plan while it runs: the running leg is neither replaced nor cancelled; the new
        # plan is refused (as in-process: a plan is refused while one runs); the running leg still reports its result
        qx = np.asarray(arm.state().q)
        bridge.on_plan(plan_message('x1', qx, qx + [.1, 0, 0], 1.5, gated=True))
        bridge.on_control(pack_control_update(execute='x1'))
        time.sleep(.4)
        bridge.on_plan(plan_message('x2', qx, qx + [.05, 0, 0], 1.0, gated=True))
        refused_x2 = [e for e in cap.events('plan_staged') if e['plan_id'] == 'x2']
        assert refused_x2 and refused_x2[0]['ok'] is False and 'still executing' in refused_x2[0]['reason'], refused_x2
        pump(lambda: [e for e in cap.events('leg_result') if e['plan_id'] == 'x1'], 'leg_result x1', 15.)
        assert [e for e in cap.events('leg_result') if e['plan_id'] == 'x1'][0]['ok'] is True, 'the running leg was cancelled'
        assert not [e for e in cap.events('plan_cleared') if e['plan_id'] == 'x1']
        # cancelling a RUNNING leg reports a failed leg_result (as in-process), not only a clear
        qy = np.asarray(arm.state().q)
        bridge.on_plan(plan_message('y1', qy, qy + [.1, 0, 0], 3.0, gated=True))
        bridge.on_control(pack_control_update(execute='y1'))
        time.sleep(.5)
        bridge.on_control(pack_control_update(cancel=True, reason='operator'))
        pump(lambda: [e for e in cap.events('leg_result') if e['plan_id'] == 'y1'], 'leg_result y1')
        failed = [e for e in cap.events('leg_result') if e['plan_id'] == 'y1'][0]
        assert failed['ok'] is False and 'cancelled' in failed['reason'], failed
        # a targeted cancel of a staged plan answers plan_cleared; one aimed at another plan does not clear it
        q2 = np.asarray(arm.state().q)
        bridge.on_plan(plan_message('c1', q2, q2 + [.1, 0, 0], 1.0, gated=True))
        pump(lambda: [e for e in cap.events('plan_staged') if e['plan_id'] == 'c1'], 'plan_staged c1')
        bridge.on_control(pack_control_update(cancel=True, cancel_plan_id='c1', reason='operator'))
        pump(lambda: [e for e in cap.events('plan_cleared') if e['plan_id'] == 'c1'], 'plan_cleared c1')
        assert [e for e in cap.events('plan_cleared') if e['plan_id'] == 'c1'][0]['ok'] is True
        # named-preset gains are accepted (mode event with the new gains), other gains refused
        before = len(cap.events('mode'))
        bridge.on_control(pack_control_update(gains=FIRM))
        modes = cap.events('mode')[before:]
        assert any(m['ok'] and m['mode_state']['kp'] == FIRM['kp'] for m in modes), modes
        before = len(cap.events('mode'))
        bridge.on_control(pack_control_update(gains=dict(kp=[7., 7., 7.], kd=[1., 1., 1.])))
        refused = [m for m in cap.events('mode')[before:] if not m['ok']]
        assert refused and 'not a named controller preset' in refused[0]['reason'], cap.events('mode')[before:]
        # jog = a supervisor setpoint (setpoint/1): the arm follows while it is re-sent, and holds where it is once it stops
        assert arm.extensions.get('setpoint') == 1, arm.extensions
        assert arm.setpoint(arm.state().q) == .3, 'the jog deadman is not the mode config\'s jog.timeout_s'
        before = len(cap.events('mode'))
        target = np.asarray(arm.state().q) + [.08, 0, 0]
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            bridge.on_jog(pack_jog(q=target))
            bridge.pump()
            time.sleep(.05)
        reached = np.asarray(arm.state().q)
        assert abs(reached[0] - target[0]) < .02, (reached, target)
        assert not [m for m in cap.events('mode')[before:] if not m['ok']], 'an accepted jog raised a notice'
        time.sleep(.7)  # past the jog timeout (0.3 s): the hold anchors at the MEASURED pose, not at the last jog target (no coasting on)
        held = arm.state()
        assert abs(held.selected_q[0] - reached[0]) < .005 and abs(held.selected_q[0] - target[0]) > .005, (held.selected_q, reached, target)
        assert held.q[0] < target[0] - .005, (held.q, target)

        def refused(q, why, client=arm):
            try:
                client.setpoint(q)
            except Refused as exc:
                assert why in str(exc), (why, exc)
            else:
                raise AssertionError(f'a setpoint was accepted; expected {why!r}')
        refused(reached + [10., 0, 0], 'position limit')
        refused(reached[:2], 'finite joint positions')
        refused([np.nan, 0., 0.], 'finite joint positions')
        other = ArmClient('sdk_arm', socket_path=sock)
        try:
            refused(reached, 'owned by another client', client=other)
        finally:
            other.close()
        bridge.on_plan(plan_message('j1', reached, reached + [.1, 0, 0], 3.0, gated=True))
        bridge.on_control(pack_control_update(execute='j1'))
        pump(lambda: handle.state().executing, 'j1 running')
        refused(reached, 'execution is live')
        before = len(cap.events('mode'))
        for _ in range(5):  # through the bridge: a refused burst is one notice, with the reason
            bridge.on_jog(pack_jog(q=reached))
        notices = [m for m in cap.events('mode')[before:] if not m['ok']]
        assert len(notices) == 1 and 'execution is live' in notices[0]['reason'], notices
        bridge.on_control(pack_control_update(cancel=True, reason='operator'))
        pump(lambda: [e for e in cap.events('leg_result') if e['plan_id'] == 'j1'], 'leg_result j1')
        # a jog supersedes a staged, unreleased plan (planned from the pose the jog leaves): plan_cleared, then the jog moves the arm
        qs = np.asarray(arm.state().q)
        bridge.on_plan(plan_message('s1', qs, qs + [.1, 0, 0], 1.0, gated=True))
        pump(lambda: [e for e in cap.events('plan_staged') if e['plan_id'] == 's1'], 'plan_staged s1')
        refused(qs, 'execution is live')  # a staged plan is a live execution: a direct setpoint is refused (the bridge cancels first)
        before = len(cap.events('mode'))
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            bridge.on_jog(pack_jog(q=qs - [.05, 0, 0]))
            bridge.pump()
            time.sleep(.05)
        assert [e for e in cap.events('plan_cleared') if e['plan_id'] == 's1'], 'the superseded plan was not cleared'
        assert abs(arm.state().q[0] - (qs[0] - .05)) < .02, (arm.state().q, qs)
        assert not [m for m in cap.events('mode')[before:] if not m['ok']], 'the supersede showed a refusal'
        time.sleep(.6)
        # the console kept s1 as pending: its execute answers a failed leg_result, never silence
        bridge.on_control(pack_control_update(execute='s1'))
        failed = [e for e in cap.events('leg_result') if e['plan_id'] == 's1']
        assert failed and failed[0]['ok'] is False and 'superseded by a jog' in failed[0]['reason'], failed
        # a target beyond the policy's abort_pos_err_rad (0.35 here) is refused, not taken and then dropped by the controller
        refused(np.asarray(arm.state().q) + [.5, 0, 0], 'abort_pos_err_rad')
        # a failed settle freezes the controller (a completion goal the plan never reaches): setpoints and plans are refused until
        # the console's Hold clears it through the supervisor (`hold`), still ARMed -- as Stop/cancel did in-process
        qf = np.asarray(arm.state().q)
        frozen_plan = pack_plan(plan_id='f1', phase='move', gated=False, times=[0., .5], positions=np.stack([qf, qf]),
                                velocities=np.zeros((2, 3)), kp=KP.tolist(), kd=KD.tolist(),
                                completion=dict(vel_tol=.05, goal_q=(qf + [.1, 0, 0]).tolist(), residual_max=.005))
        bridge.on_plan(frozen_plan)
        pump(lambda: [e for e in cap.events('leg_result') if e['plan_id'] == 'f1'], 'leg_result f1 (settle timeout)', 15.)
        assert [e for e in cap.events('leg_result') if e['plan_id'] == 'f1'][0]['ok'] is False
        refused(qf, 'frozen')
        bridge.on_control(pack_control_update(hold=True, reason='operator'))
        assert arm.setpoint(np.asarray(arm.state().q)) > 0 and handle.state().armed, 'Hold did not clear the frozen hold'
        time.sleep(.5)
        # the gripper is not served by an arm supervisor: refused with a notice, a burst makes one notice
        before = len(cap.events('mode'))
        for _ in range(5):
            bridge.on_gripper(pack_json_message('gripper', {}))
        notices = [m for m in cap.events('mode')[before:] if not m['ok']]
        assert len(notices) == 1 and 'gripper' in notices[0]['reason'], notices
        time.sleep(1.1)  # past the notice period: the next refusal is told again
        bridge.on_gripper(pack_json_message('gripper', {}))
        assert len([m for m in cap.events('mode')[before:] if not m['ok'] and 'gripper' in m['reason']]) == 2
        # malformed input never raises
        bridge.on_plan(pack_json_message('plan', {'plan_id': 'broken'}))
        bridge.on_control(pack_json_message('not_control', {}))
        assert [e for e in cap.events('plan_staged') if not e['ok'] and 'unreadable' in e['reason']]
        # stop: cancel and DISARM
        bridge.on_control(pack_control_update(stop=True, reason='operator stop'))
        pump(lambda: not handle.state().armed and bridge._armed_sent is False, 'the DISARM edge')
        assert not arm.authority().arm_epoch
        assert [unpack_json_message(v)['armed'] for t, v in cap.out if t == 'arm'] == [False, True, False]
        refused(reached, 'not armed')
        # an owner that leaves takes its setpoint with it: the arm holds where it is, not at the departed owner's target
        arm.arm()
        end = time.monotonic() + 5.
        while True:
            try:
                arm.setpoint(np.asarray(arm.state().q) + [.3, 0, 0])
                break
            except Refused as exc:
                assert 'fresh measured pose' in str(exc) and time.monotonic() < end, exc
                time.sleep(.02)
        arm.unclaim()
        left = np.asarray(arm.state().q)
        time.sleep(.25)  # inside the 0.3 s deadman: without the clear the arm would still be travelling to the target
        assert abs(arm.state().q[0] - left[0]) < .03, (arm.state().q, left)
        arm.claim()
        arm.disarm()
        for t, v in cap.out:
            if t == 'arm':
                pass
        # a supervisor fault is a motor_health latch (recoverable: the console clears it from motor_health), never the terminal `fault` event;
        # it clears when the supervisor reports none
        from arm_control.multi_arm.types import ArmControlState

        class FakeHandle:
            def __init__(self):
                self.fault, self.events = None, []

            def poll_events(self):
                out, self.events = self.events, []
                return out

            def state(self):
                return ArmControlState('arm', position=np.zeros(3), velocity=np.zeros(3), armed=True, ready=True, executing=False,
                                       frozen=False, fault_reason=self.fault, session_id='s')
        fake, cap2 = FakeHandle(), Capture()
        fb = DoraArmBridge(arm, fake, presets, cap2, arm_id='arm')
        fake.events.append(dict(kind='fault', plan_id='', ok=False, reason='cmd_lost', runtime_state={}))
        fake.fault = 'cmd_lost'
        fb.pump()
        health = [unpack_json_message(v) for t, v in cap2.out if t == 'motor_health']
        assert health and health[-1]['latched_fault'] == 'cmd_lost' and health[-1]['any_fault'] is True, health
        assert not cap2.events('fault') and [m for m in cap2.events('mode') if not m['ok'] and 'cmd_lost' in m['reason']]
        fake.fault = None
        fb._clock = lambda: time.monotonic() + 100  # past the health period
        fb.pump()
        assert unpack_json_message([v for t, v in cap2.out if t == 'motor_health'][-1])['latched_fault'] == '', 'the latch never cleared'
        n_health = len([1 for t, _ in cap2.out if t == 'motor_health'])
        fb.pump()
        assert len([1 for t, _ in cap2.out if t == 'motor_health']) == n_health, 'motor_health repeated without a change or the period'
        fb._clock = lambda: time.monotonic() + 200
        fb.pump()
        assert len([1 for t, _ in cap2.out if t == 'motor_health']) == n_health + 1, 'no motor_health heartbeat (the console treats silence as stale)'
        # the supervisor is lost (the SDK channel dies): UNKNOWN is never published as DISARMED, a terminal fault is, the node ends, and
        # a new session DISARMS what the vanished owner left armed
        bridge.on_control(pack_control_update(arm=True))
        pump(lambda: handle.state().ready, 'armed before the loss')
        arm_edges_before = [unpack_json_message(v)['armed'] for t, v in cap.out if t == 'arm']
        arm._channel.close()
        pump(lambda: bridge.lost, 'the loss')
        assert [unpack_json_message(v)['armed'] for t, v in cap.out if t == 'arm'] == arm_edges_before, 'unknown was published as an arm edge'
        lost_events = [e for e in cap.events('fault') if 'supervisor lost' in e['reason']]
        assert len(lost_events) == 1 and any(unpack_json_message(v).get('unknown_authority') for t, v in cap.out if t == 'motor_health')
        bridge.pump()
        assert len([e for e in cap.events('fault') if 'supervisor lost' in e['reason']]) == 1, 'the loss was reported twice'
        watcher = ArmClient('sdk_arm', socket_path=sock)
        assert watcher.authority().lifecycle.value == 'armed', 'the supervisor holds an ownerless arm armed (by design)'
        said = release_after_loss('sdk_arm', sock)
        assert 'DISARMED' in said, said
        assert watcher.authority().lifecycle.value != 'armed'
        watcher.close()
        ok = True
    finally:
        if handle is not None:
            handle.close()
        if arm is not None:
            try:
                arm.disarm(timeout_s=3.0)
            except Exception:  # noqa: BLE001
                pass
            arm.close()
        sup.terminate()
        try:
            sup.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sup.kill()
        log.close()
        if ok:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    print('PASS: the Dora controller contract (plan/control/jog/gripper in; controller_event/arm/motor_state/model_revision out) over a '
          'supervisor on a simulated plant: ARM edge and ready, gated and ungated plans, targeted cancel, preset gains, refusals, stop, drift')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
