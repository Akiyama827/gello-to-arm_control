"""The supervisor over a simulated plant (SimLink), driven through the SDK, with the plant's servo semantics checked directly.

    python tools/check/check_sim_supervisor.py

No arm_rt_server and no robot: ``arm_control.supervisor.service --plant sim`` runs the same RobotRuntime over a MuJoCo
single-model plant on the generated 3-joint fixture. Part A drives ``SimLink`` in-process: ARM epochs are monotonic and a
disarmed or faulted servo ignores commands, the deadman holds after hold_ms and latches CMD_LOST after fault_ms, ARM is
refused while latched and DISARM clears it, a plant failure latches FAULT_PLANT, and a missing sim_servo key refuses to start.
Part B drives the supervisor process: extensions, claim/arm, a move that COMPLETES through SENT and APPLIED, a cancel, DISARM
and a second ARM in a new epoch, and the negotiated state_history/trace reads. Nothing here is a hardware or physics claim.
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sdk_supervisor import JOINTS, completed_chain, move, write_config  # noqa: E402

from arm_control.config import load_robot_config  # noqa: E402
from arm_control.execution.contract import CartesianImpedance, JointGains, Lifecycle, Phase, Start  # noqa: E402
from arm_control.plants.mujoco.backend import MuJoCoBackend  # noqa: E402
from arm_control.plants.remote_rt import protocol as rtp  # noqa: E402
from arm_control.plants.remote_rt.client import RtConfig, RtLinkError  # noqa: E402
from arm_control.plants.sim_link import SimLink  # noqa: E402
from arm_control.sdk import ArmClient, Refused  # noqa: E402

# hold_ms/fault_ms are the fake-RT bench's launch values; state_hz and the hold gains are the RT server's own defaults
SERVO = dict(state_hz=250, hold_ms=100, fault_ms=600, hold_kp=50, hold_kd=5)


def plant(urdf):
    return MuJoCoBackend(joint_names=list(JOINTS), single_model_path=str(urdf), timestep=0.001, control_period=0.004,
                         ground_z=None, gravcomp_prefixes=("",))


def wait(predicate, timeout=5., what=''):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.005)
    raise AssertionError(f'timed out waiting for {what}')


def command(q, kp=100., kd=20.):
    return dict(position=np.asarray(q, float), velocity=np.zeros(3), torque=np.zeros(3), kp=np.full(3, kp), kd=np.full(3, kd))


def plant_model_checks(tmp):
    """The single-model plant: a robot's stated armature (`sim_armature`) is applied and nothing otherwise, and the servo law's torque slew is per
    1 ms RT tick at ANY plant step (it is applied per step, so the per-step budget scales with the step)."""
    for step, armature in ((0.001, 0.1), (0.0005, 0.0)):
        b = MuJoCoBackend(joint_names=list(JOINTS), single_model_path=str(tmp / 'three.urdf'), timestep=step, control_period=step,
                          ground_z=None, arm_armature=armature)
        b.load()
        assert np.allclose(b.model.dof_armature[b._vadr], armature), (armature, b.model.dof_armature)
        before = np.array(b.data.ctrl, dtype=float)
        b.step(dict(position=np.full(3, 1.0), velocity=np.zeros(3), torque=np.zeros(3), kp=np.full(3, 1e4), kd=np.zeros(3)))
        moved = np.max(np.abs(np.array(b.data.ctrl, dtype=float) - before))
        assert abs(moved - b.slew_per_tick * step / 0.001) < 1e-9, (step, moved)  # one step: exactly the scaled budget, saturated
        b.close()


def observe_unit(tmp):
    """observe/1 over SimLink: off unless stated; refused if malformed; decimated from the plant ticks; a bounded ring with an exact
    cursor, ``dropped`` and ``limit``; every field from the tick that stepped the plant; a failing observation never stops a tick."""
    cfg = RtConfig(instance_id='sim_arm', state_timeout_s=.5)
    assert not SimLink(cfg, JOINTS, object(), servo=SERVO).observing
    for bad in (dict(rate_hz=0, ring_s=1), dict(rate_hz=251, ring_s=1), dict(rate_hz=50, ring_s=0), dict(rate_hz=50),
                dict(rate_hz='fast', ring_s=1), dict(rate_hz=float('nan'), ring_s=1), dict(rate_hz=50, ring_s=10.5)):
        try:
            SimLink(cfg, JOINTS, object(), servo=SERVO, observe=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted a bad sim_observe {bad}')
    from types import SimpleNamespace
    from arm_control.supervisor.runtime import RobotRuntime
    plain = SimpleNamespace(_native=None, _stream_admission=None, link=SimpleNamespace())  # an RT link: no observe/1
    assert 'observe' not in RobotRuntime.extensions.fget(plain), RobotRuntime.extensions.fget(plain)
    p = plant(tmp / 'three.urdf')
    p.load()
    p.data.qvel[:] = 2.0   # the arm moves, so a stale position cannot pass for this tick's
    link = SimLink(cfg, JOINTS, p, servo=SERVO, observe=dict(rate_hz=50, ring_s=.2))   # every 5th tick; 10 observations
    assert (link.observe_rate_hz, link._obs.maxlen) == (50., 10), (link.observe_rate_hz, link._obs.maxlen)
    for i in range(100):
        link._tick(i * .004)
        if (i + 1) % 5 == 0:  # an observing tick: the observation and the state sample come from the same step
            last = link._obs[-1]
            assert np.allclose([last['actor_q']['sim_arm'][j] for j in JOINTS], link.latest_state()[0].q), i
            assert last['sim_time_s'] == float(p.data.time) and last['seq'] == (i + 1) // 5, (i, last['sim_time_s'], p.data.time)
    first = link.observations(0)
    assert first['schema'] == 'observe/1' and [o['seq'] for o in first['samples']] == list(range(11, 21)), first['samples']
    assert (first['latest_seq'], first['dropped'], first['capacity'], first['rate_hz']) == (20, 0, 10, 50.), first
    times = [o['sim_time_s'] for o in first['samples']]
    assert np.allclose(np.diff(times), 5 * .004), times                                   # one per 5 plant ticks
    assert link.observations(5)['dropped'] == 5                                          # 6..10 fell off the ring
    page = link.observations(10, limit=3)
    assert [o['seq'] for o in page['samples']] == [11, 12, 13] and page['latest_seq'] == 13, page
    assert link.observations(20)['samples'] == [] and link.observations(20)['latest_seq'] == 20
    assert link.observations(99)['latest_seq'] == 20                                     # a stale cursor is clamped
    assert set(first['samples'][-1]['actor_q']['sim_arm']) == set(JOINTS) and first['samples'][-1]['object_poses'] is None
    moved = [first['samples'][k]['actor_q']['sim_arm']['j1'] for k in (0, -1)]
    assert abs(moved[1] - moved[0]) > 1e-6, moved                                    # the consistency check is not vacuous
    big = SimLink(cfg, JOINTS, plant(tmp / 'three.urdf'), servo=SERVO, observe=dict(rate_hz=250, ring_s=1))
    big._plant.load()
    for i in range(300):
        big._tick(i * .004)
    capped = big.observations(0, limit=1000)
    assert len(capped['samples']) == 64 and capped['latest_seq'] == capped['samples'][-1]['seq'] == 300 - 250 + 64, capped['latest_seq']
    big._plant.close()
    p.observe = lambda: 1 / 0
    for i in range(100, 105):
        link._tick(i * .004)                                                             # the tick survives a failing observation
    assert link.observations(20)['errors'] == 1 and link.latest_state()[0].state_seq == 105
    p.close()
    print('observe/1 over SimLink: off unless stated, decimated, bounded ring, exact cursor/dropped/limit, one tick per sample: PASS',
          flush=True)


def link_unit(tmp):
    cfg = RtConfig(instance_id='sim_arm', state_timeout_s=.5)
    for bad in ({k: v for k, v in SERVO.items() if k != 'hold_ms'}, dict(SERVO, fault_ms=50), dict(SERVO, fault_ms=100),
                dict(SERVO, state_hz=0), dict(SERVO, state_hz=0.5), dict(SERVO, state_hz=1e7), dict(SERVO, hold_kp=float('nan')), dict(SERVO, hold_kp=-5),
                dict(SERVO, hold_kd=-1), dict(SERVO, hold_ms='soon'), dict(SERVO, hold_ms=None), dict(SERVO, hold_ms=float('inf'))):
        try:
            SimLink(cfg, JOINTS, object(), servo=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted a bad sim_servo {bad}')
    link = SimLink(cfg, JOINTS, plant(tmp / 'three.urdf'), servo=SERVO)
    link.open()
    # the profiles a supervisor offers over it: this plant has no EE, so no cartesian_impedance; a plant that runs the law offers
    # it on streamed legs and never on the native reference (joints only)
    from arm_control.supervisor.runtime import offered_profiles
    assert offered_profiles(link, native=False) == ('joint_impedance',)
    lawful = type('L', (), {'supports_cartesian': True})()
    assert offered_profiles(lawful, native=False) == ('joint_impedance', 'cartesian_impedance')
    assert offered_profiles(lawful, native=True) == ('joint_impedance',)
    # a plant without the Cartesian law refuses a spec rather than running the joint word alone (the Dora plant path), and
    # atomically: the word it was running stays installed
    before = link._plant._last_command
    try:
        link._plant.apply_motor_command(dict(command([0, 0, 0]), cartesian=dict(frame='base', task_R=np.eye(3), kc=np.zeros(6),
                                                                                 dc=np.zeros(6))))
    except ValueError:
        pass
    else:
        raise AssertionError('a plant without the Cartesian law accepted a spec')
    assert link._plant._last_command is before, 'a refused Cartesian word was partly installed'
    try:
        state, _ = link.latest_state()
        assert state.session_id == link.session_id and not state.armed and state.arm_epoch == 0, state
        # what the simulated link does not implement is refused, not ignored
        for extra in (dict(cartesian={'x': 1}), dict(pose_hold={'id': 1})):
            try:
                link.apply_command(dict(command([0, 0, 0]), **extra))
            except RtLinkError:
                pass
            else:
                raise AssertionError(f'{list(extra)} accepted by a joint-only link')
        assert link.negotiate_reference('token') is False and not link.group_capable and link.clock_domain is None
        link._cmd_seq = 0
        # a disarmed servo ignores commands (the sequence is still handed out)
        seq = link.apply_command(command([.1, 0, 0]))
        time.sleep(.05)
        assert seq == 1 and link.latest_state()[0].last_cmd_seq == 0 and abs(link.latest_state()[0].q[0]) < 1e-3
        # ARM: a new epoch, holding the measured pose until commands flow
        link.enable_all()
        s = link.latest_state()[0]
        assert s.armed and s.arm_epoch == 1 and s.holding and not s.faulted, s
        link.enable_all()  # already armed: acknowledged, no new epoch
        assert link.latest_state()[0].arm_epoch == 1 and link._epochs == 1
        # holding since ARM, before any command: the configured hold gains
        wait(lambda: np.allclose(link._plant._last_command['kp'], SERVO['hold_kp']), 1., 'the hold gains after ARM')
        # commands are accepted and tracked; last_cmd_seq is the accepted sequence
        seq = None
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            seq = link.apply_command(command([.2, .1, 0]))
            time.sleep(.01)
        s = link.latest_state()[0]
        assert not s.holding and s.last_cmd_seq == seq and abs(s.q[0] - .2) < .03, (s.holding, s.last_cmd_seq, seq, s.q)
        # deadman: after hold_ms the plant holds where it is, with the gains of the last accepted command (as the RT does);
        # after fault_ms CMD_LOST latches, still holding
        wait(lambda: link.latest_state()[0].holding, .5, 'the hold')
        assert np.allclose(link._plant._last_command['kp'], 100.) and np.allclose(link._plant._last_command['kd'], 20.)
        held = list(link.latest_state()[0].q)
        assert not link.latest_state()[0].faulted
        wait(lambda: link.latest_state()[0].faulted, 1.5, 'CMD_LOST')
        s = link.latest_state()[0]
        assert s.fault_code == rtp.FAULT_CMD_LOST and s.holding and np.allclose(s.q, held, atol=.03), (s.fault_code, s.q, held)
        assert link.motor_health()['any_fault'] and link.motor_health()['latched_fault']
        # a faulted servo ignores commands and refuses ARM; DISARM clears it, and the next ARM is a NEW epoch
        accepted = link.latest_state()[0].last_cmd_seq
        link.apply_command(command([0, 0, 0]))
        time.sleep(.03)
        assert link.latest_state()[0].last_cmd_seq == accepted
        try:
            link.enable_all()
        except RtLinkError as exc:
            assert 'arm refused' in str(exc), exc
        else:
            raise AssertionError('ARM accepted while a fault is latched')
        assert link.safe_stop()
        s = link.latest_state()[0]
        assert not s.armed and not s.faulted and s.stop_epoch == 1 and not link.motor_health()['any_fault'], s
        assert link.motor_health()['arm_epoch'] == 0  # the link's own view is disarmed
        link.enable_all()
        assert link.latest_state()[0].arm_epoch == 2 and link.latest_state()[0].armed  # never reused
        assert link.safe_stop() and link.latest_state()[0].stop_epoch == 2
        assert link.safe_stop() and link.latest_state()[0].stop_epoch == 3  # every DISARM counts, as on the RT server
        # a confirmed DISARM RELEASES the plant: its last word is zero-gain, zero-torque (step(None) would keep the old one)
        link.enable_all()
        end = time.monotonic() + .5
        while time.monotonic() < end:
            link.apply_command(command([.5, .4, .3], kp=200., kd=30.))
            time.sleep(.01)
        assert np.allclose(link._plant._last_command['kp'], 200.)
        assert link.safe_stop()
        time.sleep(.05)
        last = link._plant._last_command
        assert not last['kp'].any() and not last['kd'].any() and not last['torque'].any(), last
        assert not link.latest_state()[0].tau_cmd[0] and not link.latest_state()[0].armed
        # a plant that fails latches FAULT_PLANT, not a dead thread
        link.enable_all()
        real_step = link._plant.step

        def failing(*args, **kw):
            raise RuntimeError('plant exploded')
        link._plant.step = failing
        wait(lambda: link.latest_state()[0].faulted or link._fault_code, 1., 'plant fault')
        link._plant.step = real_step
        wait(lambda: link.latest_state()[0].fault_code == rtp.FAULT_PLANT, 1., 'FAULT_PLANT in the state')
        assert 'plant exploded' in link.motor_health()['latched_fault']
        assert link.safe_stop()
    finally:
        link.close()


class Gated:
    """A plant whose step can be held open: the race between an ARM/DISARM/command and a step that already took its snapshot."""

    def __init__(self, inner):
        self.inner, self.entered, self.release, self.gate = inner, threading.Event(), threading.Event(), False
        self.fail_on_release = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, command=None):
        if self.gate:
            self.gate = False
            self.entered.set()
            assert self.release.wait(10), 'the gated step was never released'
            if self.fail_on_release:
                self.fail_on_release = False
                raise RuntimeError('plant failed mid-step')
        return self.inner.step(command)


def link_races(tmp):
    cfg = RtConfig(instance_id='sim_arm', state_timeout_s=.5)
    gated = Gated(plant(tmp / 'three.urdf'))
    real_step_global = gated.inner.step
    link = SimLink(cfg, JOINTS, gated, servo=SERVO)
    link.open()
    try:
        link.enable_all()
        for _ in range(20):
            link.apply_command(command([.1, 0, 0]))
            time.sleep(.01)
        before = link.latest_state()[0]
        assert before.armed and before.stop_epoch == 0
        # a DISARM then an ARM land inside one step: that step ran ARMED, so no sample may show the DISARM yet
        gated.entered.clear(), gated.release.clear()
        gated.gate = True
        assert gated.entered.wait(2)
        result = {}
        t = threading.Thread(target=lambda: result.update(stopped=link.safe_stop()))
        t.start()
        wait(lambda: link._stop_epoch == 1, 2., 'the DISARM request')
        seq_before = link._cmd_seq
        accepted_during = link.apply_command(command([.2, 0, 0]))  # disarmed now: handed a sequence, not accepted
        assert accepted_during == seq_before + 1
        armer = threading.Thread(target=link.enable_all)  # the ARM lands in the same step as the DISARM
        armer.start()
        wait(lambda: link._armed, 2., 'the ARM request')
        gated.release.set()
        t.join(10)
        armer.join(10)
        # no step ran released, so the DISARM is NOT confirmed (False), and no sample may claim it (stop_epoch still 0)
        assert result['stopped'] is False, result
        s = link.latest_state()[0]
        assert s.armed and s.arm_epoch == 2 and s.stop_epoch == 0, s
        assert all(h.stop_epoch == 0 for h, _ in link.recent_states() if h.armed), 'an armed sample showed the DISARM'
        assert link.safe_stop()
        assert not link.latest_state()[0].armed and link.latest_state()[0].stop_epoch == 2
        # a command accepted while a step is in flight is not reported by the sample of that step
        link.enable_all()
        link.apply_command(command([.1, 0, 0]))
        time.sleep(.05)
        gated.entered.clear(), gated.release.clear()
        gated.gate = True
        assert gated.entered.wait(2)
        published = link._state_seq  # the gated step will publish sample published + 1
        seq = link.apply_command(command([.1, 0, 0]))
        in_flight = link.latest_state()[0].last_cmd_seq
        assert in_flight != seq
        gated.release.set()
        wait(lambda: link.latest_state()[0].last_cmd_seq == seq, 1., 'the command in a later sample')
        step_sample = next(h for h, _ in link.recent_states() if h.state_seq == published + 1)
        assert step_sample.last_cmd_seq == in_flight, (step_sample.last_cmd_seq, in_flight, seq)  # not the later command
        assert link.safe_stop()
        # a step that FAILS publishes the authority it ran under: an ARM or DISARM that landed during it is not confirmed by
        # that sample, and the failure is not latched against the new authority
        def failing_step_with(request):
            link.enable_all()
            for _ in range(20):
                link.apply_command(command([.1, 0, 0]))
                time.sleep(.01)
            gated.entered.clear(), gated.release.clear()
            gated.gate, gated.fail_on_release = True, True
            assert gated.entered.wait(2)
            request()
            gated.release.set()
        outcome = {}
        failing_step_with(lambda: (threading.Thread(target=lambda: outcome.update(stopped=link.safe_stop())).start(),
                                   wait(lambda: link._stop_epoch == 4, 2., 'the DISARM request')))
        wait(lambda: 'stopped' in outcome, 3., 'the DISARM to settle')
        history = [h for h, _ in link.recent_states()]
        failure = [h for h in history if h.fault_code == rtp.FAULT_PLANT]
        assert len(failure) == 1 and failure[0].armed and failure[0].stop_epoch == 3 and not failure[0].holding, failure
        assert outcome['stopped'] is True and not link.latest_state()[0].armed and link.latest_state()[0].stop_epoch == 4
        assert link._fault_code == 0 and not link.latest_state()[0].faulted, 'a failure of a step under old authority latched'
        # DISARM then ARM inside the failing step: the ARM is not confirmed by the failure sample
        thread = []
        failing_step_with(lambda: (thread.append(threading.Thread(target=lambda: outcome.update(stopped=link.safe_stop()))),
                                   thread[0].start(), wait(lambda: link._stop_epoch == 5, 2., 'the DISARM request'),
                                   thread.append(threading.Thread(target=link.enable_all)), thread[1].start(),
                                   wait(lambda: link._armed, 2., 'the ARM request')))
        thread[0].join(5), thread[1].join(5)
        s_ = link.latest_state()[0]
        assert s_.armed and not s_.faulted and link._fault_code == 0, s_
        assert not [h for h, _ in link.recent_states() if h.arm_epoch == s_.arm_epoch and h.faulted], 'the new epoch inherited the fault'
        assert link.safe_stop()
        # a plant that fails every tick keeps the publish rate, latches the FIRST message and reports the last good velocity
        good = link.latest_state()[0]
        counter = iter(range(1, 10**6))

        def always(*args, **kw):
            raise RuntimeError(f'boom {next(counter)}')
        gated.inner.step = always
        seq0 = link._state_seq
        time.sleep(1.0)
        rate = link._state_seq - seq0
        assert 150 <= rate <= 350, f'{rate} samples in 1 s while the plant fails'
        assert 'boom 1' in link.motor_health()['latched_fault'], link.motor_health()['latched_fault']
        assert link.latest_state()[0].dq == good.dq and link.motor_health()['state_fresh']
        gated.inner.step = real_step_global
        assert link.safe_stop() and link.latest_state()[0].fault_code == 0
        # a plant failure AFTER another cause is latched still releases the plant (the first cause stays the published one)
        link.enable_all()
        for _ in range(20):
            link.apply_command(command([.1, 0, 0], kp=100.))
            time.sleep(.01)
        wait(lambda: link.latest_state()[0].fault_code == rtp.FAULT_CMD_LOST, 2., 'CMD_LOST')
        assert np.allclose(gated.inner._last_command['kp'], 100.)  # holding with the last command's gains
        gated.inner.step = (lambda *a, **k: (_ for _ in ()).throw(RuntimeError('late plant failure')))
        wait(lambda: not link._plant_ok, 1., 'the plant failure')
        time.sleep(.05)  # failing ticks keep publishing: every one of them carries the FIRST cause, never FAULT_PLANT
        assert not [h for h, _ in link.recent_states() if h.fault_code == rtp.FAULT_PLANT]
        assert {h.fault_code for h, _ in list(link.recent_states())[-8:]} == {rtp.FAULT_CMD_LOST}
        gated.inner.step = real_step_global
        wait(lambda: not gated.inner._last_command['kp'].any(), 1., 'the release after a plant failure under CMD_LOST')
        s = link.latest_state()[0]
        assert s.fault_code == rtp.FAULT_CMD_LOST and not s.holding and 'command stream lost' in link.motor_health()['latched_fault'], s
        assert link.safe_stop()
        link.enable_all()  # a new ARM resets the per-epoch plant flag
        assert link._plant_ok
        assert link.safe_stop()
        # a plant fault while disarmed is faulted but neither armed nor HOLDING (the plant is released)
        real_step = gated.inner.step

        def failing(*args, **kw):
            raise RuntimeError('plant exploded')
        gated.inner.step = failing
        wait(lambda: link.latest_state()[0].fault_code == rtp.FAULT_PLANT, 1., 'FAULT_PLANT while disarmed')
        s = link.latest_state()[0]
        assert s.faulted and not s.armed and not s.holding, s
        gated.inner.step = real_step
        assert link.safe_stop() and not link.latest_state()[0].faulted
        # an ARMED plant fault releases the plant (zero word) instead of holding it, and is not reported as HOLDING
        link.enable_all()
        for _ in range(20):
            link.apply_command(command([.1, 0, 0], kp=200.))
            time.sleep(.01)
        gated.inner.step = failing
        wait(lambda: link.latest_state()[0].fault_code == rtp.FAULT_PLANT, 1., 'FAULT_PLANT while armed')
        gated.inner.step = real_step
        wait(lambda: not gated.inner._last_command['kp'].any(), 1., 'the released word after a plant fault')
        s = link.latest_state()[0]
        assert s.armed and s.faulted and not s.holding, s
        assert link.safe_stop()
    finally:
        gated.release.set()
        link.close()
    # close() with a step stuck: bounded, and the plant is left open rather than closed under the running step
    stuck = Gated(plant(tmp / 'three.urdf'))
    link = SimLink(cfg, JOINTS, stuck, servo=SERVO)
    link.open()
    stuck.gate = True
    assert stuck.entered.wait(2)
    began = time.monotonic()
    link.close()
    assert time.monotonic() - began < 8, 'close() hung on a stuck step'
    assert stuck.inner.model is not None, 'the plant was closed under a running step'
    stuck.release.set()
    link._thread.join(5) if link._thread is not None else None


def scene_plant_unit(tmp):
    """The supervisor hosts a scene world (step 2b): SimLink's authority over ONE actor's slice; the other actors keep the world's
    spawn hold; observe/1 names every actor and the free objects; a refused config names what it lacks."""
    from arm_control.plants.mujoco.scene_core import ScenePlant

    root = tmp / 'scene'
    root.mkdir()
    hinge = "<body name='l{i}' pos='0 0 .1'><joint name='j{i}' type='hinge' axis='0 1 0' armature='.1'/><geom type='capsule' size='.02 .05'/>"
    (root / 'arm.xml').write_text("<mujoco><worldbody>" + "".join(hinge.format(i=i) for i in (1, 2, 3))
                                  + "</body>" * 3 + "</worldbody></mujoco>")
    (root / 'peer.xml').write_text("<mujoco><worldbody>" + hinge.format(i=1) + "</body></worldbody></mujoco>")
    (root / 'item.xml').write_text("<mujoco><worldbody><body name='part'><freejoint name='free'/><geom type='box' "
                                   "size='.02 .02 .02'/></body></worldbody></mujoco>")
    (root / 'scene.yaml').write_text(
        "version: 1\nscene:\n  actors:\n"
        "    arm: {path: arm.xml, joints: [j1, j2, j3], pos: [0, 0, 0], rpy: [0, 0, 0]}\n"
        "    peer: {path: peer.xml, joints: [j1], pos: [1, 0, 0], rpy: [0, 0, 0], q: [.3]}\n"
        "  objects:\n    item: {path: item.xml, pos: [.5, .5, .2], rpy: [0, 0, 0]}\n")
    base = write_config(tmp, 0, 0).read_text().replace("  native_reference: true\n", "")
    servo = ("sim_servo: {" + ", ".join(f"{k}: {v}" for k, v in SERVO.items()) + "}\nsim_observe: {rate_hz: 250, ring_s: 1}\n")

    def cfg(extra):
        path = tmp / 'scene_cfg.yaml'
        path.write_text(base + servo + extra)
        return load_robot_config(path)

    saved = os.environ.get('WORKCELL_SCENE')
    os.environ['WORKCELL_SCENE'] = str(root / 'scene.yaml')
    try:
        for extra, reason in (("idle_timeout_sec: 0.1\n", 'sim_actor'), ("sim_actor: arm\n", 'idle_timeout_sec'),
                              ("sim_actor: ghost\nidle_timeout_sec: 0.1\n", 'not an actor'),
                              ("sim_actor: peer\nidle_timeout_sec: 0.1\n", 'has 1 joints'),
                              ("sim_actor: arm\nidle_timeout_sec: -1\n", 'finite and positive')):
            try:
                SimLink.from_config(cfg(extra))
            except ValueError as exc:
                assert reason in str(exc), (reason, exc)
            else:
                raise AssertionError(f'a scene plant was built without {reason}')
        link = SimLink.from_config(cfg("sim_actor: arm\nidle_timeout_sec: 0.1\n"))
    finally:
        if saved is None:
            os.environ.pop('WORKCELL_SCENE', None)
        else:
            os.environ['WORKCELL_SCENE'] = saved
    scene = link._plant
    assert isinstance(scene, ScenePlant) and scene.actor == 'arm' and link.joint_names == list(JOINTS)
    core = scene._core
    peer = core.arm_slices['peer']
    peer_span = slice(peer['start'], peer['start'] + peer['n'])
    spawn_kp = core.command['kp'][peer_span].copy()
    world = (scene.backend, core.arm_slices, list(scene.backend.joint_names), core.n)
    try:
        ScenePlant(world, 'arm', ['j2', 'j1', 'j3'], period=.004, idle_timeout=.1)
    except ValueError as exc:
        assert 'in order' in str(exc), exc
    else:
        raise AssertionError('a scene plant drove the actor with its joints out of order')
    # Before the servo thread: the slice is written BEFORE the world steps, so the step a sample reports ran under that
    # sample's command (a DISARM is confirmed only by a released step).
    arm = core.arm_slices['arm']
    arm_span = slice(arm['start'], arm['start'] + 3)
    seen, inner = [], scene.backend.step
    scene.backend.step = lambda cmd=None: (seen.append(np.asarray(cmd['kp'])[arm_span].copy()), inner(cmd))[1]
    for kp in (7., 0.):
        scene.step(command([0, 0, 0], kp=kp, kd=0.))
        assert np.allclose(seen[-1], kp), (kp, seen[-1])
    scene.backend.step = inner
    # Object poses are this tick's, [x, y, z, qw, qx, qy, qz] in the world: equal to an independent kinematics pass.
    import mujoco
    for _ in range(50):
        scene.step(command([0, 0, 0]))
    poses = scene.observe()['object_poses']
    m, d = scene.backend.model, scene.backend.data
    mujoco.mj_kinematics(m, d)
    (name, pose), = poses.items()
    assert np.allclose(pose[:3], d.body(name).xpos) and np.allclose(pose[3:], d.body(name).xquat), (pose, d.body(name).xpos)
    assert pose[2] < .19, f'the part did not fall: {pose}'
    link.open()
    try:
        link.enable_all()
        target = np.array([.3, -.2, .1])
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            link.apply_command(command(target))
            time.sleep(.01)
        s = link.latest_state()[0]
        assert s.armed and not s.faulted and np.allclose(s.q, target, atol=.03), s.q
        # ONLY the arm's slice is written: the peer keeps the world's spawn hold (its own writer is step 3)
        assert np.array_equal(core.command['kp'][peer_span], spawn_kp) and np.all(spawn_kp > 0), core.command['kp']
        assert np.allclose(core.command['position'][arm['start']: arm['start'] + 3], target)
        link.safe_stop()
        wait(lambda: not np.any(core.command['kp'][arm['start']: arm['start'] + 3]), 1., 'the arm slice released on DISARM')
        assert np.array_equal(core.command['kp'][peer_span], spawn_kp), 'DISARM of the arm released another actor'
        batch = link.observations(0)
        last = batch['samples'][-1]
        world_names = list(scene.backend.joint_names)
        assert set(last['actor_q']) == {'arm', 'peer'}, last['actor_q']
        assert list(last['actor_q']['arm']) == world_names[arm['start']: arm['start'] + 3], last['actor_q']
        assert abs(list(last['actor_q']['peer'].values())[0] - .3) < .02, last['actor_q']['peer']
        poses = last['object_poses']
        assert poses and len(poses) == 1 and len(next(iter(poses.values()))) == 7, poses
        assert last['topology'] is not None and batch['errors'] == 0, (last['topology'], batch['errors'])
        # a world that fails to step is a plant fault, latched against the epoch it ran under, as for one model
        link.enable_all()
        scene._core.tick = lambda now: (_ for _ in ()).throw(RuntimeError('scene step failed'))
        wait(lambda: link.latest_state()[0].fault_code == rtp.FAULT_PLANT, 1., 'FAULT_PLANT from a failing scene step')
    finally:
        link.close()
    print('ScenePlant: one actor under SimLink authority, others keep their hold, actor-keyed observations, plant fault: PASS',
          flush=True)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='sim_supervisor_'))
    cfg_path = write_config(tmp, 0, 0)  # also writes three.urdf; the rt ports are unused by the sim plant
    plant_model_checks(tmp)
    link_unit(tmp)
    observe_unit(tmp)
    link_races(tmp)
    scene_plant_unit(tmp)
    try:
        SimLink.from_config(load_robot_config(cfg_path))  # the fixture config has no sim_servo yet
    except ValueError as exc:
        assert 'sim_servo' in str(exc), exc
    else:
        raise AssertionError('a sim plant was built without sim_servo')
    print('SimLink servo semantics: PASS', flush=True)
    # the single-model plant builder shared with the Dora mujoco_interface node reads its keys from the config
    from arm_control.plants.mujoco.single import build_single_model_backend
    keyed = cfg_path.read_text().replace("  ee_frame: l3\n", "  ee_frame: l3\n  gripper_joints: []\n")
    keyed_path = tmp / 'keyed.yaml'
    keyed_path.write_text(keyed + f"sim_model_path: {tmp / 'three.urdf'}\nsim_timestep: 0.002\nsim_gravcomp: true\n"
                          "sim_default_joint_positions: {j1: 0.3}\n")
    built = build_single_model_backend(load_robot_config(keyed_path), control_period=0.01)
    assert (built.timestep, built.control_period, built.ground_z, built.gravcomp_prefixes) == (0.002, 0.01, None, ("",)), built
    assert built.default_joint_positions == {'j1': 0.3} and built.ee_body == 'l3' and built.joint_names == list(JOINTS)
    keyed_path.write_text(keyed + f"sim_model_path: {tmp / 'three.urdf'}\n")
    plain = build_single_model_backend(load_robot_config(keyed_path), control_period=0.01)
    assert plain.timestep == 0.001 and plain.gravcomp_prefixes == () and plain.default_joint_positions == {}, plain
    text = cfg_path.read_text().replace("  ee_frame: l3\n", "  ee_frame: l3\n  gripper_joints: []\n")
    text = text.replace("  native_reference: true\n", "")
    text += (f"sim_model_path: {tmp / 'three.urdf'}\nsim_gravcomp: true\nsim_timestep: 0.001\n"
             f"sim_servo: {{state_hz: {SERVO['state_hz']}, hold_ms: {SERVO['hold_ms']}, fault_ms: {SERVO['fault_ms']}, "
             f"hold_kp: {SERVO['hold_kp']}, hold_kd: {SERVO['hold_kd']}}}\n"
             "sim_observe: {rate_hz: 250, ring_s: 2}\n")
    cfg_path.write_text(text)
    sock = tmp / 'sdk_arm.sock'
    log = open(tmp / 'supervisor.log', 'w')
    sup = subprocess.Popen([sys.executable, '-m', 'arm_control.supervisor.service', '--config', str(cfg_path), '--socket',
                            str(sock), '--plant', 'sim'], stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
    arm = None
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
        # the simulation advertises what it implements: no native reference, so no stream/group
        assert arm.extensions.get('controller') == 1 and arm.extensions.get('state_history') == 1 \
            and arm.extensions.get('trace') == 1, arm.extensions
        assert 'stream' not in arm.extensions and 'group' not in arm.extensions, arm.extensions
        # observe/1: read-only, no claim needed; the cursor advances and never repeats a sample
        assert arm.extensions.get('observe') == 1, arm.extensions
        seen = wait(lambda: arm.observe()['samples'], what='an observation')
        batch = arm.observe()
        assert batch['rate_hz'] == 250. and batch['samples'][-1]['actor_q']['sdk_arm'].keys() == set(JOINTS), batch
        later = wait(lambda: arm.observe(after=batch['latest_seq'])['samples'], what='a later observation')
        assert later[0]['seq'] > batch['latest_seq'] and later[0]['sim_time_s'] > batch['samples'][-1]['sim_time_s'], later
        assert seen
        from arm_control.sdk import _observation
        bad = dict(batch, samples=[dict(batch['samples'][-1], actor_q=[1, 2])])
        try:
            _observation(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('the SDK decoded an observation whose actor_q is not a mapping')
        # A slow reader cannot slow the plant (design 2026-10-08 §3 rule 1): four tight readers at the cap, on their own
        # connections, while the plant's own clock is measured from the observations (its sim time against wall time).
        # Separate PROCESSES: reader threads in one process throttle each other on that process's own interpreter lock and
        # never load the supervisor enough to show the effect.
        loop = ("import sys\nfrom arm_control.sdk import ArmClient\nr = ArmClient('sdk_arm', socket_path=sys.argv[1])\n"
                "while True:\n    r.observe(0, 1024)\n")
        readers = [subprocess.Popen([sys.executable, '-c', loop, str(sock)], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, env=dict(os.environ)) for _ in range(4)]
        time.sleep(.5)
        t0, a = time.monotonic(), arm.observe(after=10**9)['latest_seq']
        s0 = arm.observe(after=a - 1)['samples'][-1]['sim_time_s']
        time.sleep(3.0)
        t1, b = time.monotonic(), arm.observe(after=10**9)['latest_seq']
        s1 = arm.observe(after=b - 1)['samples'][-1]['sim_time_s']
        assert all(r.poll() is None for r in readers), 'a reader died: the plant was not under load'
        for r in readers:
            r.kill()
            r.wait(timeout=5)
        speed = (s1 - s0) / (t1 - t0)
        assert speed >= .95, f'four observe readers slowed the plant to {speed:.3f}x real time'
        print(f'observe/1 under 4 tight readers: plant at {speed:.3f}x real time', flush=True)
        assert arm.authority().lifecycle is Lifecycle.CONNECTED
        arm.claim()
        assert arm.authority().lifecycle is Lifecycle.CLAIMED
        arm.arm()
        authority = arm.authority()
        assert authority.lifecycle is Lifecycle.ARMED and authority.arm_epoch >= 1, authority
        # a move completes through SENT and APPLIED, ends near the target
        q0 = np.asarray(arm.state().q)
        status = arm.execute(move('m1', q0, q0 + [.2, -.1, .1], 1.5, start=Start.ON_ADMIT))
        assert status.phase is not Phase.REJECTED, status
        done = arm.wait(status.execution_id, 15)
        assert done.phase is Phase.COMPLETED, (done.phase, [r.phase.value for r in arm.receipts(status.execution_id)])
        completed_chain(arm, status.execution_id)
        assert np.allclose(arm.state().q, q0 + [.2, -.1, .1], atol=.05), (arm.state().q, q0)
        # cartesian_impedance (RT Cartesian 2b): offered because the twin runs the law; a leg runs it through execution and
        # settling (trace/1 law_pc) and every word after it is the joint law; a leg under the joint-kp floor is refused
        assert 'cartesian_impedance' in arm.capabilities.profiles, arm.capabilities
        law = CartesianImpedance(np.eye(3), [500., 500., 500., 0., 0., 0.], [40., 40., 40., 0., 0., 0.])
        q1 = np.asarray(arm.state().q)
        weak = arm.execute(move('c0', q1, q1 + [.1, .1, -.1], 1.5, start=Start.ON_ADMIT, profile='cartesian_impedance',
                                cartesian=law, gains=JointGains([10.] * 3, [5.] * 3)))
        assert weak.phase is Phase.REJECTED and 'joint kp >= 20' in weak.reason, weak
        cart = arm.execute(move('c1', q1, q1 + [.1, .1, -.1], 1.5, start=Start.ON_ADMIT, profile='cartesian_impedance',
                                cartesian=law))
        assert cart.phase is not Phase.REJECTED, cart
        assert arm.wait(cart.execution_id, 15).phase is Phase.COMPLETED, [r.phase.value for r in arm.receipts(cart.execution_id)]
        assert np.allclose(arm.state().q, q1 + [.1, .1, -.1], atol=.05), (arm.state().q, q1)
        time.sleep(.3)
        ticks = arm.trace_samples(0)['samples']
        mine = [t for t in ticks if t['plan_id'] == 'c1']
        assert mine and {t['law_pc'] for t in mine} == {'cartesian_impedance'}, {t['law_pc'] for t in mine}
        after = [t['law_pc'] for t in ticks if t['seq'] > mine[-1]['seq']]
        assert after and set(after) == {'joint'}, set(after)
        # a cancel mid-run is CANCELLED, the arm holds, and a later move still runs
        q1 = np.asarray(arm.state().q)
        running = arm.execute(move('m2', q1, q1 + [.3, .3, .3], 4.0, start=Start.ON_ADMIT))
        time.sleep(.5)
        assert arm.cancel(running.execution_id).phase is Phase.CANCELLED
        # the negotiated reads work over the simulated link
        history = arm.state_history()
        assert history['samples'] and history['clock_domain'] is None, history['clock_domain']
        trace = arm.trace_samples(0)
        assert trace['samples'] and trace['errors'] == 0 and trace['samples'][-1]['seq'] == trace['latest_seq']
        # DISARM, then ARM again: a new ARM epoch
        first_epoch = arm.authority().arm_epoch
        arm.disarm()
        assert arm.authority().lifecycle is Lifecycle.CLAIMED
        arm.arm()
        assert arm.authority().arm_epoch > first_epoch, (first_epoch, arm.authority())
        try:
            arm.open_stream(None)
        except Refused:
            pass
        else:
            raise AssertionError('a stream opened on a simulation with no native reference')
        arm.disarm()
    finally:
        if arm is not None:
            arm.close()
        sup.terminate()
        try:
            sup.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sup.kill()
    print('PASS: the supervisor over a simulated plant (SimLink): servo semantics, claim/arm, a move through SENT and APPLIED, '
          'a cartesian_impedance leg (the law through its leg only; under the kp floor refused), '
          'cancel, DISARM and a new ARM epoch, state_history/trace, no stream or group advertised')


if __name__ == '__main__':
    main()
