"""The supervisor over a simulated plant (SimLink), driven through the SDK, with the plant's servo semantics checked directly.

    python tools/bench/check_sim_supervisor.py

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
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sdk_supervisor import JOINTS, completed_chain, move, write_config  # noqa: E402

from arm_control.config import load_robot_config  # noqa: E402
from arm_control.execution.contract import Lifecycle, Phase, Start  # noqa: E402
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


def link_unit(tmp):
    cfg = RtConfig(instance_id='sim_arm', state_timeout_s=.5)
    for bad in ({k: v for k, v in SERVO.items() if k != 'hold_ms'}, dict(SERVO, fault_ms=50), dict(SERVO, fault_ms=100),
                dict(SERVO, state_hz=0), dict(SERVO, state_hz=1e7), dict(SERVO, hold_kp=float('nan')), dict(SERVO, hold_kp=-5),
                dict(SERVO, hold_kd=-1), dict(SERVO, hold_ms='soon'), dict(SERVO, hold_ms=None), dict(SERVO, hold_ms=float('inf'))):
        try:
            SimLink(cfg, JOINTS, object(), servo=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted a bad sim_servo {bad}')
    link = SimLink(cfg, JOINTS, plant(tmp / 'three.urdf'), servo=SERVO)
    link.open()
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


def main():
    tmp = Path(tempfile.mkdtemp(prefix='sim_supervisor_'))
    cfg_path = write_config(tmp, 0, 0)  # also writes three.urdf; the rt ports are unused by the sim plant
    link_unit(tmp)
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
             f"hold_kp: {SERVO['hold_kp']}, hold_kd: {SERVO['hold_kd']}}}\n")
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
          'cancel, DISARM and a new ARM epoch, state_history/trace, no stream or group advertised')


if __name__ == '__main__':
    main()
