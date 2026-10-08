"""Fake RT regression for v5 streamed motion across a stalled PC tick.

The server uses the legacy streamed command path. A 100 ms RT hold must end
the current execution rather than let its wall-clock trajectory resume.
"""
from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import re

from arm_control.config import load_robot_config
from arm_control.execution.contract import Lifecycle, Phase, TrajectoryRequest
from arm_control.motion import JointTrajectory
from arm_control.supervisor.runtime import RobotRuntime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_sdk_supervisor import JOINTS, free_port, write_config
from _bench_target import parse_target
from _envelope_age import EnvelopeAgeMonitor


def until(fn, timeout=12.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = fn()
        if value:
            return value
        time.sleep(.005)
    raise AssertionError("timed out waiting for " + str(fn))


def scenario(binary, stall_s, *, delayed_health=False, remote=None):
    udp = remote.udp_port if remote else free_port(socket.SOCK_DGRAM)
    tcp = remote.tcp_port if remote else free_port(socket.SOCK_STREAM)
    instance = remote.instance_id if remote else 'sdk_arm'
    joints = ('j0', 'j1', 'j2') if remote else JOINTS
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode="w+") as log:
        path = write_config(Path(tmp), udp, tcp)
        cfg_text = path.read_text().replace("native_reference: true", "native_reference: false")
        cfg_text = cfg_text.replace("done_pos_tol: 0.02", "done_pos_tol: 0.06")
        cfg_text = cfg_text.replace('sdk_arm', instance)
        if remote:
            cfg_text = cfg_text.replace('host: 127.0.0.1', f'host: {remote.host}')
            cfg_text = re.sub(r'\bj([123])\b', lambda m: f'j{int(m[1])-1}', cfg_text)
            urdf = Path(tmp) / 'three.urdf'
            urdf.write_text(re.sub(r'\bj([123])\b',
                                   lambda m: f'j{int(m[1])-1}', urdf.read_text()))
        path.write_text(cfg_text)
        proc = (None if remote else subprocess.Popen([
            binary, "--backend", "fake", "--n", "3", "--instance-id", instance,
            "--joint-names", ",".join(joints), "--udp-port", str(udp),
            "--tcp-port", str(tcp), "--hold-ms", "100", "--fault-ms", "1000",
        ], stdout=log, stderr=log))
        runtime = RobotRuntime(load_robot_config(path))
        ages = EnvelopeAgeMonitor(runtime.link, include_all_states=True)
        pause = threading.Event()
        entered = threading.Event()
        finished = threading.Event()
        original_step = runtime._step
        commands = []
        entered_at = []
        original_apply = runtime.link.apply_command
        original_health = runtime.link.motor_health
        original_history = runtime.link.recent_states

        def stalled_step():
            if pause.is_set():
                pause.clear()
                old_health = original_health()
                old_history = original_history()
                entered_at.append(time.monotonic())
                entered.set()
                time.sleep(stall_s)
                finished.set()
                if delayed_health:
                    # The UDP receiver has seen the RT hold, but the resumed
                    # controller tick has not consumed that health/state edge.
                    runtime.link.motor_health = lambda: old_health
                    runtime.link.recent_states = lambda: old_history
                    try:
                        return original_step()
                    finally:
                        runtime.link.motor_health = original_health
                        runtime.link.recent_states = original_history
            return original_step()

        def record_command(command):
            state, _ = runtime.link.latest_state()
            commands.append((time.monotonic(), np.array(command["position"], copy=True),
                             np.array(command["velocity"], copy=True),
                             bool(state and state.holding),
                             np.array(state.q_cmd, copy=True) if state else None))
            return original_apply(command)

        runtime._step = stalled_step
        runtime.link.apply_command = record_command
        try:
            time.sleep(.15)
            runtime.start()
            assert runtime.link.backend_name == 'fake', 'bench refuses a non-fake RT backend'
            assert runtime.link.hold_ms == 100, runtime.link.hold_ms
            assert runtime._native is None
            runtime.claim("stall-check")
            gen = runtime.arm("stall-check")
            assert runtime.await_authority(gen, armed=True, wait_s=3).lifecycle is Lifecycle.ARMED
            q0 = runtime.state().q
            # Constant 0.4 rad/s on one joint; the endpoint lies within the
            # analytical fixture's limits and leaves ample time for the stall.
            t = np.array([0., 1.2])
            q = np.stack([q0, q0 + np.array([.48, 0., 0.])])
            v = np.array([[.4, 0., 0.], [.4, 0., 0.]])
            request = TrajectoryRequest(
                f"stream-{int(stall_s * 1000)}", instance, joints,
                JointTrajectory(t, q, v),
            )
            ex = runtime.submit("stall-check", request)
            until(lambda: Phase.APPLIED in runtime.status(ex.execution_id).reached)
            until(lambda: any(c[2][0] > .35 for c in commands))
            until(lambda: runtime.link.latest_state()[0].q_cmd[0] - q0[0] > .04)
            assert not runtime.link.latest_state()[0].holding
            pause.set()
            assert entered.wait(2)
            if stall_s >= .1:
                until(lambda: (
                    runtime.link.latest_state()[0]
                    if runtime.link.latest_state()[0].holding else None), timeout=2)
                hold_time = time.monotonic()
            assert finished.wait(2)
            status = until(lambda: (
                runtime.status(ex.execution_id)
                if runtime.status(ex.execution_id).terminal else None), timeout=7)
            if stall_s >= .1:
                assert status.phase is Phase.FAULTED, status
                expected = ("PC streamed command gap" if delayed_health
                            else "RT command-staleness hold")
                assert expected in status.reason, status
                after = [c for c in commands if c[0] >= hold_time]
                assert after, "no post-hold PC command"
                assert all(np.max(np.abs(c[2])) < 1e-12 for c in after), after[-3:]
                # A delayed in-flight packet may briefly release and then
                # re-enter RT hold. Compare against the RT-selected target at
                # the actual PC handoff, not the first observed holding frame.
                prior = [c for c in commands if c[0] < entered_at[0]]
                expected_target = prior[-1][1] if delayed_health else after[0][4]
                assert after[0][3] and np.max(np.abs(after[0][1] - expected_target)) < 1e-8, (
                    "PC handoff stepped from the last sent/RT hold target", after[0])
                assert max(np.max(np.abs(c[1] - after[0][1])) for c in after) < 1e-8, (
                    "advancing q_des after RT hold", after[-3:])
            else:
                assert status.phase is Phase.COMPLETED, status
                assert not any(c[3] for c in commands if c[0] >= entered_at[0]), (
                    "short stall entered RT hold")
            assert runtime.authority().lifecycle is Lifecycle.ARMED
            print(f"streamed_hold stall={int(stall_s * 1000)}ms: {status.phase.value}"
                  f" commands={len(commands)}")
            ages.report(f'streamed_hold {int(stall_s * 1000)}ms'
                        f'{" delayed-health" if delayed_health else ""}')
        except BaseException:
            log.seek(0)
            print(log.read()[-4000:])
            raise
        finally:
            ages.stop()
            runtime.close()
            if proc is not None:
                proc.terminate()
                proc.wait(3)


def main(binary, remote=None):
    for duration in (0., .05, .2, .5):
        scenario(binary, duration, remote=remote)
    scenario(binary, .115, delayed_health=True, remote=remote)
    print("streamed_hold: PASS")


if __name__ == "__main__":
    main(*parse_target(__doc__))
