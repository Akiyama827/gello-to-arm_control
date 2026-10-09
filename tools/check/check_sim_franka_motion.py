#!/usr/bin/env python3
"""The FR3 sim motion graph's supervisor comes up as `examples/run_console.py` starts it, and serves what the graph's nodes use.

    PYTHONPATH=. python tools/check/check_sim_franka_motion.py

Starts the supervisor over `examples/configs/sim_arm.yaml` with the launcher's overlay and mode config (no Dora graph), then:
* `supervised_arm`'s start check holds: the gains it resolves from the merged config are the ones the supervisor runs;
* setpoint/1 (the console jog) is offered and moves the twin: ARMed, re-sent setpoints carry joint 1 to its target;
* the Hand answers on the hand_bridge protocol at rt.host:rt.hand_port, where `franka_gripper` connects (never the RT box).
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_control.config import load_robot_config  # noqa: E402
from arm_control.sdk import ArmClient, Refused  # noqa: E402
from arm_control.supervisor.dora_arm import default_gains_and_presets  # noqa: E402
from arm_control.supervisor.launch import start_supervisor, stop_supervisor  # noqa: E402
from examples.run_console import DEFAULT_CONFIG, MODE_PROFILE, SINGLE, SUPERVISED_PROFILE  # noqa: E402


def main() -> int:
    assert "supervised_arm" in SINGLE.read_text() and "arm_controller.py" not in SINGLE.read_text(), SINGLE
    env = dict(os.environ, ARM_CONTROL_ROOT=str(ROOT), PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    supervisor, graph_env = start_supervisor(DEFAULT_CONFIG, env, overlays=(SUPERVISED_PROFILE,), control_mode=MODE_PROFILE,
                                             cwd=ROOT, timeout_s=300.0)
    arm = None
    try:
        cfg = load_robot_config(graph_env["ARM_CONTROL_CONFIG"])
        rt = cfg.get("rt")
        assert rt["host"] == "127.0.0.1" and rt["hand_port"] == cfg.get("sim_hand")["port"], rt
        arm = ArmClient(rt["instance_id"], socket_path=graph_env["ARM_CONTROL_SOCKET"])
        kp, kd, _ = default_gains_and_presets(cfg)
        law = arm.controller_state()
        assert np.allclose(law.kp, kp) and np.allclose(law.kd, kd), (law.kp, kp, law.kd, kd)
        assert arm.extensions.get("setpoint") == 1 and "cartesian_impedance" in arm.capabilities.profiles, arm.extensions
        with socket.create_connection((rt["host"], int(rt["hand_port"])), timeout=2.0) as hand:
            hand.settimeout(2.0)
            data, end = b"", time.monotonic() + 2.0
            while b"STATE " not in data and time.monotonic() < end:
                data += hand.recv(4096)
        assert b"STATE " in data, data[:200]
        arm.claim()
        arm.arm()
        q0 = np.asarray(arm.state().q)
        end = time.monotonic() + 5.0
        while True:  # the first setpoint is taken once the controller has a fresh measured pose since ARM
            try:
                arm.setpoint(q0)
                break
            except Refused as exc:
                assert "fresh measured pose" in str(exc) and time.monotonic() < end, exc
                time.sleep(.02)
        target = q0 + np.eye(len(q0))[0] * .05
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            timeout_s = arm.setpoint(target)
            time.sleep(.05)
        moved = np.asarray(arm.state().q)
        assert abs(moved[0] - target[0]) < .01, (moved, target)
        assert timeout_s > 0  # the expiry hold (measured pose, not the target) is check_dora_arm's
        arm.disarm()
    finally:
        if arm is not None:
            arm.close()
        stop_supervisor(supervisor)
    print("sim franka motion: the supervisor started as run_console does runs the gains supervised_arm resolves, offers setpoint/1 "
          "(a jog moves the FR3 twin) and cartesian_impedance, and serves the Hand at rt.host:rt.hand_port: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
