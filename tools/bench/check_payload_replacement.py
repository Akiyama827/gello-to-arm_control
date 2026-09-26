"""Offline complete-payload transitions using the configured Pinocchio model.

Use --snapshot-only before a refactor, then --baseline with that JSON afterward.
No plant or transport is constructed. Defaults use the library asset root.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from arm_control.config import CONTROL_ROOT, arm_joints, ee_frame, load_config_tree
from arm_control.control.doubles import FakeNode
from arm_control.control.factory import arm_urdf, build_arm_controller, build_executor
from arm_control.dynamics import PinocchioDynamics
from arm_control.motion import JointState, TrajectoryPoint
from arm_control.messages import pack_control_update


def snapshot(cfg):
    chunks = []
    # Both gravity-compensated and PC-gravity paths; no gravity convention changes.
    for compensated in (False, True):
        config = deepcopy(cfg)
        config["arm"]["plant_gravity_comp"] = compensated
        for shift in (0., .01, -.01, .02):
            q = np.array([0., -.785, 0., -2.356, 0., 1.571, .785]) + shift
            executor = build_executor(config)
            for mass, com in ((0., None), (.5, [.02, -.03, .12]),
                              (.3, [-.01, .04, .08]), (0., None)):
                executor.set_payload(mass, com_offset=com)
                state = JointState(q, np.zeros(7))
                point = TrajectoryPoint(0., q + .001, np.full(7, .01), np.full(7, .02))
                for command in (executor.hold_command(state), executor.step(point, state)):
                    chunks.extend(getattr(command, key).tobytes() for key in
                                  ("q_des", "qd_des", "tau_ff", "kp", "kd"))
    return dict(urdf_sha256=hashlib.sha256(Path(arm_urdf(cfg)).read_bytes()).hexdigest(),
                commands=64, commands_sha256=hashlib.sha256(b"".join(chunks)).hexdigest())


def transitions(cfg):
    state = JointState(np.array([0., -.785, 0., -2.356, 0., 1.571, .785]), np.zeros(7))
    # Real controller update decoding, real executor, real model; no device link.
    controller = build_arm_controller(cfg, FakeNode(), arm_id="payload-check")
    fresh = build_arm_controller(cfg, FakeNode(), arm_id="payload-check").executor

    def declare(mass, com=None):
        controller.on_control(pack_control_update(payload={"mass_kg": mass, "com_ee": com}))
        return controller.executor.hold_command(state).tau_ff

    unloaded = fresh.hold_command(state).tau_ff
    loaded = declare(.5, [.02, -.03, .12])
    assert np.max(abs(loaded - unloaded)) > .01
    fresh.set_payload(.5)
    at_origin = fresh.hold_command(state).tau_ff
    assert np.array_equal(declare(.5), at_origin), "new payload inherited previous CoM"
    declare(.5, [.02, -.03, .12])
    assert np.array_equal(declare(0.), unloaded), "release retained payload torque"
    assert controller.executor._payload == (0., None), "clear retained transient CoM"
    assert np.array_equal(declare(.5), at_origin), "release/regrasp retained previous CoM"
    assert np.array_equal(declare(0., [1., 2., 3.]), unloaded)
    assert controller.executor._payload == (0., None), "zero mass must discard supplied CoM"
    com = np.array([.02, -.03, .12])
    controller.executor.set_payload(.5, com_offset=com)
    com[:] = 100.
    assert np.array_equal(controller.executor.hold_command(state).tau_ff, loaded), "caller mutated payload value"
    assert controller.executor._payload_frame == ee_frame(cfg)
    try:
        controller.executor.set_payload(.5, frame_name="fr3_link8")
    except TypeError:
        pass
    else:
        raise AssertionError("transient declaration can still change construction-owned frame")
    return "controller decoding, replace, clear, regrasp, zero-offset, copied CoM, fixed frame: PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONTROL_ROOT / "examples/configs/sim_arm.yaml")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    cfg = load_config_tree(args.config)
    assert len(arm_joints(cfg)) == 7, "this focused fixture is an FR3"
    if cfg["arm"].get("max_tau") is None:
        # Generic example's operating caps normally come from its mode profile;
        # this offline dynamics comparison uses the description's effort limits.
        cfg["arm"]["max_tau"] = PinocchioDynamics(arm_urdf(cfg), arm_joints(cfg)).effort_limits.tolist()
    parity = snapshot(cfg)
    if args.snapshot_only:
        print(json.dumps(parity, indent=2))
        return
    if args.baseline:
        assert parity == json.loads(args.baseline.read_text()), "explicit-payload command parity changed"
    print(json.dumps(dict(parity=parity, transitions=transitions(cfg),
                          baseline_parity="PASS" if args.baseline else "not requested"), indent=2))


if __name__ == "__main__":
    main()
