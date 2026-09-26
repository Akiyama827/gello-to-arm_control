"""Headless FR3 Soft disturbance check; pass the freshly built servo .so.

No hardware connection. Numerical limits belong to the identified historical
fixture, not to arbitrary nominal or calibrated FR3 descriptions.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

FIXTURE_ID = "legacy-fr3-soft-20260907"
FIXTURE_SHA256 = "94bd52bf6d1b5a87312a5edbf61f3ea91c8a34a1236e0e4cc1785611a5d4957d"


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("servo_binding", nargs="?", help="freshly built arm_rt_servo .so")
    parser.add_argument("--fixture", type=Path, default=root / "franka/urdf/fr3.urdf",
                        help=f"source URDF for {FIXTURE_ID}; meshes must resolve beside it")
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    if not fixture.is_file():
        parser.error(f"fixture/provenance mismatch: missing URDF {fixture}; "
                     "controller regression NOT RUN")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if digest != FIXTURE_SHA256:
        parser.error(f"fixture/provenance mismatch for {FIXTURE_ID}: {fixture} "
                     f"has SHA256 {digest}, expected {FIXTURE_SHA256}; "
                     "controller regression NOT RUN. Supply --fixture with the "
                     "historical URDF; these thresholds do not qualify another model.")
    print(json.dumps(dict(fixture_id=FIXTURE_ID, fixture_path=str(fixture),
                          fixture_sha256=digest, fixture_identity="verified")), flush=True)
    if args.servo_binding:
        spec = importlib.util.spec_from_file_location("arm_rt_servo", args.servo_binding)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["arm_rt_servo"] = module

    from arm_control.config import load_config_tree
    from arm_control.plants.mujoco.backend import MuJoCoBackend

    cfg = load_config_tree(root / "examples/configs/sim_arm.yaml")
    mode = load_config_tree(root / "examples/profiles/motion_franka.yaml")
    soft = mode["gain_presets"]["soft"]["pose_hold"] | {"id": 1}
    plant = MuJoCoBackend(
        joint_names=cfg["arm"]["joints"], single_model_path=fixture,
        timestep=.001, control_period=.01, ee_body=cfg["arm"]["ee_frame"],
        enable_self_collision=True, default_joint_positions=cfg["sim_default_joint_positions"],
        gravcomp_prefixes=("",), gripper_joints=tuple(cfg["arm"]["gripper_joints"]),
    )
    plant.load()
    assert plant.supports_pose_hold
    q0 = plant.data.qpos[plant._qadr].copy()
    p0 = plant.data.body(plant._ee_bid).xpos.copy()
    r0 = plant.data.body(plant._ee_bid).xquat.copy()
    cmd = dict(position=q0, velocity=np.zeros(7), torque=np.zeros(7),
               kp=cfg["arm"]["kp"], kd=cfg["arm"]["kd"], pose_hold=soft)
    samples = []
    for tick in range(1000):
        # Push the elbow with a physical external joint torque, not a
        # pre-projected synthetic nullspace force.
        plant.data.qfrc_applied[:] = 0
        if 200 <= tick < 500:
            plant.data.qfrc_applied[plant._vadr[2]] = 1.0
        plant.step(cmd)
        body = plant.data.body(plant._ee_bid)
        samples.append((np.linalg.norm(body.xpos-p0),
                        2*np.arccos(np.clip(abs(body.xquat @ r0), 0, 1)),
                        np.max(np.abs(plant.data.qpos[plant._qadr]-q0)),
                        np.max(np.abs(plant.data.qvel[plant._vadr]))))
    values = np.asarray(samples)
    assert np.isfinite(values).all()
    assert values[200:500, 2].max() > .05, "elbow must be compliant"
    assert values[:, 0].max() < .02, "position departed more than 2 cm"
    assert values[:, 1].max() < .15, "orientation departed more than 0.15 rad"
    assert values[-100:, 0].max() < .005, "position did not recover"
    assert values[-100:, 1].max() < .03, "orientation did not recover"
    assert values[-100:, 3].max() < .05, "release did not settle"
    # Joint fallback must capture the current posture, not the pre-Soft one.
    q_exit = plant.data.qpos[plant._qadr].copy()
    cmd.update(pose_hold=None, position=q_exit)
    for _ in range(200):
        plant.step(cmd)
    assert np.max(abs(plant.data.qpos[plant._qadr]-q_exit)) < .01
    track_displacement = 0.0
    for tick in range(500):
        plant.data.qfrc_applied[plant._vadr[2]] = 1.0 if tick < 300 else 0.0
        plant.step(cmd)
        track_displacement = max(track_displacement, float(np.max(
            abs(plant.data.qpos[plant._qadr]-q_exit))))
    assert values[200:500, 2].max() > 10*track_displacement
    print(json.dumps(dict(max_position_error_m=float(values[:, 0].max()),
                         max_orientation_error_rad=float(values[:, 1].max()),
                         max_joint_displacement_rad=float(values[:, 2].max()),
                         final_position_error_m=float(values[-1, 0]),
                         final_orientation_error_rad=float(values[-1, 1]),
                         final_max_joint_speed_rad_s=float(values[-1, 3]),
                         track_joint_displacement_rad=track_displacement), indent=2))
    print("pose hold sim: elbow compliance, pose recovery, measured joint exit OK")


if __name__ == "__main__":
    main()
