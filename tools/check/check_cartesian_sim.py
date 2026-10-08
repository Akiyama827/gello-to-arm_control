"""The cartesian_impedance law on the simulated arm a config describes: the RT box's v8 law on the twin (operator 2026-10-09).

The spec carries the task frame in the robot BASE frame and K_c/D_c, no pose: the plant targets FK(q_des) with its own model and
rotates the task frame by its own base orientation. Checked on the robot's description (plants/mujoco/single):
* with the mount yawed 90 deg, a stiffness along base x pushes along the BASE's x in the world (a law applying the task frame in
  world coordinates fails), at K_c times the FK(q_des) displacement, with no moment when the rotational K_c is zero;
* at q_des = q and at rest the law adds nothing;
* closed loop with the bench's gains (joint kp at the 20 N.m/rad floor, K_c [1000, 200, 200, 20, 5, 5]) the EE settles on
  FK(q_des) and stays finite;
* SimLink's v8 rules: a base-frame spec reaches the plant, a world-frame or out-of-envelope one is refused, and a hold (lost
  stream) drops the Cartesian law -- the hold is the joint law only, as on the RT box.
"""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np


def wrench(plant, tau):
    """The EE wrench whose J^T is ``tau`` (world frame; J is 6 x n, full row rank on a 7-DOF arm)."""
    mujoco.mj_jacBody(plant.model, plant.data, plant._jacp, plant._jacr, plant._ee_bid)
    jac = np.vstack((plant._jacp[:, plant._vadr], plant._jacr[:, plant._vadr]))
    w, *_ = np.linalg.lstsq(jac.T, tau, rcond=None)
    assert np.allclose(jac.T @ w, tau, atol=1e-9), "tau is not in J^T's range"
    return w


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "examples/configs/sim_arm.yaml", help="robot entry config")
    args = parser.parse_args()

    from arm_control.config import arm_joints, load_robot_config
    from arm_control.plants.mujoco.single import build_single_model_backend

    cfg = load_robot_config(args.config)
    plant = build_single_model_backend(cfg, control_period=.01)
    plant.load()
    assert plant.supports_cartesian, "the twin cannot run the Cartesian law"
    model, data, n = plant.model, plant.data, len(plant.joint_names)
    base, ee = plant._base_bid, plant._ee_bid
    assert plant.ee_chain_joints and set(plant.ee_chain_joints) <= set(plant.joint_names), plant.ee_chain_joints

    # 1. Mount yawed 90 deg about world z.
    yaw = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    turned = np.empty(4)
    mujoco.mju_mulQuat(turned, yaw, model.body_quat[base].copy())
    model.body_quat[base] = turned
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    R_wb = data.xmat[base].reshape(3, 3)
    q0 = data.qpos[plant._qadr].copy()
    q_des = q0 + np.r_[.03, .02, -.02, .03, 0., .02, 0.][:n]
    target = plant._ee_pose_at(q_des)
    disp = target[:3] - data.xpos[ee]
    base_x = R_wb[:, 0]
    assert abs(base_x @ disp) > 1e-3 and np.linalg.norm(np.cross(base_x, disp)) > 1e-3, "displacement must not be degenerate"
    k = 1000.
    spec = dict(frame="base", task_R=np.eye(3), kc=np.r_[k, 0, 0, 0, 0, 0], dc=np.zeros(6))
    w = wrench(plant, plant._cartesian_tau_ff(np.zeros(n), spec, q_des))
    expected = k * base_x * (base_x @ disp)
    assert np.allclose(w[:3], expected, atol=1e-6 * k), (w[:3], expected)
    assert np.allclose(w[3:], 0, atol=1e-9), w[3:]
    world_x = np.array([1., 0, 0])
    assert np.linalg.norm(w[:3] - k * world_x * (world_x @ disp)) > .5, "base and world frames are indistinguishable here"
    # The spec's own task frame is honoured too: a task frame whose X is base y pushes along the base's y.
    swapped = dict(spec, task_R=np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]))
    w = wrench(plant, plant._cartesian_tau_ff(np.zeros(n), swapped, q_des))
    assert np.allclose(w[:3], k * R_wb[:, 1] * (R_wb[:, 1] @ disp), atol=1e-6 * k), w[:3]
    for bad in (dict(spec, frame="world"), {key: value for key, value in spec.items() if key != "frame"}):
        try:
            plant._cartesian_tau_ff(np.zeros(n), bad, q_des)
        except ValueError:
            continue
        raise AssertionError(f"a spec without the base frame was applied: {sorted(bad)}")

    # 2. Target = here, at rest: the law adds nothing.
    full = dict(spec, kc=np.r_[1000, 200, 200, 20, 5, 5], dc=np.r_[130, 40, 40, .9, .45, .45])
    assert np.allclose(plant._cartesian_tau_ff(np.zeros(n), full, q0), 0, atol=1e-9)

    # 3. Closed loop with the bench's gains: the joint law at the floor plus the Cartesian law, through the compiled servo.
    command = dict(position=q_des, velocity=np.zeros(n), torque=np.zeros(n), kp=np.full(n, 20.), kd=np.full(n, 5.),
                   cartesian=full)
    for _ in range(int(2. / plant.control_period)):
        plant.step(command)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    error = np.linalg.norm(data.xpos[ee] - plant._ee_pose_at(q_des)[:3])
    assert error < 2e-3, f"EE {error * 1e3:.2f} mm from FK(q_des) after 2 s"
    print(f"cartesian twin: base-frame task axis under a 90 deg mount, zero at the target, closed loop {error * 1e3:.2f} mm "
          "from FK(q_des)", flush=True)

    # 4. SimLink: the v8 rules and the hold.
    from arm_control.plants.remote_rt.client import RtConfig, RtLinkError
    from arm_control.plants.sim_link import SimLink

    link = SimLink(RtConfig(instance_id="sim_arm", state_timeout_s=.5), arm_joints(cfg), plant,
                   servo=dict(state_hz=250, hold_ms=100, fault_ms=600, hold_kp=50, hold_kd=5))
    link.open()
    try:
        assert link.supports_cartesian and link.identity.capabilities & 1 << 5
        link.enable_all()
        q = np.asarray(link.latest_state()[0].q)
        word = dict(position=q, velocity=np.zeros(n), torque=np.zeros(n), kp=np.full(n, 20.), kd=np.full(n, 5.))
        for bad in (dict(full, frame="world"), dict(full, kc=np.full(6, 2000.))):
            try:
                link.apply_command(dict(word, cartesian=bad))
            except (RtLinkError, ValueError):
                continue
            raise AssertionError(f"SimLink accepted {bad}")
        end = time.monotonic() + .2
        while time.monotonic() < end:
            link.apply_command(dict(word, cartesian=full))
            time.sleep(.004)
        sent = plant._last_command["cartesian"]
        assert sent is not None and sent["frame"] == "base" and np.array_equal(sent["kc"], full["kc"]), sent
        time.sleep(.3)  # past hold_ms: the servo holds on the joint law only
        assert link.latest_state()[0].holding and plant._last_command["cartesian"] is None
    finally:
        link.close()
    print("cartesian twin: SimLink admits base-frame specs only, and its hold drops the Cartesian law", flush=True)


if __name__ == "__main__":
    main()
