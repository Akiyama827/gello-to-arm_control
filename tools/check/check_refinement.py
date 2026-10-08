"""Native refinement integration: manual console, pair and exact held peer."""

import ctypes
import time

import numpy as np
from unittest.mock import patch

from arm_control.config import gripper_joints
from arm_control.multi_arm.config import load_pair_config
from arm_control.multi_arm.cooperative import CooperativeKinematics
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.ui.arm_console import (
    _close_planning_pool, _planning_pool, _plan_snapshot,
)
from arm_control.ui.dual_arm_console import plan_request


def main():
    config = load_pair_config("examples/configs/dual_arm.yaml")
    starts = {a.name: np.asarray(a.q) for a in config.scene.actors}
    first, peer = config.scene.actor_names
    fingers = {a: 0.04 for a in starts}
    goals = {
        a: dict(kind="joint", frame="base", values=np.rad2deg(q).tolist())
        for a, q in starts.items()
    }
    goals[first]["values"][0] += 5
    goals[peer]["values"][0] += 15  # Must be ignored for the held peer.
    motion, _ = plan_request(
        str(config.source),
        config.scene_token,
        starts,
        goals,
        [first],
        fingers,
        config.scene_token,
        refinement="trajopt",
    )
    np.testing.assert_array_equal(
        motion.trajectories[peer].positions,
        np.tile(starts[peer], (len(motion.combined.times), 1)),
    )
    assert np.all(motion.trajectories[peer].velocities == 0)
    print("Native pair refinement with exactly held peer: PASS", flush=True)

    world = MuJoCoCollisionWorld.from_scene(
        config.scene,
        config.scene.state(),
        first,
        held_positions={
            f"{a}__{j}": fingers[a]
            for a in starts
            for j in gripper_joints(config.robots[a])
        },
    )
    goal = np.deg2rad(goals[first]["values"])
    pool = _planning_pool()
    try:
        result = pool.apply_async(
            _plan_snapshot,
            (world, {}, starts[first], goal,
             np.full(7, 0.4), np.full(7, 0.8), "trajopt"),
        )
        gap = _wait_responsive(result)
        trajectory = result.get()
        # Native extension calls may hold the GIL for their entire duration.
        # Deliberately exercise that condition in the same worker as planning.
        result = pool.apply_async(_native_stall, (1.5,))
        gap = max(gap, _wait_responsive(result))
        result.get()
        print(f"Single planner process: max parent heartbeat gap {gap:.3f} s", flush=True)
        # Teardown must kill an active solve rather than wait for its deadline.
        pool.apply_async(_native_stall, (10,))
        time.sleep(0.2)
        before = time.monotonic()
    finally:
        _close_planning_pool(pool)
    assert time.monotonic() - before < 2, "console shutdown waited for the solver"
    np.testing.assert_array_equal(trajectory.positions[[0, -1]], [starts[first], goal])
    assert not any(world.in_collision(q) for q in trajectory.densify(0.001))
    assert np.max(trajectory.bounds()["qd_abs_max"]) <= 0.4 + 1e-8
    assert np.max(trajectory.bounds()["qdd_abs_max"]) <= 0.8 + 1e-8
    print(
        "Manual console native refinement, endpoints/collision/cubic limits: PASS",
        flush=True,
    )


def _native_stall(seconds):
    # PyDLL intentionally retains the GIL; CDLL would not test this failure.
    ctypes.PyDLL(None).usleep(int(seconds * 1_000_000))


def _wait_responsive(result):
    deadline = time.monotonic() + 180
    previous = time.monotonic()
    gap = 0.0
    while not result.ready():
        time.sleep(0.01)
        now = time.monotonic()
        gap = max(gap, now - previous)
        previous = now
        assert now < deadline, "planner child did not finish"
    assert gap < 0.25, f"planner blocked the parent for {gap:.3f} s"
    return gap


def check_rejection():
    """Native success cannot admit a curve that breaks either fixed grasp."""
    from arm_control.ui.dual_arm_console import _pose_values

    config = load_pair_config("examples/configs/dual_arm_object.yaml")
    kin = CooperativeKinematics(config)
    pose = config.shared_object.world_T_object
    starts = kin.solve(pose, {a.name: np.asarray(a.q) for a in config.scene.actors})
    target = _pose_values(pose)
    target[2] += 20

    def broken(_env, _world, seed, _targets):
        seed = seed.copy()
        seed[len(seed) // 2, 0] += 0.03
        return seed

    with patch("arm_control.planning.trajopt._optimize_constrained", broken):
        try:
            plan_request(
                str(config.source),
                config.scene_token,
                starts,
                {},
                list(starts),
                config.shared_object.fingers,
                config.scene_token,
                pose,
                target,
                mode="shared",
                refinement="trajopt",
            )
        except ValueError as exc:
            assert "grasp closure" in str(exc), str(exc)
        else:
            raise AssertionError("accepted an optimizer result that breaks the grasp")
    print("Grasp-breaking optimizer output rejected after timing: PASS", flush=True)


if __name__ == "__main__":
    main()
    check_rejection()
