"""Shared-world crossing regression and two-controller MuJoCo execution."""

from pathlib import Path
import tempfile
import numpy as np

from arm_control.scene import ActorSpec, SceneSpec
from arm_control.config import gripper_joints
from arm_control.multi_arm.planning import PairPlanner
from arm_control.multi_arm.config import load_pair_config
from arm_control.multi_arm.simulation import SharedSimulation
from arm_control.multi_arm.coordinator import BimanualCoordinator
from arm_control.multi_arm.types import PairLifecycle


def crossing():
    with tempfile.TemporaryDirectory() as directory:
        actors = []
        for a, axis in [("x", "1 0 0"), ("y", "0 1 0")]:
            path = Path(directory) / f"{a}.xml"
            path.write_text(f'''<mujoco><worldbody><body name="tip" pos="0 0 0.3">
                <joint name="slide" type="slide" axis="{axis}" range="-0.5 0.5"/>
                <geom type="sphere" size="0.05" mass="1"/></body></worldbody></mujoco>''')
            actors.append(ActorSpec(a, path, ("slide",), (0, 0, 0), (0, 0, 0), (-0.3,)))
        scene = SceneSpec(tuple(actors), (), ())
        planner = PairPlanner(
            scene,
            scene.state(),
            model_tokens={"x": "x", "y": "y"},
            scene_token="one-world",
            gains={a: ([40], [2]) for a in ("x", "y")},
            max_velocity={a: [0.4] for a in ("x", "y")},
            max_acceleration={a: [0.8] for a in ("x", "y")},
        )
        for q in np.linspace(-0.3, 0.3, 61):
            assert not planner.world.in_collision(np.array([q, -0.3]))
            assert not planner.world.in_collision(np.array([-0.3, q]))
        assert planner.world.in_collision(np.zeros(2)), (
            "moving arms must collide at shared crossing"
        )
        result = planner.plan(
            {"x": np.array([-0.3]), "y": np.array([-0.3])},
            {"x": np.array([0.3]), "y": np.array([0.3])},
        )
        assert not any(
            planner.world.in_collision(q) for q in result.combined.densify(0.005)
        )
        assert np.max(result.combined.bounds()["qd_abs_max"]) <= 0.4 + 1e-8
        held = planner.plan(
            {"x": np.array([-0.3]), "y": np.array([-0.3])},
            {"x": np.array([0.3]), "y": np.array([0.3])},
            active_arms=["x"],
        )
        assert np.all(held.trajectories["y"].positions == -0.3)
        assert np.all(held.trajectories["y"].velocities == 0)


def fr3_sim():
    config = load_pair_config("examples/configs/dual_arm.yaml")
    sim = SharedSimulation(config)
    try:
        for h in sim.handles.values():
            h.set_armed(True)
        coord = BimanualCoordinator(sim.handles, clock=lambda: sim.t)
        for _ in range(100):
            sim.step()
            coord.pump()
        planner = PairPlanner(
            config.scene,
            config.scene.state(),
            model_tokens=config.model_tokens,
            scene_token=config.scene_token,
            gains={
                a: (c.executor.kp, c.executor.kd) for a, c in sim.controllers.items()
            },
            max_velocity={a: np.full(c.n_arm, 0.4) for a, c in sim.controllers.items()},
            max_acceleration={
                a: np.full(c.n_arm, 0.8) for a, c in sim.controllers.items()
            },
            held_positions={
                f"{a}__{j}": sim.finger_positions()[a]
                for a, cfg in config.robots.items()
                for j in gripper_joints(cfg)
            },
        )
        q = {a: c.last_state.position.copy() for a, c in sim.controllers.items()}
        goals = {a: v.copy() for a, v in q.items()}
        for a in goals:
            goals[a][0] += 0.1
        motion = planner.plan(q, goals)
        assert np.array_equal(*[t.times for t in motion.trajectories.values()])
        try:
            motion.admit(motion.pair, coord.arm_states(), "changed-scene")
        except ValueError as exc:
            assert "scene changed" in str(exc)
        else:
            raise AssertionError("changed scene admitted")
        from dataclasses import replace
        from arm_control.contracts.motion import unpack_plan, pack_plan

        member = motion.pair.left
        body = unpack_plan(member.payload)
        body.pop("schema", None)
        body["positions"][1, 0] += 0.01
        changed = replace(motion.pair, left=replace(member, payload=pack_plan(**body)))
        try:
            motion.admit(changed, coord.arm_states(), config.scene_token)
        except ValueError as exc:
            assert "payload differs" in str(exc)
        else:
            raise AssertionError("unvalidated payload admitted under preview id")
        coord.admission = lambda p, s: motion.admit(p, s, config.scene_token)
        coord.stage_pair(motion.pair)
        coord.pump()
        coord.execute_pair(motion.pair.pair_id)
        for _ in range(600):
            sim.step()
            coord.pump()
            if coord.lifecycle.terminal:
                break
        assert coord.lifecycle is PairLifecycle.COMPLETE, coord.pair_state()
        error = max(
            np.max(np.abs(c.last_state.position - goals[a]))
            for a, c in sim.controllers.items()
        )
        assert error < 0.01, error
        print(
            f"FR3 pair shared-world execution: duration={motion.duration_sec:.6f}s, max joint residual={error:.6g}rad"
        )
    finally:
        sim.close()


if __name__ == "__main__":
    # Keep this regression reproducible without changing production planner
    # randomness or its public API.
    from ompl import util as ou

    ou.RNG.setSeed(42)
    crossing()
    fr3_sim()
    print(
        "pair planning: PASS (moving-arm crossing, shared clock, held member, full simulation)"
    )
