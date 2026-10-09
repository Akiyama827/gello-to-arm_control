"""Shared-world crossing regression and the simulated FR3 pair's execution on its two supervisors."""

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np

from arm_control.scene import ActorSpec, SceneSpec
from arm_control.config import gripper_joints
from arm_control.multi_arm.planning import PairPlanner
from arm_control.multi_arm.config import load_pair_config
from arm_control.multi_arm.launch import running_pair
from arm_control.multi_arm.supervised_hardware import SupervisedHardware
from arm_control.multi_arm.coordinator import BimanualCoordinator
from arm_control.multi_arm.types import PairLifecycle
from arm_control.execution.contract import Phase
from arm_control.sdk import Refused

ROOT = Path(__file__).resolve().parents[2]


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
        replan(planner)


def replan(planner):
    """The pair planner plans a FRESH route when the finished curve of one collides (plan_and_finish)."""
    from unittest.mock import patch

    from arm_control.planning.ompl_planner import OMPLPlanner
    from arm_control.planning.retiming import RouteCollides

    start, goal = {"x": np.array([-0.3]), "y": np.array([-0.3])}, {"x": np.array([0.3]), "y": np.array([0.3])}
    grazing = np.array([[-0.3, -0.3], [0.3, 0.3]])          # straight through the crossing
    around = np.array([[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]])

    def routes(*seq):
        calls = []
        return calls, lambda self, s, g: (calls.append(1), seq[min(len(calls), len(seq)) - 1])[1]

    calls, plan = routes(grazing, around)
    with patch.object(OMPLPlanner, "plan", plan):
        result = planner.plan(start, goal)
    assert len(calls) == 2 and not any(planner.world.in_collision(q) for q in result.combined.densify(0.001)), calls
    calls, plan = routes(grazing, None)
    with patch.object(OMPLPlanner, "plan", plan):
        try:
            planner.plan(start, goal)
        except RouteCollides:
            assert len(calls) == 2
        else:
            raise AssertionError("a pair route through the crossing was flown")
    calls, plan = routes(None)
    with patch.object(OMPLPlanner, "plan", plan):
        try:
            planner.plan(start, goal)
        except RouteCollides:
            raise AssertionError("no route reported as a collision")
        except ValueError as exc:
            assert str(exc) == "no collision-free pair path found", exc
        else:
            raise AssertionError("a pair plan without a route")


def fr3_sim():
    """The simulated FR3 pair on its two supervisors (one hosts the world, one co-hosts): a shared-world plan executes, and
    with the partner's release 40 ms late both references still start at the one scheduled stamp."""
    env = dict(os.environ, ARM_CONTROL_ROOT=str(ROOT), PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    with running_pair(ROOT / "examples/configs/dual_arm.yaml", env, plant="sim", cwd=ROOT) as (staged, env):
        config = load_pair_config(staged)
        sim = SupervisedHardware(config, env=env)
        try:
            sim.connect()
            for h in sim.handles.values():
                h.set_armed(True)
            coord = BimanualCoordinator(sim.handles, release_context=sim.release_context)
            end = time.monotonic() + 10
            while not all(h.state().ready for h in sim.handles.values()) and time.monotonic() < end:
                sim.step()
                coord.pump()
                time.sleep(.01)
            gains = {a: (np.asarray(c.controller_state().kp), np.asarray(c.controller_state().kd))
                     for a, c in sim.clients.items()}
            planner = PairPlanner(
                config.scene,
                config.scene.state(),
                model_tokens=config.model_tokens,
                scene_token=config.scene_token,
                gains=gains,
                max_velocity={a: np.full(len(kp), 0.4) for a, (kp, _) in gains.items()},
                max_acceleration={a: np.full(len(kp), 0.8) for a, (kp, _) in gains.items()},
                held_positions={
                    f"{a}__{j}": sim.finger_positions()[a]
                    for a, cfg in config.robots.items()
                    for j in gripper_joints(cfg)
                },
            )
            q = {a: np.asarray(h.state().position, float) for a, h in sim.handles.items()}
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
            while coord.lifecycle is not PairLifecycle.READY and time.monotonic() < end + 10:
                coord.pump()
                time.sleep(.01)
            late = list(sim.clients.values())[1]
            release = late.release
            stamps = []

            def delayed(eid, start_at=None):
                time.sleep(.04)
                stamps.append(start_at)
                return release(eid, start_at=start_at)

            with patch.object(late, "release", delayed):
                coord.execute_pair(motion.pair.pair_id)
            assert stamps and stamps[0] is not None
            end = time.monotonic() + 30
            rows, seqs = {a: [] for a in sim.clients}, {a: 0 for a in sim.clients}
            while not coord.lifecycle.terminal and time.monotonic() < end:
                sim.step()
                coord.pump()
                for a, c in sim.clients.items():  # the supervisors' controller ticks (a 64-tick ring: read as we go)
                    reply = c.trace_samples(seqs[a])
                    rows[a] += reply["samples"]
                    seqs[a] = reply["latest_seq"]
                time.sleep(.01)
            assert coord.lifecycle is PairLifecycle.COMPLETE, coord.pair_state()
            sim.step()
            error = max(np.max(np.abs(np.asarray(h.state().position) - goals[a])) for a, h in sim.handles.items())
            assert error < 0.01, error
            # each supervisor's ticks: its reference start = tick time minus elapsed phase (the world clock is wall time
            # here: nothing pauses it). One stamp => one start, despite the 40 ms late release.
            starts = {}
            for a in sim.clients:
                plan = f"{motion.pair.pair_id}:{a}"
                moving = [r for r in rows[a] if r["plan_id"] == plan and r["trajectory_elapsed_s"]]
                assert moving, a
                starts[a] = np.median([r["pc_monotonic"] - r["trajectory_elapsed_s"] for r in moving])
            spread = max(starts.values()) - min(starts.values())
            assert spread < .002, starts
            print(
                f"FR3 pair on two supervisors: duration={motion.duration_sec:.6f}s, max joint residual={error:.6g}rad, "
                f"40 ms late release -> reference starts {spread * 1e3:.2f} ms apart"
            )
            hung_partner(sim, coord, planner, goals, q)
            start_refusals(sim)
        finally:
            sim.close()


def hung_partner(sim, coord, planner, start, goals):
    """A partner release that is not accepted before the start stamp -- hung past it, or refused at selection (a terminal
    status, not an exception) -- cancels the accepted member before it moves and faults the pair; a hung release that lands
    afterwards is refused (its stamp has passed). Neither arm plays the leg."""
    first, late = list(sim.clients.values())
    release = late.release
    for case in ("hung", "refused"):
        end = time.monotonic() + 5
        while coord.pair_state().cancellation_pending and time.monotonic() < end:  # the previous case's cancels confirmed
            coord.pump()
            coord.cancel_pair()
            time.sleep(.01)
        sim.step()
        motion = planner.plan(start, goals)
        coord.admission = lambda p, s, motion=motion: motion.admit(p, s, sim.config.scene_token)
        coord.stage_pair(motion.pair)
        end = time.monotonic() + 10
        while coord.lifecycle is not PairLifecycle.READY and time.monotonic() < end:
            coord.pump()
            time.sleep(.01)
        assert coord.lifecycle is PairLifecycle.READY, coord.pair_state()
        before = {a: np.asarray(h.state().position, float) for a, h in sim.handles.items()}
        outcome = []

        def partner(eid, start_at=None):
            if case == "refused":  # as the supervisor answers a selection refusal: a terminal status
                return replace(late.status(eid), phase=Phase.REJECTED, reason="refused for the test")
            time.sleep(.4)  # past the stamp (150 ms ahead)
            try:
                outcome.append(release(eid, start_at=start_at))
            except Refused as exc:
                outcome.append(exc)

        why = "not confirmed before the scheduled start" if case == "hung" else "refused for the test"
        with patch.object(late, "release", partner):
            coord.execute_pair(motion.pair.pair_id)
            assert coord.lifecycle is PairLifecycle.FAULTED, coord.pair_state()
            assert why in coord.pair_state().reason, coord.pair_state().reason
            time.sleep(.6)
        if case == "hung":
            assert outcome and isinstance(outcome[0], Refused) and "ahead" in str(outcome[0]), outcome
        executions = {a: sim.handles[a]._inner.execution_of(m.plan_id) for a, m in motion.pair.members().items()}
        for a, c in sim.clients.items():
            status = c.status(executions[a])
            assert status.terminal and status.phase.value != "completed", (case, a, status)
        time.sleep(.3)
        sim.step()
        for a, h in sim.handles.items():
            np.testing.assert_allclose(h.state().position, before[a], atol=1e-3)  # neither arm played the leg
        print(f"{case} partner release: the accepted member cancelled before its start, neither arm moved: PASS")


def start_refusals(sim):
    """scheduled_release/1 bounds: under 10 ms, over 150 ms, another clock domain refused; no extension -> the SDK refuses."""
    from arm_control.execution.contract import Stamp, Start, TrajectoryRequest
    from arm_control.motion import JointTrajectory

    a, c = next(iter(sim.clients.items()))
    q = np.asarray(c.state().q, float)
    request = TrajectoryRequest("refusals", c.identity.instance_id, tuple(c.identity.joint_names),
                                JointTrajectory([0., 1.], np.stack([q, q]), np.zeros((2, len(q)))),
                                model_token=sim.config.model_tokens[a], start=Start.ON_RELEASE)
    eid = c.execute(request).execution_id
    now = c.now()
    for label, stamp, why in (("late", Stamp(now.ns + 2_000_000, now.clock, now.domain), "ahead"),
                              ("far", Stamp(now.ns + 400_000_000, now.clock, now.domain), "ahead"),
                              ("domain", Stamp(now.ns + 80_000_000, now.clock, "pc:other"), "clock")):
        try:
            c.release(eid, start_at=stamp)
        except Refused as exc:
            assert why in str(exc), (label, exc)
        else:
            raise AssertionError(f"{label} start stamp accepted")
    saved = c.extensions
    c.extensions = {k: v for k, v in saved.items() if k != "scheduled_release"}
    try:
        c.release(eid, start_at=Stamp(now.ns + 80_000_000, now.clock, now.domain))
    except Refused as exc:
        assert "scheduled_release" in str(exc), exc
    else:
        raise AssertionError("a scheduled release sent without the extension")
    finally:
        c.extensions = saved
    c.cancel(eid, "done")
    print("scheduled_release bounds and extension gate: PASS")


if __name__ == "__main__":
    # Keep this regression reproducible without changing production planner
    # randomness or its public API.
    from ompl import util as ou

    ou.RNG.setSeed(42)
    crossing()
    fr3_sim()
    print(
        "pair planning: PASS (moving-arm crossing, held member, supervised pair simulation with one start stamp)"
    )
