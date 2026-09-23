"""One collision world and one time parameter for two arm-keyed trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import uuid

import numpy as np

from arm_control.contracts.motion import pack_plan
from arm_control.motion import JointTrajectory
from arm_control.multi_arm.types import ArmPlan, BimanualPlan
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.planning.ompl_planner import OMPLPlanner
from arm_control.planning.refinement import finish_path, refinement_choice


@dataclass
class PairMotion:
    pair: BimanualPlan
    trajectories: dict[str, JointTrajectory]
    scene_token: str
    combined: JointTrajectory
    cooperative_metrics: dict[str, float] | None = None

    @property
    def duration_sec(self):
        return self.combined.duration_sec

    def admit(self, plan, states, scene_token, *, start_tolerance=0.02):
        if plan.pair_id != self.pair.pair_id or scene_token != self.scene_token:
            raise ValueError("pair preview or scene changed; re-plan")
        approved, requested = self.pair.members(), plan.members()
        if set(requested) != set(approved) or any(
            requested[a].plan_id != member.plan_id
            or not requested[a].payload.equals(member.payload)
            for a, member in approved.items()
        ):
            raise ValueError("pair payload differs from the validated preview")
        for a, trajectory in self.trajectories.items():
            if (
                np.max(np.abs(states[a].position - trajectory.positions[0]))
                > start_tolerance
            ):
                raise ValueError(f"{a}: measured start moved; re-plan")


class PairPlanner:
    """Generic joint planner. Consumer-supplied IK produces the goal mapping.

    A held member stays fixed throughout planning and retiming, including OMPL
    detours. Collision checks always evaluate both arms and attached payloads.
    This instance owns mutable scratch geometry; use it on one planning worker.
    """

    def __init__(
        self,
        scene,
        state,
        *,
        model_tokens,
        scene_token,
        gains,
        max_velocity,
        max_acceleration,
        held_positions=None,
        ground_z=0.0,
        workspace_xy=None,
        solve_time_sec=3.0,
        refinement="none",
    ):
        if len(scene.actors) != 2:
            raise ValueError("pair planning requires exactly two scene actors")
        self.ids = tuple(a.name for a in scene.actors)
        if set(model_tokens) != set(self.ids) or set(gains) != set(self.ids):
            raise ValueError("pair model/gain identities must match scene actors")
        self.tokens, self.scene_token, self.gains = (
            dict(model_tokens),
            scene_token,
            gains,
        )
        self.world = MuJoCoCollisionWorld.from_scene(
            scene,
            state,
            self.ids,
            held_positions=held_positions,
            ground_z=ground_z,
            workspace_xy=workspace_xy,
        )
        self.slices, offset = {}, 0
        for a in scene.actors:
            self.slices[a.name] = slice(offset, offset + len(a.joints))
            offset += len(a.joints)
        self.vmax = self.join(max_velocity)
        self.amax = self.join(max_acceleration)
        self.solve_time = float(solve_time_sec)
        self.refinement = refinement_choice(refinement)
        prefixes = tuple(f"{a}__" for a in self.ids)
        self.constrained = any(
            w.body1.startswith(prefixes) or w.body2.startswith(prefixes)
            for w in scene.welds
        )

    def join(self, values):
        if set(values) != set(self.ids):
            raise ValueError("joint mapping must name both arms")
        vector = np.concatenate(
            [np.asarray(values[a], dtype=float).ravel() for a in self.ids]
        )
        for a, sl in self.slices.items():
            if np.asarray(values[a]).shape != (sl.stop - sl.start,):
                raise ValueError(f"{a}: joint count mismatch")
        if not np.isfinite(vector).all():
            raise ValueError("joint values must be finite")
        return vector

    def plan(self, starts, goals, *, active_arms=None):
        if self.constrained:
            raise ValueError(
                "joint-goal planning cannot enforce scene welds; use cooperative planning"
            )
        start, goal = self.join(starts), self.join(goals)
        active = self.ids if active_arms is None else tuple(active_arms)
        if not active or len(set(active)) != len(active) or set(active) - set(self.ids):
            raise ValueError("select one or both distinct arms")
        for a in set(self.ids) - set(active):
            goal[self.slices[a]] = start[self.slices[a]]
        for label, q in [("start", start), ("target", goal)]:
            if np.any(q < self.world.lower) or np.any(q > self.world.upper):
                raise ValueError(f"{label} outside joint limits")
            if self.world.in_collision(q):
                contacts = self.world.collision_contacts(q)
                raise ValueError(f"{label} collision: {contacts[:3]}")
        indices = np.concatenate(
            [np.arange(self.slices[a].start, self.slices[a].stop) for a in active]
        )

        def expand(q):
            full = start.copy()
            full[indices] = q
            return full

        planner = OMPLPlanner(
            list(zip(self.world.lower[indices], self.world.upper[indices])),
            lambda q: self.world.in_collision(expand(q)),
            solve_time_sec=self.solve_time,
            simplify_time_sec=0.15,
            resolution_frac=0.002,
        )
        path = planner.plan(start[indices], goal[indices])
        if path is None:
            raise ValueError("no collision-free pair path found")
        full_path = np.asarray([expand(q) for q in path])
        if np.max(np.abs(goal - start)) < 1e-10:
            combined = JointTrajectory(
                [0.0, 0.1], np.vstack([start, start]), np.zeros((2, len(start)))
            )
        else:
            combined = finish_path(
                self.world,
                full_path,
                self.vmax,
                self.amax,
                refinement=self.refinement,
                active_indices=indices,
                label="pair",
            )
        return self.motion(combined)

    def motion(self, combined):
        """Package a checked common trajectory for the existing pair barrier."""
        limits = combined.bounds()
        if np.any(limits["q_min"] < self.world.lower - 1e-9) or np.any(
            limits["q_max"] > self.world.upper + 1e-9
        ):
            raise ValueError("interpolated pair trajectory exceeds joint limits")
        pair_id = uuid.uuid4().hex
        trajectories, members = {}, []
        for a, sl in self.slices.items():
            trajectory = JointTrajectory(
                combined.times, combined.positions[:, sl], combined.velocities[:, sl]
            )
            trajectories[a] = trajectory
            kp, kd = self.gains[a]
            payload = pack_plan(
                plan_id=f"{pair_id}:{a}",
                phase="paired_move",
                gated=True,
                times=trajectory.times,
                positions=trajectory.positions,
                velocities=trajectory.velocities,
                kp=kp,
                kd=kd,
                model_token=self.tokens[a],
            )
            members.append(ArmPlan(a, f"{pair_id}:{a}", payload))
        return PairMotion(
            BimanualPlan(pair_id, *members), trajectories, self.scene_token, combined
        )
