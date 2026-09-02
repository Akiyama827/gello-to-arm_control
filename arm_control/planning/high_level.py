"""High-level facade exposing plan_cartesian / plan_joint for orchestrator nodes."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arm_control.frames import T_to_pose_xyzquat, pose_xyzquat_to_T
from arm_control.planning.ik import PinocchioIK
from arm_control.planning.trajectory import (
    JointTrajectory,
    time_parameterize_blended,
)

if TYPE_CHECKING:
    from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
    from arm_control.planning.ompl_planner import OMPLPlanner


def _quat_of(T: np.ndarray) -> np.ndarray:
    """[w,x,y,z] of a homogeneous transform, via the wire pose convention."""
    return np.asarray(T_to_pose_xyzquat(T), dtype=float)[3:]


def _slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dot > 0.9995:  # nearly parallel: lerp is exact enough and slerp blows up
        out = q0 + s * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(dot) * s
    basis = q1 - q0 * dot
    basis = basis / np.linalg.norm(basis)
    return q0 * np.cos(theta) + basis * np.sin(theta)


def build_collision_stack(
    urdf_path,
    planned_joints: list[str],
    planner_cfg: dict,
    *,
    cache_dir,
    environment: list[dict] | None = None,
) -> tuple["MuJoCoCollisionWorld", "OMPLPlanner"]:
    """Construct the shared collision world + OMPL planner pair.

    Single construction site for teleop and orchestrator nodes so both plan
    against the same environment. Imports mujoco/ompl lazily — this module
    stays importable on hosts without them (the executor path).
    """
    from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
    from arm_control.planning.ompl_planner import OMPLPlanner

    world = MuJoCoCollisionWorld(
        urdf_path,
        list(planned_joints),
        cache_dir=cache_dir,
        self_collision_padding_m=float(
            planner_cfg.get("self_collision_padding_m", -0.002)
        ),
        environment=environment,
        held_positions={
            str(k): float(v)
            for k, v in (planner_cfg.get("held_joints") or {}).items()
        },
        ignore_bodies=[str(b) for b in (planner_cfg.get("ignore_bodies") or [])],
        cloud_obstacles=planner_cfg.get("cloud_obstacles"),
    )
    ompl = OMPLPlanner(
        list(zip(world.lower, world.upper)),
        world.in_collision,
        solve_time_sec=float(planner_cfg.get("solve_time_sec", 2.0)),
        simplify_time_sec=float(planner_cfg.get("simplify_time_sec", 0.5)),
        resolution_frac=float(planner_cfg.get("resolution_frac", 0.005)),
    )
    return world, ompl


class ArmPlanner:
    """Compose IK + (optionally OMPL) + blended time-parameterization.

    Orchestrator nodes call ``plan_cartesian(target_pose, q_start)`` to get a
    fully-timed ``JointTrajectory`` ready for execution by
    ``JointTrajectoryExecutor``.
    """

    def __init__(
        self,
        arm_id: str,
        ik: PinocchioIK,
        ompl: OMPLPlanner | None,
        max_vel: np.ndarray,
        max_acc: np.ndarray,
        world=None,
    ) -> None:
        self.arm_id = str(arm_id)
        self._ik = ik
        self._ompl = ompl
        # MuJoCoCollisionWorld | None: with a world, IK restarts are filtered
        # to collision-free branches — bare DLS happily converges into a
        # self-colliding posture and OMPL then rejects the goal outright.
        self._world = world
        self._max_vel = np.asarray(max_vel, dtype=float).ravel()
        self._max_acc = np.asarray(max_acc, dtype=float).ravel()
        if self._max_vel.shape != self._max_acc.shape:
            raise ValueError("max_vel and max_acc must have matching shape")
        # Optional callable invoked ~every 20 ms while OMPL solves in a worker
        # thread. The hardware orchestrator uses it to keep streaming a hold
        # command: OMPL's solve budget (seconds) dwarfs the bridge's 0.1 s
        # deadman, so blocking the event loop during planning would disarm the
        # arm mid-sequence. None (default) keeps planning synchronous.
        self.keepalive = None

    @property
    def ik(self) -> PinocchioIK:
        return self._ik

    def plan_cartesian(
        self,
        target_T: np.ndarray,
        q_start: np.ndarray,
        *,
        collision_check: bool = True,
        speed_scale: float = 1.0,
    ) -> JointTrajectory | None:
        target_T = np.asarray(target_T, dtype=float)
        q_start = np.asarray(q_start, dtype=float).ravel()
        validate = None
        if collision_check and self._world is not None:
            validate = lambda q: not self._world.in_collision(q)  # noqa: E731
            if self._world.in_collision(q_start):
                print(
                    f"[planner:{self.arm_id}] start pose is in collision per the "
                    f"plan world: q={np.round(q_start, 3).tolist()}",
                    flush=True,
                )
        q_goal = self._ik.solve(target_T, q_start, validate=validate)
        if q_goal is None:
            print(
                f"[planner:{self.arm_id}] IK found no"
                f"{' collision-free' if validate else ''} goal branch for "
                f"target p={np.round(target_T[:3, 3], 3).tolist()}",
                flush=True,
            )
            return None
        return self.plan_joint(
            q_goal, q_start, collision_check=collision_check, speed_scale=speed_scale
        )

    def plan_linear(
        self,
        target_T: np.ndarray,
        q_start: np.ndarray,
        *,
        collision_check: bool = True,
        speed_scale: float = 1.0,
        max_step_m: float = 0.005,
    ) -> JointTrajectory | None:
        """A STRAIGHT line in Cartesian space, not just a Cartesian goal.

        ``plan_cartesian`` solves IK for the goal and then interpolates in
        JOINT space, so the tool traces whatever curve the joints happen to
        sweep -- measured 0.34 mm of lateral bow over a 40 mm insertion, a
        third of the seat gate, and unbounded because it depends on posture.
        A connector being pushed into its socket has to travel down its own
        axis, so this samples the line itself and solves IK per sample.

        Returns None if any sample is unreachable (the caller can fall back
        to a joint-space plan) -- a partially-straight path is not a thing
        worth shipping.
        """
        target_T = np.asarray(target_T, dtype=float)
        lo, hi = self._ik.hard_limits
        q_start = np.clip(np.asarray(q_start, dtype=float).ravel(), lo, hi)
        if not 0.0 < speed_scale <= 1.0:
            raise ValueError("speed_scale must be in (0, 1]")
        start_T = self._ik.fk(q_start)
        p0, p1 = start_T[:3, 3], target_T[:3, 3]
        span = float(np.linalg.norm(p1 - p0))
        if span < 1e-9:
            return self.plan_joint(
                q_start, q_start, collision_check=collision_check,
                speed_scale=speed_scale,
            )
        steps = max(2, int(np.ceil(span / float(max_step_m))) + 1)
        q0 = _quat_of(start_T)
        q1 = _quat_of(target_T)
        if float(q0 @ q1) < 0.0:
            q1 = -q1  # shortest arc: quaternions double-cover rotations
        validate = None
        if collision_check and self._world is not None:
            validate = lambda q: not self._world.in_collision(q)  # noqa: E731
        waypoints = [q_start]
        seed = q_start
        for step in range(1, steps):
            s = step / (steps - 1)
            pose = np.empty(7)
            pose[:3] = p0 + s * (p1 - p0)
            pose[3:] = _slerp(q0, q1, s)
            q = self._ik.solve(pose_xyzquat_to_T(pose), seed, validate=validate)
            if q is None:
                print(
                    f"[planner:{self.arm_id}] straight-line insertion is not "
                    f"reachable at {s * span * 1e3:.0f} mm of {span * 1e3:.0f} mm",
                    flush=True,
                )
                return None
            waypoints.append(q)
            seed = q
        return time_parameterize_blended(
            np.vstack(waypoints), self._max_vel * speed_scale,
            self._max_acc * speed_scale,
        )

    def plan_joint(
        self,
        q_goal: np.ndarray,
        q_start: np.ndarray,
        *,
        collision_check: bool = True,
        speed_scale: float = 1.0,
    ) -> JointTrajectory | None:
        # A parked/sagged arm can measure marginally OUTSIDE the joint limits
        # (gravity sag in sim, calibration offset on hardware); OMPL then
        # rejects the start state outright. Plan from the nearest in-bounds
        # configuration — sub-degree from where the arm actually is.
        lo, hi = self._ik.hard_limits
        q_start = np.clip(np.asarray(q_start, dtype=float).ravel(), lo, hi)
        if not 0.0 < speed_scale <= 1.0:
            raise ValueError("speed_scale must be in (0, 1]")
        q_start = np.asarray(q_start, dtype=float).ravel()
        q_goal = np.asarray(q_goal, dtype=float).ravel()
        if q_start.shape != q_goal.shape:
            raise ValueError("q_start and q_goal must have the same shape")
        if np.allclose(q_start, q_goal):
            # Trivial hold: 2-point traj over 0.1 s with zero velocity.
            times = np.array([0.0, 0.1])
            positions = np.vstack([q_start, q_start])
            velocities = np.zeros_like(positions)
            return JointTrajectory(times=times, positions=positions, velocities=velocities)

        if self._ompl is not None and collision_check:
            waypoints = self._plan_with_keepalive(q_start, q_goal)
            if waypoints is None:
                return None
        else:
            waypoints = np.vstack([q_start, q_goal])
        # Blended retiming: continuous velocity along the whole path — the
        # per-segment trapezoids stopped at EVERY OMPL waypoint and crawled.
        return time_parameterize_blended(
            waypoints, self._max_vel * speed_scale, self._max_acc * speed_scale
        )

    def _plan_with_keepalive(
        self, q_start: np.ndarray, q_goal: np.ndarray
    ) -> np.ndarray | None:
        if self.keepalive is None:
            return self._ompl.plan(q_start, q_goal)
        # OMPL runs in a worker; its Python validity callback releases the GIL
        # at every collision query, so this thread gets scheduled to pump the
        # keepalive between checks.
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FutureTimeout

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._ompl.plan, q_start, q_goal)
            while True:
                try:
                    return future.result(timeout=0.02)
                except FutureTimeout:
                    self.keepalive()


def _self_check() -> None:
    """plan_linear really is straight, and _slerp really interpolates.

    Uses a fake arm whose joints ARE the tool position, so the correct path
    is exactly the straight line and any bow is the sampler's own error --
    no URDF, no solver tolerance to hide behind.

    Run: python -m arm_control.planning.high_level
    """
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(_slerp(q0, q1, 0.0), q0), "slerp must start at q0"
    assert np.allclose(_slerp(q0, q1, 1.0), q1), "slerp must end at q1"
    mid = _slerp(q0, q1, 0.5)
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-12, "slerp must stay on the unit sphere"
    assert abs(float(mid @ q0) - float(mid @ q1)) < 1e-12, "half-way must be equidistant"
    near = _slerp(q0, q0 + 1e-9, 0.5)  # the lerp branch must not divide by zero
    assert np.isfinite(near).all(), "slerp degenerates on nearly-parallel inputs"

    class _FakeIK:
        """Tool position == first three joints; orientation ignored."""

        hard_limits = (np.full(3, -10.0), np.full(3, 10.0))

        def fk(self, q):
            T = np.eye(4)
            T[:3, 3] = np.asarray(q, dtype=float)[:3]
            return T

        def solve(self, T, seed, validate=None):
            return np.asarray(T, dtype=float)[:3, 3].copy()

    planner = ArmPlanner("fake", _FakeIK(), None, np.full(3, 1.0), np.full(3, 2.0))
    goal = np.eye(4)
    goal[:3, 3] = [0.3, 0.4, 0.0]  # 0.5 m, not axis-aligned
    traj = planner.plan_linear(goal, np.zeros(3), collision_check=False)
    assert traj is not None, "straight-line plan failed on a trivially reachable goal"
    start, end = traj.positions[0], traj.positions[-1]
    axis = (end - start) / np.linalg.norm(end - start)
    bow = max(
        float(np.linalg.norm((p - start) - np.dot(p - start, axis) * axis))
        for p in traj.positions
    )
    assert bow < 1e-9, f"plan_linear bowed {bow * 1e3:.4f} mm off its own line"
    assert np.allclose(end, [0.3, 0.4, 0.0]), "plan_linear did not reach the goal"
    print(f"high_level self-check OK: {len(traj.positions)} samples, bow {bow:.2e} m")


if __name__ == "__main__":
    _self_check()
