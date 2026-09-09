"""High-level facade exposing plan_cartesian / plan_joint for orchestrator nodes."""
from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from arm_control.frames import T_to_pose_xyzquat, pose_xyzquat_to_T
from arm_control.planning.ik import PinocchioIK
from arm_control.motion import JointTrajectory
from arm_control.planning.retiming import (
    BLEND_TOLERANCE_RAD, COLLISION_STEP_RAD, RETIMING_STEPS_RAD,
    bound_trajectory, collision_checked_retiming, time_parameterize_blended,
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
    collision_world=None,
) -> tuple["MuJoCoCollisionWorld", "OMPLPlanner"]:
    """Construct the shared collision world + OMPL planner pair.

    Single construction site for teleop and orchestrator nodes so both plan
    against the same environment. Imports mujoco/ompl lazily — this module
    stays importable on hosts without them (the executor path).
    """
    from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
    from arm_control.planning.ompl_planner import OMPLPlanner

    world = collision_world if collision_world is not None else MuJoCoCollisionWorld(
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
        trajectory_refiner=None,
        *,
        ik_candidate_attempts: int = 0,
        ik_limit_margin_fraction: float = 0.05,
    ) -> None:
        self.arm_id = str(arm_id)
        self._ik = ik
        self._ompl = ompl
        # MuJoCoCollisionWorld | None: with a world, IK restarts are filtered
        # to collision-free branches — bare DLS happily converges into a
        # self-colliding posture and OMPL then rejects the goal outright.
        self._world = world
        self._trajectory_refiner = trajectory_refiner
        self._max_vel = np.asarray(max_vel, dtype=float).ravel()
        self._max_acc = np.asarray(max_acc, dtype=float).ravel()
        if self._max_vel.shape != self._max_acc.shape:
            raise ValueError("max_vel and max_acc must have matching shape")
        if isinstance(ik_candidate_attempts, bool) \
                or not isinstance(ik_candidate_attempts, (int, np.integer)) \
                or not 0 <= ik_candidate_attempts <= 128:
            raise ValueError('ik_candidate_attempts must be an integer in [0, 128]')
        margin = float(ik_limit_margin_fraction)
        if not np.isfinite(margin) or not 0 <= margin <= 0.5:
            raise ValueError('ik_limit_margin_fraction must be finite and in [0, 0.5]')
        self._ik_candidate_attempts = ik_candidate_attempts
        self._ik_limit_margin_fraction = margin
        if ik_candidate_attempts:
            lo, hi = ik.hard_limits
            span = hi - lo
            if lo.shape != self._max_vel.shape or hi.shape != lo.shape \
                    or not lo.size or not np.isfinite(lo).all() \
                    or not np.isfinite(hi).all() or not np.isfinite(span).all() \
                    or not (span > 0).all():
                raise ValueError('IK ranking requires finite positive joint spans')

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
        started = perf_counter()
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
        stage_started = perf_counter()
        if self._ik_candidate_attempts:
            candidates = self._ik.solve_candidates(
                target_T, q_start, validate=validate, attempts=self._ik_candidate_attempts
            )
            lo, hi = self._ik.hard_limits
            span = hi - lo

            def score(q):
                margin = float(np.min(np.minimum(q - lo, hi - q) / span))
                travel = float(np.linalg.norm((q - q_start) / span))
                return max(0., self._ik_limit_margin_fraction - margin), travel

            candidates.sort(key=score)
        else:
            q_goal = self._ik.solve(target_T, q_start, validate=validate)
            candidates = [] if q_goal is None else [q_goal]
        print(f'[planner:{self.arm_id}] stage=IK elapsed_s={perf_counter() - stage_started:.6f}', flush=True)
        if not candidates:
            print(
                f"[planner:{self.arm_id}] IK found no"
                f"{' collision-free' if validate else ''} goal branch for "
                f"target p={np.round(target_T[:3, 3], 3).tolist()}",
                flush=True,
            )
            return None
        for rank, q_goal in enumerate(candidates, 1):
            trajectory = self.plan_joint(
                q_goal, q_start, collision_check=collision_check, speed_scale=speed_scale
            )
            if trajectory is not None:
                if self._ik_candidate_attempts:
                    margin = float(np.min(np.minimum(q_goal - lo, hi - q_goal) / span))
                    shortfall, travel = score(q_goal)
                    print(
                        f'[planner:{self.arm_id}] IK candidate rank={rank}/{len(candidates)} '
                        f'travel={travel:.6f} margin={margin:.6f} shortfall={shortfall:.6f} '
                        f'q_start={np.round(q_start, 6).tolist()} '
                        f'q_goal={np.round(q_goal, 6).tolist()}',
                        flush=True,
                    )
                print(f'[planner:{self.arm_id}] stage=cartesian_total elapsed_s={perf_counter() - started:.6f}', flush=True)
                return trajectory
        return None

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
        if collision_check and self._world is not None:
            return collision_checked_retiming(
                np.vstack(waypoints), self._max_vel * speed_scale,
                self._max_acc * speed_scale, self._world.in_collision)
        return time_parameterize_blended(
            np.vstack(waypoints), self._max_vel * speed_scale,
            self._max_acc * speed_scale)

    def plan_joint(
        self,
        q_goal: np.ndarray,
        q_start: np.ndarray,
        *,
        collision_check: bool = True,
        speed_scale: float = 1.0,
    ) -> JointTrajectory | None:
        started = perf_counter()
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
        if self._trajectory_refiner is not None and (self._ompl is None or not collision_check):
            raise ValueError('trajectory refinement requires collision-checked OMPL planning')
        if np.allclose(q_start, q_goal):
            # Trivial hold: 2-point traj over 0.1 s with zero velocity.
            times = np.array([0.0, 0.1])
            positions = np.vstack([q_start, q_start])
            velocities = np.zeros_like(positions)
            print(f'[planner:{self.arm_id}] stage=joint_total elapsed_s={perf_counter() - started:.6f}', flush=True)
            return JointTrajectory(times=times, positions=positions, velocities=velocities)

        if self._ompl is not None and collision_check:
            stage_started = perf_counter()
            waypoints = self._ompl.plan(q_start, q_goal)
            print(f'[planner:{self.arm_id}] stage=OMPL elapsed_s={perf_counter() - stage_started:.6f}', flush=True)
            if waypoints is None:
                return None
        else:
            waypoints = np.vstack([q_start, q_goal])
        # Blended retiming: continuous velocity along the whole path — the
        # per-segment trapezoids stopped at EVERY OMPL waypoint and crawled.
        #
        # Certify only where a path was actually CLEARED. Without OMPL the
        # waypoints are a bare start->goal line nothing collision-checked, so
        # a complaint would report the absence of planning rather than a curve
        # that left its corridor. NOTE this is deliberately narrower than the
        # `self._world is not None` guards above, which gate collision-aware
        # IK CANDIDATE selection and must stay on with or without OMPL --
        # widening this condition to those cost a self-check ("colliding
        # candidate was selected") while writing it.
        certify = collision_check and self._world is not None and self._ompl is not None
        stage_started = perf_counter()
        failure = None
        for ds in self.RETIMER_STEPS_RAD:
            seed = time_parameterize_blended(
                waypoints, self._max_vel * speed_scale,
                self._max_acc * speed_scale, ds=ds,
                blend_tolerance_rad=BLEND_TOLERANCE_RAD * ds / self.RETIMER_STEPS_RAD[0],
            )
            if self._trajectory_refiner is not None:
                # Keep the proven seed geometry/sampling; replace only its
                # timing after refinement. Exceptions propagate to the
                # planner's hold path. Re-run per attempt because refinement
                # rewrites the VELOCITIES, and the flown cubic is a function
                # of those as much as of the positions.
                seed = self._trajectory_refiner(
                    seed.positions, self._max_vel * speed_scale,
                    self._max_acc * speed_scale,
                )
            seed = bound_trajectory(seed, self._max_vel * speed_scale,
                                    self._max_acc * speed_scale)
            if not certify:
                break
            try:
                self._certify_curve(seed)
            except ValueError as exc:
                # Retry the same route with finer samples: a corner's cubic
                # can depart from the cleared polyline. Refinement may also
                # change the curve, so every attempt needs its own check.
                # Exhausting this ladder rejects this candidate; it does not
                # prove that no collision-free trajectory exists.
                failure = exc
                continue
            failure = None
            break
        if failure is not None:
            raise failure
        print(f'[planner:{self.arm_id}] stage=seed_retiming '
              f'elapsed_s={perf_counter() - stage_started:.6f} '
              f'samples={len(seed.times)}', flush=True)
        print(f'[planner:{self.arm_id}] stage=joint_total elapsed_s={perf_counter() - started:.6f}', flush=True)
        return seed

    # Tighten both geometric blending and timing resolution on contact.
    RETIMER_STEPS_RAD = RETIMING_STEPS_RAD
    CURVE_COLLISION_STEP_RAD = COLLISION_STEP_RAD

    def _certify_curve(self, trajectory) -> None:
        """Collision-check what will actually be FLOWN, not its waypoints.

        OMPL clears a polyline; the retimer blends its corners and exports
        samples for the executor's cubic. Those changes need a final check.
        Equal-time curve/chord differences are not geometric deviations:
        a straight leg can change timing while staying on the same segment.
        This is a sampled collision check, not continuous collision proof.

        Raises rather than returning a flag: a trajectory that cannot be
        certified is not a slower trajectory, it is not a trajectory, and
        plan_joint's callers already treat an exception as the hold path.
        """
        for q in trajectory.densify(self.CURVE_COLLISION_STEP_RAD):
            if self._world.in_collision(q):
                raise ValueError(
                    f'[planner:{self.arm_id}] interpolated trajectory collides '
                    'between waypoints — the flown cubic leaves the cleared '
                    'path; re-plan or tighten the retimer step')


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

    # The optional refiner is planner-only and must not reshape linear legs.
    import inspect
    assert 'trajectory_refiner' in inspect.signature(ArmPlanner).parameters
    calls = []

    class _FakeOMPL:
        def plan(self, start, end):
            return np.vstack([start, end])

    def refine(seed, vmax, amax):
        calls.append((seed.copy(), vmax.copy(), amax.copy()))
        return traj

    hybrid = ArmPlanner('fake', _FakeIK(), _FakeOMPL(), np.ones(3),
                        np.full(3, 2.), trajectory_refiner=refine)
    from contextlib import redirect_stdout
    from io import StringIO

    output = StringIO()
    with redirect_stdout(output):
        refined = hybrid.plan_cartesian(goal, np.zeros(3), speed_scale=.5)
    assert np.array_equal(refined.positions, traj.positions)
    assert np.all(refined.bounds()['qd_abs_max'] <= .5 + 1e-8)
    assert np.all(refined.bounds()['qdd_abs_max'] <= 1. + 1e-8)
    for stage in ('IK', 'OMPL', 'seed_retiming', 'joint_total', 'cartesian_total'):
        assert f'[planner:fake] stage={stage} elapsed_s=' in output.getvalue(), stage
    assert len(calls) == 1 and np.all(calls[0][1] == .5) and np.all(calls[0][2] == 1.)
    hybrid.plan_linear(goal, np.zeros(3))
    assert len(calls) == 1, 'Cartesian line entered free-space refinement'
    try:
        hybrid.plan_joint(np.ones(3), np.zeros(3), collision_check=False)
    except ValueError:
        pass
    else:
        raise AssertionError('hybrid accepted collision checking disabled')

    def reject(*args):
        raise ValueError('refinement rejected')

    hybrid._trajectory_refiner = reject
    try:
        hybrid.plan_joint(np.ones(3), np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError('refinement failure silently fell back to seed')

    assert 'ik_candidate_attempts' in inspect.signature(ArmPlanner).parameters, \
        'missing opt-in candidate ranking'

    class _CandidateIK(_FakeIK):
        hard_limits = (np.full(3, -np.pi), np.full(3, np.pi))
        candidates = [np.array([3.1, 0., 0.]), np.array([2.5, 0., 0.]), np.zeros(3)]
        enumerations = 0
        solves = 0

        def solve_candidates(self, T, seed, validate=None, *, attempts=64):
            self.enumerations += 1
            assert attempts == 64
            return [q for q in self.candidates if validate is None or validate(q)]

        def solve(self, T, seed, validate=None):
            self.solves += 1
            return super().solve(T, seed, validate)

    candidate_ik = _CandidateIK()
    ranked = ArmPlanner('ranked', candidate_ik, None, np.ones(3), np.full(3, 2.),
                        ik_candidate_attempts=64)
    start = np.array([3., 0., 0.])
    chosen = ranked.plan_cartesian(goal, start, collision_check=False)
    assert np.allclose(chosen.positions[-1], [2.5, 0., 0.]), \
        'ranking must clear the margin, then prefer shorter whole-arm travel'
    candidate_ik.candidates = [np.array([-3.1, 0., 0.]), np.zeros(3)]
    ranked._ik_limit_margin_fraction = 0.
    chosen = ranked.plan_cartesian(goal, start, collision_check=False)
    assert np.allclose(chosen.positions[-1], 0), 'bounded joints must not wrap'
    candidate_ik.hard_limits = (np.array([-10., -1., -1.]), np.array([10., 1., 1.]))
    candidate_ik.candidates = [np.array([1., 0., 0.]), np.array([0., .2, 0.])]
    chosen = ranked.plan_cartesian(goal, np.zeros(3), collision_check=False)
    assert np.allclose(chosen.positions[-1], [1., 0., 0.]), 'travel must use joint ranges'
    candidate_ik.candidates = [np.array([1., .5, 0.]), np.array([1.5, 0., 0.])]
    chosen = ranked.plan_cartesian(goal, np.zeros(3), collision_check=False)
    assert np.allclose(chosen.positions[-1], [1.5, 0., 0.]), 'travel must use every joint'
    candidate_ik.hard_limits = _CandidateIK.hard_limits

    class _World:
        def in_collision(self, q):
            return q[0] > 1.

    ranked._world = _World()
    candidate_ik.candidates = [np.array([2.5, 0., 0.]), np.zeros(3)]
    chosen = ranked.plan_cartesian(goal, start)
    assert np.allclose(chosen.positions[-1], 0), 'colliding candidate was selected'
    ranked._world = None
    route_goals = []

    class _RetryOMPL:
        def plan(self, start, end):
            route_goals.append(end.copy())
            return None if len(route_goals) == 1 else np.vstack([start, end])

    ranked._ompl = _RetryOMPL()
    chosen = ranked.plan_cartesian(goal, start)
    assert len(route_goals) == 2 and np.allclose(chosen.positions[-1], 0), \
        'route search failure did not retry the next ranked goal'
    ranked._trajectory_refiner = reject
    route_goals.clear()
    route_goals.append(start)  # route exists immediately; refinement must stop retries
    try:
        ranked.plan_cartesian(goal, start)
    except ValueError as exc:
        assert str(exc) == 'refinement rejected'
    else:
        raise AssertionError('candidate ranking swallowed refinement rejection')
    assert len(route_goals) == 2, 'refinement failure retried another candidate'
    enumerations = candidate_ik.enumerations
    ranked.plan_linear(goal, np.zeros(3))
    assert candidate_ik.enumerations == enumerations and candidate_ik.solves > 0
    legacy = ArmPlanner('legacy', candidate_ik, None, np.ones(3), np.full(3, 2.))
    legacy.plan_cartesian(goal, np.zeros(3), collision_check=False)
    assert candidate_ik.enumerations == enumerations, 'default planner enumerates IK'
    for kwargs in ([{'ik_candidate_attempts': v} for v in (True, -1, 129, 1.5, '64')]
                   + [{'ik_limit_margin_fraction': v} for v in (-.1, .51, np.nan, np.inf)]):
        try:
            ArmPlanner('invalid', candidate_ik, None, np.ones(3), np.ones(3), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid ranking config: {kwargs}')
    for upper in (np.zeros(3), np.full(3, np.inf), np.full(3, np.nan)):
        candidate_ik.hard_limits = (np.zeros(3), upper)
        try:
            ArmPlanner('invalid', candidate_ik, None, np.ones(3), np.ones(3),
                       ik_candidate_attempts=64)
        except ValueError:
            pass
        else:
            raise AssertionError('ranking accepted invalid joint bounds')
    print(f"high_level self-check OK: {len(traj.positions)} samples, bow {bow:.2e} m")



def _check_curve_certificate() -> None:
    """The certificate must catch a collision the WAYPOINTS do not contain.

    A cubic between two collision-free knots can bulge into an obstacle. The
    planted world here is collision-free at every knot and blocked only in the
    band the curve bulges through, so a waypoint-only check passes it and the
    flown-curve check must not.
    """
    from arm_control.motion import JointTrajectory

    # Both ends at rest at 0.0, velocities pushing out and back: the knots sit
    # at 0.0 and the curve peaks near 0.087 (see types.py's own self-check).
    traj = JointTrajectory(np.array([0.0, 1.0]), np.array([[0.0], [0.0]]),
                           np.array([[0.6], [-0.6]]))

    class _BandWorld:
        """Blocked strictly between the knots' value and the curve's peak."""

        lower, upper = np.array([-10.0]), np.array([10.0])

        def __init__(self):
            self.calls = 0

        def in_collision(self, q):
            self.calls += 1
            return bool(0.05 < float(q[0]) < 0.09)

    world = _BandWorld()
    assert not world.in_collision(traj.positions[0]), "knots must be clear"
    assert not world.in_collision(traj.positions[1]), "knots must be clear"

    planner = ArmPlanner.__new__(ArmPlanner)
    planner.arm_id = "curve-check"
    planner._world = world
    try:
        planner._certify_curve(traj)
    except ValueError as exc:
        assert "between waypoints" in str(exc), exc
    else:
        raise AssertionError(
            "certificate passed a curve that leaves the cleared path")
    assert world.calls > 2, "certificate checked only the endpoints"

    # A clear world must still pass, and must actually sample the interior.
    class _Clear(_BandWorld):
        def in_collision(self, q):
            self.calls += 1
            return False

    clear = _Clear()
    planner._world = clear
    planner._certify_curve(traj)
    assert clear.calls >= 60, f"too coarse to certify anything ({clear.calls})"
    print("curve collision certificate: catches between-waypoint contact OK")


if __name__ == "__main__":
    _check_curve_certificate()
    _self_check()
