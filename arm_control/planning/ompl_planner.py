"""Thin RRT-Connect wrapper around OMPL's Python bindings."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError as exc:  # pragma: no cover
    raise ImportError("OMPLPlanner requires the `ompl` Python bindings.") from exc


CollisionFn = Callable[[np.ndarray], bool]


class OMPLPlanner:
    """RRT-Connect (or other ``og.*``) planner over a hyper-rectangular joint space.

    The collision function takes an ``np.ndarray`` of joint values and returns
    ``True`` when the configuration is in collision (invalid). It is queried at
    OMPL's configured validity-checking resolution.
    """

    def __init__(
        self,
        joint_limits: Sequence[tuple[float, float]],
        collision_fn: CollisionFn,
        *,
        planner_name: str = "RRTConnect",
        solve_time_sec: float = 2.0,
        simplify_time_sec: float = 0.5,
        resolution_frac: float = 0.01,
    ) -> None:
        self._limits = [(float(lo), float(hi)) for lo, hi in joint_limits]
        self._collision_fn = collision_fn
        self._solve_time = float(solve_time_sec)
        self._simplify_time = float(simplify_time_sec)
        self._planner_name = planner_name

        n = len(self._limits)
        space = ob.RealVectorStateSpace(n)
        bounds = ob.RealVectorBounds(n)
        for i, (lo, hi) in enumerate(self._limits):
            bounds.setLow(i, lo)
            bounds.setHigh(i, hi)
        space.setBounds(bounds)
        self._space = space

        si = ob.SpaceInformation(space)
        # Modern (nanobind) OMPL bindings accept a plain callable here; the
        # older ``ob.StateValidityCheckerFn`` wrapper is no longer exported.
        si.setStateValidityChecker(self._validity)
        si.setStateValidityCheckingResolution(float(resolution_frac))
        si.setup()
        self._si = si

    def _state_to_array(self, state) -> np.ndarray:
        return np.asarray(
            [state[i] for i in range(len(self._limits))], dtype=float
        )

    def _validity(self, state) -> bool:
        q = self._state_to_array(state)
        return not self._collision_fn(q)

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        q_start = np.asarray(q_start, dtype=float).ravel()
        q_goal = np.asarray(q_goal, dtype=float).ravel()
        n = len(self._limits)
        if q_start.shape != (n,) or q_goal.shape != (n,):
            raise ValueError("q_start and q_goal must match joint_limits length")

        if self._collision_fn(q_goal):
            return None
        if self._collision_fn(q_start):
            return None

        # RRT-Connect can behave oddly when start == goal; short-circuit it so
        # the wrapper always returns a sensible 2-point path.
        if np.allclose(q_start, q_goal):
            return np.vstack([q_start, q_goal])

        pdef = ob.ProblemDefinition(self._si)
        s_start = self._space.allocState()
        s_goal = self._space.allocState()
        for i in range(n):
            s_start[i] = float(q_start[i])
            s_goal[i] = float(q_goal[i])
        pdef.setStartAndGoalStates(s_start, s_goal)

        planner_cls = getattr(og, self._planner_name)
        planner = planner_cls(self._si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        solved = planner.solve(self._solve_time)
        if not solved:
            return None
        path = pdef.getSolutionPath()
        if path is None:
            return None

        try:
            og.PathSimplifier(self._si).simplify(path, self._simplify_time)
        except Exception:
            # Simplification is a best-effort polish; keep the raw path if it
            # fails for any reason (rare; e.g. interrupted by timeout).
            pass

        count = path.getStateCount()
        waypoints = np.array(
            [self._state_to_array(path.getState(i)) for i in range(count)],
            dtype=float,
        )
        if waypoints.shape[0] < 2:
            return None
        # Pin endpoints exactly to requested start/goal.
        waypoints[0] = q_start
        waypoints[-1] = q_goal
        return waypoints
