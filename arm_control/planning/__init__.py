"""Planning: IK, OMPL, trajectories, and the high-level ArmPlanner."""

# The OMPL planner's settings when a config states none: the single statement (configs state only a deviation).
# padding absorbs shallow convex-hull overlap at the rest pose (MuJoCo hulls concave STLs); resolution_frac is the validity-check
# step as a fraction of the joint-space extent.
PLANNER_DEFAULTS = {"self_collision_padding_m": -0.002, "solve_time_sec": 2.0, "simplify_time_sec": 0.5,
                    "resolution_frac": 0.005}
