# A route that cannot be certified is re-planned (sim-add's intermittent move_to_clear failure)

The project suite's gated sim-add run failed about one run in three, one mode being `planning_failed:move_to_clear:
interpolated trajectory collides between waypoints`. `ArmPlanner.plan_joint` planned ONE OMPL route, retimed it through the
blend/timing ladder and certified the flown curve at 1 mrad; when every rung failed, the leg failed.

Measured (temporary diagnostic, 6 runs, 2 failures): in both failures the raw OMPL polyline itself collided at the certify
step (12 and 78 samples; segments up to 2.1 rad). OMPL validates a segment only every `resolution_frac` x the joint space's
extent (0.005 x 12.98 rad, about 65 mrad on the FR3), and shortcutting makes long segments, so a route can graze an obstacle
between its checks. No retiming rung can fix such a route; certification was right to refuse it.

Operator decision: re-plan on certification failure (not a finer OMPL resolution on every leg).
* `plan_joint` tries at most `ROUTE_ATTEMPTS` (3) fresh OMPL routes. Each is first checked as a straight polyline at the
  certify step: one that grazes an obstacle is replaced at once (the ladder would only spend ~3 s failing). Otherwise it is
  retimed through the same ladder (`_retime`); if no rung certifies, the next route is planned. OMPL finding no route still
  returns None at once; refinement exceptions still propagate (the hold path). Certification is unchanged and still gates
  every flown curve.
* `_check_replan`: a grazing route is replaced by a fresh one and the result is certified; a planner that only ever gets the
  grazing route refuses after exactly 3 attempts. Mutant `ROUTE_ATTEMPTS = 1`: caught.

The coordinated planners' `refinement.finish_path` has its own copy of the ladder and no OMPL access; it is consolidated with
this one in the planning-layout slice that follows.
