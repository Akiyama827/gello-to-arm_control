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

Review round 1 (ACCEPT, two findings closed here):
* `_check_replan` never reached the re-plan after an exhausted timing ladder (its grazing route is always caught by the
  polyline pre-check): mutants "raise instead of re-plan" and "no pre-check" both survived. It now forces one ladder
  exhaustion on a clear route (fresh route planned) and checks the refiner case; both mutants caught. Round 2 (ACCEPT): "no
  pre-check" still survived (a grazing route then fails the ladder and is re-planned anyway; the pre-check only saves the
  ladder's cost); the first case now asserts the grazing polyline never reaches the ladder, and that mutant is caught. With a
  refiner, a grazing seed costs up to 3 x the ladder's TrajOpt calls before the last failure: latency, not correctness.
* The polyline pre-check is skipped when a refiner is set: TrajOpt's collision terms can push a grazing seed clear, and a
  seed it cannot fix fails the ladder, which re-plans as well.
Noted, unchanged: if a later attempt's OMPL returns None, `plan_joint` returns None and drops the earlier certification
error, so `plan_cartesian` moves on to its next IK candidate where it used to stop (the coordinator maps both to
`planning_failed`).
Not this change: `tools/check/check_refinement.py` fails when run with `ARM_CONTROL_ROOT` set to the project (the project's
FR3 carries a wrist camera, `fr3_gemini305`, which the shared plate of `examples/configs/dual_arm_object.yaml` clips by 1.97 mm
at the start; "shared-object start collision" instead of the asserted "grasp closure"). It passes with the library alone, as
the library suite runs it, on this commit and on `c9030de`. Owner: the library example fixture; blocking effect: none for
the library suite; a project-rooted run of that check is red until the fixture's grasp clears the camera.
