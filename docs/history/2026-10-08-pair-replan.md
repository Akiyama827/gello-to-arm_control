# Two-arm and console planning re-plan a route whose finished curve collides

Follow-up of 2026-10-08-plan-replan.md (operator: "two-arm re-plan first"). `ArmPlanner.plan_joint` re-plans an OMPL route that
cannot be certified; the other OMPL callers of `finish_path` did not: the pair joint planner (`multi_arm/planning.py`) and the
manual console's `plan_trajectory` (`ui/arm_console.py`) raised on the first route that grazed an obstacle between OMPL's
segment checks (the pair planner checks at `resolution_frac` 0.002, the console at the stack's).

* `planning/retiming.py`: `RouteCollides(ValueError)` marks "this route's finished curve collides" (a fresh route may not).
  `collision_checked_retiming` raises it when no rung clears, and TrajOpt's canonical collision validation
  (`trajopt._check_collision`) raises it too. Every other failure (refiner optimization, timing, limits, input) stays a plain
  `ValueError` and is not re-planned, as in `ArmPlanner` (refiner exceptions propagate). Existing `except ValueError` callers
  are unchanged.
* `plan_and_finish(plan_route, world, vmax, amax, **finish)`: plans, finishes, and on `RouteCollides` plans a fresh route,
  `ROUTE_ATTEMPTS` (3) routes in all; returns None only when OMPL found no route at all; a re-plan that finds none raises the
  earlier collision (the reason), not "no path". The pair planner and the console go through it. `ArmPlanner.ROUTE_ATTEMPTS`
  is the same constant.
* Not changed: the cooperative shared-object path (`multi_arm/cooperative.py`) is IK interpolation, not an OMPL route; a fresh
  attempt would be the same path, so it keeps `finish_path`. No polyline pre-check here (unlike `ArmPlanner`): a grazing route
  fails the ladder and is re-planned anyway; the pre-check only saves the ladder's cost.

Checks: `retiming` self-check `_check_plan_and_finish` (graze then around; all grazing -> 3 routes then RouteCollides; failed
re-plan keeps the collision; no route -> None; a non-collision failure is neither re-planned nor reported as a collision);
`trajopt` self-check: interior collision is `RouteCollides`, invalid input is not. Mutants (trajopt collision as plain
ValueError; no re-plan; failed re-plan returns None): all caught. Related: `planner`, `multi_arm.planning`, `control.factory`,
`check_retiming`, `check_pair_planning`, `check_dual_modes` (+`--refine`), `check_cooperative`, `check_dual_console`,
`arm_console --self-check`; project `check_manual_carry`, `check_base_profile`. All pass.
