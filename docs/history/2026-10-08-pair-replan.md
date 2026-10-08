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

Review round 1: ACCEPT. OMPL is not seeded in production (each `plan()` draws a new seed), so a fresh route does differ;
re-planning is probabilistic, not guaranteed (the simplifier can pull routes toward the same corner). Taken:
* The wiring was untested: mutants "the pair planner / the console reuse one route" and "no `None` guard" survived. Now
  `check_pair_planning` (`replan`: graze then around through `PairPlanner.plan`; graze then no route raises the collision; no route
  keeps "no collision-free pair path found") and `check_retiming`'s console seam (graze then around; no route keeps the
  operator's text) cover them; both route-reuse mutants caught.
* `ArmPlanner` with a TrajOpt refiner did NOT re-plan on a refiner-detected collision: the refiner's `RouteCollides` escaped
  `_retime` (only `_CertifyFailed` was caught), so the comment "a route it cannot fix fails the ladder below, which re-plans" was
  false. Now a refiner `RouteCollides` is a failed rung (other refiner errors still propagate), so the ladder runs out and the
  route is re-planned; `_check_replan` asserts it (mutant: caught). Behaviour change on the refiner path only: a TrajOpt
  collision used to refuse the leg at once; it now tries the finer rungs and up to 3 routes, each still certified.
Noted, unchanged: with Cartesian targets `collision_fn` is the task-path validator, so a task-path deviation also raises
`RouteCollides`; only the cooperative path passes targets and it does not re-plan. Worst-case console planning is about 3x
per click against its fixed 180 s pool timeout (trajopt mode).
