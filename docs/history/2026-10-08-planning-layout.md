# Planning package layout (operator approved 2026-10-08: "yes and yes")

One file per job. No behaviour change: moves and merges only.

| Was | Now | Why |
|---|---|---|
| `planning/high_level.py` | `planning/planner.py` | it is the arm planner (`ArmPlanner`); "high level" named nothing |
| `planning/trajectory.py` (compat shim) | deleted | no importer anywhere (library, `Control/`, perceptions); its ARCHITECTURE row said retire when that held |
| `planning/refinement.py` | `finish_path`, `refinement_choice` in `planning/retiming.py` | path finishing is retiming plus the opt-in refiner; 77 lines did not need a module |
| `planning/projection.py` | `project_world`, `_workspace_bound` in `planning/trajopt.py` | the Tesseract projection exists only for TrajOpt; all its imports were already lazy, so importing `trajopt` stays as light |
| `planning/preview_rerun.py` | `viz/preview.py` | a Rerun preview is visualization, not planning; `scene_geoms` keeps the rerun-free scene geometry |
| `mujoco_collision` re-export of `build_mujoco_model` | dropped | nothing imported it through `mujoco_collision`; the two names it uses itself stay imported |

Kept: `scene_geoms` (rerun-free geometry the planner needs), `stream_target`, `jog`, `ik`, `ompl_planner`, `stack`.
Every importer and current doc was repointed; dated records under `docs/history/` keep the names they were written with.
Not in this slice (behaviour, its own slice): the coordinated planners' `finish_path` does not re-plan an OMPL route that
cannot be certified, as `ArmPlanner.plan_joint` now does (2026-10-08-plan-replan.md).

Related checks run (library alone, FR3 assets linked): `python -m` planner, retiming, trajopt, viz.preview, scene_geoms,
mujoco_collision, stack, assets, grasp_visual, multi_arm.cooperative, control.factory; `check_retiming`,
`check_ui_scene_injection`, `check_pair_planning`, `check_dual_modes --refine`, `check_cooperative`, `arm_console --self-check`.
All pass.
