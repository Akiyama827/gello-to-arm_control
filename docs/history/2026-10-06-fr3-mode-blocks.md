# FR3 mode basics and planner defaults, stated once (config single source, step 3, library)

* **`robots/fr3/control_mode.yaml`** holds the FR3's controller stiffness ceilings (`kp_max` 3000, `kd_max` 200, `torque_limit_max` 90)
  and the console's gain presets (`float`, `soft`, `track`). These were stated identically in the library's sim motion mode
  (`examples/profiles/motion_franka.yaml`) and in the deployment's real mode, which includes this file in the project step. The sim mode
  includes it and keeps only its own tracking stiffness and jog envelope. Values unchanged.
* **A conflicting description resolved** (operator 2026-10-06: the real mode keeps Float and Soft): the sim mode said "the real FR3 uses a
  deployment-owned mode ... it deliberately has no presets". The real mode has carried the presets since 2026-09-10. The shared block
  states that the RT box's servo law consumes them on the real arm, and marks `soft` as not hardware-certified (verify before first use;
  hardware track list).
* **Planner defaults have one home**: `arm_control.planning.PLANNER_DEFAULTS` (padding -0.002 m, solve 2.0 s, simplify 0.5 s, resolution
  0.005), used by `build_collision_stack` and the planning stack through `ompl_settings`. The same four numbers were written twice in code
  and restated in both example motion modes. They are deleted there, and a mode states one only to deviate.

Checked: `tools/inspect/config_snapshot.py` (parent) shows only the planner-default keys absent from the two example modes; the effective
values are the code defaults, which are equal. `arm_console --self-check`, `run_console.py --self-check`, offline suite.
`check_pose_hold_sim.py` does not run here (it needs the historical URDF fixture; it is not in the suite).

Review (ACCEPT) note, folded in before merge: the planner defaults were still written into the `OMPLPlanner` constructor (resolution
0.01, not 0.005) and the `MuJoCoCollisionWorld` constructor (padding). `arm_control.planning.PLANNER_DEFAULTS`, which lives in the package
`__init__` so that neither importer needs the OMPL bindings, now supplies those constructor defaults too. Every existing caller passes
explicit values (`multi_arm/planning.py` deliberately uses its own two-arm settings), so nothing changes at runtime. A future bare
`OMPLPlanner(...)` plans at 0.005, not 0.01.
