# The last Dora controller/plant nodes retired; the FR3 build in the offline suite; config ownership debt (CTRL-053)

Every graph's arm runs on its supervisor now (sim-add, merged `723dcaf`, was the last controller user). This slice deletes what
no graph runs and closes the offline gaps the run left open.

Library:
* Deleted: `nodes/arm_controller.py` + `arm_control/control/adapter.py` (the in-process Dora controller; `ArmController` itself
  stays, the supervisor runs it), `nodes/rt_interface.py` + `arm_control/plants/remote_rt/adapter.py` (the Dora RT plant;
  `RtBackend` stays, the supervisor's link), `nodes/mujoco_interface.py` + `arm_control/plants/mujoco/adapter.py` (the Dora
  MuJoCo plant; `SceneCore`/`ScenePlant`/`SimLink` host every simulated world).
* `dataflows/sim_view_arm.yml` (`run.py sim view`): `supervised_arm` + `visualizer` over the FR3's supervisor (never ARMed by
  the graph), instead of the plant node.
* `arm_console --self-check` (`console_graphs`, failing since the controller node left the last graph): checks each
  console's one authority feedback source and that no graph wires the retired node.
* `tools/check/suite.py` builds `rt/` a second time with `WITH_FRANKA=ON` (`<build>-franka`) and runs its CTest (23, with
  `hand_bridge_selfcheck`); without libfranka that FAILs with the reason. This PC's `/usr` libfranka is 0.21.2 (the box's).
  Verified by hand: configure, build and 23/23 CTest pass.
* Comments that called the retired nodes live are corrected.

Project:
* `sim modular` (and its `single` alias, the generated rep1d arm) deleted, operator 2026-10-09; with it
  `configs/{entries/sim_single_arm,simulation/single_arm}.yaml` and the launcher's asset machinery (`ASSET_COMMANDS`,
  `--regenerate-assets`, `--skip-assets`; it built rep1d only).
* `sim view` is supervised (`supervised=True`, the FR3 motion profile).
* Config ownership (allow-list OPEN debt, step 4b), every resolved config before/after identical (62 files; the reviewer re-ran it from scratch trees):
  - the RT box (`rt.host`, `rt.state_timeout_s`, `rt.max_fault_ms`, `listen_mode`): `configs/deployments/rt_box.yaml`,
    included by the FR3 deployment and the base hardware (the registry's `rt_host.address` in `instances.yaml` stays the
    second statement of the address, held equal by `apply_instance` and the deployment self-check);
  - `arm.plant_gravity_comp: true`: the FR3's own description (`robots/fr3/description.yaml`), not the deployment, the bench
    or the library example (`examples/configs/franka.yaml`);
  - `sim_self_collision: false` (with its reason): `configs/simulation/hosted_world.yaml`.
* Checks updated: check_add_contracts builds the controller with the factory (as the supervisor), check_node_shutdown drops
  the controller node, check_supervised_sims expects the observer's constraint_force (from the sim-add slice),
  check_franka_ownership names rt_box.yaml as rt.host's owner.

Checks: check_node_shutdown, check_arm_controller, check_scene_core, arm_console self-check, check_supervised_sims (3 sim
scenarios incl. view), check_supervised_real, check_base_supervisor, check_add_contracts, check_calibration_contract,
check_franka_ownership (config duplicates 91 in 26 groups), deployment self-check, check_rt_deployment, check_bench_twin,
the WITH_FRANKA build + CTest, `run.py sim view --dry-run`, ruff: PASS.

Also fixed here, left failing by the sim-add merge: `console_graphs` (above) and `check_supervised_sims` (the observer's new
`constraint_force`).

Review round 1 (opus): ACCEPT. SHOULDs taken:
* Two safety keys now come from an include, so a later layer could override them silently. `check_franka_ownership` pins
  their owners on the real FR3 entries: `arm.plant_gravity_comp` -> the FR3 description, `rt.state_timeout_s` and
  `rt.max_fault_ms` -> `rt_box.yaml`. `check_bench_twin` asserts `sim_self_collision` is false for every config that hosts a
  composed world.
* Live docs (`rt/README.md`, `docs/ARCHITECTURE.md`) no longer describe the retired nodes as current.
* The console self-check's lost "unhandled input" rule now applies to `supervised_arm`: `dora_arm.INPUTS` is the node's input
  table, `unhandled_inputs(graph)` is checked over the library graphs (console self-check) and the project graphs
  (`check_add_contracts`); none is unhandled.
* NITs taken: stale comments (franka_bench, hosted_world, check_add_contracts, pyproject, the base unit file's comment,
  arm_planner, RUNBOOK), the RT README's CTest count, the supervisor's sim log prefix `[mujoco_interface]` -> `[sim_world]`,
  the unused `REPO_ROOT` in run.py.
* NIT not taken: an offline SIGTERM/STOP -> DISARM check for `supervised_arm` (the gap predates this slice).

OPEN:
* 8 allow-list groups, all the legacy `franka_bench` restating other files (camera parameters, Hand task, pick choreography,
  fixtures, nominal mount, completion tolerances): retire with the composed-scene work on franka_bench (pulling those files in
  whole would add keys the bench does not use).
* `BaseDriver`'s in-process executor/`base_command` path (no node runs it; the supervised client's decision core): fold into
  `SupervisedBaseDriver` when its checks move onto the client.
* `ArmController.run` (the Dora loop) has no Dora caller left; the supervisor ticks the controller itself.
