# Thinning before hardware, and CTRL-050's online mode for the live consumer

Operator decisions 2026-10-10: a thinning bundle runs before the hardware session (retire compatibility layers, dead code,
dead configuration); ArmController's Dora-Arrow message interface waits until after the first hardware session.
2026-10-09: live consumers use the online query mode (CTRL-050).

CTRL-050 (project `assembly/runtime/planner.py`). The planner is the only live perception consumer (the console consumes
none; perceptions' `vision_pipeline`/`sensor_fusion` nodes run in no graph). It judged a module pose fresh by its ARRIVAL;
it now uses the producer's Unix capture stamp (`stamp`, required by `pack_object_observation`): the capture-to-receipt
latency is taken off the receipt on the planner's own clock, so `perception.stale_s` bounds the capture's age, and a pose
captured "after now" is a clock fault and never used. check_assembly_planner `_check_online_module_pose`: a 0.2 s old
capture is fresh with its age kept, a 1.5 s old capture arriving now is stale, a future capture is ignored.

Retired (each had one replacement):
* `arm_control.messages` (re-exports of `contracts/*`, 40 importers across this library, `Control/` and perceptions) ->
  each importer names its `arm_control.contracts.<module>` (rewritten mechanically; all 181 imported names verified to
  resolve). Its wire round-trip self-check moved to `python -m arm_control.contracts` (`contracts/__main__.py`).
* `arm_control.rt_protocol`, `arm_control.simulation.mujoco_backend`, `arm_control.simulation.scene_backend` (aliases with
  no or one importer) -> `plants.remote_rt.protocol`, `plants.mujoco.backend`, `plants.mujoco.composer` (perceptions'
  `sim_perception` now imports the composer). The ARCHITECTURE alias table is gone; check_franka_ownership's alias guard
  went with it (its "documented module does not exist" check covers a stale command).
* The Dora pair path: `multi_arm/adapter.py` (`TopicArmHandle`, `run`), `contracts/multi_arm.py`,
  `nodes/multi_arm_coordinator.py`, and `multi_arm.BimanualInterface` -> the pair runs through `SupervisedHardware` over
  `SdkArmHandle`s. None was wired into a graph, and driving a pair through them would release the members without the
  common start stamp (`release_context`). check_bimanual_contract keeps the coordinator cases and drops the topic one.
* `nodes/motor_state_logger.py` and the topic logger (`viz/motor_state_logger.main/cli/_row`, which read a `motor_command`
  topic no supervised graph has) -> `trace_main` (`nodes/trace_logger.py`, in `real_franka_motion`). check_node_shutdown's
  logger cases now run `trace_main` against a fake trace/1 client.
* Dead names (referenced nowhere): `ArmController.plant_time`, `MuJoCoBackend.release_constraint_if_force_exceeds` and
  `aggregate_inertial`, `ScenePlant.apply_gripper`, `SceneState.set_actor_q`, `JogPanel.max_speeds`,
  `calibration_console.ConsoleAuthority` and `ActorSpec.command_port/arm_port`, `pack/unpack_controller_settings`,
  `pack_object_poses` (perceptions produces object_poses itself).
* `RobotConfig`'s serial-era typed fields read by nothing (`serial_port`, `baud_rate`, `update_rate_hz`, `listen_mode`,
  `rx_timeout_ms`, `debug_tx/rx` and their throttles, `torque_scale`, `gui_publish_rate_hz`, `gc_alpha`,
  `torque_ramp_sec`, `rerun_app_id`; the visualizer reads `rerun_app_id` from the raw config, unchanged) and their
  validation.
* Configuration keys no code reads, from every file that stated them: top-level `serial_port`, `baud_rate`,
  `update_rate_hz`, `listen_mode` (the live one is `control_mode.plant.listen_mode`), `sim_backend`, `real_backend`,
  `hardware_update_rate_hz`, `sim_update_rate_hz`, `ompl_solve_sec`. Every resolved entry, simulation and example config
  (24) was compared before/after: nothing else changed. Two config-duplicate groups disappeared with them.
* Project: `nodes/calibration_console.py` (a "legacy motion-console entry point" nothing launched);
  `assembly/planning/dock_targets.py`'s `SEAT_TOL_*` re-export (the planner imports `assembly.model.mate`).

Left, deliberately: `ArmController`'s Arrow-message handlers (operator: after the first hardware session);
`planning/stream_target.py` (stream/1's composition strategy, a feature without a production caller yet); `banks` and
`simulation_only` in two project configs (hardware/storage descriptors, unread, harmless); `assembly/runtime/base_driver.py`
as `SupervisedBaseDriver`'s parent (its decisions are the base client's).

Checks: every package module imports (179); all 46 library module self-checks; library checks jog_ramp, execution_policy,
pose_hold, node_shutdown, bimanual_contract, arm_controller, controller_mode_feedback, dora_arm, sim_supervisor,
scene_cohost, pair_planning, dual_console, cooperative, bimanual_sync, supervisor_ledger, trace_retention,
supervised_hardware, sdk_supervisor, dual_supervisor, state_history; project assembly_planner, config_duplicates,
bench_twin, dual_bench, supervised_sims, add_contracts, completion, manual_carry, dataflows, franka_ownership,
calibration_contract, workcell_scene, model_binding, stream_admission, base_supervisor, supervised_real; perceptions
check_sensor_contracts; `run.py` dry-runs (sim bench, sim-add, real franka motion, real-add); sim-add e2e passed (70.9 s).
