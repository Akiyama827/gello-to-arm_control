# Scene hosting step 2c: the sim bench runs on the arm's supervisor (refactored, not run end to end)

Operator 2026-10-09: step 2c go; the assembly task's end-to-end runs are DEFERRED (the camera holder and perhaps the gripper will
change), so this slice is the refactor with startup and wiring checks, not a task run. Goal yardstick: one control authority per
arm (the supervisor), the project composing library pieces rather than owning a plant.

Library:
* `arm_control/supervisor/launch.py`: `start_supervisor(config, env, overlays=, control_mode=, plant=, cwd=, stopped=)` and
  `stop_supervisor`, moved out of `examples/run_console.py` (which now calls them with its example overlay and mode) so a
  project launcher starts a supervisor the same way. Behaviour unchanged: one merged config for the supervisor and every node
  (`ARM_CONTROL_CONFIG`), `ARM_CONTROL_SOCKET`, own session, SIGTERM (DISARM) then kill, log kept on failure.
* `arm_control/supervisor/scene_observer.py` + `nodes/scene_observer.py`: the supervisor-hosted world (observe/1, read-only, no
  ownership) republished as the plant node's `sim_qpos` (0.2 s) and `inhand_pose` (0.1 s) messages, same encodings, so world
  readers keep their inputs when the plant moves into the supervisor. Self-check in the library suite.

Project:
* `dataflows/sim_bench.yml`: `operator_arm`, `sim_perception`, `arm_planner` unchanged; `supervised_arm` (the arm_controller
  contract from the supervisor), `franka_hand` (`franka_gripper`, the bench's own Hand client, wired as `real_workcell_add` does)
  and `scene_observer` replace the Dora `arm_controller`, `sim_bridge` and `mujoco_interface`. RETIRED in this graph: the Dora
  arm_controller (CTRL-053's per-graph retirement), the Dora plant node, `sim_bridge`'s SimHand/GraspGate path.
* `configs/simulation/franka_bench.yaml`: includes `robots/fr3/description.yaml` (its restated FR3 facts deleted: joints, gripper
  joints, URDF, TCP, torque ratings, num_motors, motor_ids), `robots/fr3/execution.yaml` (a supervisor requires a policy; its own
  `arm.max_vel/max_acc`, 0.8-1.0 rad/s, double the FR3 policy's 0.4, deleted: the bench now plans at the real FR3's limits) and
  the library's `sim_supervised.yaml` (the simulated servo = the RT server's launch defaults); states `rt` (instance `sim_bench`,
  the Hand at 127.0.0.1:47812), `sim_observe`, `sim_actor: assembler`, `sim_hand: {port: 47812}`.
* `tools/run.py`: `Scenario.supervised`; `sim bench` starts the supervisor over the entry config (with the scene injections its
  world needs, `SCENE_INJECTIONS`) before the graph and stops it after.
* `docs/ARCHITECTURE.md`, `docs/status/2026-10-06-config-single-source-plan.md` (its deferred bench item is done).

Not retired, with conditions: `sim_bridge` (SimHand, SimHandMove, GraspGate) and the Dora `mujoco_interface` stay for
`sim_workcell_add` until it moves (step 4). Deferred with the assembly task: keying the plant's dock release on the measured jaws
(the sim Hand's ramped OPEN reaches the "commanded open" trigger ~0.7 s later than SimHand's instant one; only an end-to-end run
can show the effect), the sim dock re-measure at K_c 1000, and the bench pick geometry (2026-10-09-rt-cartesian-pc.md).

Checks (related only; no end-to-end run, by the operator's deferral):
* new `tools/check/check_dataflows.py` (project suite): every project graph's node paths exist and every input names a declared
  output (6 graphs; the check's own probe catches a dangling input and a missing path).
* new `tools/check/check_bench_supervised.py` (project suite): the supervisor started over the bench entry exactly as `run.py`
  does hosts the composed world (FR3 identity, both actors), offers `cartesian_impedance`, its observe/1 yields what
  `scene_observer` publishes (full qpos, the module-in-EE pose), and the Hand answers on hand_bridge at rt.host:rt.hand_port.
* `scene_observer` self-check; `examples/run_console.py --self-check` (also without PYTHONPATH); `check_bench_twin`,
  `check_franka_ownership`, `check_config_duplicates` (allow-list pruned: 2 groups no longer duplicated), `check_add_contracts`,
  `check_assembly_planner`, `check_route_selection`, `check_calibration_sim`, `check_workcell_scene`, `check_launcher_shutdown`;
  ruff.
