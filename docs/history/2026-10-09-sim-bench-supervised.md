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
  contract from the supervisor), `franka_hand` (`franka_gripper`, the bench's own Hand client; its grasp path ungated as the old SimHand path was, without
  `real_workcell_add`'s hand_move wiring)
  and `scene_observer` replace the Dora `arm_controller`, `sim_bridge` and `mujoco_interface`. RETIRED in this graph: the Dora
  arm_controller (CTRL-053's per-graph retirement), the Dora plant node, `sim_bridge`'s SimHand/GraspGate path.
* `configs/simulation/franka_bench.yaml`: includes `robots/fr3/description.yaml` (its restated FR3 facts deleted: joints, gripper
  joints, URDF, TCP, torque ratings, num_motors, motor_ids), `robots/fr3/execution.yaml` (a supervisor requires a policy; its own
  `arm.max_vel/max_acc`, 0.8-1.0 rad/s, double the FR3 policy's 0.4, deleted: the bench now plans at the real FR3's limits) and
  the library's `sim_supervised.yaml` (the simulated servo = the RT server's launch defaults); states `rt` (instance `sim_bench`,
  the Hand at 127.0.0.1:47812), `sim_observe` (125 Hz, a divisor of the 250 Hz servo), `sim_actor: assembler`,
  `sim_hand: {port: 47812}`; drops `motor_state_period_s`, `sim_update_rate_hz`, `sim_scene_rerun` (read only by the retired
  plant path).
* `tools/run.py`: `Scenario.supervised`; `sim bench` starts the supervisor over the entry config with the FR3 control mode (with
  the scene injections its world needs, `SCENE_INJECTIONS`; up to 300 s for a cold mesh cache; SIGTERM/SIGHUP during the start
  stop it) before the graph and stops it after; `--dry-run` says so.
* `assembly/runtime/planner.py`: `motor_health` input; a latched arm fault stops a running sequence (wired in `sim_bench`).
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

Review round 1: BLOCK. Taken:
* (BLOCK) Including the FR3 description gave the config top-level joint names, so `resolve_gains` no longer fell back to the arm
  table: the bench's 600/30 resolved CLAMPED to the generic ceilings 500/5, on the supervisor and in the planner's phase gains
  alike (shoulder damping ~6x lower). The scenario now carries the FR3 control mode (kp_max 3000 / kd_max 200), which reaches the
  supervisor's merged config and the nodes' `ARM_CONTROL_MODE_CONFIG`; `check_bench_supervised` asserts the live supervisor gains
  AND the planner's baseline equal `arm.kp/kd` (mutant "no mode" caught). The config-plan doc's "the 9-motor concern is gone" was
  wrong in exactly this way; corrected there.
* `motor_state_period_s: 0.005` paced the planner by sim_bridge's 200 Hz republish; `supervised_arm` publishes ~100 Hz, so the
  planner's timeouts would run at half speed: removed (wall clock).
* A supervisor fault reaches Dora as a recoverable `motor_health` latch (a checked decision in the bridge), which the planner did
  not read, so a fault between legs only surfaced at the next refused plan: the planner now stops a running sequence on a latched
  fault (`on_motor_health`, checked: idle and clean reports do not stop it).
* `run.py` could orphan the supervisor (Ctrl-C/SIGTERM mid-start); `start_supervisor` now stops the child on ANY exception or
  interrupt during the start (checked live: interrupted at 0.3 s, no supervisor left, its log kept), and keeps the log on every
  failure path (a timeout used to delete the log it pointed at). Cold-cache start: `run.py` and the check allow 300 s.
* `scene_observer` ends with one line, not a traceback, when observe/1 is refused or the supervisor vanishes.
* Stale comments (mujoco_interface, SimHand) fixed.
Named, not changed (owners):
* The twin's ground-truth Rerun mirror (`sim_scene_rerun`) lived in the retired plant node and does not run on the supervised
  bench; the operator note says so. Running it inside the supervisor would put visualisation threads in the control authority's
  process (the observe/1 design keeps readers out of the plant's way); a scene_observer mirror needs the composed world rebuilt
  outside the plant (grafts included). Owner: the assembly task, with its deferred end-to-end runs.
* Payload during a live leg is refused by the supervisor (the reviewed console-adapter rule, CTRL-033); the in-process controller
  accepted it. On a grasp lost mid-carry the planner's payload-0 is refused and the frozen arm holds with the module's
  feedforward (~0.5 kg at the hand: a few mm, under the operator's stop). Owner: the assembly task (cancel-then-payload
  sequencing in the planner, or a bridge rule), with its end-to-end runs.
* `check_dataflows` checks wiring, not that a node reads the input NAME it is given (a renamed `grasp_request` passes), nor
  qpos length against the perception twin's nq (19 = 19 measured in review).

Review round 2: ACCEPT (live supervisor runs 600/30, preset `track`; interrupted and timed-out starts leave no supervisor and keep
the log; cold-cache start measured 93 s; observe/1 at 125 Hz). NITs taken in the project: `check_bench_supervised` now allows the
same 300 s as `run.py` (it used the 60 s default and failed on a cold cache), and both pass the mode config as
`CONTROL_ROOT / scenario.mode_config` (it resolved against the cwd, so `run.py` from the repo root failed before any start).
Accepted as is: two sub-millisecond orphan windows (between `Popen` and the guarded wait; between `run.py` restoring its signal
handlers and Dora installing its own).
