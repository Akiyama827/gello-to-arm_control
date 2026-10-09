# CTRL-053 graph 1: the FR3 sim motion graph runs on the arm's supervisor (the in-process controller graph is retired)

Operator 2026-10-09: graphs 1-8 approved in one go; jog through the supervisor first (setpoint/1, the previous slice, which this
branch is stacked on). Refactored and checked; started, not run end to end with a browser.

Library:
* `dataflows/sim_franka_motion.yml` IS the supervised graph now (the pilot `sim_franka_motion_supervised.yml` is deleted; the
  in-process `arm_controller` + `mujoco_interface` graph is retired): `arm_console`, `trajectory_replay`, `supervised_arm`, plus
  what the pilot lacked for parity with the retired graph and with `real_franka_motion.yml`: `franka_gripper` (the console's
  gripper slider and grasp requests, over the Hand's own protocol) and `visualizer`. Jog is setpoint/1.
* `plants/sim_link.py`: SimLink serves the Hand (`sim_hand`: the real hand_bridge wire, `SimHandBridge`) for EVERY sim plant,
  from the plant's MuJoCo backend, on the servo thread (a sample after each step, a command for the next) -- moved out of
  `ScenePlant` (scene_core.py), which served it only for a composed world. One place; the single-model FR3 twin now has its Hand.
* `examples/profiles/sim_supervised.yaml`: the sim Hand at `rt.host 127.0.0.1`, `rt.hand_port` == `sim_hand.port` 47812.
* `end_effectors/franka_adapter.py` (`franka_gripper`): the Hand's address is required (`rt.host`, `rt.hand_port`); it defaulted
  to the REAL RT box (172.16.1.2:47802), so a sim graph that forgot them would have driven the bench's Hand. Every real entry
  that runs the node states both (FR3 #2's are staged from the registry at launch); several SIM entries inherit the real address
  from the deployment and now override it through the profile.
* `examples/run_console.py`: one arm is always supervised (`--supervised` removed); RUNBOOK says so.
* `tools/check/check_dataflows.py` (moved here from the project, which now runs it on its own graphs): every node path exists and
  every input names a declared output, for all 9 library graphs.
* `tools/check/check_sim_franka_motion.py`: the supervisor started as run_console starts it runs the gains `supervised_arm`
  resolves, offers setpoint/1 (a jog moves the FR3 twin) and cartesian_impedance, and serves the Hand at rt.host:rt.hand_port.

Project:
* `tools/run.py`: `Scenario.arm_supervised` (per arm); `sim franka motion` runs the library graph on its supervisor (the DM arm's
  `sim motion` stays in-process: graph 2 needs a DM execution policy, an operator decision). `start_scenario_supervisor` is the
  one way run.py and the checks start a scenario's supervisor; the mode config resolves project-first, then the library's
  (`resolve_mode_config_path`; joining CONTROL_ROOT, the 2c NIT fix, missed the library's shipped modes). `ARM_CONTROL_ROOT` is
  set for the launcher's own process.
* `configs/simulation/franka.yaml`: includes `sim_supervised.yaml`, `rt.instance_id: sim_franka`.
* `configs/simulation/franka_bench.yaml`: its sim Hand address restatements removed (the profile's).
* `tools/check/check_supervised_sims.py` (was `check_bench_supervised.py`): every supervised sim scenario (`bench`,
  `motion/franka`) starts as run.py starts it: no in-process controller in its graph, gains as `supervised_arm` resolves them,
  the FR3's joints, cartesian_impedance and setpoint/1, the sim Hand; the bench's world checks as before.

Retired: the in-process FR3 sim motion graph (Dora `arm_controller` + `mujoco_interface` in it), the pilot graph, ScenePlant's
own Hand hosting, `run_console --supervised`. Not retired: `mujoco_interface` and the in-process `arm_controller` serve the
remaining graphs (sim_motion, dual_arm_sim, sim_view_arm, and the project's) until each moves.

Console parity (the setpoint/1 review's open item, taken here because this graph is the first console on the supervisor):
* `supervisor/runtime.py` + `service.py` + `sdk.py`: `hold(reason)` (controller/1 op `controller_hold`): cancels live executions,
  drops a setpoint, and clears the controller's `frozen` hold after a failed settle, holding the measured pose, still ARMed (the
  in-process controller's cancel/Stop). Before it, only DISARM then ARM recovered a failed settle on a supervised arm.
* `supervisor/dora_arm.py`: an untargeted cancel/hold/stop from the console also calls `hold`.
* `control/arm_controller.py`: the frozen refusal says what clears it.
* `check_dora_arm`: a plan whose completion goal it never reaches freezes the controller (setpoint refused 'frozen'); the console's
  Hold clears it, still ARMed. Mutant (bridge never calls hold) caught.
* `franka_gripper` address check: `check_node_shutdown` states the fixture's address and asserts a config without one never starts.
