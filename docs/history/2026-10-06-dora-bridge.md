# The first graph on the supervisor: `supervised_arm` (CTRL-053 pilot, single-arm simulation)

`dataflows/sim_franka_motion_supervised.yml` is `sim_franka_motion.yml` with the in-process `arm_controller` node and the
`mujoco_interface` plant node replaced by ONE node, `nodes/supervised_arm.py`, which serves the same Dora contract from the arm's
supervisor. The supervisor runs the same `ArmController` over a simulated MuJoCo plant (`service --plant sim`, `SimLink`); the console
and the replay node are unchanged. `python examples/run_console.py --supervised [--no-viewer]` starts the supervisor and the graph
together and stops both (SIGTERM DISARMs the plant on the way out).

**The bridge** (`arm_control/supervisor/dora_arm.py`, `DoraArmBridge`, Dora-free and benched without a Dora runtime): plan, control,
jog, gripper in; `controller_event`, `arm`, `motor_state`, `model_revision` out, through one `ArmClient`, one non-blocking
`AsyncAuthorityHandle` (moved to `multi_arm/supervised.py`, shared with `SupervisedHardware`) and the console control adapter. It owns
no authority and no control law. It refuses to start if the supervisor runs other default gains than the config it resolved itself
(`check_config`: one recipe, `default_gains_and_presets`, proven against the running supervisor).

**What it does that the in-process node did implicitly:** a new plan REPLACES a staged-but-unreleased one (the supervisor refuses a
second admitted plan, found by running the real graph); an ungated plan is released when admitted; the first fresh measured pose after
ARM becomes the `ready` event; a mode heartbeat keeps the console's mode fresh.

**Stated differences:** jog and gripper are refused with a notice (the supervisor has no raw-jog and no gripper); gains must be named
presets (`gain_presets`); `hold` cancels the live leg and leaves the arm held by the supervisor (no latched "frozen" state); `stop`
cancels and DISARMs; `motor_state` gains and measured torque are zeros where the supervisor does not publish them.

**What running the real graph found** (headless, driven over the console's HTTP API):
1. The stock example console itself did not come up: `examples/configs/franka.yaml` lacked `arm.max_tau` since the examples were split
   out (62a48b6), so the controller refused to build. Restored from the project's `configs/robots/franka/limits.yaml` (= the URDF efforts).
2. A supervisor REQUIRES an execution policy; the example had none. `examples/profiles/sim_supervised.yaml` states it (the project's FR3
   thresholds, except the speed limits, which are the ones this example's own planner already runs at; the project's real FR3 policy is
   slower and would refuse this example's plans) together with the sim servo parameters (the RT server's own launch defaults). A
   simulation example only; nothing there applies to a deployment.
3. The plant spawned with the Hand shut (the Dora node opened it after load; the shared builder did not): `open_jaws` is now shared.
4. ARM, plan, re-plan and the console's mode display work through the supervisor. EXECUTE was rejected by admission ("start point 0.377
   rad from measured pose (tol 0.1)"): **joint 7 of the simulated FR3 oscillates by about +-0.6 rad whenever it is held**, and the
   supervisor's start tolerance catches what the in-process controller (no start check) never showed. Measured on the plant alone, with
   fingers unservoed, with and without gravity compensation, with and without self-collision, at control periods 1 to 10 ms, and in the
   STOCK graph (the end-effector yaw wanders by tens of degrees once armed): not a SimLink or bridge effect. Likely the wrist's tiny
   inertia against kp 400 under the 12 N*m limit (a limit cycle); a plant/gains question for the example, not changed here.

Not done: a deterministic end-to-end execute through the real Dora graph (blocked by finding 4; the bridge contract is benched against a
real supervisor on the 3-joint fixture in `tools/bench/check_dora_arm.py`, and ARM/plan/re-plan were verified live); real-hardware graphs;
the project's composed-scene graphs (scene ownership decision, see 2026-10-06-sim-link.md); retirement of `arm_controller` for any graph.
