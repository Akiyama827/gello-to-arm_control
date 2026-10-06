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
second admitted plan, found by running the real graph; staged plans are tracked when staged, so an execute right behind a plan finds
it) and is REFUSED while one runs ("... is still executing"); an ungated plan is released when admitted; cancelling a running leg
reports a failed `leg_result` as well as the clear; the first fresh measured pose after ARM becomes the `ready` event; a mode
heartbeat keeps the console's mode fresh; `motor_health` (armed, latched fault) is published on change and twice a second, because the
console takes its armed state and its fault latch from it, and only it CLEARS a latch.

**Faults and loss.** A supervisor fault (e.g. `cmd_lost`) is a recoverable `motor_health` latch plus a log line, not the terminal
`fault` controller event (which in-process means the controller stopped for good), so the console clears it after DISARM then ARM.
If the supervisor cannot be read (the SDK channel never reconnects), no ARM/DISARM edge is published (unknown is not an edge): the
bridge emits ONE terminal `fault` and a `motor_health` carrying `unknown_authority` (the console has no consumer for that flag: its
badge reads DISARMED with the "supervisor lost" fault, and Execute is gated by the fault), the node ends non-zero, and it tries to
DISARM through a new session (a supervisor that loses its owner cancels and HOLDS, staying armed, by design). The launcher watches the supervisor and
stops the graph if it exits.

The console takes its armed state and fault latch from `motor_health` ONLY in this graph (its graph check requires exactly one authority
source), so the `arm` output is published but not wired. A failed ARM or DISARM (the handle's fault reason) is shown as the same latch.
Known, not this branch's: in the live example the simulated FR3 can slump after a long freeze and a re-ARM may drop within seconds when
the supervisor's controller rejects a servo command as invalid (`execution_policy.command_valid`), so "DISARM then ARM recovers" is not
reliable live there.

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
3. The Dora node opens the Hand after load and the shared builder did not: `open_jaws` is now shared, so both spawn alike. (It does not
   cure finding 4: the swing is the same with the jaws shut.)
4. ARM, plan, re-plan and the console's mode display work through the supervisor. EXECUTE was rejected by admission ("start point 0.377
   rad from measured pose (tol 0.1)"): **joint 7 of the simulated FR3 oscillates by about +-0.6 rad whenever it is held**, and the
   supervisor's start tolerance catches what the in-process controller (no start check) never showed. Measured on the plant alone (and
   in the STOCK graph, where the end-effector yaw wanders by tens of degrees once armed): not a SimLink, bridge or jaws effect (jaws
   open or shut, gravity compensation on or off, control period 1 to 10 ms all give the same swing). Review measured the cause: the
   explicit-damping instability of the 1 ms plant step (kd*dt/I > 2 with a wrist inertia of about 0.003 kg m^2) at the example's
   kd 10 for joint 7: kd 6 or less holds still (0.001 rad), kd 7 swings 2 rad, a 0.5 ms step holds still; the 12 N*m limit only bounds
   it into a limit cycle. A plant-step or example-gains question, not changed here.

**Contract differences still stated:** `runtime_state.stopped` is always False; `plan_cleared` is sent for every cancelled plan even when the
cancel did not name it; `ready` can fire again when measured validity flickers; a plan refused by supervisor admission gives
`plan_staged` false only (in-process also sends a `leg_result` for policy refusals); plan cancels run synchronously in the Dora loop
for up to 2 s; the same-preset-gains cancel relies on the supervisor (`live` is not passed). The review also saw, once, the supervisor
log "stopped: non-finite or invalid servo command; controller replaced" about 7 s after a re-ARM that followed a freeze; not
root-caused (the joint-7 limit cycle is a candidate).

Not done: a deterministic end-to-end execute through the real Dora graph (blocked by finding 4; the bridge contract is benched against a
real supervisor on the 3-joint fixture in `tools/bench/check_dora_arm.py`, and ARM/plan/re-plan were verified live); real-hardware graphs;
the project's composed-scene graphs (scene ownership decision, see 2026-10-06-sim-link.md); retirement of `arm_controller` for any graph.
