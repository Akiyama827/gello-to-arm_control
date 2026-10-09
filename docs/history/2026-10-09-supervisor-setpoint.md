# setpoint/1: console jog (and policy joint targets) through the supervisor

Operator 2026-10-09: graphs 1-8 of the CTRL-053 migration approved in one go; six of them (sim_franka_motion, sim_motion,
dual_arm_sim, real_franka_motion, real_franka_workcell_motion, calibration_workcell) feed console jog into the Dora
arm_controller, which the supervisor refused ("no raw-jog"). Operator chose a streamed setpoint verb on the joint law over
DeadmanTeleop/stream/1 (native-reference only, excludes Cartesian legs, payload refused) and over migrating without jog. The
same verb is the command path for policy clients (VLA/RL) that emit joint targets.

No new control law, gain, limit or timing constant: the supervisor's controller already had the in-process jog (`JogHold`:
deadman, one-shot expiry, re-anchor at the MEASURED pose); this exposes it.

* `control/arm_controller.py`: `on_jog` returns why it drops a jog (stopped, frozen, not armed, no fresh pose, Soft, a plan
  running), None when taken. The Dora node ignores the return as before.
* `control/factory.py`: `jog_timeout_s` defaults from the mode config's `jog.timeout_s` (moved from `control/adapter.py`), so
  the supervisor's deadman is the one the console reads; it used the hard-coded 0.2 s.
* `supervisor/runtime.py` + `service.py`: extension `setpoint/1` (offered when the supervisor is NOT on the native reference),
  op `setpoint(q)`: owner only, finite and full width, refused while any execution is live (a staged plan included), refused
  when it moves a joint further past its position limit than the measured pose is (the console's escape-from-a-stop rule),
  then the controller's own refusals. Returns the timeout.
* `sdk.py`: `ArmClient.setpoint(q) -> timeout_s` (unreadable reply = SupervisorError, as the rest of the SDK).
* `supervisor/dora_arm.py`: `jog` is forwarded as a setpoint; a staged, unreleased plan is cancelled first (`plan_cleared`;
  the in-process controller kept it and refused it at execute on start tolerance, the same outcome), and refusals during
  that cancel (0.5 s) are not shown. Gripper stays refused.
* `docs/ARCHITECTURE.md` (setpoint/1 section; the stream/1 checkpoint's "raw jog" note), the pilot graph's header.

Named, not changed:
* The step size is the caller's envelope (the console's `JogLimits`: stroke, workspace, singularity, collision). A policy
  client calling `setpoint` directly gets no step bound from the supervisor, exactly as any Dora node sending `jog` to the
  in-process controller did. A supervisor-side step bound (e.g. from the policy's velocity limits and the jog timeout) is a
  safety threshold: operator decision, not taken here.
* One RPC per jog on the bridge's Dora thread (the loop answers within one 10 ms tick).

Checks: `check_dora_arm` (jog follows while re-sent; on expiry the hold anchors at the measured pose, not the target; refused
past a limit, malformed, non-owner, while a plan runs (one notice through the bridge), DISARMed; a jog supersedes a staged plan
with no refusal shown; the deadman is the mode config's 0.3 s). Mutants caught: bridge refuses jog, no limit check, no
supersede, timeout not from the mode config, no live-execution refusal. Related: `check_arm_controller`,
`check_execution_policy`, `check_pose_hold`, `check_node_shutdown`, `check_sim_supervisor`, `check_stream`,
`check_sdk_supervisor`, `check_dual_supervisor`, `check_state_history` (offline server rebuilt from current rt/src),
`run_console --self-check`, ruff.
