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
* `supervisor/dora_arm.py`: `jog` is forwarded as a setpoint; a staged, unreleased plan is cancelled first (`plan_cleared`), and
  an execute that later names it answers a failed `leg_result` ('superseded by a jog'). Gripper stays refused.
* `docs/ARCHITECTURE.md` (setpoint/1 section; the stream/1 checkpoint's "raw jog" note), the pilot graph's header.

Named, not changed:
* Within `abort_pos_err_rad` the step size is the caller's envelope (the console's `JogLimits`: stroke, workspace, singularity,
  collision). A policy client calling `setpoint` directly gets no tighter bound, exactly as any Dora node sending `jog` to the
  in-process controller did; the reviewer measured a one-shot 0.3 rad step peaking at 2.26 rad/s against the policy's 1.0
  velocity limit. A supervisor-side rate bound (e.g. from the policy's velocity limits and the jog timeout) is a safety
  threshold: OPERATOR DECISION, not taken here.
* One RPC per jog on the bridge's Dora thread (the loop answers within one 10 ms tick).

Review round 1: ACCEPT, SHOULDs taken:
* An owner that leaves (disconnect, unclaim) left its setpoint live until the deadman (the arm kept travelling 0.26 rad after an
  unclaim): `_abandon` now clears it (`ArmController.clear_jog`: the measured-pose hold its expiry gives).
* A target beyond the policy's `abort_pos_err_rad` was accepted, then cancelled by the controller on the next tick (the caller
  told 'held'): refused at the op with the reason (an existing threshold, not a new one).
* An execute of a plan a jog superseded got no `leg_result` (the console keeps it pending; `moving` stayed true): it now fails
  the leg. The record's "the same outcome" was wrong for jogs within start tolerance (in-process ran the plan); corrected.
NITs taken: the 0.5 s quiet window removed (the cancel is synchronous; it only hid real refusals); a no-answer/transport error
says "outcome unknown", not "refused"; `_setpoint` refuses on a native reference itself; `ArmController.jog_timeout_s`; a plan
release clears a stale jog (it could re-anchor or resume after the leg, in-process too; not separately checked); the staged-plan
refusal is checked directly; ARCHITECTURE's stale "raw jog remains" line. Mutants caught: owner loss keeps the jog, no
abort-distance refusal, a superseded execute answered with silence.

Checks: `check_dora_arm` (jog follows while re-sent; on expiry the hold anchors at the measured pose, not the target; refused
past a limit, malformed, non-owner, while a plan runs (one notice through the bridge), DISARMed; a jog supersedes a staged plan
with no refusal shown; the deadman is the mode config's 0.3 s). Mutants caught: bridge refuses jog, no limit check, no
supersede, timeout not from the mode config, no live-execution refusal. Related: `check_arm_controller`,
`check_execution_policy`, `check_pose_hold`, `check_node_shutdown`, `check_sim_supervisor`, `check_stream`,
`check_sdk_supervisor`, `check_dual_supervisor`, `check_state_history` (offline server rebuilt from current rt/src),
`run_console --self-check`, ruff.

Review round 2: ACCEPT (every round-1 fix verified by probe). NITs taken: only `Refused` reads "jog refused", any other SDK error
"outcome unknown" (the SDK maps lost/garbled replies to SupervisorError); the superseded-plan ids are bounded (64, oldest out) and
an id staged again is forgotten; the controller's own refusal (`ArmController.jog_refusal`, pure) is checked first, so a far
target while frozen or DISARMed reports that, not the distance.
OPEN (pre-existing, found in review; owner: CTRL-053, the next slice; blocking effect: a migrated console cannot recover a failed
settle with Stop, only with DISARM then ARM): after a completion timeout the supervisor's controller stays `frozen`; the bridge's
Stop cancels only executions the handle holds, so with none nothing reaches the controller, and the SDK has no op that clears
`frozen` (the in-process controller cleared it on Stop). The "cancel to clear" text is wrong for setpoint/1 callers until then.
