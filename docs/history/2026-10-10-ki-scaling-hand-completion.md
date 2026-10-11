# kp-scaled joint integral; Cartesian legs complete on the hand

Bench session 2026-10-10 (step 13, rung 6), open item 1. Operator decisions the same day: "go with your recs", then the
general ki rule and the settle-plus-sanity-bound completion below.

**ki scales with kp.** `IntegralState(kp_ref=...)`: ki is stated at the profile kp; a leg or hold at a lower kp integrates at
`ki * min(1, kp / kp_ref)` per joint, so the integral keeps the profile's ki/kp time constant (every FR3 joint: ki = 2 kp).
At the Cartesian floor (kp 20) ki is 40, not 1200; before, the bias pinned its 3 N.m clamp against a 20 N.m/rad spring.
Never above the stated ki. `build_arm_controller` passes the executor's construction kp (after the mode's ceilings). Inside a
Cartesian leg the integral stays frozen (2026-10-10, integral freeze), so in practice this changes the hold after a
reduced-kp leg and any joint leg planned with lower gains.

**A Cartesian leg without a completion rule completes on the hand.** Rung 6 timed out and froze at 0.00 mm because K_c had
left j1/j3 32/38 mrad around the null space, outside the 0.01 rad joint rule. Now: at rest (the executor's velocity
tolerance) and the hand (the config's `ee_frame`, on the construction URDF) within a SANITY bound of FK(planned end),
default 20 mm / 5 deg (`arm.done_pose_tol_m`/`done_pose_tol_rad` override, both or neither). A tight bound was rejected: with
the integral frozen the static hand error is load/K_c (the library twin: 5.74 mm / 1.10 deg at 500 N/m), so precision is
K_c's job, not completion's. A settled leg prints its hand error; a timed-out one names it in the leg result. A leg that
carries its own completion rule (the dock) is unchanged; an arm with no hand frame (body-list) refuses a Cartesian leg
without one. The hold after completion still anchors at the planned joint end (unchanged): the null-space offset closes under
the joint law, which to first order does not move the hand.

**Gravity review mismatch names the expected value.** `bind_gravity` now prints the selected bundle's value next to the
declaration's, so activating a new bundle (RUNBOOK, MUKCa approval) can copy `base_sha256`, a digest nobody computes by hand.

Checks: `integral` self-check (scaled ki, never above, kp_ref validation); `check_execution_policy.check_cartesian_completion`
(a real FR3 executor: a 40 mrad null-space swivel fails the joint rule and passes the hand rule, moving / 0.1 rad j2 / 0.1 rad
j7 fail; the controller routes Cartesian legs to the hand rule and joint legs to the joint rule; a timed-out leg names the
hand error; no hand frame refuses; the factory wires kp_ref). Mutants: the routing removed, kp_ref dropped -- each fails it.
Also run: check_sim_supervisor, check_cartesian_sim, check_pose_hold, check_sdk_supervisor, check_dual_supervisor,
check_bimanual_sync, check_sim_franka_motion, check_scene_core, check_scene_cohost, contracts.
