# RT Cartesian law, slice 2a: one base-frame spec from plan to plant, the twin targets FK(q_des)

Operator decisions 2026-10-09 (slice 1: `2026-10-09-rt-cartesian.md`): the plant computes the target as FK(q_des) with its own
model; Cartesian legs carry a joint kp floor so the RT's fault hold (joint law only) is not limp -- 20 N.m/rad per joint on the
FR3, a robot config key; the bench's axial K_c 2000 -> 1000 N/m. Goal yardstick (operator): `arm_control` as a ros2_control-like
backend other projects (VLA, RL) use unchanged, so the library's Cartesian contract is robot-BASE-frame and project-agnostic;
world frames stay in the application.

The spec, everywhere: `{frame: "base", task_R (3x3, task axes as columns in the robot BASE frame), kc[6], dc[6]}`, checked by
`impedance.cartesian_values` (pose-hold envelope). No pose: every plant aims the spring at FK(q_des) with its own model.
* Plan (`contracts/motion.pack_plan`): the spec is validated at pack time and unpacked with its frame; a world-frame or
  out-of-envelope spec never ships. `cartesian_poses` stays, for previews only.
* `motor_command` (`contracts/motor`): the tail is the RT v8 tail, 21 values (was 28 with a world pose); unpacked as the spec.
* Controller: the law lives with its leg (execution and settling) and nowhere else. Only the running leg's words carry its spec;
  every hold, jog and keepalive is the joint law only, as the RT box's own hold is (review round 1). The pose lookup in
  `TrajectoryReference` is deleted (the target was the knot pose of the segment, piecewise constant; FK(q_des) follows the
  interpolated reference, as the RT box does).
* Admission (`ExecutionPolicy`): a Cartesian leg is refused unless the spec passes `cartesian_values` and every joint kp >=
  `cartesian_kp_min`; a policy without `cartesian_kp_min` admits no Cartesian leg. `robots/fr3/execution.yaml` sets 20. Joint
  legs are unaffected. The supervisor always has a policy; a Dora graph with none (today's `sim_bench`) has no admission at all,
  unchanged.
* Twin (`MuJoCoBackend`): the robot base is the parent of the first jointed body on the EE's chain (FR3: fr3_link0 =
  libfranka's O frame), found from the EE so a composed world's other actors cannot be taken for it; FK(q_des) runs on a scratch
  `MjData`; the task frame is rotated by the base body's orientation into MuJoCo's world. `supports_cartesian`: an EE on a
  jointed chain whose joints the plant commands, and the compiled law.
* `SimLink`: no longer refuses Cartesian commands. It applies the RT link's v8 rules (refused unless the plant runs the law; the
  spec through `cartesian_values`), advertises `CAP_CARTESIAN_TRAJECTORY` in its identity, and -- as the RT box -- holds on the
  joint law only (a hold word carries no spec). `ScenePlant` passes the spec to its slice and runs it only when the world's EE
  chain is its actor's.
* Project (`assembly/task/coordinator.py`): the task frame is converted world -> base with the arm's mount
  (`_arm_to_world`); the world axis stays in the log. `planner.py` forwards the frame; `sim_bridge.py` packs the new tail.
  `configs/simulation/franka_bench.yaml`: axial K_c 1000 N/m; D unchanged (130 N.s/m: the axial mode is now ~1.45x critical
  for the ~2 kg EE) -- an operator decision if it should follow K_c.

Not in this slice (2b): the supervisor's `cartesian_impedance` profile (Capabilities, `TrajectoryRequest`, the SDK codec,
`SdkArmHandle`/`DoraArmBridge` stop refusing), refused when the supervisor runs the native reference. Until then the supervised
path still refuses Cartesian plans, and `check_dual_supervisor` checks that refusal.

Checks (related only):
* new `tools/check/check_cartesian_sim.py` (library suite, FR3 assets; the project suite runs it on each deployed arm's own
  description): with the mount yawed 90 deg a base-x stiffness pushes along the base's x in the world at K_c times the FK(q_des)
  displacement, with no moment, and a swapped task frame along the base's y; a spec without the base frame is refused; zero at
  the target; closed loop with the floor kp and the bench K_c/D_c the EE settles 0.39 mm from FK(q_des) in 2 s; SimLink admits
  base-frame specs only and its hold drops the law. Mutants caught: task frame not rotated by the base; target = measured pose.
  Deployed descriptions: fr3_1 0.40 mm, fr3_2 0.33 mm.
* `check_execution_policy`: poses optional; bad kc shape/NaN/over-envelope, world frame and a reflection refused at admission;
  a Cartesian leg at 19.9 N.m/rad on one joint refused while the same joint leg passes; a floorless policy refuses; the sent
  spec is base-frame with no pose.
* `messages` self-check (pack refusals, the 21-value motor tail round trip), `control.reference`, `contracts.impedance`,
  `check_arm_controller`, `check_scene_core`, `check_sim_supervisor`, `check_pose_hold_sim`, `check_dual_supervisor` (its
  packed spec gains the frame), the MuJoCo backend's `--self-check`, `--demo`, `--scene-demo`; ruff.

Sim dock re-measure (slice 3's "re-measured in sim"): NOT DONE, blocked by a pre-existing failure. `tools/run.py sim bench`
(driven headless, ARM and gates pressed on the loopback panel) stops at the FIRST leg, `planning_failed:move_to_pregrasp`: "IK
found no collision-free goal branch" for the perception pregrasp (base [0.379, 0.0, 0.184]). Identical at the base revision
(parent 7237c48 / library 8173845) and at parent eed6cc6 (2026-10-06) and 258c267 (2026-10-01): not this slice. Diagnosed: all
12 IK branches self-collide, fr3_link1 against fr3_link5/6, up to 32 mm deep; raising the target 5 / 10 cm leaves 1 / 0 branches.
The sideways pregrasp at the bench's PROPOSED module placement folds the FR3 onto its own base. Owner: operator (bench scenario
geometry). Blocking: the dock leg cannot be reached in sim, so the K_c 1000 seat error is unmeasured, and step 2c (sim_bench
under the supervisor) cannot run end to end until the placement or the grasp approach changes.

Bench pick, investigated after the operator delegated the fix (2026-10-09); all changes REVERTED, findings only. Owner: the
bench task (project), decisions marked. Blocking: the sim dock re-measure and step 2c's end-to-end run.
1. Pregrasp: the CAD-registered mount (scene v3, 2026-08-06) sits 7 cm nearer the module than the 2026-07-30 placement SWEEP
   assumed (pick radius 0.479 vs 0.55). Moving module, pedestal and fallback targets 7 cm along world -y (r 0.549) gives 28/28
   collision-free pregrasp branches (0/14 before) and the leg plans and executes end to end. Config-only; the placement is
   PROPOSED, so it is a bench decision.
2. Grasp: FR3 #1's description carries the wrist camera (fr3_gemini305 + holder, added with the model bundles after the sweep).
   In the sideways grasp it hangs below the hand and hits the 8 x 8 cm pedestal (0/32 grasp branches free). Raising pedestal and
   module together does not help (the camera keeps its offset); a pedestal reaching only 1 cm toward the arm from the module's
   centre clears it by 3 mm, which is not a fixture. This applies to the real arm too: the side grasp needs a fixture designed
   around the camera, or another grasp. Decision: fixture design / grasp.
3. A 180 deg grasp roll (camera on top; parallel jaws are symmetric) plans the pregrasp, but the twin's wrist then never settles
   (joint 7 0.26 rad off and moving at 0.2 rad/s after the 5 s settle; fingers at 3-14 mm, 70-86 N while idle; the plant falls
   behind wall clock). Unverified cause; consistent with the open item "no armature on the composed bench arm" (the
   single-model FR3 showed the same joint-7 instability before `sim_armature`). The roll also changes the in-hand pose the dock
   ladder was swept for. Decision: armature on the composed arm (own item), then whether to roll.

Review round 1: BLOCK. Taken:
* (BLOCK) The law outlived its leg: `_cartesian_now`/`_cartesian_live` survived DISARM->ARM, and a gains preset after a hold or a
  re-arm streamed Cartesian words at joint kp 0-5, under the floor the leg was admitted with (the base revision had the same
  re-arm gap; the record's "disarm drops it" was false). Fixed at the root: `_cartesian_live` is deleted and only the running
  leg's words carry the spec, so no event outside the leg can leave a spring on; disarm also clears the loaded spec. Same fix
  closes the policy-less divergence (a Dora graph with no policy kept the law after completion and cancel). Checked with and
  without a policy: after completion, after DISARM->ARM and after a gains preset, every later word is the joint law; mutant
  "holds carry the law" caught.
* A plan that names itself but does not decode (e.g. kc 2000 on the wire) crashed `on_plan`; it is now refused with the reason
  (`leg_result` failed, "malformed plan: ..."), and the next good plan loads. Mutant "no guard" caught.
* `check_cartesian_sim`'s 90 deg mount could not see a transposed base rotation (x maps to the same line; k a a^T is
  sign-blind): the mount is now 30 deg yaw + 10 deg tilt and the transposed rotation's axis is asserted different; mutant caught.
  The project's `check_bench_twin` asserts on the real composed bench that the twin's base rotation times the coordinator's
  base-frame task frame equals the coordinator's world task frame (matrices, so the -90 deg mount cannot hide a transposition;
  mutant caught), and that ScenePlant runs the law only for the actor owning the EE chain and passes its spec to the plant
  (mutants "slice drops the spec" and "any actor" caught). Its planner preview is off (a check never streams to a viewer).
* A phase's Cartesian block outside the envelope now fails when the coordinator is built, not when the dock leg is packed.
* `MuJoCoBackend.apply_motor_command` refuses a Cartesian spec on a plant that cannot run the law (it ran the joint word alone,
  silently); `check_sim_supervisor` checks it on a plant with no EE. `_stream` passes `False`, not `None`.
Not taken, recorded: the DM backend ignores a Cartesian spec (no DM config admits Cartesian legs: its policy has no
`cartesian_kp_min`; changing a hardware backend's command path is its own stop); the Dora multi-slice `SceneCore.set_slice` is
last-writer-wins for the EE-level spec with no actor gate (`sim_bench` has one writer; the supervisor path is gated by
ScenePlant).
