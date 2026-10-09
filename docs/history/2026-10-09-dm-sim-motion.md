# CTRL-053: the DM arm's sim_motion on its supervisor

Operator 2026-10-09: graphs 1-8 approved, among them the DM arm's `sim_motion`. Its execution policy is decided: the FR3's
tolerances and timeouts, 0.5 rad/s and 1.0 rad/s^2, and URDF position limits. Torque is 8/8/8/3/2/1.5 N.m, the mode
profile's controller table, corrected the same day from "9/3", which I had misread as the policy's source. The DM gripper is
served over the hand_bridge wire (A1/A2, merged), with no force grasp yet and speed_max 0.10 m/s.

* `dataflows/sim_motion.yml`: `arm_console` + `supervised_arm` + `franka_gripper` (the hand_bridge client) + `visualizer`,
  the shape of `sim_franka_motion.yml`. The in-process `arm_controller` and the `mujoco_interface` plant node are gone from
  it.
* The DM gripper's fingers close as their joints rise; the Franka Hand's open. The single-arm width <-> finger conversions
  assumed the Franka's `finger = width / 2`, in five places:
  - the client node's slider input;
  - the console, three places;
  - the visualizer.
  The sim Hand summed the fingers. `joint_motor_map.finger_from_width` / `width_from_finger` (and `gripper_mimic`) read
  the direction from the arm's `joint_mimics` (`lower` = the finger at the open motor end). Without a mimic the result is
  the Franka Hand's, unchanged. All of those places use them now, and so does `open_jaws`, which was dormant for the DM
  (no open width configured).
* `joint_mimics` was overloaded: it meant both "how the finger joints run" and "the gripper is a motor slot of the arm's
  command". In the supervisor the second meaning would pack a 7-wide command for a 6-joint arm. Now
  `control.factory.gripper_command_cfg` packs a slot only when the mimic's `source` is one of the config's `motor_names`:
  - yes for the PC-side DM driver's config (`Gripper` is its 7th motor);
  - no for the sim, and none for the DM arm behind arm_rt_server, where motor 7 is hand_bridge's.
  `plants/dm/adapter.py` reuses `gripper_mimic` (its private copy is deleted).
* `plants/sim_link.py`: the sim Hand gets `sim_hand.limits` (absent: the Franka Hand) and the arm's mimic, and refuses
  `limits.width_max_m` that is not the mimic's stroke.
* `end_effectors/sim_hand_bridge.py`: width and finger targets go through the mimic.

* Never further out (operator 2026-10-09; it was the open item "holds anchored at the measured pose can still fault on a
  limit"). The DM arm's first live supervised run could not ARM. Disarmed, the twin rests on joint 2's lower stop, -0.098
  rad against the URDF's -0.087 (MuJoCo's limits are soft; the real arm rests on its stops too). The hold at the measured
  pose was then refused as outside the envelope, so the controller stopped. The old in-process DM path never hit this
  because it ran without an execution policy.
  - `ExecutionPolicy.envelope(where)` widens the envelope per joint to a pose; `command_valid(..., where)` and
    `plan_error(..., been=where)` use it (`plan_error`'s start check keeps the measured pose).
  - `ArmController` widens it PER JOINT. `_arm_pose` is where the arm rested as it ARMed; it follows the measured pose up
    to the episode's first command, because the plant is limp until its first hold and that hold is built from a sample
    taken then.
    - A joint inside the envelope there never widens: an overshoot or sag while armed grants nothing (no ratchet).
    - A joint outside it (resting on its stop) widens to its outermost measured pose since ARM (`_been`, reset at ARM).
  - `envelope_band()` is that band; its servo commands, its plan admission and the supervisor's setpoint/1 use it.
  - What failed live, in turn:
    - The latest sample alone: the arm creeps in under the hold by ~0.1 mrad per tick, so a hold anchor or a jog step
      built from an earlier sample sat "further out".
    - A fixed ARM pose: at the DM's kp 20 the arm keeps settling after ARM, joint 4 drifts past its relatch tolerance,
      and the hold re-anchors EVERY joint at the measured pose, including joint 3, which rests on its upper stop and had
      crept 0.6 mrad further out.
  - Inside the envelope nothing changes. A joint that rests outside is held where it is on its stop (wherever it has
    settled since ARM), and plans and jogs may only bring it back in. A joint that was inside never goes out.

Review round 1 (opus): BLOCK, fixed:
* M1: the finger-direction evidence was blind. Mid-stroke is the one width an inverted map gets right, and the spawn
  expectation went through the function under test (mutants of both helpers passed). The check now MOVEs to quarter
  stroke and asserts the spawn width literally: the stated stroke for a mimic'd gripper, else the configured open width.
* S1: the controller's plan admission used the latest sample only (an SDK plan from a measured start was refused as
  the arm crept in). It now uses the band.
* S2: the round-1 band was "the outermost pose since ARM" for every joint. That is broader than the operator's rule: a
  joint inside at ARM that overshoots its limit could then be jogged there. Now the band follows only joints that rest
  outside at ARM. A fixed ARM pose for every joint was tried first and failed live (see above).
* S3: the dual console's Hand path (`multi_arm.hands`, `dual_arm_console`) and the project's `ManualCarry` work in
  Franka units; both now REFUSE an arm whose gripper has `joint_mimics`, rather than invert silently.
* S4 (recorded, track list): a future DM-on-RT entry that includes hardware.yaml must override `motor_names` (drop
  `Gripper`), or the controller packs a 7th slot for a motor hand_bridge owns.
* NITs: the hold-anchor term is gone (the ARM pose covers it); the sim Hand no longer divides by zero fingers; the
  supervisor's setpoint tolerance is `ENVELOPE_TOL`; the sim profile's policy comment.

Project (pinned with this): `configs/robots/assembler/execution.yaml` (the policy), `configs/robots/assembler/gripper.yaml`
(`joint_mimics` moved out of hardware.yaml so the sim shares it), `configs/simulation/assembler.yaml` (the supervised sim
profile, `rt.instance_id: sim_assembler`, `sim_hand.limits`), and `tools/run.py`: `sim motion` is supervised for both arms.

Checks:
* project `check_supervised_sims` (the DM arm included now). The DM arm MOVES: ARM where gravity rested it, a jog (setpoint/1)
  carries joint 1 to its target, a planned move COMPLETES on observed feedback, DISARM. The supervisor starts as run.py starts it and offers the six
  joints and setpoint/1, but no `cartesian_impedance` (the policy allows no Cartesian legs). The gains match
  supervised_arm's. The sim Hand states the DM envelope; the spawned width is the open 0.0878 m (killed mutant: SimLink
  without the mimic reads 0.0). A MOVE to mid-stroke lands there, and GRASP is REJECTed `uncalibrated_force`. The FR3
  sim and the bench pass the same Hand checks.
* `check_dataflows`, `check_sim_hand`, `check_console_grasp`, `check_console_controls`, `check_node_shutdown`,
  `check_hand_grasp`.
* `check_sim_supervisor`, rerun under normal load (0.4): PASS. Its earlier observe-rate FAIL (0.84-0.88x) was the
  external load, as attributed.
* Project `check_dataflows`, `check_franka_ownership`, `check_config_duplicates`. The DM policy restates the FR3's six
  tolerances and timeouts, allowed with its reason: each arm's own safety facts, equal by decision.
* `check_execution_policy` (`check_measured_outside` and the policy cases): ARM past a limit, the hold creeping back in, a
  jog built from an older sample, a plan out of the stop admitted and one further out refused, nothing further out, an
  overshoot while armed widening nothing, a resting joint settling further out on its stop held, re-ARM (outside again,
  less far) granting nothing from before. Mutants killed: plan admission on the latest sample, widening every joint
  (the ratchet), widening none, no reset at ARM, the policy ignoring measured poses. The M1 mutants
  (`finger_from_width` / `width_from_finger` ignoring the mimic) now fail the DM scenario. `check_arm_controller`, `check_controller_mode_feedback`,
  `check_dora_arm`, `check_jog_ramp`, `check_sdk_supervisor`, `check_dual_supervisor`, `check_sim_franka_motion`,
  `check_supervisor_ledger`.

Hardware relevance: every supervised arm, the FR3 included, now ARMs and holds a joint resting past its URDF limit
instead of faulting (the RT box has no position envelope of its own: PC admission is the only one). Verify on the bench: the DM arm ARMs where it rests, and a plan out of the stop is admitted.

Not run: a live console session through the Dora graph (the check drives the supervisor through the SDK, as supervised_arm
does). `sim assembler` (the view graph) shares the config; its nodes ignore the new keys (no check covers that graph).
