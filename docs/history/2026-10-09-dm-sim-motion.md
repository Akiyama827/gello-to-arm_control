# CTRL-053: the DM arm's sim_motion on its supervisor

Operator 2026-10-09: graphs 1-8 approved, among them the DM arm's `sim_motion`. Its execution policy is decided: the FR3's
tolerances and timeouts, 0.5 rad/s and 1.0 rad/s^2, and URDF position limits. Torque is 8/8/8/3/2/1.5 N.m, the mode
profile's controller table, corrected the same day from "9/3", which I had misread as the policy's source. The DM gripper is
served over the hand_bridge wire (A1/A2, merged), with no force grasp yet and speed_max 0.10 m/s.

* `dataflows/sim_motion.yml`: `arm_console` + `supervised_arm` + `franka_gripper` (the hand_bridge client) + `visualizer`,
  the shape of `sim_franka_motion.yml`. The in-process `arm_controller` and the `mujoco_interface` plant node are gone from
  it.
* The DM gripper's fingers close as their joints rise; the Franka Hand's open. Every width <-> finger conversion assumed
  the Franka's `finger = width / 2`, in five places:
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
  - `ExecutionPolicy.envelope(where)` widens the envelope per joint to the poses the arm has been at;
    `command_valid(..., where)` and `plan_error` use it.
  - `ArmController` keeps the outermost measured pose since ARM (`_been`, reset at ARM, so a sag while disarmed grants
    nothing). Its `envelope_band()` is that band, together with the latest sample and the hold anchor.
  - Each single-pose rule failed live in turn: the latest sample alone raced the arm creeping back in under the hold, by
    ~0.1 mrad per tick (a hold anchor or jog step built from an earlier sample), and the anchor alone missed the jog.
  - The supervisor's setpoint/1 escape rule now uses the same band; it had compared against the latest sample only.
  Inside the envelope nothing changes. A joint already outside may be held where it is, or anywhere it has been since
  ARM, and plans and jogs may only bring it back in.

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
  jog built from an older sample, nothing further out, re-ARM granting nothing. Mutants killed: no history (latest pose
  only), no reset at ARM, the policy ignoring measured poses. `check_arm_controller`, `check_controller_mode_feedback`,
  `check_dora_arm`, `check_jog_ramp`, `check_sdk_supervisor`, `check_dual_supervisor`, `check_sim_franka_motion`,
  `check_supervisor_ledger`.

Hardware relevance: every supervised arm, the FR3 included, now ARMs and holds a joint resting past its URDF limit
instead of faulting. Verify on the bench: the DM arm ARMs where it rests, and a plan out of the stop is admitted.

Not run: a live console session through the Dora graph (the check drives the supervisor through the SDK, as supervised_arm
does). `sim assembler` (the view graph) shares the config; its nodes ignore the new keys (no check covers that graph).
