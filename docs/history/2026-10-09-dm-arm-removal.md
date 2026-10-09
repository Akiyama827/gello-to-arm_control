# The DM arm leaves the stack: arm_control targets the Franka FR3

Operator 2026-10-09: "For arm_control, we target only the Franka robot, the standard one. For all code related to DM arm,
locate, cleanup, and zip them to a zip file for later use ... we remove it completely from our current stack." The base
module (two DM motors behind `arm_rt_server --backend dm`) stays: its RT backend and wire codec are kept.

The removed code is archived outside the repository (`~/Documents/work/dmmotors/arm_control_dm_arm_2026-10-09.zip`: every
deleted file at its last commit, the edit patches, a README). Git history keeps it too: library `c9ed6e4`, project
`a616e8ee`.

Deleted:
* The PC-side DM driver:
  - `plants/dm/adapter.py`, `plants/dm/socketcan.py`, and the `DmHardwareBackend` / SDK transport half of
    `plants/dm/backend.py`;
  - `nodes/hardware_interface.py` and `control/safety.py` (its command-mode safety layer).
* The bridge-side grasp gate: `end_effectors/grasp.py` (`GraspGate`) and `end_effectors/torque_gripper.py`.
* The DM graphs: `real_motion`, `sim_motion`, `real_arm_listen`, `real_gripper_log`, `view_real`.
* The DM mode profiles: `examples/profiles/motion.yaml`, `command.yaml`.
* The DM gripper behind hand_bridge (A2): `hand_dm.hpp`, `hand_dm_selfcheck.cpp`, `hand_device.hpp`, `--backend dm`.
  - `hand_bridge.cpp`, its self-check, `rt/CMakeLists.txt` and the A2 parts of `rt/README.md` are back at their pre-A2
    form (`e00fc6e`), unchanged since: the Franka Hand only, built `WITH_FRANKA`.
  - The sim Hand no longer REJECTs `uncalibrated_force`.
* The gripper as a motor slot of the arm's command:
  - `joint_mimics` and its readers: `gripper_mimic`, the mimic finger maps, `pack_arm_gripper_command` /
    `unpack_motor_state_to_joint`;
  - the `ArmController`'s `gripper=` argument, gripper slot and `gripper` input;
  - `gripper_command_cfg`;
  - `LocalArmHandle`'s gripper slots. `command_gripper` now refuses, as the supervised handle does. It was a silent no-op
    for every non-DM arm.
  - The width <-> finger helpers stay as the Franka Hand's (finger = width / 2).
  - The dual config's and the project's manual-carry refusals of `joint_mimics` are gone with it.

Kept:
* `rt/src/backend_dm.cpp` and `dm_mit.hpp`, the base's servo, including the shared-bus interlock.
* `plants/dm/backend.py`: now only the MIT codec, the per-type limits and `dm_spec_from_config`, for the base's unit and
  the parity check `tools/check/rt/torque_cap.py`.
* `control/gains.py` and `tools/hw/dm_read_params.py` (the base's motors are DM).
* A1's `LIMITS` line and `HandLimits` (the FR3 Hand states its envelope). The no-force-grasp branch of A1 stays as
  generic envelope handling; no configured gripper uses it now.
* "Never further out" in the execution policy: every supervised arm, FR3 included.

Unchanged by decision: the default controller ceilings 500/5 in `node_utils` (the DM wire limits by origin). The FR3
states its own ceilings, and changing a default gain limit is the operator's call.

Checks:
* `check_arm_controller` (its gripper-slot case deleted with the slot), `check_controller_mode_feedback`,
  `check_execution_policy`, `check_jog_ramp`, `check_node_shutdown`, `check_bimanual_contract`.
* `check_sim_hand`, `check_console_grasp`, `check_console_controls`, `check_hand_grasp`, `check_dataflows`,
  `check_dora_arm`, `check_sdk_supervisor`, `check_dual_console`, `check_dual_modes`, `check_cooperative`,
  `check_sim_franka_motion`.
* `dm_backend_selfcheck`, `dm_command_selfcheck`, `tools/check/rt/torque_cap.py`, `plants/dm/backend.py`,
  `plants/base.py`.
* Rebuilt `rt/build-offline` (the stale `hand_bridge` / `hand_dm_selfcheck` binaries removed).
* `ruff`.

Review round 1 (opus): ACCEPT, 3 SHOULDs taken:
* Record SHAs filled at merge.
* `hand_bridge_selfcheck` builds only `WITH_FRANKA` again (pre-A2), so the offline suite, configured `WITH_FRANKA=OFF`, no
  longer runs it. A scratch build `WITH_FRANKA=ON` against the local libfranka passed (hand_bridge built,
  `hand_bridge_selfcheck` PASS). Open: put a `WITH_FRANKA` build into the offline suite. Until then the track list's
  FR3-Hand rebuild row covers it.
* The project launcher refuses a bare `run.py real`: a hardware graph is always named.
NITs taken: stale comments (`node_utils`, `dm_read_params`, project docs and configs); `sim modular` no longer needs the
FR3 description staged.

FAIL, not this slice: the library suite's `console_graphs` (`arm_console.py --self-check`: "no arm_controller found in
any graph") fails identically at `c9ed6e4`. The earlier CTRL-053 slices moved the library graphs onto supervisors; that
self-check still looks for the retired in-process controller.
* Owner: the CTRL-053 graph-retirement follow-up (point it at `supervised_arm`, or retire it).
* Blocking: it blocks no merge, but the library suite stays red until then.
