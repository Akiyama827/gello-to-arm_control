# hand_bridge states its gripper's envelope (LIMITS line)

Operator 2026-10-09, decision A1 of the remaining CTRL-053 migrations (project
`docs/status/2026-10-09-remaining-migrations-design.md`): the DM arm's gripper is served over the existing `hand_bridge`
wire, so the wire stops assuming the Franka Hand's envelope. Additive: no existing line changes.

* Wire: every report is now `LIMITS <width_max_m> <speed_max_mps> <force_min_n> <force_max_n>`, then STATE / CAPS / META,
  so a client sees the envelope before the first STATE it validates against it. A bridge that sends no LIMITS line is the
  Franka Hand (0.08 m, 0.10 m/s, 30-70 N), so a client keeps working against an old bridge. `force_max_n` 0 means the
  gripper has no calibrated force grasp: its GRASP is refused (A2: the DM gripper until its rung-11 N->torque map).
  * Deviation from the design note, which put the numbers on `CAPS`: `BridgeClient` matches `CAPS active_stop`
    EXACTLY, so an extended CAPS line would silently lose active-stop on every deployed client; an unknown line is
    ignored. A separate line is the compatible form.
* `end_effectors/franka_hand.py`: `HandLimits` + `FRANKA_HAND` + `parse_limits`. `resolve_grasp_parameters(config, payload,
  limits)` bounds against it (no force grasp: force is not validated and is 0); `require_force_grasp(limits, mode)` refuses
  a close, and the GRASP producers call it (`HandGraspFsm.on_request`, `submit_grasp_request`, `send_line`'s GRASP, the
  panel's Close). These replace `MIN_FORCE_N`/`MAX_FORCE_N`/
  `MAX_WIDTH_M`/`MAX_SPEED_MPS`, which had no importers outside the library.
* `end_effectors/franka_adapter.py`: `BridgeClient.limits` (reset to the Franka Hand on each connect, set from LIMITS);
  STATE width, `send_line`, the panel's MOVE speed and the grasp FSM / position move use it. The snapshot carries `limits`;
  its `force_grasp` stays the epoch-protocol flag (the gripper's own is `limits.force_grasp`).
* `end_effectors/hand_move.py`, `ui/hand_panel.py`: bound by the connected bridge's limits.
* `end_effectors/sim_hand_bridge.py`: `SimHandBridge(port, limits=FRANKA_HAND)` admits inside its limits, clamps and
  rejects widths by its stroke, HOME opens to its stroke, and it sends LIMITS. `SimLink` still builds the Franka one (a DM
  sim Hand is the DM graphs' slice).
* `rt/src/hand_admission.hpp`: `HandLimits`, `franka_hand`, `parse_hand_command(line, a, limits = franka_hand)`.
  `rt/src/hand_bridge.cpp` (still Franka-only; the device seam is A2's) sends LIMITS from `franka_hand`, and its width
  envelope reads it. `rt/README.md`: the line.

No control law, gain, limit or timing constant changed. The Franka Hand's numbers are the ones already in force.

Review round 1 (opus): BLOCK, one MUST, fixed:
* A gripper without force grasp could not MOVE, HOME or release: the close refusal sat in `resolve_grasp_parameters`
  (default mode close, which the position-move validators hit) and the snapshot's `force_grasp` (which gates every line)
  had been ANDed with the gripper's. Moved to the GRASP producers; `force_grasp` restored. Checked: on a no-force envelope
  MOVE, HOME, `HandMove` and a release go out, a close does not.
* SHOULDs taken: the checks now pin LIMITS before STATE in a report that carries STATE, and a reconnect to a bridge that
  states nothing returning to the Franka Hand. Mutants killed: LIMITS after STATE, no reset on reconnect, the M1 AND.
* NITs taken: the C++ header lists LIMITS in send order. LIMITS is FROZEN at four fields (clients refuse any other count;
  a new field goes on a new line), stated in the header and README. A release on a no-force gripper commands force 0
  rather than passing the payload's value through.

Named, not changed:
* `HandGraspFsm.__init__` and `HandPanel.__init__` resolve the configured defaults against the Franka Hand before any
  LIMITS arrives. A DM config with `open_width_m`/`grasp_width_m` above 0.08 would raise at startup. That belongs to the DM
  graphs' slice.
* Deployment order: an old client facing a DM bridge that reports a width above 0.08 m raises on STATE and reconnects in a
  loop. Clients upgrade before a DM bridge is deployed (record it in A2).
* Wire precision (`%.5f`/`%.3f`) can round a limit up. A client would then admit what the bridge refuses
  (`invalid_command`). That fails closed.

Checks: `check_sim_hand` (non-Franka parse bounds, malformed LIMITS refused, LIMITS leads the first report, client
validation before and after LIMITS, no-force-grasp refusal before the wire), `hand_admission_selfcheck` (wide envelope,
no-force GRASP), `hand_bridge_selfcheck` (built with libfranka 0.21.2 offline: LIMITS precedes STATE), `check_hand_grasp`,
`check_node_shutdown`, `check_console_grasp`; project `check_manual_carry`, `check_calibration_contract`. Nothing ran on hardware; the deployed hand_bridge on the RT box is unchanged until the bench session.
