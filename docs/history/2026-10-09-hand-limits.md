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
  limits)` bounds against it and refuses a close without force grasp. These replace `MIN_FORCE_N`/`MAX_FORCE_N`/
  `MAX_WIDTH_M`/`MAX_SPEED_MPS`, which had no importers outside the library.
* `end_effectors/franka_adapter.py`: `BridgeClient.limits` (reset to the Franka Hand on each connect, set from LIMITS);
  STATE width, `send_line`, the panel's MOVE speed and the grasp FSM / position move use it. The snapshot carries `limits`,
  and `force_grasp` is false without one.
* `end_effectors/hand_move.py`, `ui/hand_panel.py`: bound by the connected bridge's limits.
* `end_effectors/sim_hand_bridge.py`: `SimHandBridge(port, limits=FRANKA_HAND)` admits inside its limits, clamps and
  rejects widths by its stroke, HOME opens to its stroke, and it sends LIMITS. `SimLink` still builds the Franka one (a DM
  sim Hand is the DM graphs' slice).
* `rt/src/hand_admission.hpp`: `HandLimits`, `franka_hand`, `parse_hand_command(line, a, limits = franka_hand)`.
  `rt/src/hand_bridge.cpp` (still Franka-only; the device seam is A2's) sends LIMITS from `franka_hand`, and its width
  envelope reads it. `rt/README.md`: the line.

No control law, gain, limit or timing constant changed. The Franka Hand's numbers are the ones already in force.

Checks: `check_sim_hand` (non-Franka parse bounds, malformed LIMITS refused, LIMITS leads the first report, client
validation before and after LIMITS, no-force-grasp refusal before the wire), `hand_admission_selfcheck` (wide envelope,
no-force GRASP), `hand_bridge_selfcheck` (built with libfranka 0.21.2 offline: LIMITS precedes STATE), `check_hand_grasp`,
`check_node_shutdown`. Nothing ran on hardware; the deployed hand_bridge on the RT box is unchanged until the bench session.
