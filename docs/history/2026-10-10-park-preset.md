# Park preset: named joint targets in the manual console

Operator request at the bench (2026-10-10, between steps 11 and 12): before FR3 #2 moves, FR3 #1 must stand where FR3 #2's
scene declares it parked (`tools/hw/peer_parked.py`, 0.05 rad), and parking it meant typing seven joint values. Decided: a
per-arm **Park** preset, sequential, through the normal gate (no "park both" button: that is concurrent two-arm motion, which
is unqualified, and no startup motion: a graph comes up DISARMED).

Library (`ui/arm_console.py`, `ui/static/console.{js,html}`): `ControlPanel(joint_presets=...)` and `main(joint_presets=...)`
take named joint targets `{name, joints, q}`. Validated at construction: the joint list must equal the console's planned joints
(a reordered list is refused, not permuted), values finite, inside the limits (refused, not clipped: a clipped park is not the
declared posture), names unique. `/state` serves the NAMES only; the page shows one "Target: <name>" button each and posts
`/joint_preset {name}`; the server sets the target (`apply_joint_preset`: new target revision, previous plan and Execute
invalidated, pending pose/cart edits dropped). The robot does not move: Plan + preview, then Execute, gate it as for any target.
The existing pose presets (Cartesian, fill Desired) are unchanged; a Cartesian park would go through IK and could land on
another branch, which the 0.05 rad peer check rejects.

The project supplies the preset from its scene (`scene.park_q`, see the project change in the same integration).

Check: `check_ui_scene_injection` serves names only, applies a preset (target set, revision advanced, status), refuses an
unknown/None/non-string name without touching the target, refuses foreign joints, out-of-limit, NaN, wrong-length,
duplicate and incomplete entries, the page builds the row and posts `/joint_preset`, and `main` passes the list through.

Review round 1 (Opus xhigh, PASS-WITH-FIXES, no MUST; verified the gate empirically: a queued Execute is consumed, an old
revision is refused by can_execute/execute_if_current/set_plan, a late IK or cart write cannot land after the preset, HTTP
errors are 400s):
* SHOULD: the project's "peer without q" refusal passed for the wrong reason (no path either) -> it asserts the message names
  `q`; the reviewer's mutant (old zero default restored) now fails the check.
* SHOULD: RUNBOOK parking order -> parking is a motion planned around the OTHER arm's declared park: confirm that arm first
  (`peer_parked.py --scene` of the arm about to move), one arm at a time, and the bootstrap rule when neither is confirmed.
* SHOULD: no offline check saw the preset as the hardware console does -> project `check_dual_bench.check_park_presets`
  stages both FR3 entries for hardware, runs the console's own validation and asserts each park collision-free with the
  peer parked.
* NITs taken: a joint list must be a list (a string or None is refused with a ValueError, the message no longer raises);
  `.button-pair[hidden]` now hides (both preset rows showed when empty); the page keeps a Desired draft typed while a preset
  POST is in flight and ignores a preset click while a pose POST is pending; `peer_parked --self-check` derives its expected
  error from the declared park instead of a third hard-coded copy. Left: re-clicking the preset discards a preview (by
  design, documented); a preset clicked before the first measured state is overwritten by the first-state sync (as any edit).

Review round 2 (Opus xhigh, PASS-WITH-FIXES; three mutants of the new park check each fail it; the CSS change hides only the
two preset rows): SHOULD: the bootstrap eye check said "clear of the previewed path", but the preview draws the peer AT its
declared park, which the plan avoids by construction -> the RUNBOOK says to look at the other arm on the bench, not the drawn
one, and "may not be" where the model puts it. NITs taken: the peer check needs the `dual_franka` profile (named); "move it
clear in Desk" spelled out; the check imports `arm_joints` from `arm_control.config` and names a missing `park_q`. Left: the
check validates against `arm_joints(cfg)` while the console honours a mode's `planner.joints` (no mode sets it); a preset
click during a pose POST is ignored without a status line; whether one FR3 servoing while the other's server only streams
state counts as unqualified dual-FCI timing (pre-existing, operator).
