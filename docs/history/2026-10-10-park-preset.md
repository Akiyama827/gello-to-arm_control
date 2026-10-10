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
