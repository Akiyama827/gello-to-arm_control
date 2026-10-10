# Bench follow-ups: RT stdout, Hand move failures, console planning stall, ARM gap, pair skew instrument

Open items from the 2026-10-10 bench session (operator: "do these").

**RT server stdout is line-buffered** (`rt/src/main.cpp`, as `hand_bridge` already was). Under systemd stdout is a pipe,
fully buffered, so the startup/instance and connect/disconnect lines appeared only at exit. No stdout write is on the RT loop
(its gap/fault journal is a separate, flushed stream). Checked: a fake server's first line reaches a pipe reader while it
runs. Takes effect at the next RT redeploy (operator).

**A failed slider MOVE or HOME is printed** (`end_effectors/franka_adapter.py`). Requested moves and grasps already answered
their requester; an unrequested one failed silently on the PC (FR3 #2's Hand: the box logged `Command failed!`). Now
`[franka_gripper] MOVE ... failed: <reason>`. Checked in the adapter demo.

**The console no longer pickles the collision world per plan** (`ui/arm_console.py`, `planning/mujoco_collision.py`). Cause,
measured on the bench workcell: the world's MjModel pickles to 368 MB; each plan sent it to the planning process, holding the
console's GIL ~460 ms (another thread's worst gap) and taking 1.67 s end to end -- the bench's "planned in 1.91 s" and 47
dropped `motor_state` inputs. Now the world crosses once (sent at startup, before ARM; again only when the scene replaces
the world object or its model, or the worker restarted -- `PlanWorldMiss`), and a plan sends `planning_delta()`: the
world's pickled state minus the model, plus the model's contact masks, the only model fields the world toggles. Measured on
the bench world: 0.05 s per plan, worst other-thread gap 2 ms. Checked in the console self-check: the delta carries the
contact masks (collision answers change and match), held finger positions and qpos; a delta for another model and a key the
process does not hold are refused.

**The ~4 ms gap at each ARM is the session start, inherent; no change.** The bench gap lines at ARM read `armed=1
prev_write=2.9 ms robot_dt=4 cmd_success=1.000`: the first write runs `startTorqueControl()` (a command round trip with the
control box) and the read it requires, so that one tick spans ~3 robot cycles; the session's own success rate stays 1.0.
The DISARM gap (`read` ~2 ms, `prev_write` 0) is the session teardown before the idle stream resumes. Recorded in the
RUNBOOK as the expected signature so a real gap still stands out.

**Pair start skew instrument** (`tools/hw/pair_skew.py`, operator-run, read-only). Polls both supervisors' trace/1 and
state_history/1 while the operator presses Execute pair; per arm, fits one delay d of the measured joints (every 250 Hz
packet, on the RT box clock both servers share) against that arm's own reference from its anchor; physical skew =
(anchor + d) difference. Saves the raw record (`--out`, re-run with `--analyze`). Checked: `--self-check` recovers a 4.1 ms
skew either sign, common and separate anchors, from 250 Hz noisy packets; a smoke run against two fake-RT supervisors
through the pair coordinator recorded 0 trace drops and gave anchor difference + d (fake-plant lag ~39 ms, equal on both).

**Hand Home button in the manual console** (operator request, same day; transport changed in review round 2, below).
`ControlPanel.request_home` (route `/hand_home`) asks for what touching the `arm_gripper_home` trigger file already did:
the Hand node sends `HOME` to the bridge (libfranka `homing()`, a full open/close sweep). Gated as the slider is (confirmed ARM, no
fault) plus: fresh feedback, idle, `is_grasped` known false, no carried scene object; one request per click. The node only
honours the file in a graph without `hand_move_requires_arm` (the manual consoles); in a gated graph it is ignored, as the
touch file always was. A failed home prints `[franka_gripper] HOME failed: ...` (above). Checked in
`check_ui_scene_injection` (each refusal, one pending request, consumed once) and the console route check.

Review round 1 (Opus xhigh, PASS-WITH-FIXES, no MUST; covered 11c53d8..a80d5b1, not the Home button). Verified
empirically: on the real bench workcell, after every mutation the console, ManualCarry and WorkcellScene make (contact
phases, scene obstacles with exclude/off, fingers, base state) the delta-applied copy, the parent and a fresh whole pickle
give identical contacts on 400 q; a spawn-pool run (wall on/off/on, a killed worker -> `PlanWorldMiss` -> whole -> delta);
only `geom_contype`/`conaffinity` are written on a collision world outside checks; the hold after a hand-judged leg goes
through the relatch bound. Taken:
* SHOULD: a future in-place geometry write would leave the planner on stale geometry silently -> the delta carries a
  digest of the placement/shape fields (geom type/data/body/pos/quat/size/margin/gap, body parent/pos/quat, jnt_range,
  qpos0); a copy whose digest differs refuses the delta and the console resends the world whole (`PlanWorldMiss`).
  Bench world: still 0.04 s per plan, 2 ms worst gap. The apply also replaces the copy's attributes exactly (NIT).
* SHOULD: settling progress and the timeout reason of a hand-judged leg reported the joint rule -> their joint fields are
  None and progress carries `hand_error_m/_rad`, `hand_tolerance_m/_rad`. (Displays added for them were removed in round
  2: no graph forwards `leg_progress` past the supervisor, so they could not run. What reaches the operator is the timeout
  reason, in the FAULTED receipt, and the supervisor's `cartesian leg ... settled: hand X mm` line.)
* SHOULD: pair_skew's number includes tracking-lag differences (two arms started together with 10/25 ms lags read 15 ms
  with a clean fit) -> renamed `motion_delay_difference_ms` and documented as such: start skew only for like legs.
  NITs: one shared clock offset when both arms report one RT clock; `delay_at_search_edge` flags a delay at the search
  bound; `same_pair` checks both arms' first legs belong to one pair; the JSON output is checked (a numpy bool broke it).
* NIT: a configured ee frame missing from the URDF now fails at executor construction.
Left: `id()` cache keys (the held references keep them unique); K_c = 0 legs take the hand bound (documented).

Review round 2 (Opus xhigh, PASS-WITH-FIXES, no MUST; verified on the bench workcell world: parent/child digests match
across round trips and 9 mutations, so consecutive plans stay deltas: whole 2.19 s once, then 0.04-0.14 s). Taken:
* SHOULD: the Home trigger FILE is shared by every graph without an ARM_ID (`calibrate`, the sim graphs), so a sim page's
  Home could home the real Hand, and the node's file path never checked `is_grasped` -> Home is now a graph topic
  (`arm_console/hand_home` -> the Hand node that serves the console's `grasp_request`, wired in the library's
  `real_franka_motion`/`sim_franka_motion` and the project's `real_franka_workcell_motion`/`calibration_workcell`; the
  console self-check and the project's `check_dataflows` assert the wiring), and the node re-gates every home, topic or
  file: refused in a gated graph, unless available, measured and idle, and unless `is_grasped` is known false
  (`home_refusal`, in the adapter demo). The touch file still works where it did, under the same node gate.
* SHOULD: the hand-error displays were unreachable (above) -> removed; the history claim corrected.
* NIT: a HOME the bridge admission refuses is printed. The digest adds `jnt_pos`, `jnt_axis`, `geom_rbound`. Any failure
  to apply a delta (not only a geometry mismatch) drops the cached copy and is a `PlanWorldMiss`. The scene-carry gate on
  Home is checked (a mutant without it now fails).
Not taken: exporting ARM_ID for `calibrate` (it would rename that graph's lock, recording and trigger files; Home no longer
depends on it). Noted, not this bundle: `planner.py`'s timed_out -> hold path cannot fire in supervised real-add either
(no `leg_progress` forwarding); owner decision.
