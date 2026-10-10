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
