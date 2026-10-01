# Application clients (CTRL-052)

`arm_control/clients/` holds four thin clients over `ArmClient` (stdlib + numpy; the backend never imports it, checked by
`tools/bench/rt/clients.py`). Run them against fake endpoints: `cd tools/bench/rt && PYTHONPATH=../../..:. python clients.py <build>/reference_runtime_selfcheck` (or `tools/bench/rt_offline_suite.py`).

| Client | Contract |
|---|---|
| `planner.run_reviewed` | submit a reviewed `TrajectoryRequest`, release it only if it starts on release, wait; a timeout leaves it running |
| `chunks.ActionChunkClient` | a policy's chunks, spliced from the supervisor's anchor |
| `teleop.DeadmanTeleop` | setpoints while the deadman is held and fresh. Each step's HORIZON is at most the deadman timeout (a longer move is shortened toward the target and noted); that bounds what one step commits, not a stalled calling loop. A step is spliced at the selected reference's next eligible whole-segment cut (can be well ahead of now); if the loop stalls the arm continues to that cut plus one step, then the stream faults (REFERENCE_UNDERFLOW): the RT holds the measured pose with no PC feedforward, the next `command()` is refused and then returns `expired`, and motion needs a new admission. Travel is measured from the measured pose and the speed is an average (admission enforces the limits). With a live loop, release or a stale heartbeat CANCELS the stream; an unconfirmed cancel is retried and reported (`stop_failed`); `busy` while the previous revision is unresolved, `expired` when the supervisor ended the stream; never re-arms |
| `rl.StepWindow` | `step(q_target, window_s)` reports actual elapsed time and `completed` (the RT selected the revision) `/intervention/refused/not_selected/expired/faulted/unknown` (`unknown`: an infrastructure error after the revision may have been sent, motion may have happened, the id is in `report.unknown`); an intervention returns early but does NOT stop the arm; `terminated` is always False (task success is project-owned); `reset()` raises `NeedsOperator` unless the robot is armed and unfaulted and never re-arms, homes, opens the Hand, discards calibration or restarts a stream |

## Action chunks

Units are radians; joints are the stream context's names in its order (a different order is refused, never permuted);
composition is `absolute`, or `delta` on the chunk's `base_q` (the joint position at observation time); `observed_at` is a
supervisor-clock stamp. `max_age_s` and `hold_s` are required (no defaults). A chunk is dropped as **late** if its
observation is older than `max_age_s`, older than the observation of what the RT already selected, of an in-flight chunk, or of one whose fate is unknown, or it was requested before the
last `start()`/`cancel()`, or a cancel is still unconfirmed (`cancel()` returned False: the session is kept, every chunk is late until a cancel is confirmed, and `start()` refuses); a late response never restarts a session. One revision is in flight and one chunk waits; a newer
chunk **replaces** the waiting one. Nothing is clipped: the supervisor's admission has the final word and its refusal reason
is reported.

## What a `Report` says

Three separate facts per high-level action: `requested` (sent), `admitted` (the supervisor's trusted admission accepted it:
the reply to a revision only says REQUESTED, admission is asynchronous, so the clients poll the revision), `selected` (the RT
selected it, which is what governs motion). A refusal after the request, or an expiry/cancel after admission without selection, is in `refused`; a revision whose status could not be read is in `unknown` (and counts as possibly sent for the ordering rule). `selected_revision` is the stream's
current revision. Planner requests are admitted synchronously.

## Timing

A chunk is spliced at the supervisor's splice anchor (the earliest the RT can take it), not at its `observed_at`: actions
already overtaken by inference latency are not skipped, so a policy must account for its delay. The horizon of a chunk is
`len * dt_s` and is bounded by the supervisor's stream policy (`max_future_s`).

## Not done

"Limit changes visible in execution reports and datasets" (the supervisor refusal reason is reported, nothing is clipped, but there is no dataset record), LeRobot and OpenPI boundaries (project-owned, optional extras, under `Control/`), a normalization map, and the control-profile
field in a recorded dataset. These need a real policy interface to adapt and are not built here. No hardware run; a real run
needs the corresponding capability gate.
