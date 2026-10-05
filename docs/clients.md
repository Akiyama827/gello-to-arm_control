# Application clients (CTRL-052)

`arm_control/clients/` holds four thin clients over `ArmClient` (stdlib + numpy; the backend never imports it, checked by
`tools/bench/rt/clients.py`). Run them against fake endpoints: `cd tools/bench/rt && PYTHONPATH=../../..:. python clients.py <build>/reference_runtime_selfcheck` (or `tools/bench/rt_offline_suite.py`).

| Client | Contract |
|---|---|
| `planner.run_reviewed` | submit a reviewed `TrajectoryRequest`, release it only if it starts on release, wait; a timeout leaves it running |
| `chunks.ActionChunkClient` | a policy's chunks, spliced from the supervisor's anchor |
| `teleop.DeadmanTeleop` | setpoints while the deadman is held and fresh. Each step's HORIZON is at most the deadman timeout (a longer move is shortened toward the target and noted); that bounds what one step commits, not a stalled calling loop. A step is spliced at the selected reference's next eligible whole-segment cut (can be well ahead of now); if the loop stalls the arm continues to that cut plus one step, then the stream faults (REFERENCE_UNDERFLOW): the RT holds the measured pose with no PC feedforward, the next `command()` is refused and then returns `expired`, and motion needs a new admission. Travel is measured from the measured pose and the speed is an average (admission enforces the limits). With a live loop, release or a stale heartbeat CANCELS the stream; an unconfirmed cancel is retried and reported (`stop_failed`); `busy` while the previous revision is unresolved, `expired` when the supervisor ended the stream; never re-arms |
| `rl.StepWindow` | `step(q_target, window_s)` reports actual elapsed time and `completed` (the RT selected the revision) `/intervention/refused/not_selected/expired/faulted/unknown` (`unknown`: an infrastructure error once the revision may have been sent, conservatively from just before the splice anchor is read; motion may have happened, the id is in `report.unknown`; an error before anything was sent is `refused` with its reason; a non-finite action raises `ValueError`); an intervention returns early but does NOT stop the arm; `terminated` is always False (task success is project-owned); `reset()` raises `NeedsOperator` unless the robot is armed and unfaulted and never re-arms, homes, opens the Hand, discards calibration or restarts a stream |

## Action chunks

Units are radians; joints are the stream context's names in its order (a different order is refused, never permuted);
composition is `absolute`, or `delta` on the chunk's `base_q` (the joint position at observation time); `observed_at` is a
supervisor-clock stamp. `max_age_s` and `hold_s` are required (no defaults). A chunk is dropped as **late** if its
observation is older than `max_age_s`, older than the observation of what the RT already selected, of an in-flight chunk, or of one whose fate is unknown, or it was requested before the
last `start()`/`cancel()`, or a cancel is still unconfirmed (`cancel()` returned False: the session is kept, every chunk is late until a cancel is confirmed, and `start()` refuses); a late response never restarts a session. One revision is in flight and one chunk waits; a newer
chunk **replaces** the waiting one. Nothing is clipped: the supervisor's admission has the final word and its refusal reason
is reported.

## Action spec (units, normalization, profile)

`ActionSpec` states once what a policy's numbers mean and is recorded in `Report.spec`: named joints in order, `units`
(`rad`, or `normalized` with a per-joint `offset` and `scale`, both required and finite, `scale > 0`; nothing is assumed),
`composition` (`absolute`: `rad = offset + scale * v`; `delta`: `rad = scale * v`, an offset belongs to a position, not a
difference), the longest chunk horizon (`horizon_s`, a longer chunk is refused) and the control profile the run expects
(`control_profile`). Pass it as `ActionChunkClient(..., spec=...)`: chunk units and composition must match it (refused
otherwise), and `start()` raises `NeedsOperator` unless the supervisor's selected preset is the expected one (also when
it cannot report its profile): a client never switches a profile. `Report.profile_at_start` records what was selected.
Without a spec the client accepts `rad` chunks as before.

## What a `Report` says

Three separate facts per high-level action: `requested` (sent), `admitted` (the supervisor's trusted admission accepted it:
the reply to a revision only says REQUESTED, admission is asynchronous, so the clients poll the revision), `selected` (the RT
selected it, which is what governs motion). A refusal after the request, or an expiry/cancel after admission without selection, is in `refused`; a revision whose status could not be read, or whose reply was lost (chunks and teleop: the revision stays in flight, `submit` still says `sent`, teleop `command` returns `unknown`), is in `unknown` (its fate was unknown at some point; the id may later also be in `selected` or `refused`, and may be listed twice) (and counts as possibly sent for the ordering rule). `selected_revision` is the stream's
current revision. Planner requests are admitted synchronously.

## Timing

A chunk is spliced at the supervisor's splice anchor (the earliest the RT can take it), not at its `observed_at`: actions
already overtaken by inference latency are not skipped, so a policy must account for its delay. The horizon of a chunk is
`len * dt_s` and is bounded by the supervisor's stream policy (`max_future_s`).

## Not done

"Limit changes visible in execution reports and datasets": the supervisor's refusal reason is reported and nothing is
clipped, so there is no limit change to record; a dataset writer is project-owned. LeRobot and OpenPI boundaries (project-owned,
optional extras, under `Control/`): neither package is installed here, so an adapter could not be checked against its API,
and none is written; adding the extras is an explicit decision. No hardware run; a real run needs the corresponding
capability gate.
