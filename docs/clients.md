# Application clients (CTRL-052)

`arm_control/clients/` holds four thin clients over `ArmClient` (stdlib + numpy; the backend never imports it, checked by
`tools/bench/rt/clients.py`). Run them against fake endpoints: `python tools/bench/rt/clients.py <reference_runtime_selfcheck>`.

| Client | Contract |
|---|---|
| `planner.run_reviewed` | submit a reviewed `TrajectoryRequest`, release it only if it starts on release, wait; a timeout leaves it running |
| `chunks.ActionChunkClient` | a policy's chunks, spliced from the supervisor's anchor |
| `teleop.DeadmanTeleop` | setpoints while the deadman is held and fresh; release or a stale heartbeat CANCELS the stream (the executor holds); never re-arms |
| `rl.StepWindow` | `step(q_target, window_s)` reports actual elapsed time and `completed/intervention/refused/expired/faulted`; `terminated` is always False (task success is project-owned); `reset()` raises `NeedsOperator` unless the robot is armed and unfaulted and never re-arms, homes, opens the Hand, discards calibration or restarts a stream |

## Action chunks

Units are radians; joints are the stream context's names in its order (a different order is refused, never permuted);
composition is `absolute`, or `delta` on the chunk's `base_q` (the joint position at observation time); `observed_at` is a
supervisor-clock stamp. `max_age_s` and `hold_s` are required (no defaults). A chunk is dropped as **late** if its
observation is older than `max_age_s`, older than the observation of what was already sent, or it was requested before the
last `start()`/`cancel()`; a late response never restarts a session. One revision is in flight and one chunk waits; a newer
chunk **replaces** the waiting one. Nothing is clipped: the supervisor's admission has the final word and its refusal reason
is reported.

## What a `Report` says

`admitted`: high-level actions the supervisor accepted. `selected_revision`: the revision the RT has actually selected, which
is what governs motion. A buffered action can be admitted and never selected.

## Not done

LeRobot and OpenPI boundaries (project-owned, optional extras, under `Control/`), a normalization map, and the control-profile
field in a recorded dataset. These need a real policy interface to adapt and are not built here. No hardware run; a real run
needs the corresponding capability gate.
