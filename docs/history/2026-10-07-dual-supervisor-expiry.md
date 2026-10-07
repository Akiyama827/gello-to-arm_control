# Intermittent REFERENCE_EXPIRED in check_dual_supervisor: what is known, and the diagnostics added

**Observed (2026-10-07).** The offline suite on arm_control `f260179` (a config-only merge) failed `dual_supervisor` once. In the
30-pair loop, pair 22's arm_b leg faulted `REFERENCE_EXPIRED`, 8.9 ms after arm_a's release. Log: Control
`docs/status/2026-10-07-dual-supervisor-reference-expired.log`.

**Measured.** The same commit then passed 30 of 30 standalone runs, and 20 of 20 with 40 busy-loop processes on the 32-core PC. That is 1
failure in 71 runs. Plain CPU contention does not reproduce it.

**Ruled out.** Freezing arm_b's supervisor (SIGSTOP) did not reproduce it:
* mid-move, for 0.02-0.2 s: no fault;
* mid-move, for 0.3 s: a different fault, a stale-observation discard;
* across the release, for 0.02-0.2 s: no fault.

The envelope phase cannot run ahead of the RT server's by PC clock error: the PC echoes the phase from the server's own state samples.
The candidates left are the server's other freshness rules, `rt/src/server.hpp` `reference_envelope_fresh` and the first-envelope gap in
`rt/src/rt_loop.cpp`: a revision or prefix mismatch, an envelope past its phase window, or a late first envelope.

**Added.** On any failure, `check_dual_supervisor.py` now prints the last 120 lines of both arms' supervisor and arm_rt_server logs
(they were temporary files, discarded on failure). That output carries the server's fault-journal line, "reference envelope expired:
<rule> ... age_us=...", and the supervisor's "reference sender gap" records. Verified with an injected 0.3 s supervisor freeze: both
appear in the failure output.

Owner: supervisor/RT reference timing. Effect: an intermittent pair fault; fail-safe (the pair faults, the partner is cancelled).
Not root-caused; the next occurrence will say which rule fired.
