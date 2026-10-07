# Intermittent REFERENCE_EXPIRED in check_dual_supervisor: what is known, and the diagnostics added

**Observed (2026-10-07).** The offline suite on arm_control `f260179` (a config-only merge) failed `dual_supervisor` once. In the
30-pair loop, pair 22's arm_b leg faulted `REFERENCE_EXPIRED`, 8.9 ms after arm_a's release. Log: Control
`docs/status/2026-10-07-dual-supervisor-reference-expired.log`.

**Measured.** The same commit then passed 30 of 30 standalone runs, and 20 of 20 with 40 busy-loop processes on the 32-core PC. That is 1
failure in 51 runs (an earlier batch of 10 runs failed on a bad server path in the author's harness and is not counted). Plain CPU contention does not reproduce it.

**What the freeze experiments show, and do not.** Freezing arm_b's whole supervisor process (SIGSTOP) did not reproduce the fault:
* mid-move, for 0.02-0.2 s: no fault;
* mid-move, for 0.3 s: a different fault, a stale-observation discard;
* across the release, for 0.02-0.2 s: no fault.

These results rule out less than they seem to (review). A 0.2 s move reaches its END state about 0.2 s after selection, and the server
never retires END, so most mid-move freezes could not trip the age rule. SIGSTOP also freezes the sender thread, so it does not model the
likelier stall: a delayed supervisor control tick while the sender keeps repeating an old decision past the 100 ms age ceiling. Only
the control tick produces the first envelope, so a delayed tick right after release also hits the first-envelope gap. And both test
servers run without real-time priority (their `sched_fifo` setup fails, so they run degraded at SCHED_OTHER). A starved server's
state samples age, and that inflates `lag`.

The envelope phase cannot run ahead of the server's by clock error: the PC echoes `ref['phase']` from the server's own state samples
(`supervisor/runtime.py` publishes `phase=point.time`), and the server's phase only moves forward for a reference.

**Candidates.**
* A first-envelope gap on arm_b (the review's best guess: arm_b released about 9 ms after arm_a, and its first envelope depends on its
  own supervisor's control tick).
* An age-ceiling expiry ("incoming/held envelope stale") from a control-tick stall.
* Server starvation under SCHED_OTHER.
* A revision/prefix mismatch, or an envelope past its phase window.

**Added.** On any failure, `check_dual_supervisor.py` now prints, for both arms' supervisor and arm_rt_server logs (temporary files,
discarded on failure until now), the key lines from the whole log (expiry, sender gap, fault) and the last 120 lines. It reads with
`pread`, so the child's file offset does not move, and a closed log is reported rather than raised. That output carries the server's fault-journal line, "reference envelope expired:
<rule> ... age_us=..." (or `gap_age_us=` for the first-envelope gap), and the supervisor's "reference sender gap" records. Verified with an injected 0.3 s supervisor freeze: both
appear in the failure output.

Owner: supervisor/RT reference timing. Effect: an intermittent pair fault; fail-safe (the pair faults, the partner is cancelled).
Not root-caused; the next occurrence will say which rule fired.
