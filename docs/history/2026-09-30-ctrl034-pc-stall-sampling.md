# CTRL-034 item c — `pc_stall.py` S2b/S3 "resume step 0.0000" (2026-09-30)

The local `pc_stall.py` failed the offline suite about 10-15% of runs at S2b or S3 with
`resume step 0.0000 rad`. Root cause: a test-only sampling assumption, not a hold jump.

After a stall the bench commits a new reference and waits for `not holding`, then measures
the step between consecutive RUNNING samples recorded by a separate 1 ms `Recorder` thread.
`wait_for(not holding)` returns as soon as the state shows the new reference running, before
the recorder has appended a second RUNNING row. One row leaves no step to measure, so
`steps` is empty and `max(steps, default=0)` prints 0.0000.

Evidence: an instrumented copy over 40 runs failed 5 times (12.5%); every failure had
`n_running=1` (statuses `[4, 4, 1]` or 40+ rows of `4` then one `1`), never a large step.
Fix: wait for at least 10 RUNNING rows (bounded 3 s) before measuring; the step bound now
covers the whole window, so the "old reference resumed" check is stronger, not weaker.
After the fix: 40/40 sequential runs on `pc_stall.py`.

Scope: harness only. No product code, control law, gain, limit or timing constant changed.
The network gate is unaffected (it passed in every earlier round).

## Review addendum

The pre-fix test could also pass with nothing measured: in 13 of 76 passing S2b/S3 checks only 2-3 RUNNING rows, all still holding, were seen, so the max step was 0.0. A mutant that commits the new reference at the retired reference's position (~0.20 rad jump) passed S2b in 4 of 12 runs before the fix and failed 12 of 12 after it. Open, pre-existing: the resume bound (74 mrad) is loose against the ~39-57 mrad held-setpoint-to-measured jump; a tighter bound (first RUNNING q_cmd vs measured q at commit) belongs to a later card.
