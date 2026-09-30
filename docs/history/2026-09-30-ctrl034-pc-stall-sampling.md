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
