# CTRL-034 pending-COMMIT cancel race — 2026-09-30

Branch `task/control/CTRL-034-pending-commit-cancel`, base `c52034f`. Fix commit
`f8c9003`. Offline software and a fake-backend network rerun only; no FCI, no
robot, no deployed unit.

## Symptom

Native mode, a controller preset change while a stream COMMIT is pending:
`controller_extension.py` times out in `until(lambda: not runtime._native.busy)`;
the RT's `last_cmd_seq` freezes and CMD_LOST follows with no `reference envelope
expired` journal line; the PC's `candidate_selection` stays `'pending'`.
Rate at the base tree: 10/15 (reviewer); 6/6 in this session's runs.

## Root cause

Evidence: the horizon timeline (`ReferenceHorizon.diagnostics()`), the sender-gap
records, and a trace of every link call and wire send around the failure.

1. **(c) Why the candidate stays pending.** `candidate_selection` leaves
   `'pending'` in only two ways: a refused/`_NotDispatched` COMMIT, or `poll()`
   observing RUNNING/END/UNDERFLOW for the candidate revision. The failing
   order is: cancel queued behind COMMIT on the one upload worker; COMMIT
   reaches the RT and is acknowledged; the RT selects; the queued
   `cancel_reference` retires it about 1 ms later. The 100 Hz `poll()` never
   sees RUNNING (the timeline has `commit_sent`/`commit_reply` for the reference
   and no `first_selected_pc_state`), and once `_interrupt` is set `poll()` skips
   its CANCELLED handling. Nothing else touches the slot.
2. **(a) What the PC sends after the cancel: nothing.** `busy` includes
   `candidate_selection == 'pending'` for a stream. `runtime._send_command`
   reaches its tail, `h.unpublish()`, then returns `None` while `busy`, so it
   never stages the measured-hold reference either. Trace after the last cancel:
   36 `_send_command` calls, 0 non-`None` returns, 0 packets on the wire
   (ordinary or envelope), decision unpublished. The horizon's own sender is
   fenced too (interrupt set, RT status not RUNNING/END). The sender-gap
   records are empty: this is a decision, not a scheduling stall.
3. **(b) Why the RT admits none of it.** Nothing arrives. `last_cmd_seq` stays at
   the last envelope sent before the cancel (30 in the trace). The RT's reference
   was CANCELLED by an explicit CANCEL, not retired by expiry, so there is no
   expiry journal event; after `--fault-ms` it latches the generic
   "command stream lost".
4. **Why it surfaced now.** `9184f43` made the RT poll its control session
   promptly while replies are pending. COMMIT and CANCEL replies now land about
   1 ms apart, which removes the window in which the PC could observe RUNNING.
   Reviewer bisection (failures/runs): 0/21 at `425a943`–`5b06d28`, 1/6 at
   `55188d3`, 1/7 at `30037e1`, 0/6 at `5325bb8`, 3/6 at `9184f43`, 10/15 at
   `3c82090`. The PC-side hole is older; `9184f43` is not reverted.

The review lead (plain `RtLinkError` instead of `_NotDispatched` when `commit()`
sees the cancel before dispatch) is a real way to reach the same stuck state,
and is covered by ordering 1 below. It is not the only way: `client.commit_reference`
clears `_upload_cancel` on entry, so a cancel landing after `commit()`'s own check
does not stop the COMMIT (ordering 2).

## Fix

`ReferenceHorizon.poll()`: when the queued CANCEL is acknowledged and the slot is
still `'pending'`, set it to `'not_requested'`. The single worker runs CANCEL after
any COMMIT, so the ack settles a selection the PC never observed. Seven lines,
one file. A failed CANCEL still leaves the slot pending and surfaces `error`;
that is fail-stop into CMD_LOST by design, because the RT state is then unknown.

Not done: raising `_NotDispatched` from `commit()` on a pre-dispatch cancel. It
would set `candidate_result = REF_LATE` and report a cancel as an expiry. Add it
only if a consumer needs the distinction.

## Deterministic reproduction

`tools/bench/rt/pending_commit_cancel.py`, registered in `rt_offline_suite.py`
and the `rt/README.md` table. Explicit gates, no timing:

- **Ordering 1, `before_dispatch`.** The horizon's cancel Event is swapped for one
  that blocks the upload worker at its first `is_set()`; the preset change runs;
  the gate opens; `commit()` raises the plain `RtLinkError`.
- **Ordering 2, `in_transport`.** `link.commit_reference` is held; the preset
  change runs; the hold is released so COMMIT dispatches, the RT selects and the
  queued CANCEL retires it. `poll()` is fed no RT state during that window, which
  is the ~1 ms case made certain instead of probable.

After each: `busy` clears, and the link is not faulted 1.5 s later (fault-ms is
1 s). The bench fails on literal `3c82090` at ordering 1 (3/3 runs); ordering 2
also fails there on its own.

## Evidence

| Check | Result |
| --- | --- |
| `pending_commit_cancel.py` on `3c82090` | FAIL 3/3 |
| `pending_commit_cancel.py` on `f8c9003` | PASS 3/3 (and in the suite) |
| `controller_extension.py`, 20 sequential runs, `f8c9003` code | **20/20** (0/6 before) |
| `rt_offline_suite.py`, clean tree, `f8c90034551d16c82924f601f20f578afd0fceed` | **20 PASS, 0 FAIL** |
| Remote fake-RT, `reference_supervisor.py` × 10, fresh scratch server each | 10/10 PASS; schedstat `ok`; worst `rt_total` 24.0 ms, `over_50ms=0` in all 10; worst sender wake lateness 22.5 ms |
| Remote fake-RT, `pc_stall.py` × 10, fresh scratch server each | 10/10 S1–S4 PASS |

Remote setup, per `ctrl034-remote-fake-rt-runbook.md`: `ok@172.16.1.2`, scratch
`arm_rt_server` (sha256 `eaf8de56…`, verified before use), `--backend fake
--enable-reference --hold-ms 100 --fault-ms 1000`, core 0, bound to 172.16.1.2,
ports 47900/47901 (free before each run), instance `bench_fake_034`. Window
2026-09-30 00:31:19–00:34:37 +08:00. PC: 32 CPUs, load 0.58 → 0.30, kernel
recorded in the raw logs. Deployed units untouched: they were in their usual
off-arm restart loop (`NRestarts` 212280 → 212368 and 212092 → 212181).

Every `command stream lost` line in the 20 RT journals is the bench's own S4
scenario (asserted, one per `pc_stall` run). `reference_supervisor` had none;
its 10 `reference envelope expired` lines are its deliberate expiry case.

## Limits and open points

- This is a 10 + 10 spot check for the changed PC path, **not** the 20/20 network
  gate and not the 60-run matrix; neither was re-run. The earlier `pc_stall`
  19/20 record stands as recorded.
- Ordering 1 exposes a second, unfixed oddity: `approve()` sets
  `selection_requested` before COMMIT is dispatched, so `cancel()` chooses
  `cancel_reference` rather than ABORT and the RT keeps an uncommitted staged
  upload. The RT accepted the next BEGIN, so nothing failed. Worth a reviewer's
  view; not changed here.
- In this harness an idle ARMed RT reports `last_cmd_seq == 0` while the PC streams
  ordinary commands at 100 Hz (it moved only for envelopes). Unexplained and not
  investigated; it is why the bench asserts on the fault state, not that field.
- While collecting this evidence I overwrote the RT-host copies of
  `pc_stall-1..10.server.log` and `reference_supervisor-1..10.server.log` in
  `~/ctrl034-scratch` (same names as the 2026-09-29 matrix, per the runbook).
  Archived copies of those names exist in tag
  `evidence/CTRL-034-integrated-remote-evidence`.
