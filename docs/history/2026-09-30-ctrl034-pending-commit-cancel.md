# CTRL-034 pending-COMMIT cancel race — 2026-09-30

Branch `task/control/CTRL-034-pending-commit-cancel`, base `c52034f`. Round 1:
`f8c9003` (settle the slot on CANCEL ack), `1e82057` (this note). Round 2, after
review: `42d4b1a` (test gate), `1b2bd4b` (abort before dispatch). Round 3:
`b29fc28` (settle on the pre-dispatch failure itself, tolerate stale refusals, test
the replacement path). Offline software
and a fake-backend network rerun only; no FCI, no robot, no deployed unit.

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

**Round 2 (`1b2bd4b`), cancel before dispatch.** `commit_reference` no longer
clears the cancel flag on entry (each attempt's `upload_reference` already does),
and `_reference_exchange` checks it under the upload lock immediately before the
first COMMIT send. If a cancel came first it raises the new
`ReferenceCancelledBeforeDispatch` (not `_NotDispatched`, so no `REF_LATE`). An
interruption after the send stays a plain `RtLinkError`: that outcome is
uncertain. `commit()` records the distinct error, and `cancel()`'s queued job
(which runs after the COMMIT job on the one worker) sends ABORT of the sealed
upload instead of CANCEL, then still cancels a selected revision if a
replacement was the undispatched one. Round 3 tests that branch. The runtime's own deadline-expiry path
(`_NotDispatched`) still ends in CANCEL; the RT discards a matching staged
candidate on CANCEL, so nothing leaks there either.

**Round 3 (`b29fc28`).** Two review points:

1. A commit job that raised `ReferenceCancelledBeforeDispatch` is proof of
   non-selection, so `poll()` sets `candidate_selection = 'not_requested'` at once
   (not `REF_LATE`). Before, the slot depended on the queued ABORT/CANCEL
   succeeding; a failed retire would have left it pending again.
2. In `retire()`, the selected-revision `cancel_reference` tolerates
   `REF_REVISION`/`REF_RETIRED`, like the existing path. Any other refusal
   propagates and is reported as `error`; the slot is already settled.

## Deterministic reproduction

`tools/bench/rt/pending_commit_cancel.py`, registered in `rt_offline_suite.py`
and the `rt/README.md` table. Explicit gates, no timing:

- **Ordering 1, `before_dispatch`.** The horizon's cancel Event is swapped for one
  that blocks the upload worker at its first `is_set()`; the preset change runs;
  the gate opens; `commit()` raises `ReferenceCancelledBeforeDispatch`. Asserted:
  `cancel()` sent ABORT and no CANCEL.
- **Ordering 2, `in_transport`.** A gate in `_send_frame` holds the COMMIT frame
  (recognised by the type last packed on that thread), which sits after the
  client's cancel check; the preset change runs; the frame is released so COMMIT
  dispatches, the RT selects and the queued CANCEL retires it. `poll()` is fed no
  RT state during that window, which is the ~1 ms case made certain instead of
  probable. (Round 1 gated before `commit_reference`; with the round-2 client that
  point is ordering 1.) Asserted: CANCEL, no ABORT.
- **Slot settles first** (ordering 1 and the replacement cases). The queued
  ABORT is held inside the worker; while it is held the slot must already read
  `not_requested` and the cancel job must still be running. On `94a9ce8` this
  times out.
- **Replacement, `replacement_cancel_*`.** Revision 1 is selected; the
  replacement's COMMIT is held at the horizon's cancel check; the preset change
  runs; the gate opens. Asserted: ABORT, then CANCEL(1). Three variants: CANCEL(1)
  succeeds; is refused `REF_RETIRED` (tolerated, no error); is refused `REF_BUSY`
  (error `... refused` surfaced, slot settled). The refusals are scripted in the
  bench (the RT is not contacted), then the test sends the real CANCEL(1) so the
  RT is not left with a running, unfed reference. Checked to bite: removing the
  tolerance fails the tolerated variant; swallowing every refusal fails the
  refused variant.
- **No sealed upload afterwards** (all cases). With the runtime's hold upload
  suppressed, a COMMIT of the last upload must be refused by the RT; a leftover
  sealed upload would be accepted. Checked to bite: a mutant that skips the ABORT
  fails with "RT still held a sealed upload".

After each: `busy` clears, and the link is not faulted 1.5 s later (fault-ms is
1 s). The bench fails on literal `3c82090` at ordering 1 (3/3 runs); ordering 2
also fails there on its own.

## Line-108 flake (`controller_extension.py`, review finding)

A reviewer saw `assert arm.stream_revision(eid, 'pending').selection_state is
SelectionState.PENDING` fail once in 10, right after `entered.wait`. It is a test
assumption, not a product bug and not an asynchronous SDK view. The `delayed`
gate wrapped **every** `commit_reference`, and the runtime also commits its own
measured-hold references. Under CPU load one of those reached the gate first, so
`entered` fired while the stream attempt was still `NOT_REQUESTED` (`PENDING` is
set immediately before `approve()` submits COMMIT, so it cannot be skipped once
the stream's COMMIT is reached).

Reproduced by pinning 8 parallel runs plus 6 busy loops to 4 cores: 10 of 80 runs
failed, every one with status `NOT_REQUESTED` (phase ADMITTED or REQUESTED) and a
gate entry of `stream=False key=None id=4`, the hold reference. Fix `42d4b1a`:
non-stream commits pass through the gate. Same load after the fix: 0 of 80 fail
at this assertion. Not fixed and unrelated: under that 4x oversubscription 3 to 6 of 80 runs (two runs of that load) still fail earlier or at `entered.wait` (see limits).

## Evidence

| Check | Result |
| --- | --- |
| `pending_commit_cancel.py` on `c52034f` / `1e82057` | FAIL (stuck slot / CANCEL not ABORT) |
| `pending_commit_cancel.py` on `1b2bd4b`, 20 sequential runs | **20/20** (round 2) |
| `controller_extension.py` on `1b2bd4b`, 30 sequential runs | **30/30** |
| `rt_offline_suite.py`, clean tree, `1b2bd4b945032a4ff3da4a4a869508a341070b3b` | **20 PASS, 0 FAIL** |
| Remote fake-RT, `reference_supervisor.py` x 10, fresh scratch server each | 10/10 PASS; schedstat `ok` x10; worst `rt_total` 18.2 ms, `over_50ms=0` x10; worst sender wake lateness 21.3 ms |
| Remote fake-RT, `pc_stall.py` x 10, fresh scratch server each | 10/10 S1-S4 PASS |

Round 3, on `b29fc28` (clean tree):

| Check | Result |
| --- | --- |
| `pending_commit_cancel.py` (with the replacement cases), 20 sequential runs | **20/20** |
| `controller_extension.py`, 10 sequential runs | **10/10** |
| `rt_offline_suite.py`, `b29fc28fc874a1070a2c8c6f2d89a4d77d35aa16` | **20 PASS, 0 FAIL** |
| Remote fake-RT, `reference_supervisor.py` x 10 | 10/10 PASS; schedstat `ok` x10; worst `rt_total` 21.8 ms, `over_50ms=0` x10; worst sender wake lateness 23.5 ms |
| Remote fake-RT, `pc_stall.py` x 10 | 10/10 S1-S4 PASS |

Remote setup as in round 2 (scratch `arm_rt_server` sha256 `eaf8de56...`, ports free
before each run, fresh scratch server per run, logs saved as
`pcc3-<bench>-<run>.server.log`). Window 2026-09-30 10:44:48-10:48:06 +08:00, PC
revision `b29fc28` clean, 1-min load 0.23 at start. Deployed units untouched
(`NRestarts` 228640 -> 228729 and 228453 -> 228542). The 10 `command stream lost`
lines are `pc_stall`'s S4 case, one per run; `reference_supervisor` had none.

Round-1 and round-2 evidence (`f8c9003`, `1b2bd4b`) is superseded by the above.

Remote setup, per `ctrl034-remote-fake-rt-runbook.md`: `ok@172.16.1.2`, scratch
`arm_rt_server` (sha256 `eaf8de56...`, verified before use), `--backend fake
--enable-reference --hold-ms 100 --fault-ms 1000`, core 0, bound to 172.16.1.2,
ports 47900/47901 (free before each run), instance `bench_fake_034`. Window
2026-09-30 01:36:17-01:39:36 +08:00, PC revision `1b2bd4b` clean, 32 CPUs, 1-min
load 1.51 at start (the contention runs were finishing). Deployed units untouched
(`NRestarts` 214013 -> 214101 and 213826 -> 213914, their usual off-arm loop). RT
server logs were saved as `pcc2-<bench>-<run>.server.log`, so nothing on the RT
host was overwritten this time.

Every `command stream lost` line in the 20 RT journals is the bench's own S4
scenario (10, one per `pc_stall` run; asserted). `reference_supervisor` had none.

## Open CTRL-034 items

**(a) BEGIN refused `REF_REVISION` while a reference is RUNNING.** Under heavy load
the runtime's own measured-hold reference can still be RUNNING when a new stream's
BEGIN arrives, and the stream is REJECTED (`reference operation 32 refused: 4`).
From the RT (`rt/src/reference_upload.hpp` `begin()`, `control_session.cpp`): BEGIN
compares `expected_revision` with the selected revision only if a reference of
this session **and ARM epoch** is RUNNING or END (`new_after_end` excepts a
higher-id first revision after END); otherwise it compares with 0. So any
first-revision BEGIN issued while an earlier reference of the same ARM epoch is
still RUNNING is refused. Can a single non-stream first reference hit it? **No,
not in a fresh ARM epoch:** nothing is selected, so `active` is false and the
expected revision is 0. The runtime also stages no hold reference before the first
reference (until then the idle hold stays on the v5 command path, `_send_command`).
It applies to a second or later first-revision reference (a new stream, or the
runtime's hold reference) started while the previous one is still RUNNING. This
reasoning is from the RT source and the trace above; it has not been run as a
dedicated test. Multi-window refills of one reference are unaffected: they carry
`expected_revision = selected revision` for the same id. A related fact from the
same function: a BEGIN while any upload is present returns `REF_BUSY`, so a
leftover sealed upload would have blocked the next upload.

**(b) Design follow-up: an unresolved `pending` selection should be reported.**
While `candidate_selection == 'pending'` the runtime's tail returns `None` and
sends nothing, so an unresolved selection silently suppresses every command until
the RT latches CMD_LOST. The fixes so far make `pending` resolve in the known
cases; they do not bound it. Report an unresolved `pending` explicitly, with a
bounded wait (a fault or error the operator sees) instead of silence. Not
implemented; needs a decision on the bound and on whether the horizon or the
runtime owns it.

## Limits and open points

- This is a 10 + 10 spot check for the changed PC path, **not** the 20/20 network
  gate and not the 60-run matrix; neither was re-run. The earlier `pc_stall`
  19/20 record stands as recorded.
- **Correction to round 1.** This note said ordering 1 leaves an uncommitted staged
  upload on the RT after CANCEL. I inferred that from the trace and had not read
  the RT. It is wrong: on CANCEL the RT discards a matching staged candidate
  (`control_session.cpp`, `upload->discard()`), so nothing leaked. Round 2's ABORT
  is the precise call, not a leak fix.
- **Unfixed, under starvation only.** With 8 runs and 6 hogs on 4 cores, 3 to 6
  of 80 `controller_extension.py` runs still fail: two at harness startup (no
  HELLO / authority timeout), and about three at `entered.wait` because the stream
  is REJECTED within ~10 ms with `reference operation 32 refused: 4` (a BEGIN
  refused `REF_REVISION`) while the runtime's own hold reference is still RUNNING.
  That is a real race between opening a stream and the runtime's hold reference,
  visible only when the hold's END is delayed. Not touched here; widening the
  1 s wait did not help, so I did not keep it.
- In this harness an idle ARMed RT reports `last_cmd_seq == 0` while the PC streams
  ordinary commands at 100 Hz (it moved only for envelopes). Unexplained and not
  investigated; it is why the bench asserts on the fault state, not that field.
- Round 1 overwrote the RT-host copies of `pc_stall-1..10.server.log` and
  `reference_supervisor-1..10.server.log` in `~/ctrl034-scratch` (same names as
  the 2026-09-29 matrix, per the runbook). Archived copies of those names exist in
  tag `evidence/CTRL-034-integrated-remote-evidence`.
