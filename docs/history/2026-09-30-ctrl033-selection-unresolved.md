# CTRL-033 4b — report an unresolved pending selection (2026-09-30)

Follow-up from CTRL-034 (see `2026-09-30-ctrl034-pending-commit-cancel.md`): while a
stream candidate is `'pending'` the supervisor's tail sends nothing, so any way of
leaving it pending starves the RT to CMD_LOST. The CTRL-034 fixes resolve the known
cases; this bounds the rest.

- `ReferenceHorizon.poll()` raises `SelectionUnresolved` (an `RtLinkError`) once a stream
  candidate has been pending longer than `SELECTION_UNRESOLVED_S` = 250 ms (operator
  decision). It raises after finishing its own bookkeeping, and first moves the slot to
  `'unresolved'`, which is not `'pending'`, so it can never again suppress commands.
- `RobotRuntime._selection_unresolved()` fails the execution with a receipt
  (`CANCELLED` if still `ADMITTED`, otherwise `FAULTED`; reason `selection unresolved: ...`),
  marks the pending attempt `CANCELLED` and, on a live selected stream, the selected attempt
  `FAULTED` (selection states left as is: structurally uncertain, the RT may have selected), and retires the reference through the existing
  `_drop_staged` path, which sends an explicit CANCEL (or ABORT if COMMIT was never sent)
  and cancels the controller plan. It does not set `_native_failed`, so once the slot is
  free the ordinary tail stages the measured hold and commands resume.
- `tools/bench/rt/selection_unresolved.py` (in the offline suite): COMMIT succeeds, the
  PC's poll is fed no RT state. Asserts the failure receipt, that the slot was seen
  `pending` then `unresolved`, an explicit CANCEL, a newer hold reference selected, and no
  fault over 1.5 s. Control run with the timeout disabled: the same setup does starve to
  CMD_LOST. On `a85e169` the bench fails.

Not changed: gains, limits, RT timing constants, the wire protocol. A candidate that is
legitimately pending longer than 250 ms (a very slow COMMIT round trip) is now failed
and retired; that is the chosen trade-off.

## Review round 1 (BLOCK, fixed in follow-up commits)

1. **Expiry evidence wins.** The raise ran after the same `poll()` had recorded
   `REFERENCE_EXPIRED`, so a PC stall that let the RT select and then expire the
   reference was reported as "selection unresolved" and an implicit hold healed it,
   contradicting the CTRL-034 decision (a) rule. The raise now requires `not self.error`.
   `selection_expiry_precedence.py` pins it with explicit hooks (no envelopes; poll fed
   only the RT's CANCELLED evidence; the pending age advanced at that poll); a mutant
   without the guard fails it.
2. **Contract phase.** `REJECTED` means "never moved"; the RT may have selected the
   candidate, so an ADMITTED execution now ends `CANCELLED` (dropped; holds if it had
   been selected) and a SELECTED one `FAULTED`. The receipt reason carries the cause;
   the operator may prefer a dedicated phase or evidence field. Attempts of the
   execution, including the selected one, are updated the same way.

Recorded, not changed:
- The explicit CANCEL is real on the wire, but a reference the PC has not observed is
  never fed envelopes, so the RT has usually already expired it by 250 ms; the bench
  therefore shows the CANCEL is sent and acknowledged, not that it retired a still
  RUNNING reference. That case rests on reasoning.
- The "measured hold" is the controller's hold command staged as a reference by the
  runtime tail, not a separate mechanism.
- A failed CANCEL still ends in CMD_LOST if the reference is still RUNNING (the tail's
  own `cancel()` returns early once interrupted): the same safe outcome as before.
- The 250 ms rule caps a stream COMMIT wait: `cancel()` interrupts the in-flight
  exchange, so the previous worst case (`ack_timeout_s` 2 s with one retry) no longer
  applies.
- `pc_stall.py` and `first_envelope_probe.py` call `poll()` on streams they stage and can
  now see `SelectionUnresolved` if a candidate stays pending 250 ms; that is a real
  failure there and is not swallowed.

## Review round 2 (ACCEPT)

Both blockers verified closed; suite 22/0 at `4e2947d`; the new benches 10/10 under CPU
load, and the mutant without the guard fails. Recorded, not changed after ACCEPT:
- **Guard breadth.** `not self.error` suppresses the rule on any earlier error, not only
  evidence from this poll. A tighter guard is `self.error not in ('REFERENCE_UNDERFLOW',
  'REFERENCE_EXPIRED')`. The reviewer could not make a stuck slot starve with it as
  written (a failed CANCEL still recovers). Open item.
- On `a85e169` `selection_expiry_precedence.py` fails only because the hook needs
  `_pending_since`; its real evidence is the mutant and the stall probe (identical to the
  parent: FAULTED `REFERENCE_EXPIRED`, latched, no implicit hold).
- Phase choice (CANCELLED for ADMITTED, FAULTED for SELECTED) is legal and reasonable; a
  single CANCELLED for both is a legal alternative. The terminal receipt is issued before
  the CANCEL/ABORT is acknowledged, as `_cancel_execution` already does. Operator's call.
- `stream_supervisor.py`'s lost-ack revision case holds a candidate pending across several
  SDK calls with an unmeasured margin to 250 ms: the likeliest future flake.

## Repeated review on the designated reviewer (2026-10-01)
Round 1 BLOCK (one blocking finding, otherwise mechanism, phase mapping and benches held): the guard `not
self.error` suppressed the 250 ms rule for ANY error, so a stream pending with an unobserved selection whose
CANCEL was refused (or lost, or that raced a DISARM) stayed `pending` forever: busy, no commands, CMD_LOST, and
DISARM/ARM did not recover it (reproduced 4/4). Not a regression (the parent deadlocks too), but 4b claimed to
bound every stuck-pending case. Fixed: only `REFERENCE_UNDERFLOW` and `REFERENCE_EXPIRED` (RT evidence recorded in the
same poll) take precedence. Second finding (non-blocking, fixed): the 250 ms rule pre-empted an operator cancel that
was waiting for the RT's confirmation, rewriting its reason to "selection unresolved"; when `ex.drop` is set it now
only frees the slot and the receipt (and the operator's reason) stay with `_confirm_drops`. Bench:
`selection_unresolved_cancel.py`. Known and unchanged: `reference_revision=h.revision` is 0 whenever feedback is
unobserved, so that evidence key is uninformative; "commands resume" also depends on the CANCEL ack arriving within
the client's ack timeout (a 600 ms ack still latched CMD_LOST in the reviewer's run); a state stream delivered more
than 250 ms late can trip the rule (safe: an explicit CANCEL follows, and an unobserved reference gets no envelopes
and is retired by the RT at ~100 ms anyway); explicit CANCEL of a still-RUNNING reference is untested.
