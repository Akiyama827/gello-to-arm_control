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
  (`REJECTED` if still `ADMITTED`, otherwise `FAULTED`; reason `selection unresolved: ...`),
  marks the attempt `FAULTED` with its selection state left `PENDING` (structurally
  uncertain: the RT may have selected), and retires the reference through the existing
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
