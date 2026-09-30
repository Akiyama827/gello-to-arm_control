# CTRL-021 transition slice 1: armed-hold vs executing, and confirmed cancels (2026-10-01)

Scoping: `Control/docs/ctrl021-scoping-for-ctrl040.md`. Findings on the fake RT:

- No new state was needed for one arm: `lifecycle_states.py` pins which existing signals
  distinguish armed-hold (ARMED, `holding` True, RT reference id 0), executing (RT reference
  RUNNING, `holding` False), post-cancel (a newer measured-hold reference; `holding` was seen
  False there, so it is not a reliable armed-hold marker; read the reference id/status),
  disarmed (CLAIMED, `arm_epoch` 0, RT not armed) and re-armed (new epoch, reference 0).
  CTRL-040's group record must therefore derive "held, not moving" from the RT reference
  state, not from `holding`.
- The cancel receipt gap was real for planned (non-stream) native executions too: the
  controller's own leg cancel produced CANCELLED without the RT's confirmation. Now the same
  rule as for streams (`cancel_on_ack.py`, see `2026-10-01-ctrl033-cancel-on-ack.md`): CANCELLED
  only on a confirmed drop; a refused CANCEL of a planned execution surfaces as the existing
  FAULTED "reference operation refused", never as CANCELLED.
- Not done: no enum or wire field was added for "armed-hold"; nothing would consume it before
  CTRL-040's group record exists.

Bench fallout, recorded as measured: the first full suite on `e8cfd00` was 24 PASS / 1 FAIL
(`controller_extension` line 70 asserted CANCELLED immediately after a preset change; standalone
2/10 failed). `reference_supervisor` had three such immediate asserts and failed 9/15 standalone.
All four now wait for the CANCELLED that follows the drop's confirmation (the rule changed on
purpose; the tests' assumption did not hold). After the fix: `reference_supervisor` 25/25,
`controller_extension` 15/15, `console_adapter`/`selection_unresolved`/`stream_supervisor`/
`reference_horizon`/`stream_races` 15/15 each.
