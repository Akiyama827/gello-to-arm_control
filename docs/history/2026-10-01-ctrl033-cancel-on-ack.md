# CTRL-033: CANCELLED is written on a confirmed drop (2026-10-01)

Operator check: could a stream execution end CANCELLED without a confirmed drop? Yes.
`_cancel_execution` wrote CANCELLED at the request and queued the CANCEL/ABORT on the
upload worker; a refused or lost CANCEL only set `horizon.error` (bench evidence:
`pending_commit_cancel.py` REF_BUSY case asserted CANCELLED with the CANCEL refused). The
contract text ("cancellation confirmed: the executor dropped it") is unchanged.

Now, for a stream whose reference may be on the RT (`selection_requested`), the execution
stays non-terminal after the cancel request (`_Execution.drop = (reason, reference id)`) and
`RobotRuntime._confirm_drops` writes CANCELLED (receipt evidence `drop_confirmed`) on the
first of: `ack` (CANCEL acknowledged, or refused `REF_REVISION` which the retire tolerates;
`REF_RETIRED` on a dispatched CANCEL is not tolerated and stays unconfirmed until RT evidence),
`undispatched` / `refused` (a first-revision COMMIT provably never selected), or
`rt_observed` (the RT's own state shows the reference retired or replaced), `rt_epoch` (session
or ARM epoch changed: every RT reference dies with it) or `rt_disarmed`. Round-1 review found
that without the epoch rule a lost CANCEL plus a missed state plus DISARM/ARM left the
execution non-terminal forever (set_payload/admission blocked); fixed and benched. A proof
(`undispatched`/`refused`/`ack`) is never overwritten by a later `unconfirmed`; `ex.drop` is
cleared when the execution ends any other way. A refused CANCEL
with no RT evidence stays non-terminal; the controller's own leg cancel and `_reconcile` no
longer write CANCELLED for such an execution. Executions with no RT reference in play keep
the immediate CANCELLED.

Consequences: `set_payload` and new admissions see the execution as current until confirmed;
`ArmClient.cancel` may time out on an unconfirmed drop (the console adapter reports it).
Benches that asserted immediate CANCELLED now assert not-CANCELLED-yet, then the confirmation
(`controller_extension` waits for CANCELLED; `pending_commit_cancel` also asserts the
`drop_confirmed` value); new `cancel_on_ack.py` covers ack, the no-evidence case (stays
non-terminal), the epoch change, and proof-not-downgraded. `rt_observed` after a refused CANCEL
may be caused by the envelope deadman or by the mode-change retire; the bench does not tell
them apart. `_selection_unresolved` is
unchanged (ADMITTED CANCELLED / SELECTED-or-later FAULTED, operator decision 2026-10-01) and is
the **one explicit exception** to "CANCELLED means confirmed": an ADMITTED execution whose
COMMIT worker is stuck can be CANCELLED before the CANCEL is sent (the RT may hold it; CANCEL is
queued and its outcome is not awaited).
The bench reopens a stream after a BEGIN refusal (open CTRL-034 item 4a).
