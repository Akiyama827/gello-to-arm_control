# CTRL-033: CANCELLED is written on a confirmed drop (2026-10-01)

Operator check: could a stream execution end CANCELLED without a confirmed drop? Yes.
`_cancel_execution` wrote CANCELLED at the request and queued the CANCEL/ABORT on the
upload worker; a refused or lost CANCEL only set `horizon.error` (bench evidence:
`pending_commit_cancel.py` REF_BUSY case asserted CANCELLED with the CANCEL refused). The
contract text ("cancellation confirmed: the executor dropped it") is unchanged.

Now, for a stream whose reference may be on the RT (`selection_requested`), the execution
stays non-terminal after the cancel request (`_Execution.drop = (reason, reference id)`) and
`RobotRuntime._confirm_drops` writes CANCELLED (receipt evidence `drop_confirmed`) on the
first of: `ack` (CANCEL acknowledged or tolerated `REF_RETIRED`/`REF_REVISION`),
`undispatched` / `refused` (a first-revision COMMIT provably never selected), or
`rt_observed` (the RT's own state shows the reference retired or replaced). A refused CANCEL
with no RT evidence stays non-terminal; the controller's own leg cancel and `_reconcile` no
longer write CANCELLED for such an execution. Executions with no RT reference in play keep
the immediate CANCELLED.

Consequences: `set_payload` and new admissions see the execution as current until confirmed;
`ArmClient.cancel` may time out on an unconfirmed drop (the console adapter reports it).
Benches that asserted immediate CANCELLED now assert not-CANCELLED-yet, then the confirmation
and its `drop_confirmed` value (`controller_extension`, `pending_commit_cancel`); new
`cancel_on_ack.py` (30/30) covers ack and the no-evidence case. `_selection_unresolved` is
unchanged (ADMITTED CANCELLED / SELECTED-or-later FAULTED, operator decision 2026-10-01).
The bench reopens a stream after a BEGIN refusal (open CTRL-034 item 4a).
