# group/1 — RT group execution through a supervisor (2026-10-02)

Operator-approved dual-arm migration step (negotiated SDK opt-in, run log 2026-10-02). A supervisor advertises
`extensions['group'] == 1` when its RT server was started with a group record. The SDK then offers `group_join`,
`group_release`, `group_report`, `group_cancel`, `group_events`; `arm_control.multi_arm.supervised_group.SdkGroupCoordinator`
drives two or more supervisors and returns the same `GroupStartRecord` as the direct-link `GroupCoordinator`.

- The supervisor owns the arm's single RT session, so group JOIN/COMMIT_GROUP/CANCEL go through it. A group member is a planned
  `ON_RELEASE` execution; `group_release` marks it so the native reference handle COMMITs through `COMMIT_GROUP` (one shared
  budget) instead of the ordinary COMMIT. The RT's reply (selection and deadline stamps on the RT host clock) or its refusal
  is read with `group_report`.
- Honest difference from the direct coordinator: a supervisor uploads the reference AFTER release, so members reach
  COMMIT_GROUP at slightly different times and the RT budget must cover that spread. A peer that has joined but not yet
  committed holds the lone COMMIT_GROUP to the group deadline: the lone member fails after about the budget (measured
  ~0.13 s): usually `REF_LATE`, sometimes the peer verdict or the RT retiring the member (the link reports the interrupted
  exchange); never a selection. This is the normal failure path through supervisors. `pc_send_skew_sec` is the release RPC skew, not the COMMIT send
  skew. The group RPCs run on the caller's thread, not the 100 Hz loop; a group joined in one ARM epoch cannot be released in
  another; the supervisor refuses a budget above half the RT acknowledgement timeout. The coordinator lives in
  `arm_control/multi_arm/supervised_group.py` (the decision text said `arm_control/clients`; it sits beside slice 1's handle).
- After a cancel, a new group is refused at JOIN (GROUP_REFUSED) or at BEGIN (REF_BUSY, the open CTRL-034 item 4a race) until
  the RT has retired the running members and their peers (about PEER_MAX_AGE plus a tick): a safe refusal the caller
  retries after a short settle; the bench waits 0.4 s.
- **OPEN: cancelling RUNNING group members can latch an arm.** The RT retires the peer's reference; the supervisor's
  measured-hold BEGIN is then refused while the RT still holds the old reference (the open CTRL-034 item 4a race), so that
  arm's PC command stream is silent and the RT latches CMD_LOST after `--fault-ms`. Measured in the bench about 1 run in 5
  (always the second arm); recovery is DISARM then ARM (the existing hold→latch ladder). This is a survivor-semantics /
  RT-side question (CTRL-042), not fixed here, and it makes a routine group cancel on hardware unacceptable until decided.
  What this slice does fix: a group start that fails with no running member (a lone refused COMMIT_GROUP) latches nothing (the
  bench asserts no fault for longer than `--fault-ms`); `group_cancel` cancels the executions released for the group, so a
  member cannot fall back to a plain COMMIT and start alone; a cancel the RT did not confirm can be retried; a join whose
  outcome is uncertain stays recorded so a cancel still reaches the RT; a hold upload disturbed by a group cancel is
  re-staged, not treated as a native failure. The group RPCs may block their connection thread up to the link's
  acknowledgement timeouts (longer than the service's 1 s wait bound).
- Not a qualification: fake RT only, approved constants as defaults, no unit file may pass `--group-*` until the FR3-host
  tick-gap measurement and the CTRL-042 decision. Bench: `tools/bench/check_dual_group.py`.
