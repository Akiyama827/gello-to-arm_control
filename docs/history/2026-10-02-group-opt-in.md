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
- **Group cancel and the CMD_LOST latch (diagnosed in review, PC-side).** Cancelling running group members used to latch
  CMD_LOST on an arm (about 1 cycle in 10). Cause: the link's `group_cancel` set the interrupt flag it shares with every
  upload and COMMIT, so it interrupted the member's own measured-hold COMMIT (the RT had already selected that hold); the
  supervisor then read the interrupted exchange as a native failure and stopped commanding, and the RT latched after
  `--fault-ms`. The RT's GROUP_CANCEL also retires whatever reference the member has selected, including a standing hold
  (an RT-side scoping question for CTRL-042; the PC works around it, see below). Now: (1) the link's `group_cancel`
  interrupts only a COMMIT_GROUP wait that is actually in flight; (2) the supervisor's `group_cancel` cancels the members'
  executions itself, holds off staging a measured hold, waits for its member-cancel jobs to leave the link worker, sends the
  RT CANCEL, then re-enables the hold; (3) a hold retired by that RT CANCEL is re-staged during a 1 s grace instead of ending
  the command stream; (4) `SdkGroupCoordinator.cancel` cancels the group on every member in parallel via `group_cancel`
  (not execution-first); a member may end FAULTED instead of CANCELLED when the RT's peer rule retires it first. The bench
  asserts no fault for longer than `--fault-ms` after a cancel of running members and after every failed start. Survivors of
  mutation testing (defence in depth, no bench case reaches them): the 'interrupt only if committing' flag and the
  staging hold-off during the RT CANCEL.
  **Survivor latch (deterministic, measured).** Through the supervisor the surviving member is LATCHED, not held, whenever
  the peer retires, faults or is lost, when only one member's `group_cancel` reaches the RT, or when the members' cancels
  are more than about 25 ms apart (the RT retires the peer, and the survivor's command stream ends). Measured 20/20 on
  this tip, 10/10 on `e1fead7`, 10/10 on `991429a`: not a regression. Cancel skew 10 ms: 0/15 latch; 25 ms: 14/15; 50 ms:
  14/15. Recovery is DISARM then ARM. This deviates from the ctrl040-design section 9 policy of holding the survivor;
  CTRL-042 evaluates it. The bench's no-fault assertion covers only the coordinator's near-simultaneous parallel cancel.
  Cancelling a committing group waits up to the budget (at most 1 s) for the RT's answer. The re-stage grace is armed when
  the supervisor takes the group, before the RT exchange.
  What else this slice fixes: a refused JOIN restores the previous group (it no longer wipes a group the RT is still joined
  to, and no released member falls back to a plain COMMIT); joining a second group while one is recorded is refused (cancel
  the first); a cancel the RT did not confirm can be retried; an uncertain
  JOIN stays recorded so a cancel still reaches the RT; the supervisor's and the RT servers' output goes to files in the
  bench (a full pipe blocks a child). The group RPCs may block their connection thread up to the link's acknowledgement
  timeouts (longer than the service's 1 s wait bound). A failed start with a peer that never joined can latch the
  first arm (a lone COMMIT_GROUP whose peer is unknown): recovery is DISARM then ARM; not claimed otherwise.
- Not a qualification: fake RT only, approved constants as defaults, no unit file may pass `--group-*` until the FR3-host
  tick-gap measurement and the CTRL-042 decision. Bench: `tools/bench/check_dual_group.py`.
