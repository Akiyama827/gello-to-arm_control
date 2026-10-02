# group/1 — RT group execution through a supervisor (2026-10-02)

Operator-approved dual-arm migration step (negotiated SDK opt-in, run log 2026-10-02). A supervisor advertises
`extensions['group'] == 1` when its RT server was started with a group record. The SDK then offers `group_join`,
`group_release`, `group_report`, `group_cancel`, `group_events`; `arm_control.multi_arm.supervised_group.SdkGroupCoordinator`
drives two or more supervisors and returns the same `GroupStartRecord` as the direct-link `GroupCoordinator`.

- The supervisor owns the arm's single RT session, so group JOIN/COMMIT_GROUP/CANCEL go through it. A group member is a planned
  `ON_RELEASE` execution; `group_release` marks it so the native reference handle COMMITs through `COMMIT_GROUP` (one shared
  budget) instead of the ordinary COMMIT. The RT's reply (selection and deadline stamps on the RT host clock) or its refusal
  is read with `group_report`.
- Honest difference from the direct coordinator: a supervisor uploads the reference AFTER release, so a peer that has not been
  released has staged nothing and a lone COMMIT_GROUP is refused (`REF_NOT_ADMITTED`) at once, not after the budget; members
  reach COMMIT_GROUP at slightly different times and the RT budget must cover that spread.
- Not a qualification: fake RT only, approved constants as defaults, no unit file may pass `--group-*` until the FR3-host
  tick-gap measurement and the CTRL-042 decision. Bench: `tools/bench/check_dual_group.py`.
