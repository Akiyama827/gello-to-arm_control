# CTRL-040 slice 1: RT-host group record and group commit (2026-10-01)

Design: `Control/docs/ctrl040-design.md` (approved with amendments; timing constants NOT approved).
Scope of this slice: the RT-side mechanism and the PC client ops, proven on two fake servers. The
PC coordinator (`multi_arm/coordinator.py`) is not yet using it.

What was built
- `rt/include/arm_rt/group_record.hpp`: one POSIX shared-memory segment per host, opened, mapped
  (`MAP_POPULATE`) and `mlock`ed before any thread starts. One seqlock slot per member (fields are
  relaxed atomics; a slot has two writers, the control thread and the servo tick, both entering by
  CAS) plus a per-slot heartbeat written by the tick alone. The tick's whole share is bounded: heartbeat
  store, `read_slot` with a fixed 4-try retry, `peer_verdict`, and single-attempt `try_write_slot`
  (a contended write is retried on the next tick). No syscall, allocation or lock in the tick.
  The segment is scoped by `group_id` (PC-assigned, never reused): a slot from another group,
  including a fault left over from an earlier epoch, is `PEER_UNKNOWN`, never faulted or healthy.
- Group deadline: NOT shared state. Each member records its own proposed deadline (commit time +
  budget) in its slot; the group deadline is the earliest deadline of any member committed for the
  group (`group_deadline`), so the first committer sets it and the others converge on it, with no
  shared header, no CAS race and nothing a dead process can leave wedged. (Round 1 of the review
  found the first design, a CAS-claimed header pair, racy: two members committing together while an
  older group held the record got REF_GROUP_REFUSED or adopted the old group's deadline.) The wire
  bounds the budget at 10 s (`GROUP_MAX_BUDGET_NS`, a validity bound, not a timing constant). It is
  checked at COMMIT (refused `REF_LATE` past it or with less than the minimum lead left) and by the
  waiting member; once both members have committed the release is unconditional.
- Wire (additive, v7 negotiated path only): `CTL_GROUP_JOIN/COMMIT/CANCEL` and `CTL_GROUP_STATUS`
  (reply and unsolicited event), results `REF_GROUP_REFUSED`/`REF_GROUP_PEER`, HELLO ` g=1`. C++ and
  Python mirrors, 15 byte-exact goldens and 799 cross-language accept/malformed cases
  (`reference_protocol.py`). First-revision references only.
- Server (`control_session.cpp`, `rt_loop.cpp`, `main.cpp`): off unless `--group-shm --group-member
  --group-members --group-peer-max-age-ms --group-min-lead-ms` are ALL given (and `--enable-reference`);
  there are no default timing values. The control thread holds a group COMMIT until every peer slot
  is COMMITTED-or-later and live, then publishes it to the RT as the existing COMMIT does; the tick
  stamps the selection time into the slot, mirrors the latch (FAULTED), heartbeats, and on a bad peer
  (FAULTED/CANCELLED/STALE/UNKNOWN) retires its running reference (hold at the measured position),
  writes CANCELLED and reports an event. A session end, DISARM or new epoch ends the group.
- Client (`client.py`): `group_join/group_commit/group_cancel`, `group_capable`, `group_events`.

Checks: `group_record_selfcheck` (ctest: setup and member-count refusal, seqlock consistency under two
writers on one slot with a reader, verdicts, group deadline as a function of the slots, cross-process
visibility, conditional writes, recovery of an odd sequence), `reference_protocol.py`, `tools/bench/rt/group_execution.py` (two fake servers: one
deadline and per-member stamps, peer never commits, late commit, cancel after staging, peer process
loss by PID, peer cancel, peer fault then a NEW group, second group / other epoch / wrong member).
The bench's timing values (`--group-peer-max-age-ms 40`, `--group-min-lead-ms 5`, budget 250 ms) are
bench values, not proposals.

Not done / not claimed
- No hardware, no RT host measurement; fake backend only. The selection skew printed by the bench is
  the fake server's, on a non-RT host.
- The peer-loss rule has no tested behaviour on a real dual-FR3; whether holding the survivor is right
  with a rigidly grasped object is the CTRL-042 open question.
- Constants (`PEER_MAX_AGE`, minimum lead, budget/H) are unproposed here; see the measurement note when
  written. Coordinator integration, PC-side group lifecycle and the report of start/progress skew per
  member are the next slice. Only first-revision references can be group-committed.
- Deployed server binary hash changes; staged manifests must record it.

Round-1 review (BLOCK) and fixes
- `grp.is_commit` leaked past every abort path, so the next plain COMMIT was answered on the group
  opcode (the arm moved while the PC saw a refusal); cleared in `group_leave`. Bench: plain COMMIT
  after a group abort (killing the fix hangs the bench).
- An identical group COMMIT retry (client resends after its 2 s ack timeout) got BUSY; now absorbed
  while held and answered from the stored reply after completion. Bench: budget 5 s, peer commits at
  2.6 s (the mutant fails with BUSY).
- Deadline race: see above. Bench: 12 consecutive groups on the same servers, both committing at once.
- Seqlock writer: a release fence after the entering CAS (weakly ordered CPUs); writes carry an
  `allowed_from` state mask, so the tick's RUNNING/DONE/CANCELLED never overwrites a state recorded
  meanwhile and FAULTED is sticky (not replaced by a peer-induced CANCELLED).
- New state `GS_DONE` (reference ran to its end): final, peers ignore its heartbeat, and a member's
  GROUP_CANCEL after DONE only leaves the group (does not retire the peer). Bench: 0.3 s vs 3 s references.
- Group ops on a server without a group record are refused locally (`group_capable`); a killed
  predecessor's odd sequence is recovered at start (`reset_slot`).
- Known and not changed: a peer's slot writer is the non-real-time control thread, so a preempted
  writer or four failed reads in one tick can trip a member spuriously (design "failed read =
  unknown"); a member of a joined group is not prevented from plain COMMITs (design 3.1's reservation
  rule is not implemented); after both members are committed a waiting member can still abort
  REF_LATE within `min_lead` of the deadline (fails safe); a segment left behind by a killed
  process persists in /dev/shm until unlinked; a leaked bench segment from an earlier session was
  seen and not deleted; `GROUP_CANCEL` does not wait for the RT retire, so a BEGIN right after a
  cancel can be refused REF_REVISION (the known CTRL-034 item 4a; the bench waits for the retire).

Round-2 review (BLOCK) and fixes: a stale `REFERENCE_END` of the member's previous reference could set
`group_done` (the DONE write was refused, but any non-CONTENDED result was treated as done), which turned
peer enforcement off for every group that followed a normally finished reference (measured by the reviewer:
peer killed, survivor still running 1 s later). `group_done` is now set only when the DONE write is
WRITTEN. Bench `group_after_finished_reference`. Also: a replayed COMMIT reply is turned into REF_RETIRED if
a cancel has happened since; `group_leave` abandons an in-flight group COMMIT and writes only over a live
state (a concurrent FAULTED/DONE stays); the killed-mid-write recovery is now really simulated.

The bench case for it found a second cause in the same place: the tick's per-group locals (`group_done`, `group_tripped`, ...) were reset only when the tick observed `group_active` false, but a group that ends and a new one that joins within one tick interval is never seen as inactive, so the previous group's `group_done` carried over. The locals are now reset whenever the group id changes.

Round-3 note (ACCEPT): if the ARM epoch is lost after a group COMMIT was released but before the RT selected it, the abandoned COMMIT gets no reply; the PC client falls back on its own session/epoch check or its retry (REF_IDENTITY), so a waiter can stall up to one ack timeout (2 s).
