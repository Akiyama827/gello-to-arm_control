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
- Group deadline: the first member to commit sets one absolute deadline on the shared host clock
  (`claim_deadline`), the others adopt it; an expired previous group's deadline is taken over.
  It is checked at COMMIT (refused `REF_LATE` past it or with less than the minimum lead left) and by the
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
writers on one slot with a reader, verdicts, first-committer deadline with threads, cross-process
visibility), `reference_protocol.py`, `tools/bench/rt/group_execution.py` (two fake servers: one
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
