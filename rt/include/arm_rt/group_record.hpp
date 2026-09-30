// CTRL-040: per-arm group records shared by the RT servers of ONE host.
//
// Design: Control/docs/ctrl040-design.md (sections 3.3, 3.4) and
// Control/docs/ctrl021-slices-2-3-audit.md. One POSIX shared-memory segment holds a small header
// and one slot per member. Each server writes only its own slot; a peer's servo tick READS it.
//
// What the servo tick may do (and only this): `read_slot` (a seqlock read with a FIXED retry count),
// `peer_verdict`, and the single relaxed store of its own heartbeat. No syscall, allocation, lock
// or unbounded loop. The segment is created/opened, mapped, prefaulted (MAP_POPULATE) and
// mlock'ed by `GroupSegment::open` BEFORE the RT loop starts; nothing in the tick can page-fault
// it in for the first time. Writers of the other operations (state changes, the group deadline)
// run on the non-RT control thread.
//
// Scope: every slot is valid only for the group_id it was written under (PC-assigned, never
// reused), with the writer's own session/arm epoch recorded. A reader that sees another group gets
// PEER_UNKNOWN, never "healthy" and never "faulted".
// The record mirrors the wire latch, not the fault journal (audit note): a member sets FAULTED when
// its own latch is set and clears it exactly where the latch is cleared.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

namespace arm_rt {

constexpr uint32_t GROUP_MAGIC = 0x31505247;  // "GRP1"
constexpr uint32_t GROUP_MAX_MEMBERS = 4;
constexpr int GROUP_READ_TRIES = 4;  // fixed seqlock retry count in the servo tick

enum GroupState : uint32_t {
  GS_NONE = 0,       // slot unused / cleared
  GS_STAGED = 1,     // PREPARE accepted, upload sealed
  GS_READY = 2,      // horizon preloaded, waiting for the group commit
  GS_COMMITTED = 3,  // COMMIT_GROUP accepted before the group deadline
  GS_RUNNING = 4,    // this member's reference is selected
  GS_FAULTED = 5,    // this member's wire latch is set
  GS_CANCELLED = 6,  // group cancelled by this member (or by the PC)
};

// One value the seqlock protects. All fields are relaxed atomics so the concurrent read is
// race-free by the language rules; the seqlock only provides the consistent snapshot.
struct alignas(64) GroupSlot {
  std::atomic<uint32_t> seq{0};  // odd while a writer is inside
  std::atomic<uint32_t> state{GS_NONE};
  std::atomic<uint32_t> arm_epoch{0};
  std::atomic<uint32_t> fault_code{0};
  std::atomic<uint64_t> group_id{0};
  std::atomic<uint64_t> session_id{0};
  std::atomic<uint64_t> selected_ns{0};  // server clock at this member's selection (0 = not yet)
  // Written by the servo tick alone, read by peers: a lone 64-bit value needs no seqlock.
  alignas(64) std::atomic<uint64_t> heartbeat_ns{0};
};

struct GroupHeader {
  std::atomic<uint32_t> magic{0};
  std::atomic<uint32_t> members{0};
  // Group deadline on the shared host clock: set once per group by the FIRST member to commit
  // (claim_deadline), adopted by the others. group id is stored before the deadline, so
  // deadline_ns == 0 with a matching id means "being set".
  alignas(64) std::atomic<uint64_t> deadline_group_id{0};
  std::atomic<uint64_t> deadline_ns{0};
};

struct GroupLayout {
  GroupHeader header;
  GroupSlot slots[GROUP_MAX_MEMBERS];
};

// A consistent copy of one slot, or nothing.
struct GroupSnapshot {
  bool ok = false;
  uint32_t state = GS_NONE, arm_epoch = 0, fault_code = 0;
  uint64_t group_id = 0, session_id = 0, selected_ns = 0, heartbeat_ns = 0;
};

enum PeerVerdict : uint32_t {
  PEER_OK = 0,
  PEER_UNKNOWN = 1,    // unreadable, other session/epoch/group, or not yet written
  PEER_FAULTED = 2,
  PEER_CANCELLED = 3,
  PEER_STALE = 4,      // heartbeat older than the maximum age
};

class GroupSegment {
 public:
  GroupSegment() = default;
  GroupSegment(const GroupSegment&) = delete;
  GroupSegment& operator=(const GroupSegment&) = delete;
  ~GroupSegment() { close_segment(); }

  // Non-RT setup. `name` is a POSIX shm name ("/arm_rt_group_x"). Maps, prefaults and locks the
  // pages; returns false (and maps nothing) on any failure. `create_if_missing` initialises the
  // header of a segment nobody has yet (several servers may race; O_EXCL decides the creator).
  bool open(const std::string& name, uint32_t members, bool unlink_first = false) {
    close_segment();
    if (members < 2 || members > GROUP_MAX_MEMBERS) return false;
    if (unlink_first) shm_unlink(name.c_str());
    bool created = true;
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
      created = false;
      fd = shm_open(name.c_str(), O_RDWR, 0600);
      if (fd < 0) return false;
    }
    if (created && ftruncate(fd, off_t(sizeof(GroupLayout))) != 0) {
      close(fd);
      shm_unlink(name.c_str());
      return false;
    }
    struct stat st{};
    // A non-creator may arrive before the creator's ftruncate: wait (setup, not RT) for the size.
    for (int i = 0; i < 200 && (fstat(fd, &st) != 0 || size_t(st.st_size) < sizeof(GroupLayout)); ++i)
      usleep(5000);
    if (size_t(st.st_size) < sizeof(GroupLayout)) { close(fd); return false; }
    void* p = mmap(nullptr, sizeof(GroupLayout), PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return false;
    if (mlock(p, sizeof(GroupLayout)) != 0) { /* best effort: MAP_POPULATE already faulted it in */ }
    layout_ = static_cast<GroupLayout*>(p);
    name_ = name;
    if (created) {
      layout_->header.members.store(members, std::memory_order_relaxed);
      layout_->header.magic.store(GROUP_MAGIC, std::memory_order_release);
    } else {
      for (int i = 0; i < 400 && layout_->header.magic.load(std::memory_order_acquire) != GROUP_MAGIC; ++i)
        usleep(5000);
      if (layout_->header.magic.load(std::memory_order_acquire) != GROUP_MAGIC ||
          layout_->header.members.load(std::memory_order_relaxed) != members) {
        close_segment();
        return false;
      }
    }
    members_ = members;
    return true;
  }

  bool mapped() const { return layout_ != nullptr; }
  uint32_t members() const { return members_; }
  void unlink() const { if (!name_.empty()) shm_unlink(name_.c_str()); }

  // ---- control-thread writers (non-RT) ------------------------------------------------------
  // One consistent replacement of a member's slot value (the heartbeat is separate). TWO threads
  // of one server write its slot (control thread, servo tick), so the seqlock is entered by CAS:
  // `try_write_slot` makes exactly one attempt (the tick: on false it retries next tick, never
  // spins); `write_slot` retries until it gets in (control thread only).
  bool try_write_slot(uint32_t member, GroupState state, uint64_t group_id, uint64_t session_id,
                      uint32_t arm_epoch, uint32_t fault_code = 0, uint64_t selected_ns = 0) {
    GroupSlot& s = layout_->slots[member];
    uint32_t q = s.seq.load(std::memory_order_relaxed);
    if ((q & 1u) || !s.seq.compare_exchange_strong(q, q + 1, std::memory_order_acquire,
                                                   std::memory_order_relaxed))
      return false;
    s.state.store(state, std::memory_order_relaxed);
    s.arm_epoch.store(arm_epoch, std::memory_order_relaxed);
    s.fault_code.store(fault_code, std::memory_order_relaxed);
    s.group_id.store(group_id, std::memory_order_relaxed);
    s.session_id.store(session_id, std::memory_order_relaxed);
    s.selected_ns.store(selected_ns, std::memory_order_relaxed);
    s.seq.store(q + 2, std::memory_order_release);
    return true;
  }
  void write_slot(uint32_t member, GroupState state, uint64_t group_id, uint64_t session_id,
                  uint32_t arm_epoch, uint32_t fault_code = 0, uint64_t selected_ns = 0) {
    while (!try_write_slot(member, state, group_id, session_id, arm_epoch, fault_code, selected_ns))
      std::atomic_signal_fence(std::memory_order_seq_cst);
  }
  void clear_slot(uint32_t member) { write_slot(member, GS_NONE, 0, 0, 0); }

  // First member to commit sets the group deadline (now + budget); later members adopt it.
  // Returns 0 = refused (another live group holds the record, or it could not be read),
  // otherwise the group's deadline in host-clock ns. A previous group whose deadline lies in
  // the past is taken over.
  uint64_t claim_deadline(uint64_t group_id, uint64_t now_ns, uint64_t budget_ns) {
    GroupHeader& h = layout_->header;
    for (int attempt = 0; attempt < 8; ++attempt) {
      uint64_t owner = h.deadline_group_id.load(std::memory_order_acquire);
      if (owner == group_id) {
        for (int i = 0; i < 2000; ++i) {  // the setter is between two stores: microseconds
          const uint64_t d = h.deadline_ns.load(std::memory_order_acquire);
          if (d) return d;
          usleep(1);
        }
        return 0;
      }
      if (owner != 0) {
        const uint64_t d = h.deadline_ns.load(std::memory_order_acquire);
        if (d == 0 || now_ns < d) return 0;  // a live group owns it
      }
      // Free (or expired): take it, publishing an unset deadline first.
      if (h.deadline_group_id.compare_exchange_strong(owner, group_id, std::memory_order_acq_rel)) {
        h.deadline_ns.store(0, std::memory_order_release);
        const uint64_t deadline = now_ns + budget_ns;
        h.deadline_ns.store(deadline, std::memory_order_release);
        return deadline;
      }
    }
    return 0;
  }
  void release_deadline(uint64_t group_id) {
    GroupHeader& h = layout_->header;
    uint64_t owner = group_id;
    if (h.deadline_group_id.compare_exchange_strong(owner, 0, std::memory_order_acq_rel))
      h.deadline_ns.store(0, std::memory_order_release);
  }

  // ---- servo-tick operations (bounded, no syscalls) -----------------------------------------
  void heartbeat(uint32_t member, uint64_t now_ns) {
    layout_->slots[member].heartbeat_ns.store(now_ns, std::memory_order_relaxed);
  }

  GroupSnapshot read_slot(uint32_t member) const {
    GroupSnapshot out;
    const GroupSlot& s = layout_->slots[member];
    for (int tries = 0; tries < GROUP_READ_TRIES; ++tries) {
      const uint32_t a = s.seq.load(std::memory_order_acquire);
      if (a & 1u) continue;  // a writer is inside
      out.state = s.state.load(std::memory_order_relaxed);
      out.arm_epoch = s.arm_epoch.load(std::memory_order_relaxed);
      out.fault_code = s.fault_code.load(std::memory_order_relaxed);
      out.group_id = s.group_id.load(std::memory_order_relaxed);
      out.session_id = s.session_id.load(std::memory_order_relaxed);
      out.selected_ns = s.selected_ns.load(std::memory_order_relaxed);
      out.heartbeat_ns = s.heartbeat_ns.load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      if (s.seq.load(std::memory_order_relaxed) == a) {
        out.ok = true;
        return out;
      }
    }
    out.ok = false;
    return out;
  }

 private:
  void close_segment() {
    if (layout_) munmap(layout_, sizeof(GroupLayout));
    layout_ = nullptr;
    members_ = 0;
  }
  GroupLayout* layout_ = nullptr;
  uint32_t members_ = 0;
  std::string name_;
};

// What a member concludes about ONE peer slot. Pure. The group id (PC-assigned, never reused) is
// the scope: a slot from another group, including a fault left over from an earlier epoch, is
// PEER_UNKNOWN, never healthy and never faulted. Members are separate servers with their own
// session ids and ARM epochs, so those are checked only when the caller knows the peer's values
// (`expect_session`/`expect_epoch` != 0); each server also ends its own slot when its epoch ends.
inline PeerVerdict peer_verdict(const GroupSnapshot& peer, uint64_t group_id, uint64_t now_ns,
                                uint64_t max_age_ns, uint64_t expect_session = 0,
                                uint32_t expect_epoch = 0) {
  if (!peer.ok) return PEER_UNKNOWN;
  if (peer.group_id != group_id) return PEER_UNKNOWN;
  if ((expect_session && peer.session_id != expect_session) ||
      (expect_epoch && peer.arm_epoch != expect_epoch))
    return PEER_UNKNOWN;
  if (peer.state == GS_FAULTED) return PEER_FAULTED;
  if (peer.state == GS_CANCELLED) return PEER_CANCELLED;
  if (peer.state == GS_NONE) return PEER_UNKNOWN;
  if (peer.heartbeat_ns == 0) return PEER_STALE;
  // A heartbeat "from the future" is the host clock read a moment apart: not stale.
  if (now_ns < peer.heartbeat_ns) return PEER_OK;
  return now_ns - peer.heartbeat_ns > max_age_ns ? PEER_STALE : PEER_OK;
}

}  // namespace arm_rt
