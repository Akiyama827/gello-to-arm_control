// CTRL-040 group record: segment setup and prefault, seqlock consistency under a hammering writer,
// scoping by (group, session, epoch), peer verdicts, the first-committer group deadline (threads and
// processes), cross-process visibility. No sockets, no hardware.
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>

#include "arm_rt/group_record.hpp"

using namespace arm_rt;

static std::string shm_name(const char* tag) {
  return "/arm_rt_group_selfcheck_" + std::to_string(getpid()) + "_" + tag;
}

static void check_setup_and_prefault() {
  const std::string name = shm_name("setup");
  GroupSegment a, b, wrong;
  assert(a.open(name, 2, true) && a.mapped() && a.members() == 2);
  assert(b.open(name, 2));         // second server: same segment
  assert(!wrong.open(name, 3));    // a member-count mismatch is refused, nothing mapped
  assert(!wrong.mapped());
  GroupSegment bad;
  assert(!bad.open(shm_name("bad"), 1) && !bad.open(shm_name("bad"), GROUP_MAX_MEMBERS + 1));
  a.write_slot(0, GS_STAGED, 1, 2, 3);
  a.unlink();
  const GroupSnapshot s = b.read_slot(0);
  assert(s.ok && s.state == GS_STAGED && s.group_id == 1 && s.session_id == 2 && s.arm_epoch == 3);
}

static void check_seqlock_consistency() {
  const std::string name = shm_name("seq");
  GroupSegment w, r;
  assert(w.open(name, 2, true) && r.open(name, 2));
  std::atomic<bool> stop{false};
  // Two writers to ONE slot, as in a server (control thread retries until in; the servo tick
  // makes single attempts and skips on contention): a reader still never sees a mixed value.
  std::atomic<uint64_t> skipped{0};
  std::thread writer([&] {
    for (uint64_t k = 1; !stop.load(); k += 2)
      w.write_slot(1, GroupState(1 + k % 6), k, k * 3, uint32_t(k * 7), uint32_t(k * 5), k * 11);
  });
  std::thread ticker([&] {
    for (uint64_t k = 2; !stop.load(); k += 2)
      if (!w.try_write_slot(1, GroupState(1 + k % 6), k, k * 3, uint32_t(k * 7), uint32_t(k * 5), k * 11))
        skipped.fetch_add(1);
  });
  uint64_t ok = 0, torn = 0, unread = 0;
  for (int i = 0; i < 3'000'000; ++i) {
    const GroupSnapshot s = r.read_slot(1);
    if (!s.ok) { ++unread; continue; }
    ++ok;
    if (s.group_id == 0) continue;  // before the first write
    const uint64_t k = s.group_id;
    if (s.session_id != k * 3 || s.arm_epoch != uint32_t(k * 7) || s.fault_code != uint32_t(k * 5) ||
        s.selected_ns != k * 11 || s.state != 1 + k % 6)
      ++torn;
  }
  stop = true;
  writer.join();
  ticker.join();
  r.unlink();
  assert(torn == 0);  // a returned snapshot is never mixed; fixed retries may return not-ok
  assert(ok > 0);
  std::printf("  seqlock: %llu consistent reads, %llu unreadable (fixed retry), 0 torn, %llu tick writes skipped\n",
              (unsigned long long)ok, (unsigned long long)unread, (unsigned long long)skipped.load());
}

static void check_verdicts() {
  GroupSnapshot p;
  p.ok = true; p.state = GS_RUNNING; p.group_id = 9; p.session_id = 4; p.arm_epoch = 2;
  p.heartbeat_ns = 1'000'000;
  const uint64_t age = 20'000'000;
  assert(peer_verdict(p, 9, 1'000'000 + 5'000'000, age) == PEER_OK);
  assert(peer_verdict(p, 9, 1'000'000 + 21'000'000, age) == PEER_STALE);
  assert(peer_verdict(p, 9, 500'000, age) == PEER_OK);   // heartbeat from the "future": not stale
  assert(peer_verdict(p, 8, 2'000'000, age) == PEER_UNKNOWN);  // other group
  // Callers that know the peer's own session/epoch scope by them too.
  assert(peer_verdict(p, 9, 2'000'000, age, 4, 2) == PEER_OK);
  assert(peer_verdict(p, 9, 2'000'000, age, 5, 2) == PEER_UNKNOWN);
  assert(peer_verdict(p, 9, 2'000'000, age, 4, 3) == PEER_UNKNOWN);
  GroupSnapshot f = p; f.state = GS_FAULTED;
  assert(peer_verdict(f, 9, 2'000'000, age) == PEER_FAULTED);
  f.group_id = 3;  // a fault left over from an earlier group says nothing about this one
  assert(peer_verdict(f, 9, 2'000'000, age) == PEER_UNKNOWN);
  GroupSnapshot c = p; c.state = GS_CANCELLED;
  assert(peer_verdict(c, 9, 2'000'000, age) == PEER_CANCELLED);
  GroupSnapshot n = p; n.state = GS_NONE;
  assert(peer_verdict(n, 9, 2'000'000, age) == PEER_UNKNOWN);
  GroupSnapshot z = p; z.heartbeat_ns = 0;
  assert(peer_verdict(z, 9, 2'000'000, age) == PEER_STALE);
  GroupSnapshot bad;  // unreadable
  assert(peer_verdict(bad, 9, 2'000'000, age) == PEER_UNKNOWN);
}

static void check_deadline() {
  const std::string name = shm_name("dl");
  GroupSegment a, b;
  assert(a.open(name, 2, true) && b.open(name, 2));
  const uint64_t d = a.claim_deadline(7, 1'000, 500);
  assert(d == 1'500);
  assert(b.claim_deadline(7, 1'200, 999) == 1'500);  // adopted, never recomputed
  assert(a.claim_deadline(8, 1'300, 500) == 0);      // a live group owns it
  const uint64_t d2 = b.claim_deadline(8, 2'000, 500);  // group 7's deadline is past: taken over
  assert(d2 == 2'500);
  b.release_deadline(7);                              // not the owner: no effect
  assert(a.claim_deadline(7, 2'100, 500) == 0);
  b.release_deadline(8);
  assert(a.claim_deadline(9, 2'100, 500) == 2'600);
  // Two members racing to commit one group agree on one deadline.
  a.release_deadline(9);
  uint64_t r1 = 0, r2 = 0;
  std::thread t1([&] { r1 = a.claim_deadline(11, 5'000, 700); });
  std::thread t2([&] { r2 = b.claim_deadline(11, 5'003, 700); });
  t1.join(); t2.join();
  assert(r1 != 0 && r1 == r2 && (r1 == 5'700 || r1 == 5'703));
  a.unlink();
}

static void check_cross_process() {
  const std::string name = shm_name("proc");
  GroupSegment parent;
  assert(parent.open(name, 2, true));
  const pid_t pid = fork();
  assert(pid >= 0);
  if (pid == 0) {
    GroupSegment child;
    if (!child.open(name, 2)) _exit(2);
    child.write_slot(1, GS_COMMITTED, 42, 5, 6);
    child.heartbeat(1, 123456);
    _exit(0);
  }
  int status = 0;
  assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  const GroupSnapshot s = parent.read_slot(1);
  assert(s.ok && s.state == GS_COMMITTED && s.group_id == 42 && s.session_id == 5 &&
         s.arm_epoch == 6 && s.heartbeat_ns == 123456);
  parent.unlink();
  // The segment outlives its writer: an old-epoch record is still readable, and scoped away.
  assert(peer_verdict(s, 41, 200000, 1'000'000) == PEER_UNKNOWN);
}

int main() {
  check_setup_and_prefault();
  check_seqlock_consistency();
  check_verdicts();
  check_deadline();
  check_cross_process();
  std::puts("group_record_selfcheck: PASS");
  return 0;
}
