// Fault journal (server.hpp): first cause, secondary causes, explicit
// truncation, full ring, stalled consumer and concurrent producers. No
// sockets, no backend, no devices.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#include "server.hpp"

namespace arm_rt {
uint64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000ull + uint64_t(ts.tv_nsec);
}
}  // namespace arm_rt

namespace {
using namespace arm_rt;

std::string drain(ServerCtx& ctx, int* count = nullptr) {
  char* buf = nullptr;
  size_t len = 0;
  FILE* out = open_memstream(&buf, &len);
  const int n = drain_fault_journal(ctx, out);
  std::fclose(out);
  std::string text(buf, len);
  free(buf);
  if (count) *count = n;
  return text;
}

void disarm(ServerCtx& ctx) {  // what CTL_DISARM does to the latch
  ctx.armed.store(false);
  assert(ctx.clear_latch());
}

void check_causes() {
  ServerCtx ctx;
  ctx.armed.store(true);
  ctx.latch(FAULT_PLANT, "read: non-libfranka exception: Net Exception", ctx.arm_gen.load());
  for (int i = 0; i < 1000; ++i)  // a persisting cause re-latches every tick
    ctx.latch(FAULT_PLANT, "read: non-libfranka exception: Net Exception", ctx.arm_gen.load());
  ctx.latch(FAULT_CTL_LOST, "control session lost while armed", ctx.arm_gen.load());
  ctx.latch(FAULT_CTL_LOST, "control session lost while armed", ctx.arm_gen.load());
  assert(ctx.fault_code == FAULT_PLANT);  // wire latch keeps the FIRST cause
  disarm(ctx);  // GUI clears the latch before anything drained
  assert(ctx.fault_text[0] == '\0');
  int n = 0;
  const std::string log = drain(ctx, &n);
  assert(n == 2);
  assert(log.find("first cause code=3") != std::string::npos);
  assert(log.find("text=\"read: non-libfranka exception: Net Exception\"") !=
         std::string::npos);
  assert(log.find("secondary cause code=2") != std::string::npos);
  assert(log.find("armed=1") != std::string::npos);
  assert(drain(ctx, &n).empty() && n == 0);  // consumed exactly once

  // After DISARM a new latch is a new first cause; secondary codes reset.
  ctx.latch(FAULT_CMD_LOST, "command stream lost (staleness > fault-ms)", ctx.arm_gen.load());
  ctx.latch(FAULT_CTL_LOST, "control session lost while armed", ctx.arm_gen.load());
  const std::string again = drain(ctx, &n);
  assert(n == 2 && again.find("first cause code=1") != std::string::npos);
  assert(again.find("secondary cause code=2") != std::string::npos);
}

void check_truncation() {
  ServerCtx ctx;
  const std::string medium(150, 'm');  // fits the journal, not the wire
  ctx.latch(FAULT_PLANT, medium.c_str(), ctx.arm_gen.load());
  assert(std::strlen(ctx.fault_text) == sizeof(ctx.fault_text) - 1);
  assert(std::strcmp(ctx.fault_text + sizeof(ctx.fault_text) - 4, "...") == 0);
  std::string log = drain(ctx);
  assert(log.find(medium + "\"\n") != std::string::npos);  // full text kept
  assert(log.find("(truncated)") == std::string::npos);
  disarm(ctx);
  const std::string huge(5000, 'h');
  ctx.latch(FAULT_PLANT, huge.c_str(), ctx.arm_gen.load());
  log = drain(ctx);
  assert(log.find(std::string(FAULT_RECORD_TEXT - 1, 'h') + "\" (truncated)") !=
         std::string::npos);
  disarm(ctx);
  const std::string exact(sizeof(ctx.fault_text) - 1, 'e');  // fits exactly
  ctx.latch(FAULT_PLANT, exact.c_str(), ctx.arm_gen.load());
  assert(exact == ctx.fault_text);
}

// Bounded text work, and an uncontended claim costs no contention count.
// Claim exhaustion under contention is exercised by check_concurrent, which
// accounts every record as delivered, dropped (full) or contended.
void check_bounded_claim() {
  ServerCtx ctx;
  const char* text = "x";
  // bounded_copy scans at most cap bytes, even for longer input.
  char small[8];
  assert(bounded_copy(small, sizeof small, "0123456789") && std::strcmp(small, "0123456") == 0);
  assert(!bounded_copy(small, sizeof small, "0123456") && std::strcmp(small, "0123456") == 0);
  ctx.journal(FAULT_SECONDARY, 1, text, ctx.arm_gen.load());
  assert(ctx.fault_head == 1 && ctx.fault_contended == 0);
}

void check_full_ring() {
  ServerCtx ctx;  // stalled consumer: producers never wait, never overwrite
  for (uint32_t i = 0; i < FAULT_RING + 5; ++i) {
    const std::string text = "cause " + std::to_string(i);
    ctx.journal(FAULT_STOP_FAILED, FAULT_PLANT, text.c_str(), ctx.arm_gen.load());
  }
  assert(ctx.fault_dropped == 5);
  int n = 0;
  const std::string log = drain(ctx, &n);
  assert(n == int(FAULT_RING));
  assert(log.find("text=\"cause 0\"") != std::string::npos);  // oldest kept
  assert(log.find("text=\"cause 15\"") != std::string::npos);
  assert(log.find("text=\"cause 16\"") == std::string::npos);  // new dropped
  assert(log.find("stop failed") != std::string::npos);
  ctx.journal(FAULT_STOP_FAILED, FAULT_PLANT, "after drain", ctx.arm_gen.load());
  assert(drain(ctx).find("after drain") != std::string::npos);
}

// Two producers (RT loop + control thread) against one draining consumer:
// every record is either delivered intact exactly once or counted dropped.
void check_concurrent() {
  ServerCtx ctx;
  constexpr int PER = 20000;
  std::atomic<int> started{0};
  std::atomic<bool> done{false};
  auto producer = [&](char tag) {
    ++started;
    while (started.load() < 3) {}
    for (int i = 0; i < PER; ++i) {
      char text[32];
      std::snprintf(text, sizeof text, "%c%06d%c", tag, i, tag);
      ctx.journal(FAULT_SECONDARY, uint32_t(tag), text, ctx.arm_gen.load());
    }
  };
  std::vector<std::string> got;
  std::thread consumer([&] {
    ++started;
    while (started.load() < 3) {}
    for (;;) {
      const bool finished = done.load();
      char* buf = nullptr;
      size_t len = 0;
      FILE* out = open_memstream(&buf, &len);
      drain_fault_journal(ctx, out);
      std::fclose(out);
      got.emplace_back(buf, len);
      free(buf);
      if (finished) break;
    }
  });
  std::thread a(producer, 'a'), b(producer, 'b');
  a.join();
  b.join();
  done.store(true);
  consumer.join();
  int delivered[2] = {0, 0};
  int last[2] = {-1, -1};
  for (const std::string& chunk : got) {
    for (size_t at = chunk.find("text=\""); at != std::string::npos;
         at = chunk.find("text=\"", at + 1)) {
      const char* t = chunk.c_str() + at + 6;
      const int who = t[0] == 'a' ? 0 : 1;
      assert((t[0] == 'a' || t[0] == 'b') && t[7] == t[0] && t[8] == '"');  // intact
      const int i = std::atoi(std::string(t + 1, 6).c_str());
      assert(i > last[who]);  // per-producer order, no duplicates
      last[who] = i;
      ++delivered[who];
    }
  }
  // Every record is delivered or accounted: ring full or claim contention.
  assert(delivered[0] + delivered[1] + int(ctx.fault_dropped.load()) +
             int(ctx.fault_contended.load()) == 2 * PER);
  assert(delivered[0] > 0 && delivered[1] > 0);
  std::printf("concurrent: delivered %d + %d, dropped %u, contended %u\n", delivered[0],
              delivered[1], ctx.fault_dropped.load(), ctx.fault_contended.load());
}

// R3 (review 2026-09-24): a latch writer that already won the claim is paused
// before it writes; DISARM must not clear underneath it, a second cause is a
// secondary record, and after the clear an immediate re-arm latches a new first
// cause without the old one being reassigned to the new epoch.
struct Gate {
  std::mutex m;
  std::condition_variable cv;
  bool in = false, go = false;
};
void hold_after_claim(void* arg) {
  auto* g = static_cast<Gate*>(arg);
  std::unique_lock<std::mutex> lock(g->m);
  g->in = true;
  g->cv.notify_all();
  g->cv.wait(lock, [g] { return g->go; });
}

std::string line_with(const std::string& log, const std::string& key) {
  const size_t at = log.find(key);
  if (at == std::string::npos) return {};
  const size_t start = log.rfind('\n', at);
  return log.substr(start == std::string::npos ? 0 : start + 1,
                    log.find('\n', at) - (start == std::string::npos ? 0 : start + 1));
}

void check_latch_disarm_race() {
  ServerCtx ctx;
  ctx.armed.store(true);
  ctx.arm_gen.store(1);
  Gate gate;
  ctx.latch_seam = hold_after_claim;
  ctx.latch_seam_arg = &gate;
  std::thread rt([&] { ctx.latch(FAULT_PLANT, "first cause, epoch 1", 1); });
  {
    std::unique_lock<std::mutex> lock(gate.m);
    gate.cv.wait(lock, [&] { return gate.in; });
  }
  ctx.latch_seam = nullptr;
  ctx.latch(FAULT_CTL_LOST, "second cause while the first is written", ctx.arm_gen.load());
  std::atomic<bool> done{false};
  bool cleared = false;
  std::thread control([&] {  // CTL_DISARM
    ctx.armed.store(false);
    cleared = ctx.clear_latch();
    done.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  assert(!done.load() && ctx.latch_state.load() == ServerCtx::LATCH_WRITING);
  {
    std::lock_guard<std::mutex> lock(gate.m);
    gate.go = true;
  }
  gate.cv.notify_all();
  rt.join();
  control.join();
  assert(cleared && !ctx.fault.load() && ctx.fault_code.load() == 0);
  assert(ctx.fault_text[0] == '\0' && ctx.latch_state.load() == ServerCtx::LATCH_FREE);

  ctx.arm_gen.store(2);  // immediate re-arm
  ctx.armed.store(true);
  ctx.latch(FAULT_CMD_LOST, "epoch 2 cause", 2);
  assert(ctx.fault.load() && ctx.fault_code.load() == FAULT_CMD_LOST);
  assert(std::strcmp(ctx.fault_text, "epoch 2 cause") == 0);
  const std::string log = drain(ctx);
  const std::string first = line_with(log, "first cause code=3");
  assert(first.find("arm_epoch=1") != std::string::npos &&
         first.find("text=\"first cause, epoch 1\"") != std::string::npos);
  assert(line_with(log, "secondary cause code=2").find("arm_epoch=1") != std::string::npos);
  const std::string second = line_with(log, "first cause code=1");
  assert(second.find("arm_epoch=2") != std::string::npos &&
         second.find("text=\"epoch 2 cause\"") != std::string::npos);
}

// A writer that never finishes: DISARM gives up after its bound and leaves
// the latch intact rather than clearing underneath it.
void check_clear_bounded() {
  ServerCtx ctx;
  ctx.latch_state.store(ServerCtx::LATCH_WRITING);
  const auto t0 = std::chrono::steady_clock::now();
  assert(!ctx.clear_latch());
  const auto waited = std::chrono::steady_clock::now() - t0;
  assert(waited >= std::chrono::milliseconds(90) && waited < std::chrono::seconds(2));
  assert(ctx.latch_state.load() == ServerCtx::LATCH_WRITING);
}

// The actual interaction under load (run it under TSan too): the servo
// latching every tick while DISARM clears. The end state is coherent.
void check_latch_stress() {
  ServerCtx ctx;
  ctx.armed.store(true);
  std::atomic<bool> stop{false};
  std::thread rt([&] {
    while (!stop.load()) ctx.latch(FAULT_PLANT, "persisting cause", ctx.arm_gen.load());
  });
  std::thread drainer([&] {  // state_tx keeps consuming the journal
    FILE* sink = std::fopen("/dev/null", "w");
    while (!stop.load()) drain_fault_journal(ctx, sink);
    std::fclose(sink);
  });
  // Authority is raised only after the word is FREE again: never armed while
  // ARMING. A sample counts only if no new round began during it -- a round
  // opens after its DISARM, so armed read within it is this round's ARM. The
  // seam holds the gap between the two stores open long enough to sample;
  // back to back, x86 publishes them together.
  ctx.arm_seam = [] {
    timespec pause{0, 20'000};
    nanosleep(&pause, nullptr);
  };
  std::atomic<int> round{0};
  std::atomic<long> samples{0}, armed_while_arming{0};
  std::thread sampler([&] {
    while (!stop.load()) {
      const int r = round.load();
      const bool a = ctx.armed.load(std::memory_order_acquire);
      const uint32_t s = ctx.latch_state.load(std::memory_order_acquire);
      if (round.load() != r) continue;
      ++samples;
      armed_while_arming += a && s == ServerCtx::LATCH_ARMING;
    }
  });
  int cleared = 0, armed = 0;
  for (int i = 0; i < 2000; ++i) {  // DISARM, then an ARM attempt
    ctx.armed.store(false);
    round.store(i + 1);
    cleared += ctx.clear_latch();
    uint64_t prepared = 0;
    const bool ok = ctx.arm_unlatched([&] { prepared = ctx.arm_gen.fetch_add(1) + 1; });
    assert(ok == (prepared != 0));  // refused = nothing prepared
    armed += ok;
  }
  stop.store(true);
  rt.join();
  drainer.join();
  sampler.join();
  assert(cleared == 2000);
  assert(samples > 0 && armed_while_arming == 0);
  std::printf("latch stress: %d ARMs admitted of 2000 against a latching servo\n", armed);
  const uint32_t s = ctx.latch_state.load();
  if (s == ServerCtx::LATCH_SET)
    assert(ctx.fault.load() && ctx.fault_code.load() == FAULT_PLANT &&
           std::strcmp(ctx.fault_text, "persisting cause") == 0);
  else
    assert(s == ServerCtx::LATCH_FREE && !ctx.fault.load() && ctx.fault_code.load() == 0);
}

// ARM admission is one transition on the latch word (arm_unlatched).
bool try_arm(ServerCtx& ctx, bool* prepared = nullptr) {
  bool ran = false;
  const bool ok = ctx.arm_unlatched([&] {
    ran = true;
    ctx.arm_gen.fetch_add(1);
  });
  if (prepared) *prepared = ran;
  return ok;
}

void check_arm_admission() {
  ServerCtx ctx;
  ctx.arm_gen.store(3);
  ctx.armed.store(true);
  ctx.latch(FAULT_PLANT, "latched in epoch 3", 3);
  ctx.armed.store(false);
  bool prepared = true;
  assert(!try_arm(ctx, &prepared) && !prepared && !ctx.armed.load());  // SET
  assert(ctx.fault_gen.load() == 3 && ctx.clear_latch());
  assert(try_arm(ctx) && ctx.armed.load() && ctx.arm_gen.load() == 4);
  assert(ctx.latch_state.load() == ServerCtx::LATCH_FREE);

  // DISARM's clear times out BEFORE the writer published fault=true: the
  // re-ARM must still be refused, and the fault keeps its own generation.
  Gate gate;
  ctx.latch_seam = hold_after_claim;
  ctx.latch_seam_arg = &gate;
  std::thread rt([&] { ctx.latch(FAULT_CMD_LOST, "epoch 4, written late", 4); });
  {
    std::unique_lock<std::mutex> lock(gate.m);
    gate.cv.wait(lock, [&] { return gate.in; });
  }
  ctx.latch_seam = nullptr;
  ctx.armed.store(false);
  const uint64_t t0 = mono_ns();
  assert(!ctx.clear_latch());  // monotonic deadline, writer never finished
  assert(mono_ns() - t0 >= ServerCtx::LATCH_CLEAR_BUDGET_NS);
  assert(!ctx.fault.load());  // not published yet -- the old ARM gate passed here
  assert(!try_arm(ctx, &prepared) && !prepared && !ctx.armed.load());
  assert(ctx.arm_gen.load() == 4);
  {
    std::lock_guard<std::mutex> lock(gate.m);
    gate.go = true;
  }
  gate.cv.notify_all();
  rt.join();
  assert(ctx.fault.load() && ctx.fault_gen.load() == 4);  // its own epoch
  assert(!try_arm(ctx, &prepared) && !prepared);           // now SET
  assert(ctx.clear_latch() && try_arm(ctx) && ctx.arm_gen.load() == 5);

  // An old-generation latch still in flight while the ARM is prepared cannot
  // claim the word: it is journaled with its own generation, the new
  // generation starts clean, and a fault of the new generation then latches.
  ctx.armed.store(false);
  assert(ctx.clear_latch());
  drain(ctx);
  assert(ctx.arm_unlatched([&] {
    ctx.arm_gen.fetch_add(1);  // epoch 6 is being prepared ...
    ctx.latch(FAULT_PLANT, "epoch 5 cause arriving during ARM", 5);  // ... epoch 5's latch lands
  }));
  assert(ctx.armed.load() && !ctx.fault.load() && ctx.arm_gen.load() == 6);
  const std::string log = drain(ctx);
  assert(line_with(log, "secondary cause code=3").find("arm_epoch=5") != std::string::npos);
  ctx.latch(FAULT_PLANT, "epoch 6 cause", 6);
  assert(ctx.fault.load() && ctx.fault_gen.load() == 6);
}

// CTL_FAULT delivery (control thread): one notice per latch serial, never the
// text of a cleared latch, and an older acknowledgement never discards a
// newer notice.
void check_fault_notice() {
  ServerCtx ctx;
  ServerCtx::FaultNotice n{};
  assert(!ctx.pending_fault_notice(n));
  ctx.latch(FAULT_PLANT, "A", 1);
  assert(ctx.pending_fault_notice(n) && n.code == FAULT_PLANT && std::strcmp(n.text, "A") == 0);
  const uint64_t a = n.seq;
  assert(ctx.pending_fault_notice(n) && n.seq == a);  // failed send: still owed
  assert(ctx.clear_latch());                          // DISARM before delivery
  assert(!ctx.pending_fault_notice(n));               // nothing cleared is sent
  ctx.latch(FAULT_CMD_LOST, "B", 2);
  assert(ctx.pending_fault_notice(n) && std::strcmp(n.text, "B") == 0);
  const uint64_t b = n.seq;
  assert(b != a);
  // B's frame is in flight; meanwhile B is cleared and C latches.
  assert(ctx.clear_latch());
  ctx.latch(FAULT_CTL_LOST, "C", 3);
  ctx.ack_fault_notice(b);  // late acknowledgement of B
  assert(ctx.pending_fault_notice(n) && std::strcmp(n.text, "C") == 0);
  const uint64_t c = n.seq;
  ctx.ack_fault_notice(c);
  assert(!ctx.pending_fault_notice(n));
  ctx.ack_fault_notice(a);  // stale acknowledgement cannot move it back
  assert(!ctx.pending_fault_notice(n));
  assert(ctx.clear_latch());
  ctx.latch(FAULT_PLANT, "D", 4);
  assert(ctx.pending_fault_notice(n) && std::strcmp(n.text, "D") == 0);
}
}  // namespace

int main() {
  check_causes();
  check_truncation();
  check_full_ring();
  check_bounded_claim();
  check_concurrent();
  check_latch_disarm_race();
  check_clear_bounded();
  check_latch_stress();
  check_arm_admission();
  check_fault_notice();
  std::puts("PASS fault journal: first/secondary causes survive DISARM, explicit "
            "truncation, full ring drops new, concurrent producers intact, latch/DISARM "
            "lifecycle (paused writer, bounded clear, re-arm, stress), ARM admission, fault "
            "notices");
}
