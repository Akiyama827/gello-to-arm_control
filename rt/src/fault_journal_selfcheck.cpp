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
  ctx.latch(FAULT_PLANT, "read: non-libfranka exception: Net Exception");
  for (int i = 0; i < 1000; ++i)  // a persisting cause re-latches every tick
    ctx.latch(FAULT_PLANT, "read: non-libfranka exception: Net Exception");
  ctx.latch(FAULT_CTL_LOST, "control session lost while armed");
  ctx.latch(FAULT_CTL_LOST, "control session lost while armed");
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
  ctx.latch(FAULT_CMD_LOST, "command stream lost (staleness > fault-ms)");
  ctx.latch(FAULT_CTL_LOST, "control session lost while armed");
  const std::string again = drain(ctx, &n);
  assert(n == 2 && again.find("first cause code=1") != std::string::npos);
  assert(again.find("secondary cause code=2") != std::string::npos);
}

void check_truncation() {
  ServerCtx ctx;
  const std::string medium(150, 'm');  // fits the journal, not the wire
  ctx.latch(FAULT_PLANT, medium.c_str());
  assert(std::strlen(ctx.fault_text) == sizeof(ctx.fault_text) - 1);
  assert(std::strcmp(ctx.fault_text + sizeof(ctx.fault_text) - 4, "...") == 0);
  std::string log = drain(ctx);
  assert(log.find(medium + "\"\n") != std::string::npos);  // full text kept
  assert(log.find("(truncated)") == std::string::npos);
  disarm(ctx);
  const std::string huge(5000, 'h');
  ctx.latch(FAULT_PLANT, huge.c_str());
  log = drain(ctx);
  assert(log.find(std::string(FAULT_RECORD_TEXT - 1, 'h') + "\" (truncated)") !=
         std::string::npos);
  disarm(ctx);
  const std::string exact(sizeof(ctx.fault_text) - 1, 'e');  // fits exactly
  ctx.latch(FAULT_PLANT, exact.c_str());
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
  ctx.journal(FAULT_SECONDARY, 1, text);
  assert(ctx.fault_head == 1 && ctx.fault_contended == 0);
}

void check_full_ring() {
  ServerCtx ctx;  // stalled consumer: producers never wait, never overwrite
  for (uint32_t i = 0; i < FAULT_RING + 5; ++i) {
    const std::string text = "cause " + std::to_string(i);
    ctx.journal(FAULT_STOP_FAILED, FAULT_PLANT, text.c_str());
  }
  assert(ctx.fault_dropped == 5);
  int n = 0;
  const std::string log = drain(ctx, &n);
  assert(n == int(FAULT_RING));
  assert(log.find("text=\"cause 0\"") != std::string::npos);  // oldest kept
  assert(log.find("text=\"cause 15\"") != std::string::npos);
  assert(log.find("text=\"cause 16\"") == std::string::npos);  // new dropped
  assert(log.find("stop failed") != std::string::npos);
  ctx.journal(FAULT_STOP_FAILED, FAULT_PLANT, "after drain");
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
      ctx.journal(FAULT_SECONDARY, uint32_t(tag), text);
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
  std::thread rt([&] { ctx.latch(FAULT_PLANT, "first cause, epoch 1"); });
  {
    std::unique_lock<std::mutex> lock(gate.m);
    gate.cv.wait(lock, [&] { return gate.in; });
  }
  ctx.latch_seam = nullptr;
  ctx.latch(FAULT_CTL_LOST, "second cause while the first is written");
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
  ctx.latch(FAULT_CMD_LOST, "epoch 2 cause");
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
    while (!stop.load()) ctx.latch(FAULT_PLANT, "persisting cause");
  });
  std::thread drainer([&] {  // state_tx keeps consuming the journal
    FILE* sink = std::fopen("/dev/null", "w");
    while (!stop.load()) drain_fault_journal(ctx, sink);
    std::fclose(sink);
  });
  int cleared = 0;
  for (int i = 0; i < 2000; ++i) cleared += ctx.clear_latch();
  stop.store(true);
  rt.join();
  drainer.join();
  assert(cleared == 2000);
  const uint32_t s = ctx.latch_state.load();
  if (s == ServerCtx::LATCH_SET)
    assert(ctx.fault.load() && ctx.fault_code.load() == FAULT_PLANT &&
           std::strcmp(ctx.fault_text, "persisting cause") == 0);
  else
    assert(s == ServerCtx::LATCH_FREE && !ctx.fault.load() && ctx.fault_code.load() == 0);
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
  std::puts("PASS fault journal: first/secondary causes survive DISARM, explicit "
            "truncation, full ring drops new, concurrent producers intact, latch/DISARM "
            "lifecycle (paused writer, bounded clear, re-arm, stress)");
}
