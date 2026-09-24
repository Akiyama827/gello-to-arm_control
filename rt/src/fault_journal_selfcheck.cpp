// Fault journal (server.hpp): first cause, secondary causes, explicit
// truncation, full ring, stalled consumer and concurrent producers. No
// sockets, no backend, no devices.
#include <atomic>
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

void disarm(ServerCtx& ctx) {  // the CTL_DISARM clear, in its order
  ctx.armed.store(false);
  ctx.fault.store(false);
  ctx.fault_code.store(0);
  ctx.fault_text[0] = '\0';
  ctx.secondary_codes.store(0);
  ctx.fault_claim.store(false);
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
  assert(delivered[0] + delivered[1] + int(ctx.fault_dropped.load()) == 2 * PER);
  assert(delivered[0] > 0 && delivered[1] > 0);
  std::printf("concurrent: delivered %d + %d, dropped %u\n", delivered[0],
              delivered[1], ctx.fault_dropped.load());
}
}  // namespace

int main() {
  check_causes();
  check_truncation();
  check_full_ring();
  check_concurrent();
  std::puts("PASS fault journal: first/secondary causes survive DISARM, explicit "
            "truncation, full ring drops new, concurrent producers intact");
}
