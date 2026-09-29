// Control-thread mode-selection semantics with NO RT loop attached: the
// bounded wait, the PENDING/BUSY bookkeeping and the pre-flight refusals are
// only observable when the acknowledgement is deliberately absent, which the
// running fake server never produces (it confirms within a tick).
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <atomic>
#include <chrono>
#include <thread>

#include "control_session.cpp"

namespace arm_rt {
uint64_t mono_ns() {
  timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}
}  // namespace arm_rt

using namespace arm_rt;

static ControlPacket frame(uint16_t type, uint32_t seq, const void* body) {
  ControlPacket p{};
  p.magic = MAGIC_CTL; p.version = VERSION; p.type = type; p.seq = seq;
  if (body) std::memcpy(p.text, body, sizeof(p.text));
  return p;
}

static void send_all(int fd, const ControlPacket& p) {
  const char* b = reinterpret_cast<const char*>(&p);
  size_t left = sizeof(p);
  while (left) {
    const ssize_t n = ::write(fd, b, left);
    assert(n > 0);
    b += n; left -= size_t(n);
  }
}

// Read frames until one of `want` arrives or the deadline passes. PING/STATUS
// are lifecycle noise here; a PING is answered so the session stays alive.
static bool read_type(int fd, uint16_t want, ModeSelectReplyPayload& out,
                      int timeout_ms) {
  const uint64_t end = mono_ns() + uint64_t(timeout_ms) * 1000000ull;
  ControlPacket p{};
  size_t have = 0;
  while (mono_ns() < end) {
    pollfd pfd{fd, POLLIN, 0};
    const int left = int((end - mono_ns()) / 1000000ull);
    const int ready = poll(&pfd, 1, left > 0 ? left : 1);
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0) return false;
    if (ready == 0) continue;
    const ssize_t got = ::read(fd, reinterpret_cast<char*>(&p) + have,
                               sizeof(p) - have);
    if (got <= 0) return false;
    have += size_t(got);
    if (have < sizeof(p)) continue;
    have = 0;
    if (p.type == CTL_PING) { send_all(fd, frame(CTL_PONG, 0, nullptr)); continue; }
    if (p.type != want) continue;
    std::memcpy(&out, p.text, sizeof(out));
    return true;
  }
  return false;
}

struct Fixture {
  ServerCtx ctx;
  int sv[2]{};
  std::thread thread;
  explicit Fixture(bool armed, bool reference, uint32_t latch) {
    ctx.cfg.backend = "fake"; ctx.backend_n.store(16); ctx.backend_ready.store(true);
    ctx.supports_reference.store(reference); ctx.session_id.store(123);
    ctx.claim_ok.store(true); ctx.armed.store(armed); ctx.arm_gen.store(5);
    ctx.latch_state.store(latch);
    std::strcpy(ctx.instance_id, "fake"); std::strcpy(ctx.model_id, "fake");
    std::strcpy(ctx.backend_name, "fake"); ctx.joint_order_hash = 456;
    ctx.identity_hash = identity_hash(ctx.instance_id, 456);
    assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    thread = std::thread([this] { serve_client(ctx, sv[0], 123); });
  }
  ~Fixture() {
    ctx.shutdown.store(true);
    ::shutdown(sv[1], SHUT_RDWR);
    thread.join();
    ::close(sv[0]); ::close(sv[1]);
  }
  void select(uint32_t seq, uint64_t generation, uint16_t mode,
              uint32_t arm_epoch = 5, int reserved0 = 0,
              const uint8_t* digest = nullptr) {
    ModeSelectPayload s{};
    s.session_id = 123; s.generation = generation;
    s.expected_predecessor = generation - 1; s.arm_epoch = arm_epoch;
    s.desired_mode = mode; s.reserved0 = uint16_t(reserved0);
    if (digest) std::memcpy(s.model_digest, digest, 32);
    send_all(sv[1], frame(CTL_MODE_SELECT, seq, &s));
  }
};

int main() {
  const uint8_t digest[32] = {1};

  // --- no RT acknowledgement -> bounded wait, then PENDING ------------------
  {
    Fixture f(true, true, ServerCtx::LATCH_FREE);
    ModeSelectReplyPayload r{};
    f.select(1, 1, MODE_STREAMED);
    // Still inside the budget: no premature answer.
    assert(!read_type(f.sv[1], CTL_MODE_STATUS, r, 80));
    // Past the documented 250 ms budget: PENDING, exactly once.
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 600));
    assert(r.result == MODE_SEL_PENDING && r.generation == 1);
    // Identical retry while still unresolved: PENDING again, not a new attempt.
    f.select(2, 1, MODE_STREAMED);
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 600));
    assert(r.result == MODE_SEL_PENDING && r.generation == 1);
    // A DIFFERENT request while unresolved is BUSY.
    f.select(3, 2, MODE_REFERENCE, 5, 0, digest);
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 600));
    assert(r.result == MODE_SEL_BUSY && r.generation == 1);
  }

  // A servo acknowledgement must not sit behind the idle 100 ms socket poll.
  // This fixture supplies RT's reply immediately after observing publication.
  {
    Fixture f(true, true, ServerCtx::LATCH_FREE);
    f.select(4, 1, MODE_STREAMED);
    const uint64_t publication_deadline = mono_ns() + 100'000'000;
    while (f.ctx.mode_published.load(std::memory_order_acquire) != 1 &&
           mono_ns() < publication_deadline)
      std::this_thread::yield();
    assert(f.ctx.mode_published.load(std::memory_order_acquire) == 1);
    ModeSelectReplyPayload r{};
    // Drain HELLO/IDENTITY/initial PING and answer that PING before measuring.
    // Otherwise its queued PONG wakes even the old 100 ms poll immediately.
    assert(!read_type(f.sv[1], CTL_MODE_STATUS, r, 10));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ModeSelectInfo info{};
    info.reply.session_id = 123;
    info.reply.arm_epoch = 5;
    info.reply.generation = 1;
    info.reply.requested_mode = MODE_STREAMED;
    info.reply.selected_mode = MODE_STREAMED;
    info.reply.result = MODE_SEL_OK;
    f.ctx.mode_feedback.write(info);
    const uint64_t done_at = mono_ns();
    f.ctx.mode_done.store(1, std::memory_order_release);
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 50));
    const uint64_t reply_us = (mono_ns() - done_at) / 1000;
    assert(reply_us < 10'000);
    assert(r.result == MODE_SEL_OK && r.generation == 1);
    std::printf("mode-select reply latency %llu us (<10000 us)\n",
                static_cast<unsigned long long>(reply_us));
  }

  // --- pre-flight refusals --------------------------------------------------
  {
    // Disarmed: REFUSED, and nothing is published.
    Fixture f(false, true, ServerCtx::LATCH_FREE);
    f.select(1, 1, MODE_STREAMED);
    ModeSelectReplyPayload r{};
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 600));
    assert(r.result == MODE_SEL_REFUSED && f.ctx.mode_published.load() == 0);
  }
  {
    // Latched: REFUSED.
    Fixture f(true, true, ServerCtx::LATCH_SET);
    f.select(1, 1, MODE_STREAMED);
    ModeSelectReplyPayload r{};
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 600));
    assert(r.result == MODE_SEL_REFUSED);
  }
  {
    // Capability off (enable_reference: false): UNSUPPORTED, never published.
    Fixture f(true, false, ServerCtx::LATCH_FREE);
    f.select(1, 1, MODE_STREAMED);
    ModeSelectReplyPayload r{};
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 600));
    assert(r.result == MODE_SEL_UNSUPPORTED && f.ctx.mode_published.load() == 0);
  }
  {
    // Malformed body (reserved must be zero): MALFORMED, never published.
    Fixture f(true, true, ServerCtx::LATCH_FREE);
    f.select(1, 1, MODE_STREAMED, 5, 1);
    ModeSelectReplyPayload r{};
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 600));
    assert(r.result == MODE_SEL_MALFORMED && f.ctx.mode_published.load() == 0);
  }
  {
    // Stale ARM epoch (the epoch changed, or the request outlived its ARM):
    // REFUSED, never published.
    Fixture f(true, true, ServerCtx::LATCH_FREE);
    f.select(1, 1, MODE_STREAMED, /*arm_epoch=*/4);
    ModeSelectReplyPayload r{};
    assert(read_type(f.sv[1], CTL_MODE_STATUS, r, 600));
    assert(r.result == MODE_SEL_REFUSED && f.ctx.mode_published.load() == 0);
  }

  std::printf("mode_select session: bounded wait + PENDING/BUSY/refusals PASS\n");
  return 0;
}
