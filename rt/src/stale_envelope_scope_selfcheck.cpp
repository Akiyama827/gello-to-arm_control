// Exercise the actual servo loop with a persistent UDP mailbox across ARM
// epochs. No socket, device, or wall-clock timing is involved.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "server.hpp"

namespace arm_rt {
namespace {
enum Case { STALE_FIRST, RECENT_REARM, CURRENT_FIRST, INCOMING_STALE, HELD_STALE };
ServerCtx* ctx;
Case scenario;
int tick;
uint64_t skew_ns;
constexpr uint64_t A = 11, B = 22;
constexpr double limits[] = {50, 50, 50};

void commit(uint64_t session, uint32_t epoch, uint64_t ticket) {
  ReferenceCommit c{};
  auto& b = c.admitted.identity;
  b.session_id = session;
  b.arm_epoch = epoch;
  b.upload_id = b.reference_id = 1;
  b.revision = 1;
  b.knots = 2;
  b.n = 3;
  b.flags = REFERENCE_HERMITE;
  auto& curve = c.admitted.curve;
  curve.n = 3;
  curve.k = 2;
  curve.hermite = true;
  curve.times[0] = 0;
  curve.times[1] = 2;
  auto& p = c.request;
  p.session_id = session;
  p.arm_epoch = epoch;
  p.upload_id = p.reference_id = 1;
  p.revision = 1;
  c.ticket = ticket;
  ctx->reference_in.write(c);
  ctx->reference_published.store(ticket);
}

void envelope(uint64_t session, uint64_t epoch, uint32_t seq, double phase,
              uint32_t age_us = 0, uint64_t reference_id = 1) {
  CommandIngress packet{};
  packet.command.magic = MAGIC_CMD;
  packet.command.version = REFERENCE_VERSION;
  packet.command.n = 3;
  packet.command.seq = seq;
  packet.command.arm_epoch = uint32_t(epoch);
  packet.command.session_id = session;
  for (int j = 0; j < 3; ++j) {
    packet.command.kp[j] = 2;
    packet.command.kd[j] = .1;
  }
  ReferenceEnvelopeBinding binding{};
  binding.reference_id = reference_id;
  binding.revision = 1;
  binding.phase = phase;
  binding.valid_for_us = binding.phase_window_us = 100'000;
  binding.age_us = age_us;
  ctx->cmd_in.write(CommandEnvelope{packet, mono_ns(), epoch, session, binding});
}

ReferenceStatePacket state() {
  ReferenceStatePacket out{};
  assert(ctx->reference_state_out.read(out));
  return out;
}

void expect_expiry_text(const char* rule, bool envelope_detail) {
  assert(ctx->fault_head.load() == 1);
  const auto& slot = ctx->fault_ring[0];
  assert(slot.ready.load() == 1);
  assert(slot.rec.code == FAULT_REFERENCE_EXPIRED);
  const char* text = slot.rec.text;
  assert(std::strstr(text, rule) != nullptr);
  if (envelope_detail) {
    assert(std::strstr(text, "arm_gen=2") != nullptr);
    assert(std::strstr(text, "session_id=0000000000000016") != nullptr);
    assert(std::strstr(text, "reference_id=1 revision=1") != nullptr);
    assert(std::strstr(text, scenario == INCOMING_STALE
                                  ? "age_us=110000" : "age_us=111000") != nullptr);
  } else {
    assert(std::strstr(text, "reference_id=1 revision=1 gap_age_us=102000") != nullptr);
  }
}

class ScriptedBackend final : public Backend {
 public:
  const char* name() const override { return "stale-envelope-selfcheck"; }
  int n() const override { return 3; }
  double tick_s() const override { return .001; }
  const double* tau_limit() const override { return limits; }
  const std::string& fault_text() const override { return error_; }
  bool stop() override { return true; }
  bool write(const double*, int count) override { assert(count == 3); return true; }

  bool read(PlantState& plant) override {
    ++tick;
    plant = {};
    plant.n = 3;
    if (tick == 1) {
      ctx->session_id.store(A);
      ctx->reference_session.store(A);
      const auto epoch = begin_arm(*ctx);
      assert(epoch == 1);
      commit(A, uint32_t(epoch), 1);
      envelope(A, epoch, 1, 0);
    } else if (tick == 2) {
      assert(state().status == REFERENCE_RUNNING);
      ctx->armed.store(false);
      ctx->disarm_gen.fetch_add(1);
      ctx->reference_session.store(0);
    } else if (tick == 3) {
      // STALE_FIRST and CURRENT_FIRST exceed the old envelope's 100 ms
      // validity. RECENT_REARM keeps it fresh at selection so the check
      // catches an illicit mark_envelope(), not only immediate retirement.
      if (scenario != RECENT_REARM) skew_ns += 120'000'000;
      ctx->session_id.store(B);
      ctx->reference_session.store(B);
      const auto epoch = begin_arm(*ctx);
      assert(epoch == 2);
      commit(B, uint32_t(epoch), 2);  // same (reference_id, revision) = (1,1)
      // cmd_in is intentionally unchanged: udp_rx writes nothing disarmed.
    } else if (tick == 4) {
      const auto previous = state();
      assert(previous.state.session_id == B && previous.state.arm_epoch == 2);
      assert(previous.reference_id == 1 && previous.revision == 1);
      assert(previous.status == REFERENCE_RUNNING);
      if (scenario == CURRENT_FIRST || scenario == INCOMING_STALE ||
          scenario == HELD_STALE) {
        skew_ns += 49'000'000;  // B+50 ms; a real B envelope arrives
        envelope(B, 2, 2, .05, scenario == INCOMING_STALE ? 110'000 : 0);
      } else {
        skew_ns += 98'000'000;  // B+99 ms; no B envelope yet
      }
    } else if (tick == 5) {
      const auto previous = state();
      if (scenario == INCOMING_STALE) {
        assert(previous.status == REFERENCE_CANCELLED);
        expect_expiry_text("incoming envelope stale", true);
        ctx->shutdown.store(true);
        return true;
      }
      assert(previous.status == REFERENCE_RUNNING);
      if (scenario == CURRENT_FIRST) {
        assert(previous.state.last_cmd_seq == 2);
        assert(!(previous.state.flags & FLAG_HOLDING));
        skew_ns += 59'000'000;  // B+110 ms, B envelope still valid
      } else if (scenario == HELD_STALE) {
        skew_ns += 110'000'000;  // accepted B envelope is now old
        envelope(B, 2, 3, .16, 0, 2);  // wrong id: not a new active envelope
      } else {
        skew_ns += 2'000'000;  // B+102 ms: only the first-envelope limit retires
      }
    } else if (tick == 6) {
      const auto previous = state();
      assert(previous.status == (scenario == CURRENT_FIRST
                                     ? REFERENCE_RUNNING : REFERENCE_CANCELLED));
      if (scenario == HELD_STALE) expect_expiry_text("held envelope stale", true);
      else if (scenario != CURRENT_FIRST)
        expect_expiry_text("first-envelope gap", false);
      ctx->shutdown.store(true);
    }
    return true;
  }

 private:
  std::string error_;
};
}  // namespace

uint64_t mono_ns() { return 10'000'000'000ull + uint64_t(tick) * 1'000'000 + skew_ns; }
std::unique_ptr<Backend> make_fake_backend(int) {
  return std::make_unique<ScriptedBackend>();
}
std::unique_ptr<Backend> make_franka_backend(const std::string&, double,
                                            const double[3]) { std::abort(); }
std::unique_ptr<Backend> make_dm_backend(const std::string&, uint32_t, double) {
  std::abort();
}
}  // namespace arm_rt

int main() {
  using namespace arm_rt;
  for (Case which : {STALE_FIRST, RECENT_REARM, CURRENT_FIRST,
                     INCOMING_STALE, HELD_STALE}) {
    ServerCtx context;
    ctx = &context;
    scenario = which;
    tick = 0;
    skew_ns = 0;
    context.cfg.backend = "fake";
    context.cfg.hold_ms = 100;
    context.cfg.fault_ms = 1000;
    context.cfg.rt_priority = 0;
    context.supports_reference.store(true);
    rt_loop(context);
    assert(!context.failed && tick == (which == INCOMING_STALE ? 5 : 6));
  }
  std::puts("PASS RT stale envelope scope: old session/epoch cannot retire or "
            "mark a new (1,1), current envelope tracks, expiry journal identifies rule");
}
