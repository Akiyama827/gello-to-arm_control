// CTRL-010 bounded loop-gap diagnostics, against the actual RT loop with
// scripted state/time and no device access. No sockets are opened.
//
// Covers the card's device-free acceptance: forced gap, trace-full ring,
// stalled consumer, correct sequence association reconstructed from a
// fixture, and instrumentation that does not change behavior (the law's
// optional raw export must leave the limited output byte-identical).
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "server.hpp"
#include "arm_rt/servo_law.hpp"

namespace arm_rt {
namespace {
ServerCtx* ctx;
int tick, writes, stops;
constexpr double native_limits[] = {28, 10};
constexpr uint64_t SESSION = 7;
uint64_t skew = 0;  // simulated elapsed time on top of the per-tick clock
// Iterations 3..72 each read 3 ms late, so each records a loop gap.
constexpr int GAP_FIRST = 3, GAP_LAST = 72;
uint32_t arm_gen_value = 0, last_seq = 0;

// Tick-2 tracking fixture: known command, known measurement, tau_ff far past
// the limit so the clamp engages on joint 0 and not the slew.
constexpr double Q[] = {0.1, -0.05}, DQ[] = {0.5, -0.25};
constexpr double TAU[] = {1.5, -2.0}, TAU_REF[] = {3.0, -4.0};
constexpr double Q_DES[] = {0.4, 0.35}, QD_DES[] = {1.0, -0.5};
constexpr double TAU_FF[] = {50., -50.}, KP[] = {100., 100.}, KD[] = {10., 10.};

void verify_first_gap() {
  // Gap #1 was written at the top of iteration 3 and describes iteration 2,
  // the tracking cycle above. `state_seq` points at the state packet
  // published in the SAME iteration as the record (the current measurement).
  assert(ctx->gap_head.load() - ctx->gap_tail.load() == 1);
  const LoopGap g = ctx->gaps[0];
  ctx->gap_tail.store(1, std::memory_order_release);
  assert(g.period_ns > LOOP_GAP_NS);
  assert(g.read_ns >= LOOP_GAP_NS / 2);
  assert(g.armed && g.diag_valid);
  assert(g.session_id == SESSION);
  assert(g.arm_epoch == arm_gen_value);
  assert(g.last_cmd_seq == last_seq);
  StatePacket previous{};
  assert(ctx->state_out.read(previous) && previous.state_seq == g.state_seq);
  // Reconstructed from the fixture: the previous cycle's command vs measurement.
  auto near = [](double a, double b) { return std::fabs(a - b) < 1e-12; };
  assert(near(g.diag.q_err_max, 0.4));    // max(|0.4-0.1|, |0.35+0.05|)
  assert(near(g.diag.qd_err_max, 0.5));   // max(|1.0-0.5|, |-0.5+0.25|)
  assert(near(g.diag.tau_ff_max, 50.));
  assert(near(g.diag.tau_raw_max, 85.));  // 100*0.3 + 10*0.5 + 50
  assert(near(g.diag.tau_out_max, 28.));  // clamped to the joint-0 limit
  assert(near(g.diag.tau_ref_max, 4.));   // vendor echo, separate from tau_out
  assert(near(g.diag.tau_meas_max, 2.));
  assert(g.diag.limiter_joint == 0);
  assert(g.diag.clamp_active && !g.diag.slew_active);
}

void verify_full_ring() {
  // 70 forced gaps, one drained: 65 fit, the newest 5 are dropped and
  // counted, and the 64 retained records are the OLDEST in arrival order
  // (drain order wraps the ring: tail, tail+1, ... mod capacity).
  assert(ctx->gap_dropped.load() == 5);
  assert(ctx->gap_head.load() - ctx->gap_tail.load() == 64);
  const uint32_t tail = ctx->gap_tail.load();
  uint64_t prev = 0;
  for (uint32_t i = 0; i < 64; ++i) {
    const LoopGap g = ctx->gaps[(tail + i) % LOOP_GAP_RING];
    assert(g.t_read_ns > prev);
    prev = g.t_read_ns;
    assert(g.diag_valid && g.armed);
  }
  ctx->gap_tail.store(ctx->gap_head.load(), std::memory_order_release);
}

class ScriptedBackend final : public Backend {
 public:
  const char* name() const override { return "gap-selfcheck"; }
  int n() const override { return 2; }
  double tick_s() const override { return .001; }
  const double* tau_limit() const override { return native_limits; }
  const std::string& fault_text() const override { return error_; }
  bool stop() override { ++stops; return true; }

  bool read(PlantState& state) override {
    ++tick;
    state = {};
    state.n = 2;
    state.robot_time_ms = uint64_t(tick);
    state.cmd_success_rate = 1.0;
    for (int j = 0; j < 2; ++j) {
      state.q[j] = Q[j];
      state.dq[j] = DQ[j];
      state.tau[j] = TAU[j];
      state.tau_ref[j] = TAU_REF[j];
    }
    if (tick == 1) arm_gen_value = uint32_t(begin_arm(*ctx));
    if (tick == 2) {  // tracking command with a known feedforward overshoot
      CommandIngress packet{};
      auto& command = packet.command;
      command.n = 2;
      command.version = VERSION;
      command.seq = uint32_t(tick);
      for (int j = 0; j < 2; ++j) {
        command.q_des[j] = Q_DES[j];
        command.qd_des[j] = QD_DES[j];
        command.tau_ff[j] = TAU_FF[j];
        command.kp[j] = KP[j];
        command.kd[j] = KD[j];
      }
      last_seq = command.seq;
      ctx->cmd_in.write(CommandEnvelope{packet, mono_ns(), ctx->arm_gen.load(), SESSION});
    }
    if (tick >= GAP_FIRST && tick <= GAP_LAST) skew += 3'000'000;  // forced gap
    if (tick == 74) skew += 3'000'000;  // one disarmed gap for diag=absent
    if (tick == 4) verify_first_gap();
    if (tick == 73) {
      verify_full_ring();
      ctx->armed.store(false);  // the next gap must report absent context
    }
    if (tick == 75) {  // the gap written at 74 describes a disarmed cycle
      assert(ctx->gap_head.load() - ctx->gap_tail.load() == 1);
      const LoopGap g = ctx->gaps[ctx->gap_tail.load() % LOOP_GAP_RING];
      ctx->gap_tail.store(ctx->gap_head.load(), std::memory_order_release);
      assert(g.armed == false && g.diag_valid == false);
    }
    if (tick == 76) ctx->shutdown.store(true);
    return true;
  }

  bool write(const double* tau, int count) override {
    assert(count == 2);
    ++writes;
    for (int j = 0; j < count; ++j)
      assert(std::isfinite(tau[j]) && std::fabs(tau[j]) <= native_limits[j]);
    return true;
  }

 private:
  std::string error_;
};
}  // namespace

uint64_t mono_ns() { return 10'000'000'000ull + uint64_t(tick) * 1'000'000 + skew; }
std::unique_ptr<Backend> make_fake_backend(int, const std::vector<double>&) {
  return std::make_unique<ScriptedBackend>();
}
// Hardware constructors deliberately unavailable in this executable.
std::unique_ptr<Backend> make_franka_backend(const std::string&, double,
                                             const double[3]) {
  std::abort();
}
std::unique_ptr<Backend> make_dm_backend(const std::string&, uint32_t, double) {
  std::abort();
}
}  // namespace arm_rt

int main() {
  using namespace arm_rt;
  // Instrumentation must not change behavior: the same law call with and
  // without the raw export produces identical limited output, and the raw
  // value is exactly the pre-clamp PD+FF the limiter consumed.
  {
    double a[2], b[2], raw[2];
    servo_torque(2, Q, DQ, Q_DES, QD_DES, TAU_FF, KP, KD, TAU_REF,
                 native_limits, 1000., a);
    servo_torque(2, Q, DQ, Q_DES, QD_DES, TAU_FF, KP, KD, TAU_REF,
                 native_limits, 1000., b, raw);
    assert(std::memcmp(a, b, sizeof(a)) == 0);
    assert(raw[0] == 85. && raw[1] == -12.5);
    assert(b[0] == native_limits[0] && b[1] == -native_limits[1]);
  }
  {
    ServerCtx context;
    ctx = &context;
    context.session_id.store(SESSION);
    tick = writes = stops = 0;
    skew = 0;
    context.cfg.slew = 1000;
    context.cfg.tau_max = 0;
    context.cfg.rt_priority = 0;
    rt_loop(context);
    assert(!context.failed && stops == 2);
    assert(writes == GAP_LAST);  // every armed iteration wrote torque
  }
  std::puts("PASS loop-gap diagnostics: forced gap with fixture-reconstructed "
            "sequence association, full ring drops newest, stalled consumer "
            "keeps control running, disarmed gap reports absent context, law "
            "raw export is behavior-neutral");
}
