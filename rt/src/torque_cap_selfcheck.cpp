// Exercise the actual RT loop, with scripted state/time and no device access.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "server.hpp"

namespace arm_rt {
namespace {
ServerCtx* ctx;
int tick, writes, stops;
constexpr double native_limits[] = {28, 10};
// Backend-declared torque-RATE ceiling, and whether this run is the case that
// exercises it. Infinity in every pre-existing scenario, so those keep
// saturating at the magnitude cap exactly as before.
double slew_max = std::numeric_limits<double>::infinity();
bool slew_case = false;

class ScriptedBackend final : public Backend {
 public:
  const char* name() const override { return "cap-selfcheck"; }
  int n() const override { return 2; }
  double tick_s() const override { return .001; }
  const double* tau_limit() const override { return native_limits; }
  double tau_slew_max() const override { return slew_max; }
  const std::string& fault_text() const override { return error_; }
  void stop() override { ++stops; }

  bool read(PlantState& state) override {
    ++tick;
    if (tick > 1) {
      StatePacket previous{};
      assert(ctx->state_out.read(previous));
      assert(previous.version == VERSION);
      for (int j=0; j<2; ++j) {
        const double expected = (tick == 4 || tick == 5) ? .125 * (j+1) : 0.;
        assert(previous.qd_cmd[j] == expected);
      }
    }
    state = {};
    state.n = 2;
    state.dq[0] = -100;
    state.dq[1] = 100;
    if (tick == 1) {  // initial hold, before the first command
      ctx->armed.store(true);
      ctx->arm_gen.fetch_add(1);
      ctx->last_cmd_rx_ns.store(mono_ns());
    }
    if (tick == 2) {  // opposite initial-hold torque
      state.dq[0] = 100;
      state.dq[1] = -100;
    }
    if (tick == 3 || tick == 4) {
      PoseHoldCommandPacket packet{};
      auto& command = packet.command;
      command.n = 2;
      command.version = VERSION;
      command.seq = uint32_t(tick);
      for (int j = 0; j < 2; ++j) {
        const double sign = (j == 0 ? 1 : -1) * (tick == 3 ? 1 : -1);
        command.tau_ff[j] = sign * 100;
        command.kd[j] = 20;
        command.qd_des[j] = .125 * (j+1);
        state.dq[j] = 0;
        // Echo outside the cap: post-slew clamping must still win.
        if (tick == 4) state.tau_ref[j] = sign * 10000;
      }
      ctx->cmd_in.write(packet);
      ctx->last_cmd_rx_ns.store(mono_ns());
    }
    if (tick == 5) ctx->last_cmd_rx_ns.store(mono_ns() - 200'000'000);
    if (tick == 6) ctx->last_cmd_rx_ns.store(mono_ns() - 2'000'000'000);
    if (tick == 7) {
      assert(ctx->fault_code == FAULT_CMD_LOST);
      StatePacket previous{};
      assert(ctx->state_out.read(previous));
      assert(previous.flags & FLAG_HOLDING);
      ctx->fault.store(false);
      ctx->fault_claim.store(false);
      ctx->latch(FAULT_CTL_LOST, "scripted communication fault");
      ctx->last_cmd_rx_ns.store(mono_ns());  // fresh commands cannot bypass latch
    }
    if (tick == 8) {  // plant failure suppresses torque writes
      assert(ctx->fault_code == FAULT_CTL_LOST);
      state.q[0] = std::numeric_limits<double>::quiet_NaN();
    }
    if (tick == 9) ctx->armed.store(false);
    if (tick == 10) ctx->shutdown.store(true);
    return true;
  }

  bool write(const double* tau, int count) override {
    assert(count == 2 && tick >= 1 && tick <= 7);
    ++writes;
    for (int j = 0; j < count; ++j) {
      const double cap = ctx->cfg.tau_max > 0
          ? std::min(ctx->cfg.tau_max, native_limits[j]) : native_limits[j];
      const double sign = (j == 0 ? 1 : -1) * (tick == 2 || tick == 4 ? -1 : 1);
      assert(std::isfinite(tau[j]) && std::abs(tau[j]) <= cap);
      // Tick 4 deliberately scripts tau_ref far OUTSIDE the cap, so the
      // ramp there starts from +-10000 and the final magnitude clamp is what
      // bounds it -- that tick still checks saturation below. Every other
      // tick has tau_ref == 0, so the rate ceiling is the only thing that can
      // bound the output, and it must, however reckless --slew was.
      if (slew_case && tick != 4) {
        assert(std::abs(tau[j]) <= slew_max + 1e-9);
        assert(std::abs(tau[j]) < cap);  // rate-bound, NOT magnitude-bound
        continue;
      }
      assert(tau[j] == sign * cap);  // each phase actually reaches saturation
    }
    return true;
  }

 private:
  std::string error_;
};
}  // namespace

uint64_t mono_ns() { return 10'000'000'000ull + uint64_t(tick) * 1'000'000; }
std::unique_ptr<Backend> make_fake_backend(int) {
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
  for (double cap : {27., 0., 100., .1}) {
    ServerCtx context;
    ctx = &context;
    tick = writes = stops = 0;
    context.cfg.tau_max = cap;
    context.cfg.slew = 1000;
    context.cfg.rt_priority = 0;
    rt_loop(context);
    assert(!context.failed && writes == 7 && stops == 2 && tick == 10);
  }
  // A --slew larger than the backend allows must be CLAMPED, not honoured:
  // main.cpp checks only `> 0`, so this is the guard that a launch flag
  // cannot defeat a vendor torque-rate limit.
  {
    ServerCtx context;
    ctx = &context;
    tick = writes = stops = 0;
    slew_max = 0.5;
    slew_case = true;
    context.cfg.tau_max = 0;      // magnitude cap out of the way
    context.cfg.slew = 1000;      // reckless flag
    context.cfg.rt_priority = 0;
    rt_loop(context);
    assert(!context.failed && writes == 7 && stops == 2 && tick == 10);
    slew_max = std::numeric_limits<double>::infinity();
    slew_case = false;
  }
  std::puts("PASS RT torque caps: tracking, initial/stale/fault holds, echo, "
            "disarm, backend slew ceiling");
}
