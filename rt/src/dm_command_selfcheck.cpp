// Exercise native command dispatch through the real RT loop. No device access.
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "server.hpp"

namespace arm_rt {
namespace {
ServerCtx* ctx;
int tick, writes, stops;
constexpr double limits[] = {28, 28};
constexpr double targets[] = {0.12, -0.78};
constexpr double velocity[] = {0.2, -0.3};
constexpr double feedforward[] = {1.5, -2.0};
constexpr double stiffness[] = {50, 60};
constexpr double damping[] = {2, 3};

class CommandBackend final : public Backend {
 public:
  const char* name() const override { return "native-command-selfcheck"; }
  int n() const override { return 2; }
  double tick_s() const override { return .002; }
  const double* tau_limit() const override { return limits; }
  const std::string& fault_text() const override { return error_; }
  void stop() override { ++stops; }
  bool write(const double*, int) override {
    std::fputs("FAIL: RT discarded native MIT fields and called torque-only write\n", stderr);
    std::abort();
  }

  bool read(PlantState& state) override {
    ++tick;
    state = {};
    state.n = 2;
    state.q[0] = .1;
    state.q[1] = -.75;
    if (tick == 1) {
      ctx->armed.store(true);
      ctx->arm_gen.fetch_add(1);
      ctx->last_cmd_rx_ns.store(mono_ns());
    }
    if (tick == 2) {
      PoseHoldCommandPacket packet{};
      auto& cmd = packet.command;
      cmd.n = 2;
      cmd.version = VERSION;
      cmd.seq = 42;
      for (int j = 0; j < 2; ++j) {
        cmd.q_des[j] = targets[j]; cmd.qd_des[j] = velocity[j];
        cmd.tau_ff[j] = feedforward[j];
        cmd.kp[j] = stiffness[j]; cmd.kd[j] = damping[j];
      }
      ctx->cmd_in.write(packet);
      ctx->last_cmd_rx_ns.store(mono_ns());
    }
    // No new PC packet at tick 3: RT must still forward the native command.
    if (tick == 4) ctx->last_cmd_rx_ns.store(mono_ns() - 200'000'000);
    if (tick == 5) ctx->last_cmd_rx_ns.store(mono_ns() - 2'000'000'000);
    if (tick == 6) {
      assert(ctx->fault_code == FAULT_CMD_LOST);
      // A fresh stray zero-gain packet cannot change the latched hold.
      PoseHoldCommandPacket stray{};
      stray.command.n = 2;
      ctx->cmd_in.write(stray);
      ctx->last_cmd_rx_ns.store(mono_ns());
    }
    if (tick == 7) ctx->armed.store(false);
    if (tick == 8) ctx->shutdown.store(true);
    return true;
  }

  bool write_command(const CommandPacket& command, double slew, double* tau) override {
    assert(command.n == 2 && slew == ctx->cfg.slew);
    ++writes;
    for (int j = 0; j < 2; ++j) {
      if (tick == 2 || tick == 3) {
        assert(command.q_des[j] == targets[j]);
        assert(command.qd_des[j] == velocity[j]);
        assert(command.tau_ff[j] == feedforward[j]);  // no external PD added
        assert(command.kp[j] == stiffness[j] && command.kd[j] == damping[j]);
      } else {
        assert(command.q_des[j] == (j == 0 ? .1 : -.75));
        assert(command.qd_des[j] == 0 && command.tau_ff[j] == 0);
        assert(command.kp[j] == (tick == 1 ? ctx->cfg.hold_kp : stiffness[j]));
        assert(command.kd[j] == (tick == 1 ? ctx->cfg.hold_kd : damping[j]));
      }
      tau[j] = .125 * (j + 1);  // native backend reports its encoded prediction
    }
    return true;
  }

 private:
  std::string error_;
};
}  // namespace

uint64_t mono_ns() { return 10'000'000'000ull + uint64_t(tick) * 2'000'000; }
std::unique_ptr<Backend> make_fake_backend(int) {
  return std::make_unique<CommandBackend>();
}
std::unique_ptr<Backend> make_franka_backend(const std::string&, double, const double[3]) {
  std::abort();
}
std::unique_ptr<Backend> make_dm_backend(const std::string&, uint32_t, double) {
  std::abort();
}
}  // namespace arm_rt

int main() {
  using namespace arm_rt;
  ServerCtx context;
  ctx = &context;
  context.cfg.tau_max = 27;
  context.cfg.rt_priority = 0;
  rt_loop(context);
  assert(writes == 6 && stops == 2 && tick == 8);
  std::puts("PASS native RT dispatch: tracking, repeated reference, holds, latch, disarm");
}
