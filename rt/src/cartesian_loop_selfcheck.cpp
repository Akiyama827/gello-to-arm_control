// The RT loop's v8 Cartesian path end to end, with a scripted plant (operator 2026-10-09): a v8 command with valid
// sensing writes exactly servo_torque of the commanded joint word whose tau_ff carries cartesian_impedance at the target
// FK(q_des); when the backend cannot evaluate FK(q_des) the loop latches FAULT_PLANT and holds. (Decode, capability and
// envelope are protocol_selfcheck's; the controller's arithmetic is controller_selfcheck's.)
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "arm_rt/servo_law.hpp"
#include "server.hpp"

namespace arm_rt {
namespace {
ServerCtx* ctx;
int tick, writes;
bool fk_ok = true, law_checked = false;
constexpr uint64_t SESSION = 7;
constexpr double limits[] = {50, 50};
const double q_meas[2] = {.1, .2}, dq_meas[2] = {.01, -.02};
const double q_cmd[2] = {.11, .18}, tau_cmd[2] = {.5, -.5}, kp_cmd[2] = {10, 10}, kd_cmd[2] = {1, 1};
// R_task = 90 deg about z (columns = task axes in base), Kc/Dc inside the envelope.
const double spec[CARTESIAN_VALUES] = {0, -1, 0, 1, 0, 0, 0, 0, 1, 300, 150, 0, 0, 0, 0, 20, 10, 0, 0, 0, 0};

// The scripted FK: x = (q0 + .3, q1 + .1, .2), identity orientation; J = d x / d q.
void fk(const double* q, double* pose) {
  const double p[7] = {q[0] + .3, q[1] + .1, .2, 1, 0, 0, 0};
  std::memcpy(pose, p, sizeof(p));
}

class CartesianBackend final : public Backend {
 public:
  const char* name() const override { return "cartesian-selfcheck"; }
  int n() const override { return 2; }
  double tick_s() const override { return .001; }
  const double* tau_limit() const override { return limits; }
  const std::string& fault_text() const override { return error_; }
  bool stop() override { return true; }
  bool supports_cartesian() const override { return true; }
  bool ee_pose_at(const double* q, double* pose) override {
    if (!fk_ok) return false;
    fk(q, pose);
    return true;
  }

  bool read(PlantState& state) override {
    ++tick;
    state = {};
    state.n = 2;
    std::copy_n(q_meas, 2, state.q);
    std::copy_n(dq_meas, 2, state.dq);
    fk(state.q, state.pose);
    state.pose_valid = true;
    state.jacobian[0 * 2 + 0] = 1;  // dx/dq0
    state.jacobian[1 * 2 + 1] = 1;  // dy/dq1
    state.jacobian_valid = true;
    if (tick == 1) begin_arm(*ctx);
    if (tick == 2 || tick == 3) {
      if (tick == 3) fk_ok = false;  // the backend loses FK(q_des) mid-leg
      CommandIngress packet{};
      auto& command = packet.command;
      command.n = 2;
      command.version = CARTESIAN_VERSION;
      command.seq = uint32_t(tick);
      std::copy_n(q_cmd, 2, command.q_des);
      std::copy_n(tau_cmd, 2, command.tau_ff);
      std::copy_n(kp_cmd, 2, command.kp);
      std::copy_n(kd_cmd, 2, command.kd);
      std::memcpy(packet.cartesian, spec, sizeof(spec));
      ctx->cmd_in.write(CommandEnvelope{packet, mono_ns(), ctx->arm_gen.load(), SESSION});
    }
    if (tick == 4) {
      assert(ctx->fault_code == FAULT_PLANT);  // latched, not a silent drop to joint PD
      StatePacket previous{};
      assert(ctx->state_out.read(previous));
      assert(previous.flags & FLAG_HOLDING);
      ctx->shutdown.store(true);
    }
    return true;
  }

  bool write(const double* tau, int count) override {
    assert(count == 2);
    ++writes;
    if (tick == 2) {  // exactly the servo law on the joint word + the Cartesian term at FK(q_des)
      double pose_des[7], twist[6] = {}, zero[6] = {}, pose[7], ff[2], want[2], raw[2];
      fk(q_cmd, pose_des);
      fk(q_meas, pose);
      double J[12] = {};
      J[0] = 1;
      J[3] = 1;
      for (int r = 0; r < 6; ++r)
        for (int j = 0; j < 2; ++j) twist[r] += J[r * 2 + j] * dq_meas[j];
      std::copy_n(tau_cmd, 2, ff);
      cartesian_impedance(2, J, spec, pose, pose + 3, pose_des, pose_des + 3, twist, zero, spec + 9, spec + 15, ff);
      const double zero_qd[2] = {0, 0}, tau_ref[2] = {0, 0};
      servo_torque(2, q_meas, dq_meas, q_cmd, zero_qd, ff, kp_cmd, kd_cmd, tau_ref, limits, ctx->cfg.slew, want, raw);
      for (int j = 0; j < 2; ++j) assert(std::fabs(tau[j] - want[j]) < 1e-12);
      assert(std::fabs(ff[0] - tau_cmd[0]) > 1e-3);  // the Cartesian term really contributed
      law_checked = true;
    }
    for (int j = 0; j < count; ++j) assert(std::isfinite(tau[j]) && std::abs(tau[j]) <= limits[j]);
    return true;
  }

 private:
  std::string error_;
};
}  // namespace

uint64_t mono_ns() { return 10'000'000'000ull + uint64_t(tick) * 1'000'000; }
std::unique_ptr<Backend> make_fake_backend(int, const std::vector<double>&) { return std::make_unique<CartesianBackend>(); }
std::unique_ptr<Backend> make_franka_backend(const std::string&, double, const double[3]) { std::abort(); }
std::unique_ptr<Backend> make_dm_backend(const std::string&, uint32_t, double) { std::abort(); }
}  // namespace arm_rt

int main() {
  using namespace arm_rt;
  ServerCtx context;
  ctx = &context;
  context.session_id.store(SESSION);
  context.cfg.tau_max = 0;     // the backend's own limits
  context.cfg.slew = 1000;     // not binding at these torques
  context.cfg.rt_priority = 0;
  rt_loop(context);
  assert(!context.failed && tick == 4 && writes >= 3 && law_checked);
  assert(context.supports_cartesian.load());
  std::puts("PASS RT v8 Cartesian loop: servo law on the joint word + Cartesian term at FK(q_des); "
            "lost FK latches FAULT_PLANT and holds");
}
