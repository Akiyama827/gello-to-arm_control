// Exercise native command dispatch through the real RT loop. No device access.
#include <vector>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "server.hpp"
#include "arm_rt/pose_hold.hpp"

namespace arm_rt {
namespace {
ServerCtx* ctx;
int tick, writes, stops;
bool pose_case=false;
PoseHold expected_pose;
PlantState measured;
constexpr double limits[] = {28, 28};
constexpr double targets[] = {0.12, -0.78};
constexpr double velocity[] = {0.2, -0.3};
constexpr double feedforward[] = {1.5, -2.0};
constexpr double stiffness[] = {50, 60};
constexpr double damping[] = {2, 3};
constexpr uint64_t SESSION = 7;
uint64_t skew = 0;  // simulated elapsed time on top of the per-tick clock
CommandEnvelope last_command{};
// No packet for `by` ns: time passes, so the last command's own ingress stamp
// and this generation's deadline both age.
void age_stream(uint64_t by) { skew += by; }

class CommandBackend final : public Backend {
 public:
  const char* name() const override { return "native-command-selfcheck"; }
  int n() const override { return 2; }
  double tick_s() const override { return .002; }
  const double* tau_limit() const override { return limits; }
  const std::string& fault_text() const override { return error_; }
  bool stop() override { ++stops; return true; }
  bool supports_pose_hold() const override { return pose_case; }
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
    if (pose_case) return read_pose(state);
    if (tick == 1) {
      begin_arm(*ctx);
    }
    if (tick == 2) {
      CommandIngress packet{};
      auto& cmd = packet.command;
      cmd.n = 2;
      cmd.version = VERSION;
      cmd.seq = 42;
      for (int j = 0; j < 2; ++j) {
        cmd.q_des[j] = targets[j]; cmd.qd_des[j] = velocity[j];
        cmd.tau_ff[j] = feedforward[j];
        cmd.kp[j] = stiffness[j]; cmd.kd[j] = damping[j];
      }
      last_command = CommandEnvelope{packet, mono_ns(), ctx->arm_gen.load(), SESSION};
      ctx->cmd_in.write(last_command);
    }
    // No new PC packet at tick 3: RT must still forward the native command.
    if (tick == 4) age_stream(200'000'000);
    if (tick == 5) age_stream(2'000'000'000);
    if (tick == 6) {
      assert(ctx->fault_code == FAULT_CMD_LOST);
      // A fresh stray zero-gain packet cannot change the latched hold.
      CommandIngress stray{};
      stray.command.n = 2;
      ctx->cmd_in.write(CommandEnvelope{stray, mono_ns(), ctx->arm_gen.load(), SESSION});
    }
    if (tick == 7) ctx->armed.store(false);
    if (tick == 8) ctx->shutdown.store(true);
    return true;
  }

  bool write_command(const CommandPacket& command, double slew, double* tau) override {
    assert(command.n == 2 && slew == ctx->cfg.slew);
    ++writes;
    if(pose_case) {
      if(tick==2 || tick==3 || tick==5) {
        double expected[2]{};
        assert(expected_pose.torque(2,measured.q,measured.dq,measured.jacobian,
          measured.pose,measured.coriolis,last_command.packet.pose_hold,.002,expected));
        for(int j=0;j<2;++j) {
          assert(command.q_des[j]==measured.q[j] && command.qd_des[j]==0);
          assert(command.kp[j]==0 && command.kd[j]==0);
          assert(command.tau_ff[j]==expected[j]);
        }
      } else {
        expected_pose.reset();
        for(int j=0;j<2;++j) {
          assert(command.q_des[j]==(tick==4?targets[j]:measured.q[j]));
          assert(command.tau_ff[j]==(tick==4?feedforward[j]:0));
          assert(command.kp[j]==(tick==1?ctx->cfg.hold_kp:stiffness[j]));
          assert(command.kd[j]==(tick==1?ctx->cfg.hold_kd:damping[j]));
        }
      }
      return true;
    }
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
  bool read_pose(PlantState& state) {
    state.pose_valid=tick!=6;
    state.jacobian_valid=true;
    state.pose[0]=.01*tick; state.pose[3]=1.;
    state.jacobian[0]=1.; state.jacobian[3]=1.;
    state.coriolis[0]=.3; state.coriolis[1]=-.2;
    measured=state;
    if(tick==1) begin_arm(*ctx);
    if(tick==2 || tick==4 || tick==5) {
      CommandIngress packet{};
      packet.command.n=2;
      packet.command.version=tick==4?VERSION:POSE_HOLD_VERSION;
      packet.command.seq=uint32_t(tick);
      double spec[15]{10,100,100,100,10,10,10,20,20,20,2,2,2,.5,2};
      std::copy_n(spec,15,packet.pose_hold);
      for(int j=0;j<2;++j) {
        packet.command.q_des[j]=targets[j]; packet.command.qd_des[j]=velocity[j];
        packet.command.tau_ff[j]=feedforward[j];
        packet.command.kp[j]=stiffness[j]; packet.command.kd[j]=damping[j];
      }
      last_command=CommandEnvelope{packet,mono_ns(),ctx->arm_gen.load(),SESSION};
      ctx->cmd_in.write(last_command);
    }
    if(tick>2) {
      StatePacket previous{}; ctx->state_out.read(previous);
      assert(previous.last_cmd_seq==uint32_t(tick<5?2:tick==5?4:5));
      if(tick==7) assert(previous.flags & FLAG_FAULTED);
    }
    if(tick==7) { assert(ctx->fault_code==FAULT_PLANT); ctx->armed.store(false); }
    if(tick==8) ctx->shutdown.store(true);
    return true;
  }
  std::string error_;
};
}  // namespace

uint64_t mono_ns() { return 10'000'000'000ull + uint64_t(tick) * 2'000'000 + skew; }
std::unique_ptr<Backend> make_fake_backend(int, const std::vector<double>&) {
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
  context.session_id.store(SESSION);
  context.cfg.tau_max = 27;
  context.cfg.rt_priority = 0;
  rt_loop(context);
  assert(writes == 6 && stops == 2 && tick == 8);
  std::puts("PASS native RT dispatch: tracking, repeated reference, holds, latch, disarm");
  ServerCtx modes;
  ctx=&modes; pose_case=true; tick=writes=stops=0; skew=0;
  modes.session_id.store(SESSION); modes.cfg.rt_priority=0;
  rt_loop(modes);
  assert(writes==6 && stops==2 && tick==8);
  std::puts("PASS RT modes: pose activation/echo, joint cancel, same-id recapture, invalid-pose fallback");
}
