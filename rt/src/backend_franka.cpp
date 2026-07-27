// FR3 torque backend over libfranka's external-control (ActiveControl) API.
//
// STATUS: compiles against libfranka >= 0.13-style ActiveControl headers and
// encodes every constraint from the 2026-07-26 franka_ros2/libfranka review,
// but it is UNVALIDATED ON HARDWARE — the first live run is a bring-up rung
// (gravity-hold, operator on the stop), not a formality. Review notes it
// implements:
//
//  - Robot is constructed ON THE LOOP THREAD: libfranka raises the calling
//    thread's scheduling in the constructor, so construct-then-move loses RT
//    priority. main.cpp therefore builds this backend inside the RT thread.
//  - Errors surface on readOnce(), not writeOnce() — read() is where faults
//    are caught and reported.
//  - tau_ref for the slew limiter is state.tau_J_d (the robot's echo of the
//    last ACCEPTED torque), which survives dropped packets; the limiter IS
//    the startup ramp, first tick included.
//  - libfranka 0.21 applies NO rate limit and NO low-pass in writeOnce —
//    the docs claiming otherwise are stale. The servo law's slew is the only
//    limiter; keep it at or under ~1 N.m per 1 ms tick.
//  - The robot adds gravity + motor friction to the commanded torque itself;
//    tau_ff arriving from the PC must be RNEA MINUS gravity (Coriolis ours).
//    That subtraction happens PC-side in the executor, not here.
//  - stop() is NOT zero torque: a loaded arm drops on zero. It re-sends the
//    last accepted torque with motion_finished, which libfranka takes as a
//    controlled stop.
#include <string>

#include "arm_rt/backend.hpp"

#ifdef ARM_RT_WITH_FRANKA

#include <franka/active_control.h>
#include <franka/exception.h>
#include <franka/robot.h>

namespace arm_rt {
namespace {

constexpr int FR3_N = 7;
constexpr double FR3_TAU_LIMIT[MAX_JOINTS] = {87, 87, 87, 87, 12, 12, 12};

class FrankaBackend final : public Backend {
public:
  explicit FrankaBackend(const std::string& ip)
      : robot_(ip, franka::RealtimeConfig::kEnforce) {
    // Bring-up-friendly collision thresholds; the arm stops on light contact
    // and the PC clears the reflex through the control channel's re-arm.
    robot_.setCollisionBehavior(
        {{20, 20, 18, 18, 16, 14, 12}}, {{40, 40, 36, 36, 32, 28, 24}},
        {{20, 20, 20, 25, 25, 25}}, {{40, 40, 40, 50, 50, 50}});
    control_ = robot_.startTorqueControl();
  }

  const char* name() const override { return "franka"; }
  int n() const override { return FR3_N; }
  double tick_s() const override { return 0.001; }
  const double* tau_limit() const override { return FR3_TAU_LIMIT; }

  bool read(PlantState& out) override {
    try {
      auto [state, duration] = control_->readOnce();  // paces the loop
      out.n = FR3_N;
      for (int j = 0; j < FR3_N; ++j) {
        out.q[j] = state.q[j];
        out.dq[j] = state.dq[j];
        out.tau[j] = state.tau_J[j];
        out.tau_ref[j] = state.tau_J_d[j];
        last_tau_ref_[j] = state.tau_J_d[j];
      }
      return true;
    } catch (const franka::Exception& exc) {
      fault_ = exc.what();
      return false;
    }
  }

  bool write(const double* tau, int n) override {
    try {
      franka::Torques torques{{0, 0, 0, 0, 0, 0, 0}};
      for (int j = 0; j < n && j < FR3_N; ++j) torques.tau_J[j] = tau[j];
      control_->writeOnce(torques);
      return true;
    } catch (const franka::Exception& exc) {
      fault_ = exc.what();
      return false;
    }
  }

  void stop() override {
    // Controlled stop: last accepted torque + motion_finished. Never zero.
    try {
      franka::Torques torques{{0, 0, 0, 0, 0, 0, 0}};
      for (int j = 0; j < FR3_N; ++j) torques.tau_J[j] = last_tau_ref_[j];
      torques.motion_finished = true;
      control_->writeOnce(torques);
    } catch (const franka::Exception& exc) {
      fault_ = exc.what();
    }
  }

  const std::string& fault_text() const override { return fault_; }

private:
  franka::Robot robot_;
  std::unique_ptr<franka::ActiveControlBase> control_;
  double last_tau_ref_[MAX_JOINTS] = {};
  std::string fault_;
};

} // namespace

std::unique_ptr<Backend> make_franka_backend(const std::string& ip) {
  return std::make_unique<FrankaBackend>(ip);
}

} // namespace arm_rt

#else // !ARM_RT_WITH_FRANKA

namespace arm_rt {
std::unique_ptr<Backend> make_franka_backend(const std::string&) {
  return nullptr; // built without libfranka; main.cpp reports it
}
} // namespace arm_rt

#endif
