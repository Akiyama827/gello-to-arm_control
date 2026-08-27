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
//  - The torque session is LAZY: while the server is disarmed, rt_loop reads
//    but never writes, and an open ActiveControl session would trip the
//    robot's control watchdog within milliseconds. Disarmed reads use the
//    plain 1 kHz state stream; the session opens on the first write (arm)
//    and closes on stop() (disarm). A session error drops back to idle reads
//    so the operator keeps seeing q/tau while the server is latched.
//  - A fault inside a session marks need_recovery_; the next arm runs
//    automaticErrorRecovery() first, so DISARM->ARM clears a collision
//    reflex. A reflex the robot entered while we were idle costs one failed
//    arm attempt (which sets the flag) — the second arm recovers it.
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
//  - loadModel() runs ONCE, in the constructor: it downloads the model
//    library from the control box (seconds of blocking network work). read()
//    then fills PlantState's Jacobian and external-wrench estimate from it —
//    the Cartesian sensing the impedance law and contact detection need.
#include <array>
#include <cstdio>
#include <string>

#include "arm_rt/backend.hpp"

#ifdef ARM_RT_WITH_FRANKA

#include <franka/active_control.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

namespace arm_rt {
namespace {

constexpr int FR3_N = 7;
constexpr double FR3_TAU_LIMIT[MAX_JOINTS] = {87, 87, 87, 87, 12, 12, 12};

class FrankaBackend final : public Backend {
public:
  // model_ is initialised from robot_ — DECLARATION ORDER matters (robot_ is
  // declared first). loadModel() DOWNLOADS the model library from the control
  // box, which is seconds of blocking network work: it must happen here, in
  // the constructor, and never inside a control loop.
  FrankaBackend(const std::string& ip, double ee_mass, const double ee_com[3])
      : robot_(ip, franka::RealtimeConfig::kEnforce), model_(robot_.loadModel()) {
    fault_.reserve(256);  // latch path must not allocate on the RT thread
    // Payload declaration. setLoad writes m_load, which is ADDITIVE to the
    // Desk-configured end effector (the Hand's m_ee) — pass the camera+mount
    // or module mass alone, never hand+camera. It is a non-realtime command:
    // legal here in the constructor, illegal once a control session is open,
    // which is why it happens once at startup and never on the arm path.
    // The inertia tensor may NOT be zero: the control box rejects a nonzero
    // mass with a zero tensor outright ("Set Load command rejected: invalid
    // argument!" — probed against this FR3 on 2026-08-06; diag(2e-4) for the
    // same 0.2 kg is accepted). Approximate the payload as a solid sphere of
    // radius R about its CoM, I = 2/5 m R^2, which is diagonal and therefore
    // trivially valid. Gravity compensation — the only thing float and hold
    // depend on — does not use it; give the real tensor here if a payload
    // ever gets heavy enough for its dynamics to matter.
    constexpr double R = 0.05;  // m, camera/module scale
    const double i = 0.4 * ee_mass * R * R;
    if (ee_mass > 0.0) {
      robot_.setLoad(ee_mass, {{ee_com[0], ee_com[1], ee_com[2]}},
                     {{i, 0, 0, 0, i, 0, 0, 0, i}});
      std::printf("[rt] franka payload declared: %.3f kg at flange "
                  "[%.3f %.3f %.3f] m\n",
                  ee_mass, ee_com[0], ee_com[1], ee_com[2]);
    } else {
      std::printf("[rt] franka payload: none declared (--ee-mass 0) — Desk's "
                  "end-effector config is the whole model\n");
    }
    // Bring-up-friendly collision thresholds: firm hand contact is fine,
    // a hard shove trips the reflex — which is the safe outcome; the
    // operator clears it with a DISARM->ARM cycle.
    robot_.setCollisionBehavior(
        {{20, 20, 18, 18, 16, 14, 12}}, {{40, 40, 36, 36, 32, 28, 24}},
        {{20, 20, 20, 25, 25, 25}}, {{40, 40, 40, 50, 50, 50}});
  }

  const char* name() const override { return "franka"; }
  int n() const override { return FR3_N; }
  double tick_s() const override { return 0.001; }
  const double* tau_limit() const override { return FR3_TAU_LIMIT; }

  bool read(PlantState& out) override {
    try {
      franka::RobotState state =
          active_ ? control_->readOnce().first  // paces the armed loop
                  : robot_.readOnce();          // idle stream, no session
      out.n = FR3_N;
      for (int j = 0; j < FR3_N; ++j) {
        out.q[j] = state.q[j];
        out.dq[j] = state.dq[j];
        out.tau[j] = state.tau_J[j];
        out.tau_ref[j] = state.tau_J_d[j];
        last_tau_ref_[j] = state.tau_J_d[j];
      }
      // O_F_ext_hat_K, NOT K_F_ext_hat_K. Both are the same estimate; the
      // choice is which frame it arrives in. Base frame wins because it is
      // the frame EVERYTHING else here already lives in: zeroJacobian is
      // base-frame, the servo's Cartesian task frame is fixed in the base
      // (a dock's insertion axis does not move with the wrist), and the PC
      // reads the wrench off the wire without a pose to rotate it by. The
      // stiffness frame K would need the live EE orientation applied at
      // every consumer — one more place to get a convention wrong, for no
      // information gained. Sign is libfranka's: POSITIVE = the robot
      // pushing on the world, so an insertion push reads positive along the
      // approach axis.
      for (int i = 0; i < 6; ++i) out.wrench[i] = state.O_F_ext_hat_K[i];
      out.wrench_valid = true;
      // zeroJacobian is COLUMN-major 6x7; PlantState wants ROW-major packed
      // at stride n. Transpose here — the one place the conversion lives.
      const std::array<double, 42> jac =
          model_.zeroJacobian(franka::Frame::kEndEffector, state);
      for (int r = 0; r < 6; ++r)
        for (int c = 0; c < FR3_N; ++c) out.jacobian[r * FR3_N + c] = jac[c * 6 + r];
      out.jacobian_valid = true;
      return true;
    } catch (const franka::Exception& exc) {
      fail(exc);
      return false;
    }
  }

  bool write(const double* tau, int n) override {
    try {
      if (!active_) {
        if (need_recovery_) {
          try {
            robot_.automaticErrorRecovery();
          } catch (const franka::Exception&) {
            // nothing to recover is fine; a real refusal fails startTorqueControl
          }
          need_recovery_ = false;
        }
        control_ = robot_.startTorqueControl();
        active_ = true;        // ownership and flag must never disagree: if
                               // the readOnce below throws, fail() must see
                               // active_ and release the live session
        control_->readOnce();  // ActiveControl: a write must follow a read
      }
      franka::Torques torques{{0, 0, 0, 0, 0, 0, 0}};
      for (int j = 0; j < n && j < FR3_N; ++j) torques.tau_J[j] = tau[j];
      control_->writeOnce(torques);
      return true;
    } catch (const franka::Exception& exc) {
      fail(exc);
      return false;
    }
  }

  void stop() override {
    // Controlled stop: last accepted torque + motion_finished. Never zero.
    if (!active_) return;
    try {
      franka::Torques torques{{0, 0, 0, 0, 0, 0, 0}};
      for (int j = 0; j < FR3_N; ++j) torques.tau_J[j] = last_tau_ref_[j];
      torques.motion_finished = true;
      control_->writeOnce(torques);
    } catch (const franka::Exception& exc) {
      fault_ = exc.what();
      need_recovery_ = true;
    }
    control_.reset();
    active_ = false;
  }

  const std::string& fault_text() const override { return fault_; }

private:
  void fail(const franka::Exception& exc) {
    // Unconditional: an error with NO session open (reflex entered while
    // idle, startTorqueControl refused) must ALSO set need_recovery_, or the
    // documented DISARM->ARM reflex recovery is unreachable — every re-arm
    // repeats the refused start forever. A spurious automaticErrorRecovery()
    // after a mere network blip is a harmless no-op.
    fault_ = exc.what();
    control_.reset();  // back to idle reads so state keeps flowing
    active_ = false;
    need_recovery_ = true;
  }

  franka::Robot robot_;
  franka::Model model_;  // MUST stay declared after robot_ (see the ctor)
  std::unique_ptr<franka::ActiveControlBase> control_;
  bool active_ = false;
  bool need_recovery_ = false;
  double last_tau_ref_[MAX_JOINTS] = {};
  std::string fault_;
};

} // namespace

std::unique_ptr<Backend> make_franka_backend(const std::string& ip, double ee_mass,
                                             const double ee_com[3]) {
  try {
    return std::make_unique<FrankaBackend>(ip, ee_mass, ee_com);
  } catch (const franka::Exception& exc) {
    // FCI off, robot unreachable, no RT permission — report, exit nonzero,
    // and let systemd's Restart=on-failure keep knocking until it's there.
    std::fprintf(stderr, "[rt] franka backend: %s\n", exc.what());
    return nullptr;
  }
}

} // namespace arm_rt

#else // !ARM_RT_WITH_FRANKA

namespace arm_rt {
std::unique_ptr<Backend> make_franka_backend(const std::string&, double,
                                             const double[3]) {
  std::fprintf(stderr, "[rt] built without libfranka (-DWITH_FRANKA=ON)\n");
  return nullptr;
}
} // namespace arm_rt

#endif
