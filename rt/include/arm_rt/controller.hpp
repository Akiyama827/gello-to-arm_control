#pragma once

#include <algorithm>
#include <cmath>
#include "backend.hpp"
#include "pose_hold.hpp"
#include "servo_law.hpp"

namespace arm_rt {

// The existing native joint/pose laws produce one selected five-field command.
// This replaces rt_loop's PoseHold state, mode field mapping and backend packet
// packing. Admission, measured-hold capture, limiting and output stay outside.
// Joint feedforward already includes the PC-owned dynamics/integral: no local
// RNEA or second integral is introduced here. n is the admitted backend count.
class Controller {
public:
  void reset() { pose_.reset(); }

  void joint(int n, const double* q_des, const double* qd_des,
             const double* tau_ff, const double* kp, const double* kd) {
    pose_.reset();
    selected_ = {};
    selected_.n = uint16_t(n);
    std::copy_n(q_des, n, selected_.q_des);
    std::copy_n(qd_des, n, selected_.qd_des);
    std::copy_n(tau_ff, n, selected_.tau_ff);
    std::copy_n(kp, n, selected_.kp);
    std::copy_n(kd, n, selected_.kd);
  }

  bool pose(int n, const PlantState& state, const double* spec,
            double dt, bool supported) {
    selected_ = {};
    selected_.n = uint16_t(n);
    if (!supported || !state.pose_valid || !state.jacobian_valid ||
        !pose_.torque(n, state.q, state.dq, state.jacobian, state.pose,
                      state.coriolis, spec, dt, selected_.tau_ff)) return false;
    // Task law already includes its feedback torque. No second joint PD.
    std::copy_n(state.q, n, selected_.q_des);
    return true;
  }

  // v8: the commanded joint word, plus the Cartesian impedance folded into its feedforward (operator 2026-10-09). The
  // target is FK(q_des) by the SAME model that measured the pose (pose_des, from Backend::ee_pose_at), so a model or
  // frame mismatch cannot appear as a spring force; twist = J dq, desired twist zero (the twin's convention). The joint
  // PD stays: Kc/Dc only add stiffness along the task axes. False (missing pose/J/FK) = the caller latches a fault.
  bool cartesian(int n, const PlantState& state, const CommandPacket& cmd, const double* spec,
                 const double* pose_des, bool supported) {
    pose_.reset();
    selected_ = {};
    selected_.n = uint16_t(n);
    if (!supported || pose_des == nullptr || !state.pose_valid || !state.jacobian_valid) return false;
    for (int i = 0; i < 6 * n; ++i) if (!std::isfinite(state.jacobian[i])) return false;
    for (int i = 0; i < 7; ++i) if (!std::isfinite(state.pose[i]) || !std::isfinite(pose_des[i])) return false;
    std::copy_n(cmd.q_des, n, selected_.q_des);
    std::copy_n(cmd.qd_des, n, selected_.qd_des);
    std::copy_n(cmd.tau_ff, n, selected_.tau_ff);
    std::copy_n(cmd.kp, n, selected_.kp);
    std::copy_n(cmd.kd, n, selected_.kd);
    double twist[6] = {}, zero[6] = {};
    for (int r = 0; r < 6; ++r)
      for (int j = 0; j < n; ++j) twist[r] += state.jacobian[r * n + j] * state.dq[j];
    cartesian_impedance(n, state.jacobian, spec, state.pose, state.pose + 3, pose_des, pose_des + 3,
                        twist, zero, spec + 9, spec + 15, selected_.tau_ff);
    return true;
  }

  // Torque backends apply servo_torque to this; native MIT backends receive
  // these original fields, never the external PD+limited torque as tau_ff.
  const CommandPacket& command() const { return selected_; }

private:
  PoseHold pose_;
  CommandPacket selected_{};
};
} // namespace arm_rt
