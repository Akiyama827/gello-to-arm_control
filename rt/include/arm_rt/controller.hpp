#pragma once

#include <algorithm>
#include "backend.hpp"
#include "pose_hold.hpp"

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

  // Torque backends apply servo_torque to this; native MIT backends receive
  // these original fields, never the external PD+limited torque as tau_ff.
  const CommandPacket& command() const { return selected_; }

private:
  PoseHold pose_;
  CommandPacket selected_{};
};
} // namespace arm_rt
