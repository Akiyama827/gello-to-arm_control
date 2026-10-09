#pragma once
// The gripper a hand_bridge serves. One device session at a time; the bridge's
// actor thread owns read() and the blocking actions, and its stop worker calls
// stop() concurrently (it must be thread-safe and interrupt a running action).
#include <cstdint>
#include <stdexcept>

#include "hand_admission.hpp"

struct HandSample {
  double width = 0;    // m, opening between the fingers
  bool grasped = false;
  uint64_t stamp = 0;  // device time or sequence: strictly increasing per new sample
};

// The session is gone: the bridge drops it and reconnects.
struct DeviceLost : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// The action ended without success; the session is fine.
struct ActionEnded : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class HandDevice {
public:
  virtual ~HandDevice() = default;
  virtual HandSample read() = 0;  // blocks until a new sample
  virtual bool move(double width_m, double speed_mps) = 0;
  virtual bool homing() = 0;
  virtual bool grasp(double width_m, double speed_mps, double force_n,
                     double eps_in_m, double eps_out_m) = 0;
  virtual bool stop() = 0;
};
