// Plant backends the RT loop drives. One interface, three implementations:
// fake (loopback integrator — the permanent test double), franka (libfranka
// torque mode), dm (native MIT over FDCAN). The backend owns loop PACING: read()
// blocks until this tick's state is due (libfranka's readOnce, the bus, or a
// deadline sleep), which is what keeps the servo phase-locked to the plant.
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "protocol.hpp"

namespace arm_rt {

struct PlantState {
  int n = 0;
  double q[MAX_JOINTS] = {};
  double dq[MAX_JOINTS] = {};
  double tau[MAX_JOINTS] = {};     // measured joint torque
  double tau_ref[MAX_JOINTS] = {}; // robot's echo of last ACCEPTED torque
                                   // (franka: tau_J_d) — the slew reference
                                   // DM: last encoded total prediction, not echo
  // Cartesian sensing. INTERNAL struct — never on the wire (the StatePacket
  // ships only the 6 wrench numbers, into a field that already exists), so
  // the 96-double Jacobian costs nothing but stack.
  double wrench[6] = {};  // external wrench on the EE, ROBOT BASE frame,
                          // [fx,fy,fz,tx,ty,tz]. Sign follows libfranka:
                          // POSITIVE = force the robot applies TO the world.
  bool wrench_valid = false;  // gates StatePacket's FLAG_WRENCH_VALID
  // EE Jacobian, base frame. Buffer is 6 x MAX_JOINTS so it fits any joint
  // count, but the rows are PACKED AT STRIDE n (J[r*n + c]) — exactly the
  // row-major layout arm_rt::cartesian_impedance documents, so this array can
  // be handed to the law with no repack. Slots past 6*n are unused.
  double jacobian[6 * MAX_JOINTS] = {};
  bool jacobian_valid = false;
  double pose[7] = {};  // measured EE pose: xyz,wxyz, base frame
  double coriolis[MAX_JOINTS] = {};  // robot supplies gravity internally
  bool pose_valid = false;
  // Robot-side clock of the returned sample (franka: RobotState::time) and
  // the robot's view of our command delivery. 0 / -1 when the plant has none.
  // Diagnostics only (loop-gap log); never on the wire.
  uint64_t robot_time_ms = 0;
  double cmd_success_rate = -1.0;
};

class Backend {
public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  virtual int n() const = 0;
  virtual double tick_s() const = 0;                 // nominal servo period
  virtual const double* tau_limit() const = 0;       // per-joint, length n

  // Vendor ceiling on per-tick torque CHANGE, N.m per tick. main.cpp only
  // checks --slew > 0, so a launch flag could otherwise ask for a torque step
  // the robot refuses (or worse, accepts). The RT loop clamps the configured
  // slew against this exactly as it clamps --tau-max against tau_limit():
  // a flag may only ever be MORE conservative than the plant's own limit.
  // Unconstrained by default; a backend with a published torque-rate limit
  // overrides.
  virtual double tau_slew_max() const {
    return std::numeric_limits<double>::infinity();
  }

  virtual bool supports_pose_hold() const { return false; }

  // The v8 Cartesian law's target: this backend's EE pose (xyz, wxyz, base frame) at a COMMANDED q, with the same
  // model read() measures with. Called on the RT thread after read(). False = no geometry: the law fails closed.
  virtual bool supports_cartesian() const { return false; }
  virtual bool ee_pose_at(const double* q, double* pose) { (void)q; (void)pose; return false; }

  // The kinematic model the RT side evaluates for task-space feedback (pose
  // hold's pose/Jacobian). "none" when the backend has no geometry. This is
  // reported, never a claim of calibration (see CAP_CALIBRATED_TASK_SPACE).
  virtual const char* task_model() const { return "none"; }

  // Fixed slots are configured once. A backend may report a subset online
  // and activate a subset for torque; conventional arms keep all slots live.
  virtual uint32_t online_mask() const {
    return n() >= 16 ? 0xFFFFu : ((1u << n()) - 1u);
  }
  virtual uint32_t active_mask() const { return online_mask(); }
  virtual bool set_active_mask(uint32_t mask) {
    return mask == active_mask();
  }

  // Block until the next tick's state is available. False = plant fault
  // (reason via fault_text()). Errors surface HERE, on the read side.
  virtual bool read(PlantState& out) = 0;

  // Apply this tick's torque (already clamped + slewed by the caller).
  virtual bool write(const double* tau, int n) = 0;

  // The authorized joint command, including locally synthesized holds.
  // Torque plants keep the existing law/output. Native MIT plants preserve
  // the five fields and update tau_out with their encoded torque prediction.
  virtual bool write_command(const CommandPacket& command, double /*slew*/,
                             double* tau_out) {
    return write(tau_out, command.n);
  }

  // Drop to zero authority safely (disarm path). Idempotent. False = the
  // stop itself failed (reason via fault_text()); the RT loop journals it as
  // a secondary event, separate from the latched first cause.
  virtual bool stop() = 0;

  virtual const std::string& fault_text() const = 0;
};

// Constructors live in their backend_*.cpp; rt_loop.cpp declares and picks.
// (No central factory: the franka one must be constructed ON the RT thread.)

} // namespace arm_rt
