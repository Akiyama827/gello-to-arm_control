// Plant backends the RT loop drives. One interface, three implementations:
// fake (loopback integrator — the permanent test double), franka (libfranka
// torque mode), dm (native MIT over FDCAN). The backend owns loop PACING: read()
// blocks until this tick's state is due (libfranka's readOnce, the bus, or a
// deadline sleep), which is what keeps the servo phase-locked to the plant.
#pragma once

#include <cstdint>
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
};

class Backend {
public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  virtual int n() const = 0;
  virtual double tick_s() const = 0;                 // nominal servo period
  virtual const double* tau_limit() const = 0;       // per-joint, length n
  virtual bool supports_pose_hold() const { return false; }

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

  // Drop to zero authority safely (disarm path). Idempotent.
  virtual void stop() = 0;

  virtual const std::string& fault_text() const = 0;
};

// Constructors live in their backend_*.cpp; rt_loop.cpp declares and picks.
// (No central factory: the franka one must be constructed ON the RT thread.)

} // namespace arm_rt
