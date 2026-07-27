// Plant backends the RT loop drives. One interface, three implementations:
// fake (loopback integrator — the permanent test double), franka (libfranka
// torque mode), dm (FDCAN, skeleton). The backend owns loop PACING: read()
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
};

class Backend {
public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  virtual int n() const = 0;
  virtual double tick_s() const = 0;                 // nominal servo period
  virtual const double* tau_limit() const = 0;       // per-joint, length n

  // Block until the next tick's state is available. False = plant fault
  // (reason via fault_text()). Errors surface HERE, on the read side.
  virtual bool read(PlantState& out) = 0;

  // Apply this tick's torque (already clamped + slewed by the caller).
  virtual bool write(const double* tau, int n) = 0;

  // Drop to zero authority safely (disarm path). Idempotent.
  virtual void stop() = 0;

  virtual const std::string& fault_text() const = 0;
};

// Constructors live in their backend_*.cpp; rt_loop.cpp declares and picks.
// (No central factory: the franka one must be constructed ON the RT thread.)

} // namespace arm_rt
