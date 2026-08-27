// Loopback plant: a damped double integrator per joint, paced at 1 kHz by an
// absolute-deadline sleep. PERMANENT test double, not bring-up scaffolding —
// it is what lets the whole server (threads, sockets, seqlocks, deadman,
// fault latching) be exercised on any machine, and what failure-injection
// tests run against. Keep it DUMB: realistic closed-loop behaviour is the
// MuJoCo twin's job on the PC, never this file's.
#include <cmath>
#include <ctime>
#include <string>

#include "arm_rt/backend.hpp"

namespace arm_rt {
namespace {

constexpr double TICK_S = 0.001;
constexpr double INERTIA = 1.0;  // kg m^2, per joint
constexpr double DAMPING = 0.8;  // N m s/rad, viscous

class FakeBackend final : public Backend {
public:
  explicit FakeBackend(int n) : n_(n) {
    for (int j = 0; j < MAX_JOINTS; ++j) tau_limit_[j] = 50.0;
    clock_gettime(CLOCK_MONOTONIC, &next_);
  }

  const char* name() const override { return "fake"; }
  int n() const override { return n_; }
  double tick_s() const override { return TICK_S; }
  const double* tau_limit() const override { return tau_limit_; }

  bool read(PlantState& out) override {
    // Absolute deadline keeps the tick rate drift-free regardless of how
    // long the servo work took (same discipline the real loop needs).
    next_.tv_nsec += 1'000'000;
    if (next_.tv_nsec >= 1'000'000'000) {
      next_.tv_nsec -= 1'000'000'000;
      next_.tv_sec += 1;
    }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_, nullptr);

    out.n = n_;
    for (int j = 0; j < n_; ++j) {
      out.q[j] = q_[j];
      out.dq[j] = dq_[j];
      out.tau[j] = tau_[j];
      out.tau_ref[j] = tau_[j];  // self-echo: last applied torque
    }
    // Cartesian sensing: EXPLICIT zeros, both flags false. This plant has no
    // geometry at all (independent per-joint integrators — there is no EE to
    // put a wrench on and no chain to build a Jacobian from), so the honest
    // answer is "not available", and the false flags keep FLAG_WRENCH_VALID
    // clear on the wire. Faking a Jacobian here would let a Cartesian bug
    // pass the loopback tests and surface first on the robot. PlantState
    // zero-initialises both arrays; the assignments below are here so that
    // stays a DECISION, not a default nobody revisited.
    for (int i = 0; i < 6; ++i) out.wrench[i] = 0.0;
    out.wrench_valid = false;
    out.jacobian_valid = false;
    return true;
  }

  bool write(const double* tau, int n) override {
    for (int j = 0; j < n; ++j) {
      tau_[j] = tau[j];
      const double qdd = (tau[j] - DAMPING * dq_[j]) / INERTIA;
      dq_[j] += qdd * TICK_S;
      q_[j] += dq_[j] * TICK_S;
    }
    return true;
  }

  void stop() override {
    for (int j = 0; j < n_; ++j) tau_[j] = 0.0;
  }

  const std::string& fault_text() const override { return fault_; }

private:
  int n_;
  double tau_limit_[MAX_JOINTS];
  double q_[MAX_JOINTS] = {};
  double dq_[MAX_JOINTS] = {};
  double tau_[MAX_JOINTS] = {};
  timespec next_ = {};
  std::string fault_;
};

} // namespace

std::unique_ptr<Backend> make_fake_backend(int n) {
  return std::make_unique<FakeBackend>(n);
}

} // namespace arm_rt
