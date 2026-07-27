// The torque servo law — THE shared truth.
//
// Header-only and dependency-free so the exact same compiled code runs in
// three places: the RT thread (1 kHz, real arm), the pybind module the MuJoCo
// plant calls (sim twin), and any offline check. If the law ever forks
// between sim and real, they drift — that is the failure this file exists to
// prevent.
//
// Per joint:   tau = kp (q_des - q) + kd (qd_des - dq) + tau_ff
// then clamp to ±tau_limit, then SLEW-limit against tau_ref.
//
// tau_ref is the robot's own echo of the last ACCEPTED torque (libfranka:
// state.tau_J_d) — not our previous output. That distinction is what
// survives dropped ticks: after a gap the ramp resumes from what the robot
// is actually doing. It also makes the limiter the startup ramp — first tick
// passes the measured tau_ref and the output climbs from there, so there is
// no separate soft-start path to get wrong. (FR3 budget: libfranka's own
// limiter allows ~1 N·m per 1 ms tick per joint; stay at or under it.)
#pragma once

namespace arm_rt {

inline double clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// All arrays length n. Writes tau_out; safe to alias tau_out with tau_ref? No
// — caller keeps them distinct (asserted by convention, not code: this runs
// at RT priority, branches are budget).
inline void servo_torque(int n,
                         const double* q, const double* dq,
                         const double* q_des, const double* qd_des,
                         const double* tau_ff,
                         const double* kp, const double* kd,
                         const double* tau_ref,
                         const double* tau_limit,
                         double slew_per_tick,
                         double* tau_out) {
  for (int j = 0; j < n; ++j) {
    // (clamp runs before the slew; the final re-clamp below closes the case
    // where tau_ref itself sits outside the limit — impossible for franka's
    // tau_J_d and the fake's self-echo, cheap insurance for the DM bus)
    double t = kp[j] * (q_des[j] - q[j]) + kd[j] * (qd_des[j] - dq[j]) + tau_ff[j];
    t = clamp(t, -tau_limit[j], tau_limit[j]);
    t = clamp(t, tau_ref[j] - slew_per_tick, tau_ref[j] + slew_per_tick);
    tau_out[j] = clamp(t, -tau_limit[j], tau_limit[j]);
  }
}

} // namespace arm_rt
