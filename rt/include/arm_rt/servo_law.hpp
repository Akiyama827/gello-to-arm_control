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

// ---------------------------------------------------------------------------
// Cartesian impedance — an anisotropic spring expressed in the task frame
// (stiff along the insertion axis, soft laterally, so the connector's lead-in
// mechanically funnels the part instead of the servo jamming it).
//
//   f    = R Kc R^T e_x + R Dc R^T edot_x        (R = blkdiag(R_task, R_task))
//   tau += J^T f
//
// ADDITIVE ON PURPOSE. This does NOT clamp or slew: the caller folds the
// result into tau_ff and the UNCHANGED servo_torque above still owns the
// per-joint limit and the ramp. That ordering is the whole safety argument —
// a Cartesian spring can ask for any torque it likes and still cannot outrun
// the joint limit, and a stiffness step still ramps instead of stepping.
//
// CONVENTIONS, pinned here because a wrong one is a silent wrong-sign spring
// rather than a crash:
//
//  * JACOBIAN LAYOUT: 6 x n, ROW-MAJOR — J[r*n + c]. Rows 0..2 linear,
//    rows 3..5 angular. Row-major because that is what both plants hand us
//    with a plain stacked copy: MuJoCo's mj_jacBody fills jacp/jacr as
//    3 x nv row-major. libfranka's zeroJacobian is COLUMN-major, so
//    backend_franka.cpp transposes on the way in — one conversion, one place.
//
//  * FRAME: J, the poses and the twists are all in the ROBOT BASE frame
//    (mj_jacBody and zeroJacobian are both base-frame; the sim's base frame
//    is the MuJoCo world). Kc/Dc are the ONLY task-frame quantities.
//
//  * POSES: [x,y,z] + quaternion [w,x,y,z]. Orientation error is the
//    axis-angle of q_err = q_des * conj(q), i.e. the rotation carrying
//    CURRENT to DESIRED — the same direction as e_pos = x_des - x, so both
//    halves of e_x push the same way. e_rot = 2 * sign(w_err) * vec(q_err).
//    The sign(w_err) factor picks the SHORTEST ARC: q and -q are the same
//    rotation, and without it a target just past 180 deg springs the long way
//    round. The small-angle 2*vec form (rather than
//    2*atan2(|vec|,w)*vec/|vec|) is deliberate — exact to third order,
//    branch- and libm-free, and a 1 kHz impedance is never meant to be
//    tracking a 170 deg error in the first place.
//
//  * Kc/Dc are TASK-FRAME diagonals [kx,ky,kz,krx,kry,krz]. R_task is 3x3
//    ROW-MAJOR with the task frame's axes as its COLUMNS in base coordinates,
//    so R_task^T maps a base vector into the task frame. Which task axis is
//    the insertion axis is the CALLER's convention (ours: task X), not this
//    function's — it only rotates.
//
//  * COMPATIBILITY: all-zero Kc and Dc reduce this to a no-op EXACTLY. Every
//    path ends in a multiply by 0.0, so tau_acc[j] += 0.0. The single
//    observable difference from not calling it at all is that a tau_acc of
//    -0.0 becomes +0.0 — same value, and no arithmetic downstream can tell.
//    (Callers still skip the call entirely when no Cartesian block is
//    configured; this guarantee is the belt to that suspenders.)
inline void cartesian_impedance(int n,
                                const double* J,        // 6 x n, row-major
                                const double* R_task,   // 3 x 3, row-major
                                const double* x,        // 3, base frame
                                const double* quat,     // 4, [w,x,y,z]
                                const double* x_des,    // 3
                                const double* quat_des, // 4
                                const double* twist,     // 6, [v; w] base
                                const double* twist_des, // 6
                                const double* Kc,        // 6, task frame
                                const double* Dc,        // 6, task frame
                                double* tau_acc) {       // n, ACCUMULATED
  double e[6], ed[6];
  e[0] = x_des[0] - x[0];
  e[1] = x_des[1] - x[1];
  e[2] = x_des[2] - x[2];
  // q_err = q_des * conj(q)  (Hamilton product, [w,x,y,z])
  const double aw = quat_des[0], ax = quat_des[1], ay = quat_des[2], az = quat_des[3];
  const double bw = quat[0], bx = -quat[1], by = -quat[2], bz = -quat[3];
  const double ew = aw * bw - ax * bx - ay * by - az * bz;
  const double s = ew < 0.0 ? -2.0 : 2.0;  // shortest arc
  e[3] = s * (aw * bx + ax * bw + ay * bz - az * by);
  e[4] = s * (aw * by - ax * bz + ay * bw + az * bx);
  e[5] = s * (aw * bz + ax * by - ay * bx + az * bw);
  for (int i = 0; i < 6; ++i) ed[i] = twist_des[i] - twist[i];

  double f[6];
  for (int b = 0; b < 6; b += 3) {  // linear block, then angular block
    double et[3], edt[3], ft[3];
    for (int i = 0; i < 3; ++i) {  // R^T v: dot v with COLUMN i of R_task
      et[i] = R_task[i] * e[b] + R_task[3 + i] * e[b + 1] + R_task[6 + i] * e[b + 2];
      edt[i] = R_task[i] * ed[b] + R_task[3 + i] * ed[b + 1] + R_task[6 + i] * ed[b + 2];
    }
    for (int i = 0; i < 3; ++i) ft[i] = Kc[b + i] * et[i] + Dc[b + i] * edt[i];
    for (int i = 0; i < 3; ++i)  // back to base: R ft
      f[b + i] = R_task[3 * i] * ft[0] + R_task[3 * i + 1] * ft[1] + R_task[3 * i + 2] * ft[2];
  }
  for (int j = 0; j < n; ++j) {  // tau += J^T f
    double t = 0.0;
    for (int r = 0; r < 6; ++r) t += J[r * n + j] * f[r];
    tau_acc[j] += t;
  }
}

} // namespace arm_rt
