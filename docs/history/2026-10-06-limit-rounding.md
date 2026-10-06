# A goal on a joint limit no longer faults the controller

Found while verifying the FR3 profile slice (record 2026-10-06-fr3-profile.md); operator decision 2026-10-06: fix both the evaluator and the
envelope comparison.

**Defect.** A plan whose goal sits exactly on a URDF joint limit was admitted and then faulted mid-move with "non-finite or invalid servo
command" (live, supervised graph, joint 4 driven to its upper limit -0.1518: the second move faulted in 6 of 9 runs; the fault carried
q_des[3] = -0.15179999999999996). Two causes together:
1. The cubic Hermite evaluator (`JointTrajectory.sample_at`, and its native twin `rt/include/arm_rt/reference.hpp`) used the weighted-sum
   form `h00*q0 + h01*q1 + ...`. With q0 = q1 the weights sum to 1 +- ulp, so a joint held still on its limit sampled one ulp past it
   (offline: 35 of 2001 samples). The linear branch had the same form.
2. `ExecutionPolicy` compared the waypoints and every servo command against the position envelope strictly, while the certified-curve
   check already allowed 1e-8 rad. A plan the curve check admitted could therefore be refused by the servo check.

**Fix.** Both evaluators use the offset form `q0 + dq*h01 + h*(v0*h10 + v1*h11)` (and `q0 + s*dq` when linear): the same polynomial, exact
on a still segment. One rounding allowance, `ENVELOPE_TOL = 1e-8` rad, now applies to the waypoints, the curve and the servo command. That is
the value the curve check already used, not a new margin (1e-8 rad is far below encoder resolution).

**Checked.**
* `check_execution_policy.py` `check_goal_on_limit`: a still joint on its limit is sampled exactly at it (Hermite and linear, 2001 samples).
  A command one ulp past the limit, or half the allowance past it, is legal; one a micro-radian past it is refused.
* `tools/bench/rt/reference_numerics.py`: the native sampler returns the limit exactly, and the Python/native parity (2e-13) holds.
* Live (headless, the example graphs with joint 4 driven onto its upper limit, implementation/control merged in for the FR3 armature): supervised 6 of 6 runs and stock 1 of 1 complete both moves (before: the second move faulted in 6 of 9 supervised runs).
* Mutants killed: the old Python evaluator, a strict servo comparison, and the old native evaluator.

**Hardware.** The native evaluator runs on the RT box for negotiated native references. The fix needs a box rebuild and redeploy, and has not
been verified on hardware.
