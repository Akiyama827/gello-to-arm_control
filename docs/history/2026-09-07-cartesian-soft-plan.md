# Cartesian Soft implementation plan

Approved behavior: Soft holds the measured end-effector position AND orientation
while allowing compliant, damped redundant-joint motion. Track remains joint
trajectory control; Float remains gravity compensation with joint damping.
No deployment or hardware motion is authorized by this implementation task.

Baseline: parent d65978189101dd65b0a8ad176ea91191832b8133;
arm_control a94f766c0e484fc74331c5fe3fabd812ecc133a1; gitlink matches.
The preceding console/deadman changes are uncommitted and must be preserved;
the perception submodule is independently dirty and out of scope.

## Design and reference

Use the established Cartesian spring/damper plus projected nullspace PD torque
structure in Franka's Cartesian impedance example, with fixed-size Eigen
computations in the reusable RT core. Capture the target at the PLANT on entry,
not in the planner or controller. Use measured pose and Jacobian every servo
tick. Smooth entry gains, retain torque magnitude/slew limits, and fail closed
on missing Cartesian sensing. Finite impedance is not an exact constraint.

Reference: https://github.com/frankarobotics/franka_ros/blob/develop/franka_example_controllers/src/cartesian_impedance_example_controller.cpp

## Tasks and boundaries

- [x] Shared RT law: add `rt/include/arm_rt/pose_hold.hpp`, compiled pybind
  entry and assert-based self-check. Fixed-size nullspace projection, quaternion
  sign invariance, independent damping, no heap allocations in servo evaluation.
- [x] RT transport: preserve byte-identical v1 joint/state/control packets;
  introduce capability-negotiated v2 pose-hold command (v1 command prefix plus
  15 doubles: id, kc[6], dc[6], nullspace kp, nullspace kd). New server accepts
  both, old server cannot receive Soft from the new client. Preserve watchdog,
  authority epoch and command acceptance semantics. C++/Python golden parity.
- [x] Package command contract: optional 15-double pose-hold tail, mutually
  exclusive with the existing 28-double Cartesian tail; strict validation.
  No imports from nodes or from Control/perception into arm_control.
- [x] Controller and console: explicit Soft entry only when armed and settled;
  stream the selected hold spec from the sole command producer. Soft persists
  until Stop/Disarm or another mode. Refuse plans/jog while holding Cartesian
  pose; tell operator to select Track first. Switching back captures measured
  joints, never an old joint setpoint. Keep current page layout and jog deadman.
- [x] MuJoCo: reuse the compiled law; no Python approximation of Soft. Preserve
  existing plain and Cartesian-trajectory paths, recompile/state-transfer code.
- [x] Profiles/docs: generic example profiles only in arm_control; no deployed
  robot identity or geometry moves. Label Soft accurately and expose capability.

## Verification (no pytest)

Run new assert harnesses before and after implementation. Build RT and pybind;
run `protocol_selfcheck`, Python protocol `--hex` parity, controller self-check,
console authority check, graph route/authority checks, AST/py_compile and Ruff.
Exercise a headless FR3 with an externally perturbed redundant configuration:
compare Soft against joint Track, record EE position/orientation errors and
joint displacement. Check release/disarm/re-entry and old-server rejection.
Run existing workcell headless and gated sim regression because the generic
plant is touched. Keep hardware validation explicitly separate.
