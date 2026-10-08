# RT Cartesian-impedance law, slice 1: the RT box runs the twin's law (command v8)

Operator decisions 2026-10-09 (step 2c is blocked on the bench's Cartesian `final_approach` leg; the twin docked with a law the
hardware could not run): implement the REAL RT Cartesian law; wire bounds = pose hold's; Cartesian legs are streamed only (the
native reference stays joint-only); lost pose/J/FK mid-leg = FAULT_PLANT + hold; the RT computes the target as FK(q_des) with its
own model, so no pose crosses the wire.

Wire (capability-negotiated, pose hold's pattern):
* Command version 8 = the v6-sized joint word + a 21-double tail: `task_R` (3x3 row-major, the task axes as COLUMNS, robot BASE
  frame), `kc[6]`, `dc[6]` (task-frame diagonals [x, y, z, rx, ry, rz]). 840 bytes. `CAP_CARTESIAN_TRAJECTORY = 1 << 5`, advertised
  in the IDENTITY capabilities only (no HELLO token; see review round 1).
* `valid_cartesian_spec` / `cartesian_values` (C++ / Python, the same rules): all finite; `task_R` orthonormal within 1e-6 and
  det +1 within 1e-6; `kc` <= 1000 N/m, 100 N.m/rad; `dc` <= 200 N.s/m, 50 N.m.s/rad; nonnegative. Python also requires
  `frame == "base"`: a world-frame spec is refused before it is packed.
* `decode_command` returns a `CommandIngress` (joint word + the version's tail): a v8 packet on a server without the capability,
  of the wrong size, or out of the envelope is dropped, as v6 is.

Law: on a fresh v8 command the controller takes the joint word as is (q_des, qd_des, tau_ff, kp, kd) and adds
`cartesian_impedance(J, task_R, x, FK(q_des), J.dq, twist_des = 0, kc, dc)` to its tau_ff; the unchanged `servo_torque` owns the
rest (joint PD, clamp, slew). It is the same compiled function the MuJoCo twin uses (`servo_law.hpp`). Like pose hold it is
recomputed every tick the stream is fresh, against that tick's measured pose and Jacobian. No measured pose or Jacobian, a
non-finite one, a backend without `supports_cartesian`, or `ee_pose_at(q_des)` failing -> FAULT_PLANT and hold.

Backends: `Backend::supports_cartesian()` / `ee_pose_at(q, pose)` default to no. The Franka backend says yes and evaluates
`model.pose(kEndEffector, q, F_T_EE, EE_T_K)` with the frames of the last state it read (the same composition libfranka
uses for `O_T_EE`, the measured pose) and refuses before the first read or on a non-finite result. The fake and DM backends
do not advertise it. **Unvalidated on hardware**: the per-tick cost of `model.pose` inside the 1 kHz loop, and the FK/measured
agreement on the real arm, are hardware rungs (operator stops). Nothing here is qualified.

Python: `pack_command(..., cartesian=...)` (exclusive with `pose_hold`); `RemoteRTClient.supports_cartesian` from the identity
capabilities; `apply_command` refuses a Cartesian spec unless the server advertises v8 and the spec is base-frame.

Not in this slice: the PC side producing v8 (ArmController, supervisor, SDK, SimLink/twin FK(q_des) with the world->base
conversion, the native reference refusing Cartesian legs) is slice 2; the bench's axial kc 2000 -> 1000 N/m and the sim dock
re-measure are slice 3. Slice 2 must also pin the end-effector frame: on the FR3 the RT's FK(q_des) and measured pose are
libfranka's EE (Desk's `F_T_EE`), while the twin's are the PC model's `ee_frame`; the spec's task frame means the same thing
on both only if those agree (the 2026-07-26 read-only check found 0.00 mm with `fr3_hand_tcp`).

Checks (related only): all 22 RT ctest self-checks (WITH_FRANKA=OFF; the franka backend through its mock), including:
* `controller_selfcheck` v8: tau_ff is bit-identical to the joint word plus a direct law call; zero gains give exactly the joint
  word; refusals for unsupported, no FK, invalid pose, invalid/NaN J and NaN FK.
* `protocol_selfcheck`: golden `CARTESIAN` hex and size 840; the capability gate, short and mis-sized packets, and bad values at
  tail indices 0, 4 (NaN), 8 (det -1), 9, 10, 12, 15, 16 (inf), 18, 20 are dropped.
* `franka_backend_selfcheck`: `supports_cartesian`; `ee_pose_at` refuses before `read()`, and after it returns the mock
  model's pose.
* `cartesian_loop_selfcheck` (new): the real `rt_loop` with a scripted plant -- the written torque equals `servo_torque` of the
  joint word with the law at FK(q_des) in tau_ff; a lost FK latches FAULT_PLANT and the loop holds. Mutant caught: the loop
  passing no FK target.
* Python: golden parity with the C++ (11/11 lines); `python -m arm_control.contracts.impedance` (new, in the library suite)
  mirrors the C++ refusals index by index plus frame/keys; the client loopback demo against the fake server: no capability
  advertised, a base spec and a world spec are refused, a world spec is refused even when advertised; ruff.

Review round 1: ACCEPT (22/22 ctest; WITH_FRANKA=ON compiles against libfranka 0.21.2 and `Model::pose(Frame, q, F_T_EE,
EE_T_K)` is column-major; golden lines identical; 7 of 9 mutants caught). Taken:
* The HELLO capability buffer (24 chars) cut a Franka's ` pose_hold=6 cartesian=8` to `cartesian=`, and the extra token spent
  the 104-byte HELLO text so a dual-arm Franka's `rt=` could be truncated. The token is gone: the client already took the
  capability from IDENTITY, which overrides the HELLO, and the server's decoder gates on its own flag. No safety effect.
* The franka mock's `pose` ignored the frame and `F_T_EE`/`EE_T_K`, so a backend computing the FLANGE pose (about 0.1 m off
  `O_T_EE` with the Hand, a ~100 N bias at kc 1000) passed. The mock now chains them along z with FR3-like offsets and the
  check expects the EE (TCP) frame; mutants caught: a zero `F_T_EE`, `kStiffness`.
* The UDP receive buffer is a union of every datagram `handle_datagram` accepts (each at an exact size), so it holds the
  largest by construction; a buffer smaller than an accepted size (reviewer mutant: the old v6 buffer with an 840-byte v8
  datagram -> `got` 840, decode reading 48 bytes past it) can no longer be written without dropping a member.
* `apply_command` refuses a command with both tails as `RtLinkError` (was a `ValueError` from the packer); demo checks it.
Open for slice 2 (operator decision; a control-law/gain question): after a v8 FAULT_PLANT or a lost stream the RT holds with the
v8 word's joint kp/kd snapshot (as v5/v6). If Cartesian legs run with low joint kp and let kc carry the stiffness, that hold is
soft (the FR3 floats under the control box's gravity compensation); a hold-gain floor or the leg's joint gains must be decided
before the dock leg runs.
