# Bench tooling for the Cartesian law's hardware rungs (track-list `8173845`)

At the bench (2026-10-10, step 13) only rung 1 (IDENTITY advertises CAP_CARTESIAN_TRAJECTORY: FR3 #1 caps 0x23, PASS) could
run: nothing exposed the box's FK and nothing could send a free-space Cartesian leg. Operator: build the tooling, then finish
the session step by step.

Library `rt/src/cartesian_probe.cpp` (WITH_FRANKA only, `$ORIGIN/lib` layout like flange_probe): rung 2, read-only. One
`readOnce()`, `loadModel()`, and the same call `backend_franka.cpp`'s `ee_pose_at` makes, `model.pose(kEndEffector, q,
F_T_EE, EE_T_K)`, at the measured q, printed next to `O_T_EE` with the position/rotation difference (one model, so it should
be ~0; anything else is a frame or model mismatch), `F_T_EE`/`EE_T_K`, and the call's cost (p50/p99/max over N calls: rung 3's
estimate on the CPU that runs it). Commands nothing; the FCI allows one client, so the RT server must be stopped. The call is
restated, not shared, with the backend (one libfranka call; a change there must be mirrored). The deployed binaries are
unchanged (a rebuild of `arm_rt_server` at this commit is byte-identical to the installed one).

The project side (rungs 3, 4, 6: `tools/hw/cartesian_rungs.py`) is in the same integration. Rung 5 (a lost FK mid-leg
latches FAULT_PLANT and holds) cannot be induced on a healthy robot; it stays covered by `cartesian_loop_selfcheck` and
`franka_backend_selfcheck`.

Review round 1 (Opus xhigh, BLOCK on one MUST, fixed): the project tool could leave the arm ARMed with its supervisor
running (a failed/timed-out SDK call or a mid-call Ctrl-C skipped both the DISARM and the supervisor stop; SIGHUP ran no
cleanup at all) -> nested finally (the supervisor stop always runs; its SIGTERM DISARMs the plant), SIGTERM/SIGHUP become
KeyboardInterrupt (run.py's pattern), the hardware supervisor log is kept; the self-check now reads the RT's armed flag at
each supervisor stop and covers a mid-leg interrupt and a failing SDK DISARM. SHOULDs taken: the start pose is re-checked after
ARM and before each GO (5 mrad); rung 6's settle FAULT at very low K_c documented; rung 2's PC half (`F_T_EE` and `O_T_EE`
against the staged URDF + bundle frames, `--rung 2 --probe`); the probe's cost is the CPU it runs on (run it on the box too);
rung 3's direct instrument (the server's loop-gap journal) named; the observer's failure recorded; the self-check's
law-engagement test (corrected in round 2). NITs taken: `--joint` range, stop after a
non-completed leg, report written even on a partial run with the export root resolved first, the run.py launcher lock, FR3 #2
commands, `rung4_agree`, the probe's non-finite message. A supervisor leaked by an earlier failed self-check run (pointed at
127.0.0.1) was found by the reviewer and stopped.

Review round 2 (Opus xhigh, PASS-WITH-FIXES; the three round-1 failure paths re-run: no leaked supervisor, RT disarmed):
* SHOULD: the law-engagement test passed a mutant that sends Cartesian requests as joint ones (two joint legs at kp 20 differ
  anyway: leg 1's residual moves leg 2's start) -> it asserts the joints the trajectory keeps still deviate by > 1 mrad more
  under a stiff K_c (sim: 4.15 mrad with the law, 0.005 mrad for the mutant).
* SHOULD: the 5 mrad start check right after ARM could refuse on the real arm (the PD-only hold sags 1-7 mrad until the
  integral clears it) -> up to 2 s to settle after ARM, 5 mrad at each GO.
* NITs taken: SIGQUIT handled too; the loop-gap lines are stderr (live in journald), the RUNBOOK's reason corrected; the
  self-check no longer takes the real per-arm launcher lock; floor gains read before ARM; rung 4 adds a third leg (the joint
  leg repeated) so `rung4_agree` is read against `baseline_agree`. Left: SIGKILL/OOM of the tool or the supervisor leaves the
  RT box latching CMD/CTL_LOST and holding (the existing RT design); the self-check keeps its supervisors' /tmp logs.
