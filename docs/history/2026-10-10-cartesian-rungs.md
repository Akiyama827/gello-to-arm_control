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
