# Native DM MIT command forwarding

> **For agentic workers:** Use superpowers:executing-plans for the scoped steps below. Work directly on main as explicitly requested; do not create a worktree or branch.

**Goal:** Send Python's position, velocity, gains and feedforward to DM's onboard MIT controller instead of replacing them with an externally computed torque.

**Architecture:** Python retains model ownership, q/qd/qdd sampling and dynamics, including model changes. The existing wire protocol is unchanged (qdd stays in Python). RT retains authority, freshness, holds and sampled torque/slew protection. Franka/fake continue using the existing torque backend. DM encodes native gains, never external PD as an additional feedforward term.

**Tech stack:** Existing C++17 RT server, SocketCAN, Python codec parity and assert-based self-checks. No pytest or new dependencies.

## Baseline and scope

- Parent main: `a176636824964d76a7d61fa0ba8cd554ca026e8b`.
- Library main and parent gitlink: `ac0225dbcf7c4682e6ae9fe2e821e5ab1404db25`.
- Prior fix branches already merged/deleted. Parent retains main/dev; library main only.
- Preserve unrelated dirty `Control/evidence/console_hand_grasp/deployment-20260908.md`.
- No RT model loading, interpolation, protocol changes, perception changes, or Franka deployment.
- No live hardware execution. Stage an immutable base-only static binary over SSH; stop before sudo/service changes.

## Command and safety contract

Normal native frames equal the existing Python MIT encoding of all five fields. Validate all active motors before any enable/command write; reject nonfinite/out-of-wire-range inputs, including gains, before encoding. Never silently clip gains.

Preserve the existing limiter as a safety exception: decode the quantized native command, predict `kp*(q_des-q)+kd*(qd_des-dq)+Tff` using fresh measured state, and keep it inside both the torque cap and the per-tick slew interval. Only if outside that interval, adjust the feedforward code to bring the prediction inside; leave position, velocity and gain codes unchanged. If no representable feedforward code can satisfy the interval, fail closed and disable rather than transmit an unsafe command. This is limiter correction, not model dynamics or double PD.

The prediction is not a guarantee of actual onboard torque between samples. DM `tau_cmd`/slew reference report the encoded command's predicted total at the last feedback sample, not a measured or motor-accepted torque. Actual torque remains the feedback field. Physical motor-side current limits require separate verification.

Holds continue capturing measured pose, using the last accepted gains, zero desired velocity and zero nominal feedforward, subject to the same native limiter. Disarm, plant faults, invalid native commands and failed writes must not leave the previous MIT authority running.

## Steps

- [x] Add a no-device C++ check of actual RT dispatch, tracking, hold, stale/fault epochs and disarm. Observe the old torque-only dispatch fail.
- [x] Add `Backend::write_command(const CommandPacket&, double slew, double* tau_out)` with the existing torque write as default. RT passes its already-authorized five-field command through it. No Franka/fake changes.
- [x] Override DM command writing; preflight complete batches and encode native fields. Extend `--mit-check` with unchanged-field, no-double-PD, cap/slew, quantization and invalid-input assertions.
- [x] Verify Release/static builds, existing torque/pose/hand self-checks, Python/C++ protocol and MIT parity. Check no model/interpolation/protocol dependency was introduced.
- [x] Update RT documentation with native DM semantics and sampled-limit qualifications. Independent review completed; both findings corrected and checked.
- Commit library on main, then record its gitlink and deployment evidence in Control without staging unrelated evidence.
- Stage the exact committed static binary and current unit under a new commit-named RT directory, verifying hashes. Do not install, restart, ARM or move hardware. Record staging completion and privileged deployment commands in the consuming project's deployment note.

## Verification commands

```bash
cmake -S libs/arm_control/rt -B /tmp/arm-dm-native-check -DWITH_FRANKA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_EXE_LINKER_FLAGS=-static
cmake --build /tmp/arm-dm-native-check -j 2
/tmp/arm-dm-native-check/dm_command_selfcheck
/tmp/arm-dm-native-check/dm_backend_selfcheck
/tmp/arm-dm-native-check/arm_rt_server --mit-check
PYTHONPATH=libs/arm_control python libs/arm_control/tools/bench/rt/torque_cap.py /tmp/arm-dm-native-check/arm_rt_server
/tmp/arm-dm-native-check/pose_hold_selfcheck
/tmp/arm-dm-native-check/hand_admission_selfcheck
git -C libs/arm_control diff --check
```

## Evidence

The commands above passed on 2026-09-09. `pose_hold_selfcheck` exits zero without
printing. The scripted RT checks deliberately use normal scheduling and print
the existing SCHED_FIFO warning; these are correctness checks, not timing evidence.

Red evidence: `dm_command_selfcheck` compiled against the original RT loop and
aborted with `RT discarded native MIT fields and called torque-only write`.
The reserved-control-word check also failed before its guard was added.
The final static build passes both regressions.

`dm_backend_selfcheck` wraps SocketCAN syscalls at link time, exercising the
actual backend without device access. It covers native frame content, all-batch
validation, deferred dynamic enable, repeated commands, partial enable/write
failure, read failure, and wrong joint count. Review found the original dynamic
enable bypass and reserved-payload collision; both are fixed and covered.

Additional commands and results:

```bash
env -u ARM_RT_DEMO_HOST PYTHONPATH=libs/arm_control ARM_RT_SERVER_BIN=/tmp/arm-dm-native-check/arm_rt_server python -m arm_control.plants.remote_rt.client
cmake -S libs/arm_control/rt -B /tmp/arm-dm-native-franka-check -DWITH_FRANKA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_PREFIX_PATH=/home/ubu/opt/libfranka-0.21.2/usr
cmake --build /tmp/arm-dm-native-franka-check -j 2
/tmp/arm-dm-native-franka-check/hand_bridge_selfcheck
/tmp/arm-dm-native-franka-check/pose_hold_selfcheck
python tools/inspect/dm_spec.py
git -C libs/arm_control diff --check
git diff --check
```

The local fake-client check passed protocol parity, tracking (0.1 mrad reported
error), stale hold, fault latching and DISARM/ARM recovery. The Franka-enabled
build and no-hardware Hand/pose checks passed. The config-derived DM specification
still matches the project unit exactly. No Python implementation or protocol
files changed.

The cap harness initially failed under sandbox socket restrictions; its approved
local-only rerun passed. The Hand socket-pair check also failed in its sandbox
run and passed outside that restriction. No real Gripper was constructed.

Static base-only binary SHA256:
`0c106c4403052c9444d9d7366a4e3564bb768505d10072217561c04ffc729f7d`.
This binary is x86-64 and statically linked; the Franka-enabled build is only
a compile/regression check and is not the deployment artifact.
