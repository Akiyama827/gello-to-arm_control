# CTRL-053: the DM arm's real_motion on its supervisor, over the RT box

Operator 2026-10-09 (the RT placement, A3): the DM arm runs on the RT box like the base. `arm_rt_server --backend dm`
drives joints 1-6 and shares isolated cpu 1 with the base. `hand_bridge --backend dm` drives the gripper (motor 7). Both
use the base's single USB2CANFD, `can0`. The two robots' CAN ids collide (base 1-2 on Master 0x11/0x12; DM 1-7 on
Master 0), and the operator's rule is one robot powered at a time. The DM motors move to a 4 Mbit/s data phase (the
base's). Motor 7 gets a distinct Master ID, flashed at the bench. Hold 20 / 0.5, tau-max 9 N.m.

* `dataflows/real_motion.yml` has the shape of `sim_motion.yml`:
  - `arm_console`, `supervised_arm`, `franka_gripper` (the hand_bridge client, at rt.host:rt.hand_port), `visualizer`;
  - `motor_state_logger` = `trace_logger` (the motor CSV from trace/1).
  The in-process `arm_controller` and the PC-side DM driver node are gone from it. GRASP is refused (`uncalibrated_force`)
  until rung 11 measures the gripper's force map.
* `rt/src/backend_dm.cpp`, the shared-bus interlock, runs on a second, unfiltered CAN socket (the probe):
  - Before the first frame (`LISTEN_NS` 100 ms, passive): a frame to one of our motor ids means another driver is
    commanding them. Refuse to start.
  - The first `PROBE_NS` 100 ms after start: one of our ids answering on a Master ID that is not ours means another
    robot's motor, powered. Latch a fault: a read failure, whose path disables any enabled motor (A2's checked path).
  Afterwards the probe is closed and the 500 Hz path is unchanged. DM motors speak only when spoken to, so the passive
  listen sees only drivers. The probe's own frames come back by loopback and are skipped once we have sent.
  Limits:
  - it detects only during those windows. A server for the other robot is refused by the units' Conflicts, but the
    other robot POWERED after the window is caught by nothing in software: the main socket filters to our Master, so the
    base's motor 1 would take our joint 1 commands silently. Only the operator rule covers it. Operator 2026-10-09:
    accepted, startup-only (the DM arm is nearly obsolete and is not used with the base);
  - it cannot tell two robots apart if they share a Master ID (the base is 0x11/0x12, the DM arm 0: distinct today).

Checks: `dm_backend_selfcheck` (new shared-bus cases: another driver before start refused; another robot's motor answering
on a foreign Master latches a fault with no motor enabled; our own looped-back frames, our replies, and another
device's id 7 are ignored). Not covered: the probe closing after its window (the bench sees it). Mutants killed: no
passive listen, no Master check. `check_dataflows`.

Project (pinned with this):
* registry instance `dm_arm`: units `arm-dm-server` (udp/tcp 47830/47831, cpu 1, Conflicts with the base and, in the
  experimental dual_franka mode, fr3_2) and `arm-dm-hand-bridge` (47832). Neither unit has [Install], so neither starts
  at boot.
* entry `configs/entries/real_assembler_rt.yaml`: six motors, so no gripper slot; the FR3 execution policy file.
* `tools/run.py real motion` is supervised over the RT plant, staging the registry instance.
* `check_supervised_real` covers the DM launch.

Not run: anything on the box or on CAN. The fake RT server has no gravity, which the DM controller feeds forward, so no
ARM offline (its closed loop is the MuJoCo twin's: `check_supervised_sims`).

Bench, unqualified: cpu 1 jitter with the DM loop; the 4 Mbit/s reflash and `hardware.yaml`'s `canfd_baudrate` (the
PC-side driver, after the reflash); the motor 7 Master ID; the interlock against the powered base; the units' install
(operator); `rt.state_timeout_s` 0.2 / `rt.max_fault_ms` 5000, taken from the FR3/base and not yet confirmed for the
DM arm.

Review: stopped by the operator before it reported (2026-10-09); merged UNREVIEWED, ahead of removing the DM arm from the stack.
