# Interactive 3D teleop (motion mode)

> The node is now `nodes/arm_console.py`, and what it is ALLOWED to make the arm
> do -- the command-owner split, gain presets, and the jog safety envelope --
> lives in [operator-console.md](operator-console.md). This file stays the
> reference for the PAGE: what it draws and how.

It serves an interactive 3D page at `http://<arm-pc>:7500`
(`nodes/console/index.html`). The node stays stdlib-HTTP; the page uses
three.js (pinned r160, CDN import map — the desk browser needs internet, the
arm PC does not) with `STLLoader` + `TransformControls`, the standard stack
for browser robot teleop. Everything downstream of the target
(Plan + preview → Execute) is unchanged.

## What the page shows (same color protocol as Rerun)

- **Real STL colors** — the live measured arm (from `motor_state`).
- **Orange (translucent)** — the target you are setting.
- **Green (translucent)** — planned-motion playback, looping after a
  successful Plan + preview; cleared on Execute/Stop.
- Fingers on every robot mirror the gripper slider.

The node serves `/scene` (mesh list + colors), `/mesh/<k>` (STL bytes) and
per-poll FK poses in `/state` — the browser does zero kinematics; Pinocchio
visual-model FK (`VisualFK`) runs server-side, same machinery as the Rerun
ghosts.

## Dragging the end effector

A `TransformControls` gizmo rides the EE. Only single-axis handles are wired
(**one direction of the 6 DOF at a time** — plane/screen handles are
ignored): arrows translate, rings rotate; buttons switch mode. Each drag
event POSTs to `/cart`:

- translation: absolute base-frame coordinate on the dragged axis;
- rotation: signed world-axis angle **increments** (composed server-side
  onto the latched rotation — no rpy, no gimbal trouble; unconsumed
  increments accumulate rather than drop).

The server runs damped-LS IK (`PinocchioIK`, `restarts=1` — only the
current-target seed, so the arm follows the nearest branch and never
fold-jumps under your hand) and writes the solution into the joint target.
The untouched axes are latched at drag-streak start so IK tolerance can't
drift them over a long drag. Unreachable → log line, gizmo snaps back to the
real EE pose on the next poll. Reachability is IK's job; self-collision
stays the planner's — motion still only happens via Plan + preview →
Execute.

Joint sliders still exist behind the **debug** toggle (same `/sliders`
protocol); they are the ground truth the planner plans to.

## Gripper

One slider (finger opening, metres, `joint_mimics`-calibrated range).
Published immediately (no plan/execute) on the `gripper` output in 2-finger
`motor_command_gripper` wire format: sim → straight to the MuJoCo plant's
finger servos; real → the trajectory executor maps finger metres → gripper
motor rad and holds that target in the 7th motor slot (kp 5 gentle hold —
grasping belongs to the grasp gate).

## Calibration end-goal

This panel is the pose-dialing tool for kinematic calibration data:

1. Drag a target, Plan + Execute, let the arm settle.
2. `motor_state_logger` (see `dataflows/real_gripper_log.yml` for wiring)
   appends the measured joint-state CSV alongside.
3. Each settled pose yields a pair: commanded target q / FK pose vs measured
   q. Later, with mocap on the EE, the same settled poses become
   (measured q, mocap EE pose) pairs — the input for calibrating link
   lengths / joint offsets against reality.

Settled poses + the motor-state CSV are the future mocap calibration pairs;
keep the workflow "dial, settle, log" rather than continuous jogging.
