# Motion bench checklist — MoveIt-lite bring-up (Jetson)

Goal: first planned, self-collision-checked motion on the real arm.
Safety posture throughout: graph comes up DISARMED; deadman (0.1 s) + fault
reflex live in command mode; executor aborts on >0.35 rad tracking error;
**Ctrl-C is always safe** (bridge sends zero-torque + disable on close).

## A. Sync + install (once)

- [ ] Jetson: `git pull` (desk must be pushed first)
- [ ] Jetson rob env: `pip install mujoco ompl rerun-sdk`
- [ ] Verify: `python -c "import mujoco, ompl; print('ok')"`
  - (MuJoCo ships aarch64 wheels — the old Drake-wheel Jetson risk is gone.)
- [ ] `PYTHONPATH=. python -m pytest tests/test_motion_planning.py tests/test_motion_executor_node.py -q`
      (proves mujoco+ompl work here; no mesh-conversion cache anymore)

## A2. Sim rehearsal (desk OR Jetson, no hardware)

- [ ] `python scripts/view.py sim motion` — same nodes against the MuJoCo plant.
      Click through the whole flow once: sliders -> Plan + preview -> Execute
      -> watch the sim arm follow in Rerun -> Stop (hold) mid-motion.
      (Open the teleop control-panel URL for sliders/buttons; all visuals —
      target robot, green live ghost, orange plan ghost — render in Rerun.)

## B. Pre-power sanity

- [ ] Motor supply kill within reach; bench area clear
- [ ] `python scripts/view.py real listen` — all 7 motors reporting,
      Rerun FK matches the physical pose, hand-move a joint and watch it track
- [ ] Exit

## C. Regression: float still works after the pull

- [ ] `python scripts/view.py real float` — hand-guide OK
      (validates gravity model + CAN path before anything commands motion)

## D. First motion session (start from a mid-workspace pose, not folded)

- [ ] `python scripts/view.py real motion`
      Expect in the logs: bridge `command mode — DISARMED`,
      `[arm_controller] idle — DISARMED`, `[arm_console] control panel at http://...:7500`
- [ ] Desk browser → `http://<jetson-ip>:7001` — robot renders,
      sliders equal the measured pose (auto-synced on first state)
- [ ] **Disarmed dry-run:** small target (+0.3 rad on Joint1) → `Plan + preview`
      → preview animation plays, terminal prints `plan OK`
- [ ] **Runaway-guard check (safe, nothing can move):** press `Execute` while
      still DISARMED → arm stays put; within ~1–2 s the executor prints
      `ABORT: tracking error ...` and drops back to the measured-pose hold.
      This is the guard that prevents a jump if you ever arm late.
- [ ] **ARM:** press Enter in the launch terminal, or `touch /tmp/arm_operator_arm`
      → `[safety] ARMED — motors enabled`. Gains + gravity ff ramp in over
      `safety.arm_ramp_sec` (1 s): expect a slight sag that firms up — NOT a
      twitch. A hard jerk at arm = gravity model or gains issue; stop and check.
- [ ] Gentle push on a link: resists softly and returns (kp=20 is deliberately soft)
- [ ] **First move:** target +0.3 rad on Joint6 (smallest link) → Plan + preview
      → check the preview → Execute. Watch Rerun `Position` vs `Position Cmd`.
- [ ] **Stop test:** start a slow multi-joint move, press `Stop (hold)` mid-motion
      → arm stops and is compliant (grab it to confirm)
- [ ] To disarm at any point: Ctrl-C the graph (motors zero-torque + disable)

## E. Gain ramp (you drive this)

Edit `configs/modes/motion.yaml` → `controller.kp/kd` between runs; restart graph.
- kp up per joint while tracking lag is too big; back off at first buzz/oscillation
- kd up if a joint overshoots/rings at waypoint stops (trapezoid stops at each one)
- Encode caps: kp ≤ 500, kd ≤ 5. Consistent small terminal error ⇒ stiction, not gains.
- [ ] Record the final table (mirror into `arm.kp`/`arm.kd` in
      `configs/real/assembler.yaml` if M1 grasp should share it)

## F. Planner exercise

- [ ] Multi-joint target across the workspace → plan → preview → execute
- [ ] Fold-region target (J2 high + J4 high, wrist bent) → expect
      `plan FAILED: target pose is in self-collision` — refusal is correct
- [ ] Near-limit targets: no physical self-contact anywhere. Hulls are
      conservative, but fingers are modeled slider-only — keep margin near
      the gripper.
- [ ] Note any pose where the real arm gets closer to itself than the preview
      suggested → we tighten padding / regen finger meshes

## Exit criteria

- [ ] Planned multi-joint motion tracks within tolerance at the tuned gains
- [ ] Stop, runaway-abort, deadman (kill executor process while armed → bridge
      latches disarm) all verified
- [ ] Gain table recorded
