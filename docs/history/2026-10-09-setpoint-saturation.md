# setpoint/1 saturation: a streamed target moves no faster than planned motion may

Operator 2026-10-09: clamp (not refuse) a setpoint's rate, from the policy's existing limits; asked whether that is standard --
it is (ros2_control's joint-limits `JointSaturationLimiter`, libfranka's `limitRate`, MoveIt Servo's velocity scaling all
saturate per control cycle). Before: a single 0.3 rad setpoint peaked at 2.26 rad/s against the policy's 1.0 (setpoint/1
review); planned motion is admitted against those limits, streamed targets were not.

* `control/jog_hold.py`: `JogRamp`: per tick the commanded target moves toward the requested one at no more than the velocity
  limit, changes speed by no more than the acceleration limit, and brakes to stop on the target (the discrete braking cap: the
  continuous sqrt(2 a d) overshoots at tick resolution, found by the self-check at 4.8 rad/s^2 against 2). It snaps onto the
  target only when that is itself within the acceleration limit, else brakes through and comes back: speed and acceleration
  stay within the limits per tick, for held targets and jittery streams alike (asserted). No jerk
  term (no jerk limit is stated anywhere; the RT server slews torque). Velocity is not fed forward: the word stays the joint
  law's hold word, as the jog always was (feeding it forward would be a control-law change).
* `control/arm_controller.py`: the jog path (console jog and setpoint/1 alike) runs through the ramp when the execution policy
  states velocity and acceleration limits; it restarts at the measured pose, at rest, whenever no jog is live (none yet,
  expired, cleared, a leg released). Without a policy (the legacy in-process DM path) nothing changes.
* Effect on the console: a joint jog (0.15 rad/s) is followed at its speed, about v^2/2a behind (13 mrad at the FR3's 0.4/0.8);
  on release it brakes (about 18 ticks) instead of stopping on a step. The bounds are per control tick (a late tick is slower,
  an early or catch-up tick briefly faster in wall-clock terms). Expiry, the measured-pose re-anchor and the abort_pos_err_rad refusal are unchanged.

Checks: `jog_hold` self-check (100 random steps: speed <= limit, acceleration <= limit but for the final stop, lands on target;
console following; restart at rest); `check_dora_arm`: one 0.3 rad setpoint on the twin peaks <= the policy's 1.0 rad/s
(mutant "no ramp" caught at 2.19); the owner-loss test now unclaims mid-travel (the ramp alone had made the slower motion pass
without the clear: mutant re-caught).

Review round 1: BLOCK. Taken:
* (BLOCK) The ramp reset only on a tick that found no jog: a jog arriving within the tick a jog was cleared (a plan released then
  cancelled, `clear_jog`) resumed the stale ramp -- reviewer's harness: a 0.25 rad commanded step back toward the pre-plan pose,
  under abort_pos_err_rad so sent. A NEW jog (none live) now resets the ramp in `on_jog`: it starts at the measured pose, at
  rest. New `tools/check/check_jog_ramp.py` drives that sequence on a real ArmController (S1 plan release+cancel, S2 clear_jog;
  mutant "no reset" caught: 0.027 commanded vs 0.271 measured).
* (SHOULD) The snap onto the target cut a moving target's speed in one tick (up to 93x the acceleration limit on a jittery
  stream): bounded as above; the self-check now asserts speed and acceleration per tick with per-joint limits for held targets
  and random-walk streams (sigma 1/10/50 mrad per tick).
* NITs: the console lag figure (13 mrad at the FR3's limits, not "a few"); the `[:n_arm]` slice was a no-op (removed); the
  bounds are per tick (stated).

Review round 2: BLOCK. The strict snap left a fixed target's last tick braking THROUGH by ~a dt^2 (1e-5 rad at the FR3's
0.4/0.8): a setpoint ON a joint limit (admitted) then commanded past the envelope and the controller stopped and DISARMed (3 of
5 distances; the round-1 ramp held) -- and a target that jumps to just ahead of the ramp brakes through by up to v^2/2a (0.16
rad). Taken (the reviewer's option b, the root): the ramp carries the policy's position envelope and never commands further
past a limit than the target or its own current command is -- at a limit it stops, as a position saturation does; an arm already
beyond a stop is still jogged off it smoothly. Speed and acceleration stay within the limits away from a limit (fuzz: worst
1.0x); the stop onto a limit is the one exception. `check_jog_ramp` S3: a setpoint on a limit from five distances at 0.4/0.8 is
reached, never passed, the arm stays ARMed (mutant "no envelope" caught at 1.0000059); the self-check covers the jump-ahead and
the escape from beyond a stop.

Review round 3: ACCEPT (limit, stale-ramp and fuzz harnesses re-run; escape from beyond a stop smooth; never past a limit). NITs
taken: the stop onto a limit spans the clipped tick and the next, up to about 2x the acceleration limit (stated, not changed);
the ramp refuses a malformed envelope loudly (unreachable from the controller, whose policy validates it). Named: jogging off a
stop from beyond a POLICY limit stays impossible through the jog path (every q_des beyond the envelope fails command_valid,
unchanged from before).
