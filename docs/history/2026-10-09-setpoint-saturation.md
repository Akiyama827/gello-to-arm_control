# setpoint/1 saturation: a streamed target moves no faster than planned motion may

Operator 2026-10-09: clamp (not refuse) a setpoint's rate, from the policy's existing limits; asked whether that is standard --
it is (ros2_control's joint-limits `JointSaturationLimiter`, libfranka's `limitRate`, MoveIt Servo's velocity scaling all
saturate per control cycle). Before: a single 0.3 rad setpoint peaked at 2.26 rad/s against the policy's 1.0 (setpoint/1
review); planned motion is admitted against those limits, streamed targets were not.

* `control/jog_hold.py`: `JogRamp`: per tick the commanded target moves toward the requested one at no more than the velocity
  limit, changes speed by no more than the acceleration limit, and brakes to stop on the target (the discrete braking cap: the
  continuous sqrt(2 a d) overshoots at tick resolution, found by the self-check at 4.8 rad/s^2 against 2). The only
  acceleration above the limit is the final stopping tick, from at most 2 a dt (0.04 rad/s at 100 Hz, a = 2): asserted. No jerk
  term (no jerk limit is stated anywhere; the RT server slews torque). Velocity is not fed forward: the word stays the joint
  law's hold word, as the jog always was (feeding it forward would be a control-law change).
* `control/arm_controller.py`: the jog path (console jog and setpoint/1 alike) runs through the ramp when the execution policy
  states velocity and acceleration limits; it restarts at the measured pose, at rest, whenever no jog is live (none yet,
  expired, cleared, a leg released). Without a policy (the legacy in-process DM path) nothing changes.
* Effect on the console: a joint jog (0.15 rad/s) is followed at its speed a few mrad behind; on release it brakes instead of
  stopping on a step. Expiry, the measured-pose re-anchor and the abort_pos_err_rad refusal are unchanged.

Checks: `jog_hold` self-check (100 random steps: speed <= limit, acceleration <= limit but for the final stop, lands on target;
console following; restart at rest); `check_dora_arm`: one 0.3 rad setpoint on the twin peaks <= the policy's 1.0 rad/s
(mutant "no ramp" caught at 2.19); the owner-loss test now unclaims mid-travel (the ramp alone had made the slower motion pass
without the clear: mutant re-caught).
