# The joint integral is frozen during a Cartesian leg

Bench, rung 6 (2026-10-10): with K_c 200 N/m the hand felt "quite stiff". The FR3 #1 fault trace showed the joint
integral (ki 1200, clamp 3 N.m, `configs/modes/franka_rt.yaml`) at its clamp on joints 1 and 3 under a sustained push: in a
Cartesian leg the joint integral drove each joint back to its reference, adding stiffness the K_c spec does not state, and the
same would hold during a contact (the dock leg). Operator decision: a Cartesian leg's stiffness is K_c's; the integral is off
for the whole leg.

`control/arm_controller.py` (running leg, trajectory and settling): while the leg carries a Cartesian spec, the integral is
updated with ZERO error (q_des := measured q). Frozen rather than zeroed: it neither accumulates against a push or contact
nor jumps; the bias the leg entered with (the hold's gravity residual, within the clamp) stays, so the leg starts without a
torque step, and its time/cycle bookkeeping stays continuous, so the hold after the leg resumes integrating without a long-gap
dt. Joint legs, jogs and holds are unchanged.

Check: `check_execution_policy` (Cartesian lifecycle): with the same 5 mrad error inside the integral window, a joint leg's
bias grows from its entry value and a Cartesian leg's stays exactly at it.

Also (project, same integration): the bench rung tool's rung-6 hold scales kd with kp at the `cartesian_kp_min` floor
(kd * sqrt(floor / kp), the profile's damping ratio; operator 2026-10-10) instead of keeping the kp-600 profile's kd, which
left every joint heavily overdamped.
