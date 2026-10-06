# arm.max_vel / arm.max_acc follow the execution policy (config single source, step 1b)

The FR3's speed limits were stated three times: `execution_policy` (the admission limit), `planner.vel_limits/acc_limits` (the console's
planner; brought under the policy in 2026-10-06-fr3-profile.md), and `arm.max_vel/max_acc`, which the planning stack
(`planning/stack.build_planner` -> `ArmPlanner`) and the dual-arm console time plans with. In the consuming project the third statement was
0.5-0.6 rad/s and 1.0-1.5 rad/s^2 against a 0.4/0.8 policy, so a plan timed at it would be refused at admission.

`control/factory.arm_speed_limits` applies the console's rule (`node_utils.planner_speed_limits`) to `arm.max_vel/max_acc`:
* with an execution policy, an unstated limit defaults to the policy, and a stated one may be slower but never faster. The planning stack
  refuses at load; the dual console refuses at plan time (its Plan fails with the message). A per-joint mapping is read by joint name;
  joints it does not name take the policy (the console's `planner.vel_limits` mapping too: it used to fill them with 0.5 / 1.0 and
  then refuse that against the policy);
* for a limit the policy leaves unbounded (no policy, or a partial one), the arm states its own, as before (`gain_vector`: "every arm declares its own"). The dual console treats an unstated
  limit as unbounded, as before (`required=False`), because it bounds speed with its own pair limits.

Effect on the consuming project at the current pin: `real_franka_workcell_motion`, `real_franka_calibration` and
`real_franka2_workcell_motion` are refused at load, because their `arm.max_vel` (0.5-0.6) exceeds their policy (0.4). The project's config
single-source step 2 removes those restatements in the same parent branch, before it merges. Library examples are unchanged: the dual
pair already states 0.4/0.8.

Checked: `arm_console --self-check` (`_check_planner_limits`: defaults to the policy, slower allowed, faster refused, no policy keeps the
arm's own, `required=False` unbounded). Mutant killed: ignoring the policy.

Review (ACCEPT) notes, folded in before merge: named per-joint limits are read by joint name (they were read by index); a partial policy no
longer brings back a 1.0 default (the arm must state the unbounded limit). For the project's step 2: `tools/demo/workcell_add.py` builds
`ArmPlanner` from the raw `arm.max_vel` and should use `arm_speed_limits`. The dual console's refusal of the project's `dual_arm.yaml` arms is
stricter than necessary (its pair caps 0.2/0.4 already kept speeds under the policy), but it fails closed, and step 2 removes the cause.
Round 2 (ACCEPT): a mapping naming no such joint is now refused (it was dropped silently, losing the slower limit it stated).
