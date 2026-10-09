# load_model/1: a revisioned rigid-body load model as the controller's RNEA (CTRL-053 B1)

Operator 2026-10-09 (B1 approved): the base's gravity comes from a revisioned body list (the base with the modules grafted onto
it), so its supervisor installs that model instead of a URDF's.

* `arm_control.dynamics.BodyListDynamics`: RNEA over a rigid-body list shipped as numbers. It is the project's
  `NumericDynamics`, moved and renamed; the project imports it from here. Joint kinds are MuJoCo's `mjtJoint` values
  (hinge 3, slide 2), written as numbers so the library imports no MuJoCo for them.
* `JointTrajectoryExecutor.set_dynamics(dynamics)`: RNEA and gravity come from the installed model.
  - Joint limits stay the construction model's.
  - Refused for another joint order, and while a payload is declared.
  - `set_payload` of a positive mass is refused once a load model carries the load.
* Supervisor `set_load_model(session, revision, model)`, wire op `load_model_set`, extension `load_model: 1`; SDK
  `ArmClient.set_load_model(revision, model)`. It is refused:
  - without the claim;
  - for a revision not above the installed one;
  - while an execution is live;
  - on a native-reference supervisor;
  - for another joint order;
  - when gravity at the measured pose is non-finite or above the execution policy's torque limits.
* The supervisor's payload op maps the executor's ValueError to `Refused` (it was an internal error before).

Checks:
* new `check_load_model`:
  - in process, body-list gravity matches a hand calculation, and the executor's hold uses the installed model;
  - over a simulated FR3 supervisor, every refusal listed above;
* `check_sdk_supervisor`, `check_arm_controller`, `check_execution_policy`, `check_sim_franka_motion`, `check_dora_arm`,
  `check_supervisor_ledger`, `ruff`.

Review round 1 (opus): BLOCK, fixed:
* MUST (project): an SDK error (unreachable, timed out, refused) escaped base_driver's loop and killed it, leaving the base
  ARMED, unguarded and deaf to the FR3's DISARM. Now every supervisor call latches a base fault on base_health instead.
  Operator 2026-10-09 on base_driver loss: keep holding (the base has no gravity-stable rest pose). Recorded on the
  track list.
* SHOULD fixes:
  - `set_load_model` refuses without a fresh measured pose; before any state, the check ran at zeros.
  - The base client also guards the supervisor's predicted output torque, and guards only while ARMed.
  - DISARM goes out before the freeze; the supervisor cancels the live leg itself.
  - run.py: the base supervisor's start is interruptible, and the stops cannot be skipped by a signal.
  - The allow-list keeps the bench's OPEN completion-tolerance debt.
* NITs: docstrings, stale "read unstaged" comments, the dry-run wording.
* Recorded, not changed:
  - there is no read-back or clear of the load-model revision (a restarted client cannot pick a revision; the supervisor
    restarts with the run);
  - load_model/1 is advertised on the FR3 too, where gravity compensation makes the model's gravity term moot;
  - the PD torque is bounded by the RT server's cap and the 0.35 rad abort, not by 5 N.m (track list).
