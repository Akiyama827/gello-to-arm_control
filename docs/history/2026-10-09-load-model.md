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
