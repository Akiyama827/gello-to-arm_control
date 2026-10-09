# RT Cartesian law, slice 2b: the supervisor's `cartesian_impedance` profile

Operator decisions 2026-10-09: the RT box runs the Cartesian law (slice 1), Cartesian legs are STREAMED only (the native reference
stays joint-only), joint kp floor for Cartesian legs (slice 2a). Goal yardstick: a ros2_control-like backend other projects use
unchanged, so the profile is a generic controller profile, base-frame, with nothing bench-specific in it.

* Contract (`execution/contract.py`): `CartesianImpedance(task_R, kc, dc)`, robot BASE frame, validated by
  `contracts.impedance.cartesian_values` (the RT envelope; a reflection or K_c 2000 is refused at construction), read-only arrays,
  `spec()` = the plan/motor form. `TrajectoryRequest.cartesian` (last field, default None) is given exactly when
  `profile == "cartesian_impedance"`. No pose: the plant targets FK(q_des). The codec decodes the field; a request from a client
  that predates it has no `cartesian` key and decodes as a joint request (`CONTRACT_VERSION` unchanged: an older supervisor
  does not offer the profile, so a Cartesian request to it is refused, never silently run as joints).
* Supervisor (`supervisor/runtime.py`): `offered_profiles(link, native=..., cartesian_kp_min=...)` -- `cartesian_impedance` when
  the link runs the law (`supports_cartesian`: the RT client from the IDENTITY bit, SimLink from its plant), the supervisor is
  not on the native reference, and the robot's policy states the joint-kp floor (review round 1). The generic `admission_refusal` then refuses it everywhere else ("controller profile 'cartesian_impedance' not
  offered"). Admission passes `request.cartesian.spec()` into the controller's plan, where `ExecutionPolicy` applies the joint
  kp floor (supervisors always have a policy).
* SDK handle (`multi_arm/supervised.plan_request`, hence `SdkArmHandle` and `DoraArmBridge`): a plan's base-frame `cartesian`
  spec becomes the profile instead of being refused; a spec without `frame: base` is refused.
* Diagnostics: the controller records the law of each word it sends (`last_law`: joint, cartesian_impedance, pose_hold) and
  trace/1 carries it as `law_pc` -- "did this leg run compliant or stiff?" is answerable from the supervisor's own ring.
* `docs/ARCHITECTURE.md`: the request mapping and the profile list.

Checks (related only):
* `execution.contract` and `execution.codec` self-checks: the profile and its settings together, admitted when offered,
  refused when not; envelope and reflection refused; JSON round trip; a pre-field request decodes; an out-of-envelope one does not.
* `check_sim_supervisor`: a supervisor over SimLink (three-link twin with an EE) offers the profile; a Cartesian leg at the
  fixture's joint gains completes on target and trace/1 shows `law_pc == cartesian_impedance` for every tick of it and `joint` for
  every tick after; a Cartesian leg at joint kp 10 is REJECTED by the floor; `offered_profiles` drops the profile for a plant
  without the law and on the native reference. Mutant "admission drops the spec" caught.
* `check_dual_supervisor`: the plan mapping (base spec -> profile; no frame -> refused); a Cartesian pair member on fake RT
  servers (no v8) is refused as "not offered" and the pair is rejected, its partner cleared.
* `check_sdk_supervisor` (plain and `--native`; its fixture policy gains `cartesian_kp_min`), `check_dual_group`,
  `check_dora_arm`, `check_stream`, `check_supervisor_ledger`, `check_console_authority`, `check_execution_policy`,
  `check_arm_controller`; ruff.

Review round 1: ACCEPT. Taken:
* (SHOULD) "Streamed only" rested on one guard, and no check pinned it where it is wired: removing both native guards let the
  dual fixture ADMIT a Cartesian member on the native path, whose upload is joints only. Now also refused at submission, first,
  with its own reason ("Cartesian legs are streamed only; this supervisor plays the native reference"); `check_dual_supervisor`
  pins that reason (its fixture runs --enable-reference; the old comment blaming the fake server's missing law was wrong).
  Mutant "no submission guard" caught. Remaining survivor, accepted: `start()` passing native=False into `offered_profiles`
  only changes what is advertised; the submission guard still refuses.
* (SHOULD) The profile was advertised on a robot whose policy has no `cartesian_kp_min`, where every such request is refused:
  now not offered (checked).
* `law_pc`: the check now reads the 64-tick ring DURING the leg (no lost ticks, > 100 executing ticks all
  cartesian_impedance, every later tick joint) and covers a Cartesian plan staged ON_RELEASE while the arm holds (its hold
  words are the joint law). Mutant "law from the loaded plan, not the word sent" caught. `last_law` is the last word SENT (after
  a link loss it keeps its last value), as its name says.
* `plan_request` validates a dict spec's exact keys (`cartesian_values`): an extra world-frame `pose` or a non-mapping is refused
  as ValueError (checked), not dropped or raised as AttributeError.
* `cartesian_values` reports a flat or scalar matrix/gain as ValueError (was TypeError); checked.
