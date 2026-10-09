# Architecture

## Repository ownership

`arm_control` owns reusable robot/control infrastructure. A consuming project
owns its robot instances, measured calibration, camera roles, concrete tool/CAD
variants, workcell geometry, assembly task, operator workflow, and deployment
units. A perception producer may use this library's contracts; this library
must not import that producer. A schema for perception output is a value
contract, not a dependency on perception implementation.

Dependencies point from consumers toward the library. Project adapters are
injected at the composition boundary; they are not imported by generic code.
The remaining simulation exception is recorded below rather than hidden by a
folder rename.

## Process and control ownership

Current motion graphs separate planning from command streaming:

```text
planner / arm console -- plan, control, jog --> arm controller --> plant
```

The controller is the sole producer of arm `motor_command` and arm authority
in those graphs. It does not own IK, OMPL, collision queries, scene loading,
HTTP, or Rerun. Planning may take unbounded time without blocking the controller
stream. `plan_id` correlation prevents execution of stale results; existing
operator approval, safe holds, authority checks, deadmen, and fail-closed
hardware behavior remain part of the runtime contract.

Use `planning.stack.build_planner` and `build_preview` for planner-side
construction, and `control.factory.build_executor` for the streaming side.
The planner builder does not instantiate an executor just to obtain gains.

The controller owns its released `control.reference.TrajectoryReference`.
Legacy streamed execution uses its PC start time and pure
`JointTrajectory.sample_at`. Negotiated native execution has **no PC start
time or active PC interpolation**: this same provider delegates to one
RT-selected feedback sample per controller tick. The planned trajectory stays
available for endpoint and Cartesian knot association. `JointTrajectoryExecutor`
receives that sample and retains RNEA, payload, gains and torque bounds; it
stores no trajectory, trajectory time or integral state. `control.integral.IntegralState`
owns the bias and update history. The canonical builder supplies its admitted
ki/i_limit to `ArmController`, which schedules one update after lifecycle,
completion and target selection and before tracking admission. Pure executor
commands receive the bias explicitly; tracking fallback reuses it. Deadline
ticks may reuse still-fresh observations; stale/nonfinite observations cannot
advance state. Event-driven cached holds update only when constructed, retaining
their historical cadence. No PC/RT clock or state owner is duplicated.

`build_execution_policy` admits the deployment's position envelope against
the robot description: explicit bounds may narrow it, never widen it. Missing
bounds inherit finite description limits; a partial declaration is refused.
Continuous joints retain the existing unset-default behavior. This admission
does not certify workspace/collision clearance or physical contact forces.
Invalid jog targets reach the existing controller fault path before any
integral update; the integral owner never accepts a nonfinite target.

For a model-bound supervisor and a capable RT server, v5 session negotiation
opts into v7. `plants.remote_rt.reference.ReferenceHorizon` uploads unchanged
windows of the admitted curve, one candidate/worker at a time. It owns transport
progress and feedback binding, not a sampler or a control law. `ReferenceSelection`
owns the RT anchor/revision and pure native Hermite evaluation. Whole retained
segments are byte-identical; replacement is C1, not necessarily C2. The bounded
RT limit is 128 knots × 16 joints; longer PC plans advance through bounded windows.

PC dynamics use measured state and the exact selected q/qd/qdd from the same
feedback tick. UDP envelopes identify session, ARM epoch, reference/revision and
phase; the supervisor initially requests 50 ms wall-age and phase-age bounds,
further capped by the RT hold deadline. No cross-host timestamp subtraction is
used. Too-old PC samples send no envelope; normal freshness fallback applies.
Terminal END retains planned q and zero qd/qdd. Settling starts on observed END;
COMPLETED still requires settling and APPLIED. UNDERFLOW faults that execution,
leaves the RT measured hold with zero PC feedforward and captured gains, and
requires a new admission. The independent session/command fault clocks remain.

The supervisor retains admission and receipts. An initial COMMIT in flight cannot
be superseded as unselected. `admit_by` is preserved with a conservative duration:
RT records the first successful SEAL; after its ACK the PC computes remaining
selection time and puts that budget in COMMIT. RT checks elapsed time from SEAL
again at selection. Network/queue delay consumes the budget; retries cannot reset
it. This can refuse early, never grant an extension, and needs no synchronized
clocks. PC waits for selection/refusal evidence rather than calling a delayed
selection receipt EXPIRED.

An unselected cancellation ABORTs only the candidate. A selected cancellation
retires active/pending generations. The controller's existing compensated
measured/contact hold is then admitted as a constant terminal reference; its
anchor/compensation still come from the existing controller/executor. It is not
confused with RT's zero-feedforward transport/underflow hold. Ordinary completion
keeps the planned endpoint throughout. No automatic hold admission heals UNDERFLOW.

v5 streaming remains for old/non-negotiated peers and unbound configurations.
Dora, RemoteHardware and SharedSimulation retain their existing construction and
execution paths; application/group migration is CTRL-040/052/053. They do not run
a second sampler for a negotiated supervisor execution. Native software checks
are not matched deployment or physical/RT timing qualification (CTRL-060/013).
Neutral values live in `motion/types.py`; retiming algorithms remain under
`planning/retiming.py`. Execution does not depend on a planning-owned value type.

## Installed APIs and executable wrappers

### Pair control and shared planning

`multi_arm` coordinates two named arm endpoints. Each retains its own
`ArmController`, joint order, model token and motor-command authority. The
pair coordinator never sends motor commands. Its local and Dora handles expose
the same staging, release, cancellation and per-arm control operations.

| Owner | Responsibility |
|---|---|
| `multi_arm/types.py`, `interface.py` | Public pair values and handles; arm ids remain attached to each member |
| `multi_arm/coordinator.py`, `synchronization.py` | Fresh-state admission, session/model binding, both-member ACK barrier, cancellation acknowledgements and timing records |
| `multi_arm/feedback.py` | Same-host timestamp alignment for the closure monitor; bounded interpolation, raw controller feedback unchanged |
| `contracts/multi_arm.py`, `multi_arm/adapter.py` | Arrow envelope and Dora transport; `nodes/multi_arm_coordinator.py` is only an entry point |
| `multi_arm/planning.py` | Combined joint-space search with both moving geometries and one retimed trajectory, split back into arm-keyed plans |
| `multi_arm/cooperative.py` | Fixed object-to-TCP transforms, continuation IK along one object path, grasp/path checks on the retimed Hermite curve |
| `multi_arm/simulation.py` | Two stock controllers driving one MuJoCo world on one simulation clock |
| `multi_arm/config.py`, `ui/dual_arm_console.py` | Compose existing profiles/scenery; run the generic console with planning in a child process |

Pair release requires explicit positive staging ACKs from both controllers,
fresh measured positions/velocities, known controller sessions, matching model
tokens and continued readiness. IDs cannot be reused. A member fault, timeout,
restart or partial release cancels both; unconfirmed cancellations block the
next pair. A targeted delayed cancellation cannot clear a newer plan.
Consumer scene admission is checked at staging and release. The console also
binds previews to their config, scene, start positions and measured Hand widths.

`PairPlanner` checks cross-arm contact, including attached collision geometry,
through the existing `MujocoCollisionWorld`. It retimes the combined path once,
then gives both controllers identical sample times. Selecting one moving arm
holds the other at its measured joints during search and execution. This is
sampled collision validation of the returned Hermite trajectory, not a proof
of continuous clearance. Scenes with arm welds refuse unconstrained joint-goal
planning. `cooperative.py` instead samples a straight SE(3) object path, solves
both arms from preceding joint seeds, and uses the same retimer and pair plan
packaging. It checks the returned Hermite curve at at most 0.001 rad joint
travel per sample, including the moving free object in the canonical collision
world. Closure and object-path deviations must stay within 1 mm / 0.5°.

`planning/retiming.finish_path` is the shared manual/pair path-finishing seam.
It selects checked retiming or the existing `planning/trajopt` refiner. Frozen
peer joints are excluded from optimization and restored with exactly zero
velocity. The task planner retains its refiner hook into that same native
implementation. The console selection is bound to the preview revision.
Both consoles isolate native planning in child processes. The single-arm console
serializes its current MuJoCo world, including mounts, collision permissions,
held joints, mocaps and boundary geometry; it does not reload a nominal URDF.
MuJoCo query data is recreated against that model without duplicating the model
in the transfer. The parent retains scene checks and rejects superseded results.

For cooperative refinement, native TrajOpt SQP's low-level constraint API adds
both TCP constraints per knot; the ordinary waypoint profile supports one TCP.
This shares projection, collision profiles, spline timing and validation with
unconstrained refinement. A planning-only rigid-body relation makes the object
follow one grasp in MuJoCo and the projected Tesseract tree. The other grasp
and commanded object path remain explicit constraints and dense validation
conditions. No free object is frozen as scenery during this optimization.
Joint endpoints remain exact; collision/grasp checks are sampled, not continuous
certification, and jerk is not bounded. Native failure never admits an unchecked
seed. The simulation/real controller code does not import an optimizer.

`SceneSpec.welds` and exact body-pair `contact_exclusions` describe simulated
rigid grasps once for both physics and planning. The object starts as an
externally supported real-mass free body; grasp constraints are inactive.
`scene_for_mode` selects the same relations for planning. The console plans a
free-arm approach from measured joints, checks pose/Hand agreement and stopped
feedback, and requires explicit grasp confirmation before transferring support.
Mode changes preserve the existing plant and controllers, mutate constraints
and payload accounting, and invalidate preview/staging identities. Returning
to independent control requires support confirmation and anchors the object
at its current pose. Active shared control assigns half the payload mass at its
center of mass declared to each existing executor. This is equal static load
sharing, with world gravity expressed in each arm's base frame; it does not
regulate internal grasp force.
Shared-mode physics pauses while either controller is disarmed. Measured TCP-to-object
error above 3 mm / 2° faults the simulated console and disarms both controllers.
The console locks individual arm targets and Hands in this mode and binds
admission to the measured object pose as well as the ordinary pair guards.
Grasp declaration, real attachment sensing and task sequencing remain consumer
policy. This console's explicit confirmation applies to simulated fixed grasps.

The console selects either `SharedSimulation` or `RemoteHardware` from its
pair configuration. Both expose measured joints, Hands, controller handles and
scene geometry through the same small runtime surface. The hardware adapter
composes the existing session-bound `RtBackend`, `BridgeClient` and stock
controllers; MuJoCo supplies collision geometry only. Hardware uses each
profile's real-mode gains and execution policy. Construction prepares models
without opening transports; the console calls `connect()` after loading its
visuals and exporting meshes. Native loading can block Python heartbeats past
the RT session deadline. Admission then applies measured arm/Hand positions to
the collision world before the UI starts. Authority RPCs stay off the
100 Hz loop, superseded ARM requests are cancelled, and fault/shutdown paths
revoke authority without requiring another tick. No simulated joint feedback
enters the hardware runtime.

Hardware independent/paired motion remains experimental; physical shared-object
mode is refused. Explicitly virtual objects allow guided preparation and fixed
TCP relationships with empty Hands: no weld, Hand command or payload is added.
Their pose follows the first time-aligned TCP; both inferred poses must agree within
3 mm / 2° or both arms are disarmed. Entry requires stationary TCPs within
10 mm / 2° of the nominal edge poses and a separate empty-Hand confirmation.
After collision admission it captures both measured object-to-TCP transforms;
no joint/object pose changes. These frames are fixed for the session, hashed
into preview identity and passed to the planning worker. Exit restores nominal
grasp frames for the next approach. Physical grasp confirmation cannot authorize
virtual coupling. Local hardware controllers share a PC reference epoch and
one sampled trajectory time per tick; both commands are computed before dispatch.
Freshness checks keep the unfrozen wall clock. Actual PC releases remain
sequential and observable; physical start skew is not measured. Scheduled RT
start remains separate work. The historical independent-world graph is an
isolation diagnostic.

The relative-pose monitor uses fresh packets bracketing a common publication
timestamp on one RT host. It refuses extrapolation, invalid/reordered packets,
wrong authority, receipt age over 50 ms, or skew/interpolation gaps over 20 ms.
Only an armed joint hold with measured speeds below 0.01 rad/s for 0.25 s may
wait briefly for a spacing gap to recover. Readiness is withheld, pending plans
are invalidated, and the existing hold target is retained. The wait ends no
later than 50 ms after the older latest packet in the last valid alignment was
received; re-reading unchanged packets cannot extend it. No closure estimate
is accepted during that interval. Active motion, invalid feedback and expired
recovery still stop both arms. Recovery requires a new plan, never auto-resume.
This removes comparison error from asynchronous streams; it does not establish
physical capture synchronization. Controllers still consume raw measurements.

The hardware adapter retains the packets consumed by each controller, so
`multi_arm.diagnostics` can keep a bounded PC history of feedback, references
and timestamps, including the reference time and interpolation brackets.
Rejected raw packets remain distinct from the controller's
last accepted measurement. Runtime faults request pair cancellation and disarm
before a background writer serializes a detached history; shutdown waits at
most one second for that writer. The closure guard keeps its 3 mm / 2° limits
and reports the measured values; trace-write failure cannot replace the fault.

### Entry points

Import implementation through `arm_control.*`, never through sibling node
files or a `sys.path` entry pointing at `nodes/`. The package owns substantial
process implementations under `control/`, `plants/`, `ui/`, and `viz/`.
Small files under `nodes/` remain Dora's executable path seam; directory depth
is not an API. CLI-only repository bootstrapping is separate from this rule.

The standalone application is also the application embedded by a deployment.
Call `ui.arm_console.main(cfg=..., collision_world=..., static_geoms=...)` and
`viz.visualizer.main(cfg=..., static_geoms=...)` from thin project wrappers;
do not copy their planning, UI or controller logic. Omitted arguments preserve
standalone config/scene loading. `static_geoms=[]` explicitly means no scenery.
`simulation.rerun_scene.static_mesh_geoms` exports a generic composed snapshot
for both web and Rerun views. Project scene roles and live revision policy stay
in the consumer, never in these entry points.

`plants/base.py` defines only the shared observable plant surface: motor count,
state, and cleanup. Backend-specific arm/disarm and command validation stay
explicit. Generic grasp semantics live in `end_effectors/`; moving these files
does not change HELD/LOST/MISSED behavior. DM gains are rejected before encoding,
not silently clipped.

`contracts/` centrally owns motor, motion, scene, grasp/gripper, and perception
wire schemas. The byte-level remote RT protocol remains separate in
`plants/remote_rt/protocol.py`.

Intentional public compatibility aliases remain:

| API | Canonical owner | Reason | Retire when |
|---|---|---|---|
| `arm_control.messages` | `arm_control.contracts` | Existing independent consumers use the codecs | Its importers here, in `Control/assembly` and in perceptions import `contracts`, and the README self-check moves with them |
| `arm_control.rt_protocol` | `arm_control.plants.remote_rt.protocol` | Preserve protocol imports and `python -m ... --hex` | No current command or importer uses it; remove one release after a deprecation notice |
| `arm_control.simulation.mujoco_backend` | `arm_control.plants.mujoco.backend` | Preserve class/helper identities and module CLI | No current command or importer uses it; remove one release after a deprecation notice |
| `arm_control.simulation.scene_backend` | `arm_control.plants.mujoco.composer` | Independent simulated perception consumes the injected builder | perceptions `nodes/sim_perception.py` imports `plants.mujoco.composer` |

Callers were audited on 2026-09-24 across this library, `Control/` and
perceptions.

These are package-level re-exports, not node-source APIs. Removing them requires
an explicit consumer migration, not a directory-cleanup sweep.

## Instances, models and calibration

One `arm_rt_server` process owns one physical plant. Protocol v5 makes that
explicit on the wire: the server is provisioned with an instance id and joint
order, sends an `IDENTITY` frame, and refuses ARM until the session's `CLAIM`
matches; every command carries the session id and ARM epoch. Observers take a
read-only UDP lease and can never acquire or revoke authority. See
[the RT reference](../rt/README.md).

`model_bundle.py` is the generic, versioned model contract (instance, joint
order, frames, nominal and model URDF hashes, tool, base, parameterization,
datasets, status) plus the one transformation chain and a joint-coordinate
conversion applied exactly once. It loads no consumer configuration and fits
nothing. `mukca.py` holds the two-socket calibration data model, the reference
objective and the evaluators; `mukca_casadi.py` is its OFFLINE solver and is
imported only by offline tools -- no CasADi, IPOPT or GUI import exists on any
control or RT path.

Installation gravity is separate from kinematic registration. Project staging
binds a registry declaration to the approved bundle/model, root frame and
installation epoch with `bind_gravity`; a canonical digest extends the existing
opaque model token. `build_executor` consumes that staged value through
`execution_gravity` and the existing `PinocchioDynamics(gravity=...)`. Bound
hardware refuses missing/stale facts. Neither RemoteHardware nor a supervisor
rotates gravity through the fitted workcell mount. Simulation explicitly uses
its own physics gravity/actor placement via `simulation_gravity`; unbound
analytical examples retain the dynamics default. No generic module reads the
project registry. Nominal installation assumptions are not measurements.

## Configuration and assets

`examples/configs/` contains minimal reusable examples, not measured deployment
facts. `examples/profiles/` contains reusable command and motion styles;
Float is a gain preset in the motion console. The retired `plant.mode: float`
bridge shortcut is rejected before opening hardware.
The consumer composes final entries from hardware identity, calibration, sensor
roles, workcell facts, task policy, and simulation overrides. Existing
`config.load_config_tree` include/deep-merge behavior remains unchanged.

With `planner.refinement: trajopt`, optional
`planner.soft_clearance_exempt_pairs: [[geom_name_a, geom_name_b]]` selects
exact MuJoCo geometry names. Each refinement resolves them against the latest
world supplied by `refinement_world()`. Missing, disabled, filtered, malformed,
or duplicate pairs (including reversed duplicates) raise an error. Selected
pairs lose only the extra 5 mm soft-cost clearance: their cost threshold becomes
the canonical self padding, or zero for environment pairs. Hard constraints,
collision exclusions, joint limits, and final canonical validation stay intact.
Omitting the option preserves the default soft cost. IK, OMPL, seed retiming,
and native refinement log completed-stage durations by arm using `perf_counter`;
successful joint, Cartesian, and refinement calls also log their total duration.

`REPO_ROOT` anchors files shipped in this checkout. `CONTROL_ROOT`, selected by
`ARM_CONTROL_ROOT`, anchors a deployment's relative asset paths; standalone it
defaults to the library checkout. Set the deployment root before importing
modules that resolve assets. These names preserve the existing path contract;
they do not authorize a reverse import into a particular project.

UI static files are package data under `arm_control/ui/static/`. Examples,
Dora wrapper paths, graphs, and tool scripts remain repository files. A wheel
provides installed APIs and web assets, not an entire deployment checkout.

Keep the README as the component index, this document as the ownership/API map,
and the runbook as current operating instructions. Add pair layouts by referencing
existing profiles and scene descriptions; do not copy gains, limits or URDFs
into another YAML. Dated reports/evidence record what was checked at that time
and should link to current instructions. Compatibility wrappers need a named
consumer and migration reason; avoid parallel implementations or new status
documents that restate these three pages.

## RT ownership and sim/real parity

The reusable C++ core keeps the protocol, mailbox, servo law, RT loop, fake/DM/
Franka backends, server, bindings, and generic Hand bridge. Concrete systemd
units, CAN-link setup, and task-specific device bridges belong to the consumer.
See [the RT reference](../rt/README.md) for core protocol and safety details;
its historical bench observations do not certify another machine.

`rt/include/arm_rt/controller.hpp` owns the RT joint/pose mode mapping and the
existing `PoseHold` capture state. It produces the selected five-field command;
`rt_loop` owns lifecycle/freshness/fallback selection, calls the unchanged limiter,
and dispatches to the plant. Native MIT receives the original fields, not a
second PD contribution folded into feedforward. Joint RNEA/integral remain PC
owned. Pose hold continues using the named vendor model and its existing
Coriolis input; no new dynamics provider or PC-to-RT state migration is implied.
The MuJoCo adapter still supplies its own measured simulation state to the same
compiled `PoseHold`/`servo_torque` primitives; it does not consume an RT wire
packet or pretend its model is the vendor model.

Simulation and hardware share control contracts without pretending their
lifecycle or physics are identical. Scene revisions and stale-command
rejection, live-state transfer across MuJoCo recompilation, contacts, and
attachment behavior must survive structural moves. Import checks alone do not
verify those behaviors; the consuming project owns its integration rehearsal.

## Execution contract (CTRL-020 v1, proposed for freeze)

`arm_control.execution.contract` holds the transport-independent values that
applications, the supervisor and transport adapters share. It depends on the
standard library, numpy and `arm_control.motion` only. It imports no transport,
so no RT protocol, Arrow, Dora, Rerun, MuJoCo, planner or device SDK is loaded.
Transports adapt to it: `plants/remote_rt/observation.py` maps RT state packets
to `RobotState` and `Authority` and derives the RT identity hash. The RT byte
protocol is unchanged, and contract v1 needs no wire change.

**Scope of v1:** one robot executing one planned joint trajectory, started
`ON_ADMIT` or `ON_RELEASE`. Group requests and a group commit (CTRL-040),
Cartesian requests, and streaming or action-chunk requests (CTRL-030) are later
contract versions.

### Where each concept already lives

| Contract value | Existing source it consolidates |
|---|---|
| `RobotIdentity.instance_id` | registry instance, RT `--instance-id`, `IDENTITY.instance_id`. Pair slot names such as `left`/`right` are roles, not identities. |
| `joint_names` | RT `--joint-names`, config `arm.joints`. The RT `joint_order_hash`/`identity_hash` are derived in the RT adapter (`observation.identity_hash`). |
| `model_id` | `IDENTITY.model_id`, the plant's task model |
| request `model_token` | `plan.model_token`, `ArmController.model_token` (model bundle and scene) |
| `Capabilities` | HELLO tokens (`hold_ms`, `fault_ms`, `active`, `pose_hold`, `rt`), `IDENTITY` capability bits, `VERSION`/`POSE_HOLD_VERSION`. What the backend advertises only. |
| qualification (not a contract value) | the deployment registry `qualification` and the evidence records, consulted by the supervisor. Never inferred from what a device advertises. |
| `Authority.session` | RT `session_id`, one per control connection. Distinct from the controller's `runtime_state.session_id` (a controller instance). |
| `Lifecycle` | `FLAG_ARMED`, `FLAG_FAULTED` (state and replies); `FLAG_CLAIMED` (control replies only); CLAIM, ARM, DISARM |
| `Authority.holding` | `FLAG_HOLDING` |
| `arm_epoch`, `stop_epoch` | ARM ack seq, `StatePacket.arm_epoch`/`stop_epoch`, `CommandPacket.arm_epoch` |
| `Fault` | `fault_code`, CTL_FAULT text, fault journal FIRST record with its ARM generation, controller `fault` event |
| `RobotState` measured | `StatePacket.q/dq/tau`, `wrench` + `FLAG_WRENCH_VALID`, `t_mono_ns`, `state_seq`, and the PC receive time |
| `RobotState.selected_q/selected_qd` | `StatePacket.q_cmd/qd_cmd`: the reference the servo selected after hold, latch and pose-hold handling |
| `RobotState.output_tau` | `StatePacket.tau_cmd`: the final torque written to the plant (DM: the backend's prediction from its encoded MIT frames) |
| `RobotState.last_command_seq` | `StatePacket.last_cmd_seq`: the freshest command the servo selected |
| `Validity` | `ArmControlState` "None is unknown", `align_feedback` age bounds, execution-policy freshness |
| `TrajectoryRequest` | `pack_plan`. `plan_id` is `request_id` today. `gated` becomes `Start.ON_RELEASE`. `kp/kd` become `gains`, and `completion` keeps its meaning. `phase` stays a label. A plan's `cartesian` spec (robot base frame) becomes `cartesian=CartesianImpedance(task_R, kc, dc)` with `profile="cartesian_impedance"`; `cartesian_poses` is visualisation only and is dropped. `pair_id` and `execute_pair` are CTRL-040. |
| `CancelRequest` | control `cancel` + `cancel_plan_id`, which produces `plan_cleared` |

### Execution phases and the evidence for each

| Phase | Meaning | Evidence today |
|---|---|---|
| REQUESTED | created by the application | — |
| ADMITTED | passed admission and staged. Not executable authority, and needs no ARM. | `plan_staged ok=True` |
| SELECTED | the executor samples it as the active reference | `_release` loads it; **no event** |
| SENT | a command derived from it left the executor for the plant transport (RT: the UDP command was sent) | `apply_command`; **no event** |
| APPLIED | the downstream plant/backend **accepted** such a command. This is acceptance evidence, not evidence of physical motion. | RT: see the rule below; **not reported** |
| COMPLETED | completion rule held on observed feedback | `leg_result ok=True` |
| REJECTED | refused at admission or selection; never moved | `plan_staged ok=False`; `leg_result ok=False` for a refused release; a wrong-id `execute` is **log-only** |
| SUPERSEDED | admitted, unselected, replaced by an admitted request naming it | a new `plan` replaces a staged one **silently and implicitly** |
| EXPIRED | not selected before `admit_by` | **none** |
| CANCELLED | cancellation confirmed; the executor dropped it, and holds if it had been selected (an unselected one is only discarded) | `plan_cleared ok=True` |
| FAULTED | stopped by a fault while it owned the reference | controller `fault`, RT latch, `leg_result ok=False` while running |

**RT APPLIED rule.** The server advances `last_cmd_seq` when it selects a
fresh command, before `backend->write_command()`. So `last_cmd_seq` alone is
selection by the servo, not acceptance.

The server keeps `last_cmd_seq` across ARM generations and client sessions,
while a new client restarts its sequence. So the RT executor numbers its
commands strictly above (serial arithmetic) the `last_cmd_seq` in the first
state it observes of its own `arm_epoch`. That is a client rule, not a wire
change. With it, a stale value can never pass for a new command.

A rejected write latches `FAULT_PLANT`, and the latch is sticky until DISARM.
DISARM clears it without changing `arm_epoch`. APPLIED therefore requires both
of these:

- an ARMED state of the same `arm_epoch` whose `last_cmd_seq` is at or past
  the first command seq sent for the execution;
- a later state of that epoch that is still ARMED and not FAULTED, received
  while the executor's control session is up and before it sends any DISARM
  in that epoch.

The later state is needed because a cause latched concurrently by another
thread can surface one tick late. The DISARM bound is needed because DISARM
clears the latch mid-tick: a state straddling it can show ARMED without
FAULTED after a rejected write. A FAULTED state seen at any point in the epoch
ends the execution FAULTED, and a later clean state never overrides it. For the FR3 backend, acceptance means
libfranka's `ActiveControl::writeOnce` returned without an exception. For DM,
it means every active motor's CAN frame was sent.

The chain REQUESTED → ADMITTED → SELECTED → SENT → APPLIED moves forward only.
A transport that cannot observe a phase may skip it, but no phase is ever
revisited. A terminal phase is final. A receipt for a phase already reached
is a redelivery and changes nothing, so receipts may be delivered at least
once.
COMPLETED requires APPLIED, so a sent packet is never "executed". Every failed
terminal phase carries a reason. `ExecutionStatus.apply` enforces this.

### Semantics proposed for freeze

- **Identity.** Every vector is in `RobotIdentity.joint_names` order. A
  request names its order and is refused on mismatch; nothing reorders.
  Instance ids are `[A-Za-z0-9_.-]{1,64}`, with no `:` because frames are
  `base:<id>`. A transport may be narrower: RT ids are under 32 bytes, and its
  adapter refuses anything longer.
- **IDs.**
  - `request_id` is the caller's idempotency key. Resubmitting it returns the
    existing execution; it is not admitted again. A resubmission with
    different content is refused at submission and creates no execution.
  - `execution_id` is assigned at admission and never reused.
- **Admission and selection are separate gates.**
  - `admission_refusal` stages. It checks, in order:
    1. instance
    2. joint order
    3. capability/version
    4. model token
    5. `admit_by` clock and deadline
    6. ownership (CLAIMED or ARMED)

    It does **not** require ARM: a gated plan is reviewed while disarmed.
  - `selection_refusal` releases, and requires ARMED.
  - The expiry check runs before it and ends the execution EXPIRED, not
    REJECTED.
  - Project admission (workcell, collision, Hand, observation freshness,
    start tolerance) is composed above both, never folded into them.
- **Start and expiry.**
  - `ON_ADMIT` selects as soon as it is admitted.
  - `ON_RELEASE` selects only on `release(execution_id)`.
  - Trajectory time 0 plays at SELECTED; review time is not charged against
    it.
  - `admit_by` is the latest selection time. It must be on the admitting
    supervisor's attested PC monotonic clock. A deadline on another clock or
    domain, or one already passed, is refused at admission.
  - Past the deadline, an admitted execution EXPIRES.
  - A selected planned trajectory then runs to COMPLETED, CANCELLED or
    FAULTED. Command freshness below that belongs to the RT deadman.
- **Supersession** (`supersession_refusal`), applied after idempotency:
  - A different request may supersede only an ADMITTED, unselected
    execution, and only by naming it in `supersedes`. A request that does not
    name the pending execution is refused while one exists; nothing is
    replaced implicitly.
  - SELECTED or later is never replaced; cancel it and wait for CANCELLED.
  - The old execution ends SUPERSEDED only once the new request is ADMITTED.
    Its receipt carries `superseded_by`.
- **Cancellation.**
  - It names one execution and never cancels another.
  - It completes with the CANCELLED receipt: the executor dropped the
    execution and, if it had been selected, holds. Cancelling an unselected
    execution only discards it.
  - Cancelling a terminal execution reports its status and changes nothing.
  - Cancel is not DISARM.
- **Holding and faults.**
  - `Authority.holding` is True while the executor holds a captured pose
    instead of tracking.
  - A FAULTED robot holds when the servo can still hold: after a
    command-stream or controller fault, or a `FAULT_PLANT` with a healthy
    plant (invalid pose-hold inputs).
  - After a plant/bus failure writes are suppressed (`holding=False`), and
    the robot's own safety owns the arm.
  - FAULTED does not imply holding, and the fault code does not say which:
    read `holding`.
- **DISARM** completes when `stop_epoch` reaches the DISARM generation, not
  when the request is acknowledged. That means the servo stopped commanding
  and ran the backend stop. It is not a claim about the plant's brakes: a
  failed backend stop still advances `stop_epoch`, and is reported as a
  `FAULT_STOP_FAILED` journal record.
- **CLAIMED** is visible only to the session that claimed. State packets carry
  `session_id` (set at connect) but not `FLAG_CLAIMED`, which appears only in
  control replies. The claiming session reports CLAIMED only for packets
  carrying its own session id: the server re-sends cached state to a new peer.
  Everyone else sees a disarmed claimed robot as CONNECTED.
- **Capability, version and qualification.**
  - An unknown controller profile or contract version is refused before
    actuation.
  - `Capabilities` is what the backend advertises. Profiles: `joint_impedance` always; `cartesian_impedance` when the plant
    runs the RT v8 Cartesian law (the RT box with `CAP_CARTESIAN_TRAJECTORY`, the MuJoCo twin) and legs are streamed (never
    on the native reference). Its `CartesianImpedance` is base-frame task axes plus K_c/D_c inside the RT envelope; the plant
    targets FK(q_des) with its own model; the joint gains must clear the robot's `execution_policy.cartesian_kp_min`; the
    law runs through the leg and its settling only, every hold is the joint law.
  - Qualification is a separate deployment/evidence record.
- **Completion.**
  - `vel_tol` (rad/s) and `residual_max` are finite and positive.
  - `residual_max` is rad per joint (`max|q - goal_q|`).
  - `goal_q` covers every joint, in the request's order.
- **Observation.** A `RobotState` carries three kinds of joint vector, never
  mixed:
  - measured `q/dq/tau`;
  - the selected reference `selected_q/selected_qd`;
  - the plant-facing output `output_tau`, the torque written to the plant.
    A native-impedance plant (DM MIT) is sent the selected reference and
    gains and closes its own PD, so for it this is the backend's prediction
    from its encoded frames.

  Selected and output are `None` while nothing is selected or written
  (disarmed, or after a plant fault); they are never zero-filled. All values
  are `None` unless VALID. Freshness is judged on the reader's own clock,
  never by subtracting RT time from PC time.

### Conventions

- **Units:** SI; `UNITS` in the module is normative.
- **Poses:** `[x, y, z, qw, qx, qy, qz]` in metres with a unit, scalar-first
  quaternion; `q` and `-q` are equal. Every pose names its frame: `world` (the
  workcell board) or `base:<instance_id>`.
- **Wrench:** `[fx, fy, fz, tx, ty, tz]`, expressed in the base frame about
  the vendor stiffness-frame origin. Positive is the wrench the robot applies
  to the environment (libfranka `O_F_ext_hat_K`).
- **Time:** a `Stamp` carries its clock (`rt_host_monotonic`, `pc_monotonic`,
  `robot`, `execution`) and that clock's identity. Stamps on different or
  unattested clocks are not subtracted (`elapsed` refuses).

### Gaps the contract exposes

The headless supervisor (CTRL-022, below) closes these. It emits every phase
from `ArmController` events, its snapshots and RT evidence, with no RT wire
change:

- selection, SENT and APPLIED (by the RT APPLIED rule)
- explicit supersession and EXPIRED
- separate request and execution ids, with idempotency
- `RtBackend` now seeds its command seq above `last_cmd_seq` at every ARM

The Dora path (`plan`/`control` topics) still reports only its old events.

Still open:

- RT and PC clocks are not attested: there is no host/boot identity on the
  wire (`Stamp.domain` is empty for RT), so cross-host timing stays unmapped.
  The supervisor's own clock is attested by the host boot id.
- RT fault text arrives only over the control session, not in state.
- A failed backend stop at DISARM is visible only in the fault journal.

## Headless supervisor and SDK (CTRL-022, first slice)

`arm_control.supervisor` runs one robot in its own process. `arm_control.sdk`
is the application side. The chain is:

`ArmClient` → named operations over a local socket → `RobotRuntime` → the
existing `ArmController` (still the only executor) → `RtBackend` → the RT
server.

- **Process and threads.**
  - The runtime steps the controller at the execution policy's 100 Hz, in the
    supervisor process.
  - Every controller call and ledger change happens on that loop while it
    holds one lock. Client threads only enqueue operations and read
    snapshots.
  - ARM/DISARM round trips run on their own worker, with
    `RemoteHardware`'s rules: a DISARM drops queued ARMs, and ARM is refused
    while a DISARM is unconfirmed.
  - A stalled client (half-sent frame, GIL-bound thread, silence) does not
    delay a tick.
- **Operations.** These names and arguments are the transport-neutral
  boundary:
  - `hello`, `now`, `state`, `authority`
  - `claim`, `unclaim`, `arm`, `disarm`
  - `submit`, `release`, `cancel`, `status` (optionally a bounded wait),
    `receipts`

  The transport today is newline-delimited JSON over a Unix socket in an
  owner-only directory, one connection per client session.
  `supervisor.service.dispatch` maps names to the runtime, and
  `execution.codec` maps contract values to JSON.
- **Ownership.**
  - One client session owns the robot (`claim`); every mutating operation
    checks it.
  - A closed session is refused at once, before its queued work runs.
  - Losing the owner cancels its live executions and holds, still armed.
    Nobody reacquires control automatically.
  - Connecting is observation only.
  - A supervisor start claims the RT server and never arms.
- **Authority as reported.**
  - ARMED/FAULTED come from RT state.
  - CLAIMED means a client owns the robot through the supervisor's RT claim.
  - CONNECTED means nobody owns it.
  - DISCONNECTED means no fresh RT state.
  - `session` is the supervisor's RT control session, as frozen; a client
    session never appears there. A client knows it owns the robot because
    its `claim` succeeded.
- **Admission.**
  - Contract admission (`admission_refusal`, then `supersession_refusal`)
    runs first.
  - Then the controller's own admission: execution policy, gains and model
    token.
  - Selection is the controller's `_release`, preceded by the contract's
    expiry and `selection_refusal`.
- **Completion.** A feedback completion becomes COMPLETED only once APPLIED is
  reached. It becomes FAULTED if acceptance evidence has not arrived within
  the link's state freshness bound (`rt.state_timeout_s`): that evidence is
  the next state packet, and past that bound the link is stale anyway.
- **One controller recipe.** `control.factory.build_arm_controller` resolves
  gains, executor, execution policy, model token, pose provenance and
  settling for the supervisor, the Dora `arm_controller` node and
  `RemoteHardware`; each passes only its node, clocks and health reporting.
  The in-process controller seam (`control.inprocess.QueueControllerNode`)
  sits under both the supervisor and `multi_arm`. The shared MuJoCo pair
  (`multi_arm.simulation`) still builds its sim controllers directly: no
  gains table, no deadline policy, simulation time.
- **Install.**
  - The minimal SDK needs numpy only (`pip install .`).
  - The supervisor needs `.[supervisor]` (pyarrow for the controller's
    codecs, pyyaml) plus Pinocchio.
  - Dora processes need `.[dora]`.
  - `tools/check/check_sdk_supervisor.py` drives all of it device-free,
    including from a clean numpy-only install.
  - `tools/check/check_supervisor_ledger.py` covers the ledger rules the fake
    server cannot provoke: completion without acceptance, ARM during a
    pending DISARM, and retention.
- **One writer.** Until the Dora adapters delegate to the supervisor, a robot
  is driven by either the supervisor or a Dora graph (PC-controlled mode),
  never both. The RT server admits one control session, so a second commander
  is refused at CLAIM.

Not in this slice:

- `RobotGroup` and group reservation (CTRL-040)
- moving the Dora `arm_controller` node and the consoles onto the SDK, and
  `RemoteHardware`'s pair execution onto supervisors (after CTRL-040, which
  gives the pair a scheduled group commit; until then the pair keeps its
  shared PC reference clock in one process)
- a client lease or heartbeat beyond connection liveness
- clearing a completion-timeout freeze without DISARM: a FAULTED settle
  leaves the controller frozen, and later submits are refused until DISARM
  (which cancels) clears it
- persisting the ledger (it keeps 512 executions)

## Known unfinished boundaries

- Motion and calibration use `arm_controller`; the duplicate standalone
  executor process is retired, not the reusable executor class. Consumers can
  explicitly select `control.execution_policy.ExecutionPolicy` for legacy
  start/tracking limits, deadline scheduling, freshness and hold-relatch guards.
  Entries without that policy retain existing event-driven behavior.
- Project-shaped scene construction has moved to the consumer. The graph
  selects `SCENE_BACKEND_FACTORY` or `WORKCELL_BACKEND_FACTORY` through the
  existing entry-point resolver. `plants/mujoco/composer.py` otherwise consumes
  a generic `SceneSpec`; it does not inspect project deployment roles.
- `plants/mujoco/backend.py` retains coupled legacy fixture-release/mating
  behavior, body-name contact exclusions, and tuned gripper/object parameters.
  Moving that core preserved physics; extracting these policies would require
  separate behavior validation. It is not yet a fully domain-free backend.

### Optional stream/1 implementation checkpoint (CTRL-033)

`execution.stream` adds negotiated values alongside execution v1. `ArmClient`
refuses all six stream operations unless HELLO advertises stream/1. RobotRuntime
owns async admission, service-session high-water marks, the bounded attempt
ledger and explicit selection uncertainty. The neutral `StreamAdmission` seam
requires trusted scene/model/policy/load facts; configuring no provider leaves
the extension unavailable. Project provider and console migration are pending.

ReferenceHorizon remains the single upload/selection owner: admitted suffixes
copy whole selected prefix knots, with no second sampler or RT wire change.
TrajectoryReference now owns the selected endpoint/end phase/terminal flag;
controller completion delegates to it rather than the original plan endpoint.
`planning.stream_target.TargetStopGrid` is the named target-generation strategy,
separate from stream policy. It does not execute or renew references.

Raw jog is not a stream: it is setpoint/1 (below), sent by the same supervisor
owner as the console's gain/Soft/payload controls (`DoraArmBridge`), so there is
one control owner. This checkpoint is not a completed console migration or
hardware qualification.

### controller/1 task checkpoint (CTRL-033)

The optional SDK/service extension lists configured named presets and observes
controller law/gains plus canonical transient payload state. RobotRuntime validates
before retiring its execution/native generation; ArmController owns the existing
law transition and emits confirmation. Soft admission is shared through
`validate_pose_hold()`, not copied into the supervisor. `control.presets` resolves
the deployment configuration; applications select names, not numerical gains.

Runtime owns payload value/revision. Replacement refuses while any execution is
nonterminal. Stream admission captures that revision and requires zero mass;
StreamFacts no longer provides a second mutable load declaration. The executor
stores only the installed compensation value, and its redundant mass-admission
accessor is removed. F5 vendor fixed-load ownership remains unresolved.

This remains unintegrated: console migration is stopped on native-to-Soft mode
arbitration. RT retains the reference identity after CANCEL and rejects all v5/v6
commands for the rest of that ARM epoch. The SDK refuses Soft in that state rather
than confirming a law RT cannot apply. No implicit DISARM/re-ARM or RT ownership
change is implemented. Existing console preset resolution and raw jog remain until
one-owner migration and its silence/cancel/expiry checks are complete.

### setpoint/1: a live joint target with a deadman

`ArmClient.setpoint(q)` holds the arm at `q` on the joint law until the next
setpoint; if none arrives within the controller's jog timeout (the mode config's
`jog.timeout_s`, the value the console's deadman reads too) the arm holds its
MEASURED pose, never the last target. It is the in-process controller's own jog
path (`JogHold`), reached through the supervisor: the owner session only, ARMed,
no live execution (a staged plan counts; cancel it first), finite and full width,
and never moving a joint further past its position limit than it already is.
Refusals carry the controller's reason and change nothing. Offered on the
streamed path only: a native-reference supervisor plays only its curve (use
stream/1 there). It is the command path for the console jog (`DoraArmBridge`
forwards `jog`, and cancels its staged plans first) and for policy clients that
emit joint targets. The step size is the CALLER's envelope (the console's
`JogLimits`: stroke, workspace, singularity, collision); the supervisor checks
only what holds for every caller.
