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

`planning/refinement.finish_path` is the shared manual/pair path-finishing seam.
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
| `arm_control.planning.trajectory` | `arm_control.motion` and `planning.retiming` | Preserve trajectory imports while values change ownership | No importer outside archived plans; remove one release after a deprecation notice |
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

Simulation and hardware share control contracts without pretending their
lifecycle or physics are identical. Scene revisions and stale-command
rejection, live-state transfer across MuJoCo recompilation, contacts, and
attachment behavior must survive structural moves. Import checks alone do not
verify those behaviors; the consuming project owns its integration rehearsal.

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
