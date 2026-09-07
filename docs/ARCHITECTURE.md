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

Import implementation through `arm_control.*`, never through sibling node
files or a `sys.path` entry pointing at `nodes/`. The package owns substantial
process implementations under `control/`, `plants/`, `ui/`, and `viz/`.
Small files under `nodes/` remain Dora's executable path seam; directory depth
is not an API. CLI-only repository bootstrapping is separate from this rule.

`plants/base.py` defines only the shared observable plant surface: motor count,
state, and cleanup. Backend-specific arm/disarm and command validation stay
explicit. Generic grasp semantics live in `end_effectors/`; moving these files
does not change HELD/LOST/MISSED behavior. DM gains are rejected before encoding,
not silently clipped.

`contracts/` centrally owns motor, motion, scene, grasp/gripper, and perception
wire schemas. The byte-level remote RT protocol remains separate in
`plants/remote_rt/protocol.py`.

Intentional public compatibility aliases remain:

| API | Canonical owner | Reason |
|---|---|---|
| `arm_control.messages` | `arm_control.contracts` | Existing independent consumers use the codecs |
| `arm_control.planning.trajectory` | `arm_control.motion` and `planning.retiming` | Preserve trajectory imports while values change ownership |
| `arm_control.rt_protocol` | `arm_control.plants.remote_rt.protocol` | Preserve protocol imports and `python -m ... --hex` |
| `arm_control.simulation.mujoco_backend` | `arm_control.plants.mujoco.backend` | Preserve class/helper identities and module CLI |
| `arm_control.simulation.scene_backend` | `arm_control.plants.mujoco.composer` | Independent simulated perception consumes the injected builder |

These are package-level re-exports, not node-source APIs. Removing them requires
an explicit consumer migration, not a directory-cleanup sweep.

## Configuration and assets

`examples/configs/` contains minimal reusable examples, not measured deployment
facts. `examples/profiles/` contains reusable command, float, and motion styles.
The consumer composes final entries from hardware identity, calibration, sensor
roles, workcell facts, task policy, and simulation overrides. Existing
`config.load_config_tree` include/deep-merge behavior remains unchanged.

`REPO_ROOT` anchors files shipped in this checkout. `CONTROL_ROOT`, selected by
`ARM_CONTROL_ROOT`, anchors a deployment's relative asset paths; standalone it
defaults to the library checkout. Set the deployment root before importing
modules that resolve assets. These names preserve the existing path contract;
they do not authorize a reverse import into a particular project.

UI static files are package data under `arm_control/ui/static/`. Examples,
Dora wrapper paths, graphs, and tool scripts remain repository files. A wheel
provides installed APIs and web assets, not an entire deployment checkout.

## RT ownership and sim/real parity

The reusable C++ core keeps the protocol, seqlock, servo law, RT loop, fake/DM/
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

- `nodes/trajectory_executor.py` remains for a consuming calibration workflow.
  Current motion graphs use `arm_controller`. Migrating that old caller needs
  a separate control-contract and safety review; do not delete the executor
  class or treat the legacy process as the template for new graphs.
- Project-shaped scene construction has moved to the consumer. The graph
  selects `SCENE_BACKEND_FACTORY` or `WORKCELL_BACKEND_FACTORY` through the
  existing entry-point resolver. `plants/mujoco/composer.py` otherwise consumes
  a generic `SceneSpec`; it does not inspect project deployment roles.
- `plants/mujoco/backend.py` retains coupled legacy fixture-release/mating
  behavior, body-name contact exclusions, and tuned gripper/object parameters.
  Moving that core preserved physics; extracting these policies would require
  separate behavior validation. It is not yet a fully domain-free backend.
