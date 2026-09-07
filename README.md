# arm_control

Reusable robot control, planning, plant backends, and Dora process adapters.
The consuming project supplies robot instances, measured calibration, workcell
geometry, task policy, and hardware deployment. This library does not require
the consuming project or a perception package to import its generic APIs.

Start with [Architecture](docs/ARCHITECTURE.md) for ownership and dependency
boundaries, and [Runbook](docs/RUNBOOK.md) for current commands. The
[historical manuals](docs/history/README.md) preserve previous designs and bench
records; they are not current operating instructions.

## Layout

| Path | Owner/responsibility |
|---|---|
| `arm_control/motion/` | Shared joint-state, command, and trajectory values |
| `arm_control/control/` | Execution, controller state, safety, gain validation |
| `arm_control/planning/` | IK, collision checking, planning, retiming, preview builders |
| `arm_control/contracts/` | Central Dora/Arrow wire contracts |
| `arm_control/plants/` | Plant protocol; DM, Franka, remote RT, MuJoCo adapters |
| `arm_control/end_effectors/` | Generic grasp and Hand behavior |
| `arm_control/ui/`, `arm_control/viz/` | Operator adapters, packaged web assets, telemetry |
| `arm_control/simulation/` | Compatibility imports, generic convex decomposition and scene mirroring |
| `nodes/` | Dora executable wrappers; package modules contain implementations |
| `examples/` | Standalone console application, example configs, reusable mode profiles |
| `tools/` | Generic DM/RT diagnostics and asset staging |
| `dataflows/` | Generic process compositions |
| `rt/` | Reusable C++ servo core, backends, protocol, and Franka Hand bridge |

## Quick offline checks

From this repository, in an environment containing the required dependencies:

```bash
PYTHONPATH=. python -B -m arm_control.config
PYTHONPATH=. python -B -m arm_control.frames
PYTHONPATH=. python -B -m arm_control.messages
PYTHONPATH=. python -B examples/run_console.py --check
```

`--check` reports local example assets and launches nothing. It is not a
simulation pass or hardware-readiness check. Installation and optional
dependencies are declared in `pyproject.toml`; see the runbook before launching.

Two migration limits remain explicit: the legacy trajectory-executor process
still has a consuming calibration caller, and the relocated MuJoCo backend
retains coupled legacy fixture/contact policy. Project scene construction now
belongs to the consumer; this does not make the physics core fully generic.
