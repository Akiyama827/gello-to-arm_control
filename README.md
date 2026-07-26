# arm_control

Real-time arm control over [Dora-rs](https://github.com/dora-rs) dataflow:
plant interfaces, safety/grasp bridge, servo execution, motion planning, and a
MuJoCo digital twin. Arm-agnostic — an in-house DM-motor arm and a Franka
Research 3 run the same graphs, differing only in a plant node and a YAML.
Adding a robot is a new config file, never a code edit: there are no
robot-shaped defaults anywhere in this package, and a config that forgets part
of its identity fails loudly by key name.

| Path | Purpose |
|------|---------|
| `arm_control/` | The installed package: frames, config, messages (wire codecs incl. grasp + module_poses), joint↔motor map, dynamics |
| `arm_control/hardware/` | DM CAN + SocketCAN transport, libfranka backend (read-only + gripper) |
| `arm_control/bridge/` | Safety controller, grasp gate, grasp controller |
| `arm_control/execution/` | Servo law / trajectory executor |
| `arm_control/planning/` | Pinocchio IK, OMPL, retiming, MuJoCo collision oracle, Rerun preview, `stack.py` (one-arm stack builder) |
| `arm_control/simulation/` | MuJoCo backend, composed-scene builder, Rerun mirror, convex decomposition |
| `nodes/` | Thin Dora node adapters, one per process (launched by path from a dataflow) |
| `dataflows/` | Dora graph YAMLs for motion / view / float / listen |
| `configs/` | `modes/` (controller modes) + `real/<arm>/{hardware,calibration}.yaml` (per-robot fragments) |
| `scripts/` | `bench_ramp.py` (DM gain ladder), `setup_fr3_assets.py`, DM bench probes |

Runnable ENTRY configs (the ones that compose hardware + calibration +
scenario into one tree) live with the deployment, not here — this repo ships
per-robot fragments a project includes.

## What this is NOT

- **Not perception.** No cameras, no point-cloud registration, no ArUco, no
  hand-eye calibration. That is
  [`perceptions`](https://github.com/chiyangW/perceptions), which depends on
  this repo for `frames`, `messages` and `config` — never the reverse.
- **Not the assembly task.** No pick-and-dock FSM, no orchestrator, no
  scenario configs, no module/base CAD, no RL. Those stay in the
  `modular_robotic_arm` project, which consumes this repo.
- **On the FR3 specifically:** there is deliberately NO motion path from this
  host. The FR3 runs torque control only, and the torque servo is a C++ 1 kHz
  loop on a realtime machine. `nodes/franka_interface.py` streams state,
  reports health, and drives the Franka Hand; `motor_command` inputs are
  dropped with a warning.

## Dependency direction

```
modular_robotic_arm  ──►  perceptions  ──►  arm_control
        └──────────────────────────────────────┘
```

Both arrows point INTO `arm_control`; this repo imports nothing from either.

## Path anchors

`arm_control.REPO_ROOT` is this checkout (shipped mode configs, `dlls/`).
`arm_control.CONTROL_ROOT` is the DEPLOYMENT root that relative asset paths in
runtime configs resolve against — it defaults to `REPO_ROOT` standalone, and a
project embedding this repo as a submodule exports
`ARM_CONTROL_ROOT=<project>/Control` so its own CAD keeps resolving.

## Install

```bash
conda activate base                     # this stack runs in conda BASE
conda install -c conda-forge pinocchio  # + ompl if you want the OMPL planner
pip install -e ".[sim,viz,mesh]"
```

`pinocchio`, `hpp-fcl`, `ompl`, `pylibfranka` and `dmcan` are **not
pip-installable** — see the comments in `pyproject.toml`. Every one of them is
imported lazily, so the package imports and the self-checks below run without
them.

## Checks

No pytest in this project by decision. Lint plus runnable self-checks:

```bash
ruff check .

python -m arm_control.frames                       # frame algebra asserts
python -m arm_control.config                       # include/merge/circular-include asserts
python -m arm_control.hardware.franka_backend      # FR3 config parsing asserts
python scripts/setup_fr3_assets.py --self-check    # asset staging asserts
```

Graphs need a config: point `ARM_CONTROL_CONFIG` at an entry YAML (there is no
default robot) and run e.g. `dora run dataflows/sim_motion.yml`.
