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
| `arm_control/planning/` | Pinocchio IK, OMPL, retiming, MuJoCo collision oracle, Rerun preview, `jog.py` (the jog safety envelope), `stack.py` (one-arm stack builder) |
| `arm_control/simulation/` | MuJoCo backend, composed-scene builder, Rerun mirror, convex decomposition |
| `nodes/` | Thin Dora node adapters, one per process (launched by path from a dataflow). `arm_console.py` is the operator page; `arm_controller.py` is the only producer of `motor_command` |
| `dataflows/` | Dora graph YAMLs for motion / view / float / listen |
| `configs/` | `modes/` (controller modes) + `real/<arm>/{hardware,calibration}.yaml` (per-robot fragments) |
| `scripts/` | `run_console.py` (the demo launcher), `bench_ramp.py` (DM gain ladder), `setup_fr3_assets.py`, DM bench probes |
| `rt/` | The RT machine's C++ 1 kHz torque server + wire protocol + the shared servo-law binding — see `rt/README.md` |

Runnable ENTRY configs (the ones that compose hardware + calibration +
scenario into one tree) live with the deployment, not here — this repo ships
per-robot fragments a project includes.

**One narrow exception: `configs/entry/`.** Those are EXAMPLES, not a
deployment. Without a single entry that runs, "arm-agnostic library" is a claim
nobody can check and a new consumer has nothing to copy. `sim_demo.yaml` drives
the console against a simulated FR3; `sim_demo_b.yaml` is a second arm that
overrides only its console port and log path. A real robot's entry config still
lives with the project that owns its CAD.

## What this is NOT

- **Not perception.** No cameras, no point-cloud registration, no ArUco, no
  hand-eye calibration. That is
  [`perceptions`](https://github.com/chiyangW/perceptions), which depends on
  this repo for `frames`, `messages` and `config` — never the reverse.
- **Not the assembly task.** No pick-and-dock FSM, no orchestrator, no
  scenario configs, no module/base CAD, no RL. Those stay in the
  `modular_robotic_arm` project, which consumes this repo.
- **On the FR3 specifically:** there is deliberately NO motion path from this
  host process. The FR3 runs torque control only; the servo is `rt/`'s
  `arm_rt_server` on a realtime machine, reached through
  `nodes/rt_interface.py` (same plant-node contract as every other bridge).
  `nodes/franka_interface.py` remains the direct-FCI read-only + gripper
  bridge; its `motor_command` inputs are dropped with a warning.

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
python -m arm_control.planning.jog                 # the five jog safety gates
python -m arm_control.execution.arm_controller     # cancel vs hold vs stop, jog expiry
python -m arm_control.console_server               # loopback / CSRF / body cap
python scripts/setup_fr3_assets.py --self-check    # asset staging asserts

python nodes/arm_console.py --self-check           # page routes + graphs can stream
python nodes/visualizer.py --self-check            # no entity path bypasses ARM_ID
python scripts/run_console.py --check              # assets, config, graph
```

## The operator console

One page per arm: plan and execute a move, jog by hand inside a checked
envelope, switch the control law, arm and disarm. It runs from a fresh clone:

```bash
python scripts/setup_fr3_assets.py --source <franka_description checkout>
python scripts/run_console.py            # console on http://127.0.0.1:7500
python scripts/run_console.py --dual     # two arms at once, :7500 and :7510
```

`docs/console-manual.md` is how to drive it — the deadman, plan/review/execute,
the jog pad, what each refusal means. `docs/operator-console.md` is why it is
built this way: the jog envelope (stroke limit, floor, joint limits,
singularity, self-collision, all checked per step) and the two independent
deadmen that stop the arm when the page goes away.

Graphs need a config: point `ARM_CONTROL_CONFIG` at an entry YAML (there is no
default robot) and run e.g. `dora run dataflows/sim_motion.yml`.
