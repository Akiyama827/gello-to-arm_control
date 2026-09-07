# Runbook

Run these commands from the `arm_control` repository root. They do not describe
the consuming project's workcell-add or hardware deployment workflow.

## Environment

The local `pyproject.toml` declares Python >=3.10, core dependencies, and the
`sim`, `viz`, and `mesh` extras. To install this checkout into your selected
environment:

```bash
python -m pip install -e ".[sim,viz,mesh]"
```

Installation can download packages and is separate from the offline checks
below. Provision Pinocchio/collision/planning and vendor hardware bindings as
needed for the selected backend; those are not satisfied by the listed extras.
Do not assume every optional module imports without its dependency installed.

## Offline checks

These commands use existing assert-based checks and do not arm hardware:

```bash
PYTHONPATH=. python -B -m arm_control.config
PYTHONPATH=. python -B -m arm_control.frames
PYTHONPATH=. python -B -m arm_control.scene
PYTHONPATH=. python -B -m arm_control.messages
PYTHONPATH=. python -B -m arm_control.planning.jog
PYTHONPATH=. python -B -m arm_control.control.arm_controller
PYTHONPATH=. python -B -m arm_control.plants.base
PYTHONPATH=. python -B -m arm_control.plants.franka.backend
PYTHONPATH=. python -B -m arm_control.plants.remote_rt.protocol
PYTHONPATH=. python -B -m arm_control.console_assets
PYTHONPATH=. python -B nodes/arm_console.py --self-check
PYTHONPATH=. python -B examples/run_console.py --self-check
PYTHONPATH=. python -B examples/run_console.py --check
PYTHONPATH=. python -B tools/assets/setup_fr3.py --check
```

Asset-status checks can report missing staged files without failing their
process. Read the output; exit zero is not evidence that a simulation launched.
Use `ruff check <changed paths>` and import/syntax checks alongside the relevant
module self-checks. They do not replace a consuming project's full gated run.

## Standalone simulated console

The examples use `examples/configs/sim_arm.yaml` and a second instance in
`sim_arm_b.yaml`, with profiles under `examples/profiles/`. They are generic
samples, not calibrated workcell scenes. First inspect asset status using the
offline commands above. If assets must be staged, explicitly select the local
source; this writes the chosen staging destination:

```bash
PYTHONPATH=. python -B tools/assets/setup_fr3.py --source /path/to/local/franka_description
```

Do not run staging over a deployed model without preserving its provenance.
With dependencies and assets ready, these commands launch simulation and UI:

```bash
PYTHONPATH=. python -B examples/run_console.py
PYTHONPATH=. python -B examples/run_console.py --dual
```

Use one launch command at a time. The console defaults to loopback port 7500;
the dual example adds 7510. For a reviewed move, sync the target, plan, inspect
the preview, then use the existing ARM/deadman/Execute controls. Jog has its
own expiring command stream and limits; it is not permission to bypass gating.
Keep HTTP loopback-bound. Stop/hold and disarm have distinct semantics; follow
the selected plant's safety policy rather than assuming process exit is safe.

`--config /absolute/path/to/entry.yaml` selects a different example entry.
Concrete deployments should use their own launcher and asset-root composition.
The older manuals are retained in [history](history/README.md) for engineering
context, not as instructions to run retired commands.

## Reusable RT build and protocol

These commands build and run offline checks only:

```bash
cmake -S rt -B /tmp/arm-control-rt-build -DWITH_FRANKA=OFF
cmake --build /tmp/arm-control-rt-build -j2
/tmp/arm-control-rt-build/arm_rt_server --mit-check
/tmp/arm-control-rt-build/protocol_selfcheck
PYTHONPATH=. python -B -m arm_control.plants.remote_rt.protocol --hex
```

The final two outputs must match exactly for wire parity. A fake/DM build is
not a libfranka-enabled build or live RT validation. Starting a server against
physical hardware, installing services, and configuring CAN belong to the
consumer's separately authorized deployment procedure.
