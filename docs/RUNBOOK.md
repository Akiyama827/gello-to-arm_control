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
the preview, press ARM, wait for the armed indication, then click Execute once.
The arm console uses this policy for both sim and hardware: there is no separate
hold-to-enable button or Spacebar deadman. A reviewed trajectory runs until it
finishes, faults, or is stopped; closing the browser does not cancel it.

Jog is press-and-hold on the negative/positive direction button. Release,
window focus loss, or expired jog updates stop jogging and hold without
disarming. The console displays configured Cartesian target speed in mm/s
(robot-base XYZ) and joint target speed in rad/s and degrees/s, not measured
velocity. These speeds are set in the mode profile's `jog` section.
Hand commands also require confirmed ARM and no fault.

Stop (hold) cancels motion but leaves authority enabled; DISARM cancels the
plan and drops authority. Plant command watchdogs, fault handling, and any
hardware enabling devices remain independent of this browser interaction.
The browser is not an emergency-stop device.
Keep HTTP loopback-bound. Stop/hold and disarm have distinct semantics; follow
the selected plant's safety policy rather than assuming process exit is safe.

Ctrl-C requests orderly node shutdown. Logger buffers and Hand/console resources
are closed; cancellation is not a successful motion result. Keep Rerun connected:
its native disconnect can block without a receiver, requiring launcher fallback
cleanup. The independent plant watchdog remains necessary.

The controller can also receive an explicit deployment `execution_policy` block
for deadline-driven execution with start/tracking/freshness/relatch limits. It
does not turn those limits on implicitly in existing examples. The retired
standalone executor node is no longer a supported process surface; its class
remains available under `arm_control.control.trajectory_executor`.

```bash
PYTHONPATH=. python tools/bench/check_execution_policy.py
PYTHONPATH=. python tools/bench/check_node_shutdown.py
```

Console policy checks (no hardware):

```bash
PYTHONPATH=. python tools/bench/check_console_authority.py
PYTHONPATH=. python tools/bench/check_console_grasp.py
PYTHONPATH=. python tools/bench/check_hand_grasp.py
PYTHONPATH=. python -m arm_control.console_server
PYTHONPATH=. python nodes/arm_console.py --self-check
PYTHONPATH=. python -m arm_control.control.arm_controller
node --input-type=module --check < arm_control/ui/static/console.js
```

`--config /absolute/path/to/entry.yaml` selects a different example entry.
Concrete deployments should use their own launcher and asset-root composition.
The older manuals are retained in [history](history/README.md) for engineering
context, not as instructions to run retired commands.

### Hand Grasp / Open

The existing slider commands finger position. Immediately below it, Grasp sends
an explicit target width (mm in the UI) and force (N); editing either input does
not move the fingers. Open uses the configured opening width and speed. Both
require confirmed ARM, no fault, and fresh measured feedback from a capable,
idle Hand. Force is a command, not a measured force readout. Speed and grasp
tolerances remain configuration values displayed beside the controls.

The Franka Hand adapter accepts 30–70 N, 0–80 mm total width and positive total
width speed up to 0.10 m/s. These are Hand limits, not a recommended force for
every object. Use the selected deployment's grasp defaults. Unsupported
simulators disable force grasp with a reason and retain their position slider.

Only one Hand action is admitted at a time. Stale/disconnected commands are
rejected, not queued for reconnection. Open acknowledgement is not completion:
the UI waits for a fresh measured opening. DISARM blocks new Hand actions but
does not open a held object or guarantee cancellation of an executing SDK call.

**Coordinated deployment required:** the updated Python adapter and compiled
`hand_bridge` use versioned command admission and observation metadata. An old
bridge cannot enable the new client's actions; the new bridge refuses old
unversioned actuator commands. Build against the target's installed libfranka
and update both ends during a separately authorized maintenance window. These
changes do not require an arm RT byte-protocol update. Local fake-peer/C++ checks
are not hardware deployment or force-grasp validation.

### Franka control modes

`float` retains gravity compensation and joint damping. `track` uses the
configured joint gains for planned moves. `soft` captures the measured EE
position AND orientation at the plant and holds them with Cartesian impedance,
plus a weak joint-posture spring and independent damping projected into the
Jacobian nullspace. It is compliant, not an exact geometric constraint: external
loads can deflect the EE. Gains ramp in over 0.5 seconds.

ARM, stop/settle, then select Soft. Select Track before planning or jogging.
Stop (hold) exits Soft into a measured joint hold; DISARM removes authority.
Neither exit pulls toward the joint posture from before Soft. Lost command
updates fall back to joint hold using configured Track gains, even if Float
preceded Soft. Hardware watchdog/fault behavior remains independent of the UI.

Soft requires the updated compiled binding in the same environment as Dora:

```bash
python -m pip install --no-deps -e rt/bindings
PYTHONPATH=. python rt/bindings/pose_hold_check.py
PYTHONPATH=. python tools/bench/check_pose_hold_contract.py
PYTHONPATH=. python tools/bench/check_pose_hold_controller.py
PYTHONPATH=. python tools/bench/check_pose_hold_sim.py
```

The single-model simulator advertises support only with an explicit EE body and
the updated binding. Composed workcells do not advertise Soft. Remote Franka
requires an updated server advertising `pose_hold=2`; older servers and DM
backends refuse this command before transmission. The DM example's existing
joint-gain preset is not Cartesian impedance. No new dependency on project
assembly or perception is introduced.

The real controller uses libfranka's configured EE pose/Jacobian and Coriolis;
Franka supplies gravity compensation. Verify the configured physical EE frame
and payload before bring-up. Example gains and sim checks are not hardware
certification. Deploying the server and validating control-loop timing, joint
limits, collision response and compliance on hardware are separate work.

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
