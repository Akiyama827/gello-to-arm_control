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
PYTHONPATH=. python -B -m arm_control.plants.base
PYTHONPATH=. python -B -m arm_control.plants.franka.backend
PYTHONPATH=. python -B -m arm_control.plants.remote_rt.protocol
PYTHONPATH=. python -B -m arm_control.console_assets
PYTHONPATH=. python -B nodes/arm_console.py --self-check
PYTHONPATH=. python -B examples/run_console.py --self-check
PYTHONPATH=. python -B examples/run_console.py --check
PYTHONPATH=. python -B tools/assets/setup_fr3.py --check
PYTHONPATH=. python -B -m arm_control.model_bundle
PYTHONPATH=. python -B -m arm_control.mukca
PYTHONPATH=. python -B -m arm_control.kinematic_fit --self-check
PYTHONPATH=. python -B tools/bench/check_rt_telemetry.py
PYTHONPATH=. python -B tools/bench/check_arm_controller.py
```

With a built `arm_rt_server` (fake backend, no hardware):

```bash
PYTHONPATH=. python -B -m arm_control.plants.remote_rt.client   # loopback rung
PYTHONPATH=. python -B tools/bench/rt/check_two_instances.py <build>/arm_rt_server
PYTHONPATH=. python -B tools/bench/rt/torque_cap.py <build>/arm_rt_server
```

`check_two_instances.py` runs two fake instances at once: simultaneous
tracking, refusal of a config aimed at the other instance's ports, an
identity-filtered observer, and per-instance fault latch and stop epoch.
`arm_control.mukca_casadi` is NOT exercised here: it needs the offline solver
environment (CasADi + forwardkinematics) that control deliberately lacks.

Asset-status checks can report missing staged files without failing their
process. Read the output; exit zero is not evidence that a simulation launched.
Use `ruff check <changed paths>` and import/syntax checks alongside the relevant
module self-checks. They do not replace a consuming project's full gated run.

## Standalone simulated console

The example uses `examples/configs/sim_arm.yaml`, with profiles under
`examples/profiles/`. These are generic
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
```

The single-arm console defaults to loopback port 7500.
For a reviewed move, sync the target, plan, inspect
the preview, press ARM, wait for the armed indication, then click Execute once.
The arm console uses this policy for both sim and hardware: there is no separate
hold-to-enable button or Spacebar deadman. A reviewed trajectory runs until it
finishes, faults, or is stopped; closing the browser does not cancel it.

Jog is press-and-hold on the negative/positive direction button. Release,
window focus loss, or expired jog updates stop jogging and hold without
disarming. The console displays configured Cartesian target speed in mm/s
(robot-base XYZ) and joint target speed in rad/s and degrees/s, not measured
velocity. The mode profile's `jog` section sets immutable startup ceilings.
Edit Cartesian mm/s or joint degrees/s and Save speeds to reduce them online;
an accepted change takes effect on the next press, never halfway through a hold.
Hand commands also require confirmed ARM and no fault.

The Control law badge reports controller-confirmed Track, Float, Soft or Custom.
It shows Unknown when feedback is unavailable or stale, and distinguishes pending
requests from applied modes. Track is the default in the supplied motion profile;
the badge reflects actual controller gains, not an assumed startup preset. This
is controller feedback, not a separate acknowledgement from the remote RT plant.
Float is selected here; the standalone `real_float.yml` graph and `float.yaml`
profile are retired. Tracking and floating share this command owner and gate.

Move shows Current as an FK estimate from fresh measured joints, not an external
pose measurement. Current and Desired use the robot base frame: XYZ in mm and
fixed-axis roll/pitch/yaw in degrees, R = Rz(yaw) Ry(pitch) Rx(roll). Copy current
fills Desired only. Apply target solves all six fields together without moving;
failed IK leaves the previous target intact. Then Plan + preview, ARM and Execute.
Edits invalidate older previews. Polling preserves unfinished input drafts.

**Refine trajectory with TrajOpt** enables the library's existing native refiner
for the next plan. It optimizes the OMPL seed, uses native spline timing, then
checks the returned curve against joint/dynamic limits and the current collision
world. Changing the checkbox invalidates the old preview; plan and review again.
Missing native dependencies, solver failure or failed validation refuse the plan
without falling back to the seed. Install the `trajopt` extra on the planning host.
The solve runs in a child process so native optimization cannot monopolize the
console's Python thread. Stop and Disarm invalidate the pending result; closing
the single-arm console terminates its planner child.

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
PYTHONPATH=. python tools/inspect/check_console_controls.py
PYTHONPATH=. python tools/inspect/check_console_page.py
PYTHONPATH=. python tools/inspect/check_controller_mode_feedback.py
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

### Dual-arm shared simulation

```bash
PYTHONPATH=. python -B examples/run_console.py --dual --check
PYTHONPATH=. python -B examples/run_console.py --dual          # http://127.0.0.1:7520
```

This launches one scene, two stock controllers and one browser console; it
opens no hardware connection. `examples/configs/dual_arm.yaml` references the
existing `sim_arm.yaml` twice and supplies only layout and pair motion limits.
For another composition use `--dual --pair-config /absolute/path/pair.yaml`;
set `ARM_CONTROL_ROOT` before launch if its profiles use deployment-relative
asset paths. An optional `scene:` loads existing `SceneSpec` static scenery.
Both actors must be configured, with matching controller joint order. The
optional `shared_object` composition below describes a supported object and
fixed grasps. Startup always preserves the configured arm joints and opens in
independent mode. Hand closure alone never establishes object ownership.

Arm both controllers. Select one or both moving arms, enter joint angles or
TCP pose (mm and fixed-axis RPY degrees, shared world or arm base). Select a
**Drag target** above the scene, or **Drag this TCP** in an arm panel, then
use **Translate / Rotate**. Drag axes are in the shared world; numeric fields
reflect the chosen coordinate frame. Dragging edits only that target. Then
**Plan + preview → review both paths → Stage pair → Execute pair**. The
unselected arm holds its measured joints on the same trajectory timeline.
Stage requires both arms armed/ready; Execute requires both staging ACKs.
Hand opening is measured separately from its command and requires ARM.

Target edits and Hand changes invalidate preview use. After reload/reconnect,
stop any staged pair and re-plan. Stop both cancels the pair and leaves the
controllers armed in hold; disarming either cancels the pair. If no browser
poll/action arrives for three seconds, an active pair and any Hand motion are
cancelled. This browser lease is additional to each hardware link's RT deadman.
Configuration/model changes require a console restart. The default example
uses nominal geometry, not the project's calibrated bench models.
For the consuming Control deployment, run `python tools/run.py sim dual` from
Control; it resolves the actual bench descriptions and approved model bindings.
Its configured object is available through the runtime **Control mode** selector.
No deployed bench values are copied into this library's example configs.

For hardware, the consumer supplies `backend: remote_rt`, two real robot
profiles with model bindings, 100 Hz execution policies and real-mode gains.
The console loads collision/display models and exports meshes before
`RemoteHardware.connect()` opens both identity-bound RT links. Fresh arm and
Hand measurements are required before the console appears; staged model poses
are replaced with those measurements. It starts disarmed. Hand moves
require the bridge's active-stop capability. Pending authority acknowledgements
block re-ARM; disarm, faults and shutdown supersede queued enables. Physical
shared-object hardware configurations are refused. A `shared_object` explicitly
marked `virtual: true` supports empty-Hand coordination without welds or payload.
The consuming Control command is
`python tools/run.py real dual` (port 7521), after its experimental RT deployment.
`--check` validates configuration without connecting. Paired hardware uses a
common PC trajectory epoch and sample time, computing both commands before
dispatch. Release/send times remain distinct; physical start skew and concurrent
FCI timing remain unqualified.

For a virtual object, choose **Virtual object**, ARM both controllers, then
**Plan to virtual grasps** → review → Stage → Execute. Keep both Hands empty
and open. Once both TCPs are within 10 mm / 2° of the nominal edge poses and
settle, confirm **Both Hands are empty; capture these TCP offsets**, then
**Enable virtual coupling**. The panel shows each residual. Entry records the
actual object-to-TCP transforms without moving either arm or the virtual plate;
these transforms stay fixed until leaving shared mode. They are included in
preview identity and passed unchanged to the planning worker. Translate/rotate
the plate and use the usual preview/stage/execute
workflow. In shared mode, the displayed pose and closure check use joint
positions interpolated to one RT publication timestamp, with fresh bracketing
samples from the same host and authority epoch. No extrapolation is allowed:
receipt age above 50 ms, publication skew or interpolation gaps above 20 ms,
or invalid feedback stops the pair. Controllers retain raw measured feedback.
The object pose is inferred from the first TCP, with agreement checked against
the second; it is not an object sensor measurement. Divergence
above 3 mm / 2° faults the console and disarms both arms. Hand commands stay
locked during coupling. Returning to **Independent arms** freezes the virtual
plate at its current pose and invalidates previews; it needs no physical support.
The next approach uses the nominal edge poses again. Capture handles a virtual
attachment only; it is not a correction for robot tracking or calibration.
Ordinary joint-goal completion retains the planned endpoint with zero desired
velocity. Timeout, cancellation and task-specific completion retain their
measured-pose hold behavior.

A shared-object runtime fault prints the measured disagreement and requests
both arms to stop. Hardware runtime faults save the last 500 PC ticks under
`$ARM_CONTROL_ROOT/.cache/dual_arm/faults/` and print the JSON path. The report
includes measured joints, PC/RT references, packet timestamps, common reference
times, interpolation brackets, captured grasps and the planned trajectories.
Receipt and RT publication times are not
synchronized physical capture times. Preserve the report before clearing the
cache; a planner pass alone does not qualify measured tracking or synchronization.

For a setup with a shared object:

```bash
PYTHONPATH=. python -B examples/run_console.py --dual \
  --pair-config examples/configs/dual_arm_object.yaml
```

This preset includes `dual_arm.yaml` and overrides only the facing layout and
fixed grasps. It uses one 200 × 300 × 12 mm, 1 kg plate; robot descriptions,
gains and limits remain in the existing arm profiles.
Choose **Shared object** in the console and arm both controllers. With open
Hands and external support, **Plan to grasp poses** plans from the current joints
with both arms, the object and scenery in collision checking. Review → Stage →
Execute, then close each Hand to its displayed grasp width. Explicitly verify
both grasps and enable shared-object mode. Already-aligned arms can skip the
approach. Entry requires 1 mm / 0.5° alignment, matching measured and commanded
Hand widths, no active operation, and 0.25 s stationary feedback.

The mode switch preserves the plant, controller instances, joints and object
pose; it transfers support and payload accounting to the two grasps. After
settling, use the object's translate/rotate controls and ordinary plan workflow.
Independent targets and Hand commands are locked while both fixed grasps are
active. Physics pauses if either controller is disarmed.

The same **Refine trajectory with TrajOpt** checkbox applies to independent
moves, grasp preparation and shared-object moves. Independent refinement optimizes
both moving arms together; an unselected peer is held exactly. Shared-object
refinement constrains both TCP poses at every interior knot of the prescribed
object path, with fixed joint endpoints. The payload follows the first grasp in
the optimization collision model. Both grasps, the object path, boundary and
collisions are checked again on the returned Hermite curve. It does not search
for object detours or control internal grasp force. Refinement is off initially;
changing it clears the preview and is refused during active pair operations.

To leave, stop and externally support the object. Choose **Independent arms**
and confirm support before disabling the two simulated grasp constraints.
Support anchors the object's current pose; then open the Hands before moving
away. Every mode change invalidates old previews and staging identities.

Object moves follow a straight translation and shortest rotation, with a
500 mm / 90° request limit. Unreachable, discontinuous, colliding or
closure-breaking paths fail instead of detouring in joint space. The preview
checks grasp and object-path errors against 1 mm / 0.5° on the retimed curve.
The running simulator faults at measured grasp error above 3 mm / 2°.
Equal payload gravity sharing uses the existing executors and torque limits;
world gravity is transformed into each model's base frame, including tilted
mounts. It is not an internal-force controller or evidence of a secure physical grasp.

Supported here: joint/TCP and fixed-grasp object targets, translate/rotate
gizmos, common retiming, shared collision checking, preview, pair staging and
release, guided grasp approach, explicit mode confirmation, per-arm ARM/DISARM
and independent-mode Hand position. Continuous jog, Soft mode, automatic
pickup and physical coupled carrying are not exposed. Hardware transport and
empty-Hand virtual coordination use the experimental adapter described above.

Device-free regression commands (planning/simulation checks need staged assets):

```bash
PYTHONPATH=. python -B tools/bench/check_bimanual_contract.py
PYTHONPATH=. python -B tools/bench/check_bimanual_sync.py
PYTHONPATH=. python -B tools/bench/check_pair_planning.py
PYTHONPATH=. python -B tools/bench/check_dual_console.py
PYTHONPATH=. python -B tools/bench/check_cooperative.py
PYTHONPATH=. python -B tools/bench/check_dual_modes.py [path/to/pair.yaml]
PYTHONPATH=. python -B tools/bench/check_dual_modes.py [path/to/pair.yaml] --refine
PYTHONPATH=. python -B tools/bench/check_refinement.py
```

The older `dataflows/dual_arm_sim.yml` deliberately runs two **independent**
worlds and consoles on 7500/7510. Retain it for process-isolation diagnostics;
it provides neither cross-arm collision checking nor coordinated execution.
`--dual` now selects the shared-world console above.

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

Measured width has a separate 0.01 mm endpoint allowance: closed jaws were
observed reporting about -0.002 mm. Measurements within that margin are
normalized to 0 or 80 mm; larger excursions and nonfinite readings remain
invalid. This is not a vendor accuracy claim and does not widen command limits.

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
requires an updated server advertising `pose_hold=4`; older servers and DM
backends refuse this command before transmission. The DM example's existing
joint-gain preset is not Cartesian impedance. No new dependency on project
assembly or perception is introduced.

Real FR3 operating gains and mode choices are deployment-owned. The consuming
Control project supplies `configs/modes/franka_rt.yaml`; the old library
`examples/profiles/motion_franka_rt.yaml` was retired. Mode paths resolve from
`ARM_CONTROL_ROOT` first, then the library for reusable example profiles.

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
