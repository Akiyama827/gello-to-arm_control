# The operator console

One page that drives one arm: set a target and plan to it, jog it by hand,
switch the control law, arm and disarm. `nodes/arm_console.py` serves it at
`console.http_port` from the robot config.

Try it without a robot — this repo ships one runnable entry config:

```bash
git clone https://github.com/frankarobotics/franka_description
python scripts/setup_fr3_assets.py --source ./franka_description   # once
python scripts/run_console.py                    # console on :7500
python scripts/run_console.py --dual             # two arms, :7500 and :7510
```

For what the 3D page draws (colour protocol, the gizmo, `/scene` and `/mesh`),
see [cartesian-teleop.md](cartesian-teleop.md). This document is about what the
console is allowed to make the arm do.

## One command owner

The console **plans and never commands the plant**. `nodes/arm_controller.py`
**servos and never plans**, and is the single producer of `motor_command` and
of the bridge's `arm` topic in every motion graph.

```
arm_console ──plan / control / jog / gripper──► arm_controller ──motor_command──► plant
```

That split is not stylistic. A node that planned *and* servoed left the arm
limp 12 times in one workcell run, because OMPL solving inside the event handler
meant nothing was streaming to the plant's deadman. The controller does a fixed
amount of work per event and can always meet a deadline; the console can take as
long as it likes to think.

## What the console can ask for

| Press | Wire | Controller |
|---|---|---|
| Plan + preview | `plan` (gated, named) | loads it, runs nothing |
| Execute | `control{execute: <that id>}` | releases that exact plan |
| Stop (hold) | `control{cancel}` | aborts the leg, holds, **stays armed** |
| ARM / DISARM | `control{arm}` | enables / lifecycle reset |
| Gains: … | `control{gains}` | cancels the leg, then re-laws |
| a held jog button | `jog` at ~10 Hz | moves, and expires |

A plan crosses **gated and named**. It runs only when an `execute` names that
exact id, so a plan superseded by a re-plan can never run — something a bare
`trajectory` topic had no way to express.

`cancel`, `hold` and `stop` are three different things and the names matter:
**cancel** aborts a leg and stays runnable (the operator's Stop button),
**hold** freezes at a milestone forever, **stop** is terminal.

## Gain presets

`gain_presets` in the mode config become buttons. `float` is `kp: 0` — the
executor's hold command then ships RNEA gravity plus payload and nothing else,
which *is* float; no separate impedance node is needed for it.

Applying a preset **cancels a running leg**, always. The plan was reviewed at
one stiffness, and float applied mid-trajectory would drop the arm through the
rest of its path.

On an arm whose wire format has gain limits (the DM MIT frame), presets are
validated by `hardware.gains.validate_hardware_gains`, which rejects rather than
clips: a slider must not be able to encode kp=600 as kp=500.

## Jog, and the envelope around it

Jog is the one motion here that no planner reviews and no operator approves in
advance. The bounds live in `arm_control/planning/jog.py` — pure, no MuJoCo, no
Dora — and **every step is checked against all five**, cheapest first:

1. **Stroke limit** — metres from where *this press* started. Release and press
   again to re-anchor. Without it a descent made of individually-legal 1 mm
   steps is unbounded.
2. **Workspace box**, whose z-minimum is the **floor** — the only environmental
   thing assumed. (`build_collision_stack` adds no ground plane, so before this
   nothing checked it.) `floor_z: null` disables it, deliberately.
3. **Joint limits** with a margin — but only against steps that make a violation
   *worse*, so an arm resting on a hard stop can always jog off it.
4. **Singularity** — σ_min of the *translational* Jacobian. This is the failure
   a velocity jog walks into that a planner never sees: 1 cm/s answered with an
   unbounded joint rate. Joint jog is exempt; it is how you back out of one.
5. **Self-collision** — last, because it is the expensive one.

A refusal names the gate, the number, and the limit, on the page:

```
jog j2 refused — stroke limit: 20.1 cm from the jog origin on z, limit 20 cm
                 — release and press again to re-anchor
```

σ_min is computed with Pinocchio, not read from a vendor API. libfranka exposes
a Jacobian, but this package drives more than one arm, and on the FR3 the motion
path lives on the RT machine while this host is read-only.

**The threshold is arm-specific and must not be inherited.** Measure it, per
arm, the same way — sample σ_min of the translational Jacobian over the joint
limits and take a low percentile:

| Arm | Extended | Bent | Median | p1 | Threshold |
|---|---|---|---|---|---|
| DM assembler | ~0.049 | ~0.18 | — | — | 0.02 (code default) |
| FR3 | 0.128 | 0.35 | 0.217 | 0.030 | 0.03 (`motion_franka.yaml`) |

The same 0.02 means very different things on those two arms — a third of the
assembler's extended reading, a sixth of the FR3's. The FR3 number is the 1st
percentile of 6000 sampled poses: it refuses the worst ~1% of the workspace
and leaves the useful envelope open.

### Two independent deadmen

A held jog button re-asserts every 100 ms. It expires **twice**, independently:

- at the console, after `JOG_STALE_S` (0.4 s)
- at the controller, after `jog.timeout_s` (0.2 s)

So a closed tab, a wedged console, a crashed browser and a cut network are all
the same event — setpoints stop arriving and the arm holds where it is. Nothing
has to notice and send a stop; **not sending is the stop.**

The jog button is deliberately *not* wired to the authority deadman (the one the
gizmo and Execute use). That one drops authority on release — Stop **and**
DISARM — which is right for a reviewed move and wrong for jogging: an operator
nudging a part into place would re-ARM between every press.

### Jog is loopback, by refusal

`console.http_bind` must be a loopback address: a non-loopback value raises at
startup rather than binding. Every endpoint on this page can move a
torque-controlled arm, unauthenticated, and jog does it without review — a LAN
bind is a robot anyone on the subnet can drive. The kernel does the real
enforcing (a socket bound to `127.0.0.1` never sees a packet off the wire);
the check just refuses to bind anywhere else.

This tightened on 2026-09-07. The panel inherited `require_loopback=False`
from `motion_teleop`, whose page could only *plan* — a move an operator
approves before it runs. Adding jog changed what a LAN bind costs.

Reach it from another desk over an SSH tunnel, which puts the authentication
in sshd where it belongs:

```bash
ssh -L 7500:127.0.0.1:7500 <arm-host>    # then open http://127.0.0.1:7500
```

## What a consuming project must supply

In the **robot** config (per instance — this is what lets two arms coexist):

```yaml
console:
  http_port: 7500        # two arms must differ; a collision fails loudly
  http_bind: 127.0.0.1
```

In the **mode** config (per control style, shared across arms of that style):
`controller.kp`/`kd`, `planner.*`, and optionally `gain_presets` and `jog`. Ports
do **not** belong here — they used to, and two arms only avoided a collision
because they happened to run different modes.

Per node, in the dataflow:

- `ARM_CONTROL_CONFIG` — which robot
- `ARM_ID` — which instance (Rerun entity root, page banner)
- `ARM_CONTROL_PLANT_HEALTH=0` — **only** on a graph whose plant has no safety
  bridge and therefore publishes no `motor_health`. Without it the controller
  waits forever for an armed edge that cannot come and simply never moves.
  `python nodes/arm_console.py --self-check` asserts every graph has exactly one
  of {health wire, opt-out}.

## Two arms

An arm is a config, so a second arm is a second config.
`configs/entry/sim_demo_b.yaml` includes the first and overrides only the
per-instance facts — console port, log path. `dataflows/dual_arm_sim.yml` runs
both: every node type twice, differing by `ARM_CONTROL_CONFIG` and `ARM_ID`.

Dora node ids namespace the topics, and `ARM_ID` namespaces the Rerun entity
paths, so both arms draw into **one** recording without overwriting each other
(`rerun_app_id` alone would give two separate recordings — not what you want
when the question is where one arm is relative to the other). Neither console
can command the other's arm: each controller subscribes only to its own.

## Checks

```bash
python -m arm_control.planning.jog            # the five gates, each on its own case
python -m arm_control.execution.arm_controller # cancel vs hold vs stop, jog expiry
python nodes/arm_console.py --self-check      # page routes + every graph can stream
python nodes/visualizer.py --self-check       # no entity path bypasses ARM_ID
python scripts/run_console.py --check         # assets, config, graph
```
