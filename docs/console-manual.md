# Operator console — user manual

How to drive an arm from the console. For *why* it is built this way — the one
command owner, the two deadmen, the jog envelope — see
[operator-console.md](operator-console.md). For the 3D page's internals see
[cartesian-teleop.md](cartesian-teleop.md).

---

## 1. Start it

```bash
git clone https://github.com/frankarobotics/franka_description
python scripts/setup_fr3_assets.py --source ./franka_description   # once, ~30 s
python scripts/run_console.py
```

Then open **http://127.0.0.1:7500**.

**You do not launch Rerun yourself.** Both launchers probe port 9876 and start
one detached viewer if none is listening — detached on purpose, so it survives
graph restarts instead of leaving you looking at a previous run's window. With
no `rerun` on PATH the console still works; you just get no visuals, and the
launcher says so. Every node in this repo only ever *connects* (`spawn=False`),
so nothing starts a viewer of its own.

`python scripts/run_console.py --check` reports what is staged and launches
nothing. `--dual` runs two arms on :7500 and :7510. `--config <path>` points at
your own entry config.

> **`dora` must be the native CLI.** If launching dies with
> `ImportError: cannot import name 'py_main' from 'dora_cli'`, you have a
> broken `dora-rs-cli` wheel first on PATH — see §8.

---

## 2. What you are looking at

| Region | What it is |
|---|---|
| **Control** (left) | Deadman, ARM/DISARM, the button strip, the jog pad, the gripper slider |
| **Scene** (middle) | The 3D robot. Drag to orbit, scroll to zoom |
| **Telemetry** (right) | Joint summary, wrench, plan status, and the last 8 log lines |

**One colour legend, on the page and in Rerun:**

| Colour | Meaning |
|---|---|
| **Real STL colours** | the live arm, where it actually is |
| **Orange** | your target — where you are asking it to go |
| **Green** | planned motion, played back |

The arm comes up **DISARMED**. Nothing moves until an operator says so.

---

## 3. The deadman

A **held** control is the only thing that authorises reviewed motion. Hold the
**Deadman** button, or hold **Spacebar** with the page focused.

Releasing it — or clicking away, switching tabs, or closing the browser — sends
**Stop + DISARM**. That is deliberate: it is the same event whether you let go
on purpose or your laptop went to sleep.

**Execute refuses unless the deadman is held.** You will see
`REFUSED: hold the deadman before Execute` in the log.

Jog buttons are *not* wired to this deadman — see §5.

---

## 4. Move somewhere: plan, review, execute

1. **Sync target to robot** — copies the measured pose into the target, so you
   start from where the arm actually is rather than from a stale orange ghost.
2. **Set a target.** Drag the gizmo on the end effector: arrows translate,
   rings rotate, one axis at a time. Each drag re-runs IK seeded from the
   current target, so the arm will not branch-jump under your hand. (The
   joint sliders behind the **debug** toggle set the target directly.)
3. **Plan + preview** — plans from measured to target and draws the result in
   green. Nothing has moved yet.
4. **Review it.** Watch the green playback. This is the only point at which a
   bad plan is free.
5. **Hold the deadman, then Execute.** The arm runs the plan you previewed —
   and only that one. Re-planning invalidates the previous plan, so a stale
   Execute cannot fire an old path.
6. **Stop (hold)** aborts the leg and holds position. The arm **stays armed**,
   so you can plan again without re-arming.

---

## 5. Jog: moving by hand

The jog pad has one row per Cartesian axis (**X / Y / Z**, world frame — "down"
means down in the room, not down along a tilted tool) and one row per joint
(**J1…Jn**). Each row has **−** and **+**.

**Press and hold.** The button is its own deadman: it re-asserts every 100 ms
while held and the arm stops within ~0.2 s of release. Nothing has to send a
stop — *not sending is the stop*. A closed tab, a wedged page, a cut network
and a lifted finger are all the same event.

Jog does **not** touch the ARM state, so nudging a part into place does not
disarm the arm between presses.

Default speed is **1 cm/s**. Every step is checked against five limits before
it is sent, cheapest first:

| Gate | The refusal it prints |
|---|---|
| **Stroke** — distance from where *this press* started | `stroke limit: 20.1 cm from the jog origin on z, limit 20 cm — release and press again to re-anchor` |
| **Workspace box**, whose floor is the z-minimum | `floor: z 1.4 cm is below the 2.0 cm limit (floor 0.0 cm + 2.0 cm clearance)` |
| **Joint limits** with a margin | `joint limit: joint 3 at -0.142 rad, usable range [-3.027, -0.167] (hard limit minus a 0.05 rad margin)` |
| **Singularity** (σ_min of the Jacobian) | `singularity: sigma_min 0.0281 below 0.0300 — jog a joint directly to back out of it` |
| **Self-collision** | `collision: the arm would hit itself` |

A refusal names the gate, the number and the limit, and prints under the jog
pad. Two rules worth internalising:

- **Release and press again to re-anchor the stroke limit.** Without that, a
  descent made of individually legal 1 mm steps would be unbounded.
- **Joint jog is the escape hatch.** It is exempt from the singularity gate,
  because backing out of a singularity is exactly what you need it for. It is
  also how you leave a joint that is resting on a hard stop — the joint gate
  only refuses steps that make a violation *worse*.

---

## 6. Changing stiffness mid-session

The **Gains: …** buttons come from `gain_presets` in the mode config. On the
FR3 sim demo:

| Preset | What it does |
|---|---|
| **Gains: float** | kp 0 — gravity and payload feedforward only. The arm goes compliant and you can push it around |
| **Gains: soft** | a tenth of the tracking stiffness; forgiving on contact |
| **Gains: track** | the configured tracking law |

**Applying a preset cancels a running leg, always.** A plan reviewed at one
stiffness must not finish at another — float applied mid-trajectory would drop
the arm through the rest of its path.

To add your own, put them in the mode config; they become buttons with no page
code:

```yaml
gain_presets:
  float: {kp: 0.0, kd: 5.0}
  soft:  {kp: 120.0, kd: 12.0}
  track: {}          # empty = the controller.kp/kd above
```

---

## 7. When it refuses

| Symptom | Cause | Do this |
|---|---|---|
| `REFUSED: hold the deadman before Execute` | Execute needs held authority | Hold Deadman or Spacebar, then click Execute |
| Nothing moves, no error | The arm is DISARMED | Press **ARM**. If there is no ARM button, this graph has no safety bridge and the controller arms on the planner's say-so |
| `jog …: no IK solution` | The target is unreachable from here — often a pose sagged against its stops | Use **joint jog** to get back to a sane posture, then **Sync target to robot** |
| `jog … refused — stroke limit: …` | You have travelled the full stroke on this press | Release and press again |
| `jog … refused — singularity: …` | Cartesian jog is gated on σ_min | Use joint jog to back out |
| `jog … refused — collision: …` | The next step would put the arm through itself | Jog the other way, or plan around it |
| Arm oscillates and never settles | kd is 0 against a non-zero kp | Set `arm.kd` in the sim config. See the sweep in `configs/entry/sim_demo.yaml` |
| Console will not start: `bind must be a loopback address` | `console.http_bind` is not `127.0.0.1` | Keep it loopback and use an SSH tunnel (§8) |
| `console port 7500 is already in use` | A previous graph is still running | `ss -ltnp \| grep 7500` for the pid. `run_console.py` refuses to start rather than let you drive the old console |
| Nodes still alive after you stop the graph | Something killed a node instead of the graph | Signal the `dora run` process — dora reaps its own nodes. `run_console.py` does this for you on exit |

---

## 8. Remote access, and one environment trap

**The console is loopback-only, and that is not negotiable.** Every endpoint can
move a torque-controlled arm, unauthenticated, and jog does it with no review.
A non-loopback `http_bind` raises at startup. To reach it from another desk,
tunnel — which puts the authentication in sshd, where it belongs:

```bash
ssh -L 7500:127.0.0.1:7500 <arm-host>     # then open http://127.0.0.1:7500
```

**The `dora` on your PATH matters.** The graphs run on the base conda
environment. Activating an env that carries its own older `dora-rs-cli` wheel
shadows the working CLI and launching fails with:

```
ImportError: cannot import name 'py_main' from 'dora_cli' (unknown location)
```

That wheel ships `dora_cli/dora_cli.abi3.so` with no `__init__.py`, so Python
reads `dora_cli` as a namespace package and the console script's
`from dora_cli import py_main` cannot resolve. Check with:

```bash
which -a dora && dora --version
```

Use the native CLI (`~/.cargo/bin/dora`) or base's, and upgrade or remove the
stale wheel from the activated env.

---

## 9. Two arms

An arm is a config, so a second arm is a second config:

```bash
python scripts/run_console.py --dual        # :7500 and :7510
```

`configs/entry/sim_demo_b.yaml` includes the first and overrides only the
per-instance facts — console port and log path. Neither console can command the
other's arm: each controller subscribes only to its own topics. Both arms draw
into **one** Rerun recording, namespaced by `ARM_ID`, so you can see where one
is relative to the other.

---

## 10. The knobs

Per **instance**, in the robot config — this is what keeps two arms apart:

```yaml
console:
  http_port: 7500        # two arms must differ; a collision fails loudly
  http_bind: 127.0.0.1   # must be loopback
```

Per **control style**, in the mode config — shared across arms of that style:

```yaml
controller: {kp: …, kd: …}
planner:    {vel_limits: 0.5, acc_limits: 1.0, …}
gain_presets: {…}
jog:
  speed_m_s: 0.01        # 1 cm/s. Deliberately slow; raise it knowingly
  joint_speed_rad_s: 0.15
  max_travel_m: 0.20     # stroke limit per press
  floor_z: 0.0           # null disables the floor gate
  sigma_min: 0.03        # ARM-SPECIFIC — measure it, never copy it
```

`sigma_min` is the one number you must not inherit from another robot. The FR3
reads 0.128 fully extended where the DM assembler reads 0.049, so the same
threshold means very different things on the two arms. Sample σ_min of the
translational Jacobian across the joint limits and take the 1st percentile.

---

## 11. Checks

```bash
python -m arm_control.planning.jog             # the five gates, each on its own case
python -m arm_control.execution.arm_controller # cancel vs hold vs stop, jog expiry
python nodes/arm_console.py --self-check       # page routes + every graph can stream
python scripts/run_console.py --check          # assets, config, graph
```
