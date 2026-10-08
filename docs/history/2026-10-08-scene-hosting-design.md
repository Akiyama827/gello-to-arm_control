# Composed scenes hosted by the supervisor: design note and the observe/1 contract

Status: ACCEPTED 2026-10-08 (D1 (a), D2 after D1, D3 `sim_bench` first); step 1 built (2026-10-08-observe-simlink.md).
Operator decision 2026-10-06: "Go" on the supervisor hosting the composed scene, with an observer
channel; start with a design note and the observer-channel contract. Plan row: CTRL-053 (supervisor as the single control authority;
Dora `arm_controller` retired per graph).

## 1. The problem

`SimLink` (2026-10-06-sim-link.md) puts a single-model MuJoCo plant behind the supervisor: one arm, one world, one owner. The
project's composed-scene graphs (`sim_workcell_add`, `sim_bench`) run a different plant: one MuJoCo world holding the FR3, the
2-DOF modular base, the Franka Hand, the row modules, the storage and the welds. The project's `mujoco_interface` node hosts it
today. In that world:

* **Writers** (inputs of the plant node):
  * the arm's motor commands (from `sim_bridge`, ultimately the Dora `arm_controller`);
  * the base's commands (`base_driver`);
  * the gripper (`sim_bridge`);
  * scene mutations, i.e. graft, weld and release (`scene_grafter`, as `scene_command` with optimistic revision checks).
* **Readers** (outputs, and who consumes them):

  | Output | Readers |
  |---|---|
  | `motor_state_franka` / `_assembler` | `sim_bridge` (to the controller) |
  | `motor_state_base` | `arm_planner`, `base_driver` |
  | `gripper_state` | `arm_planner`, `sim_bridge` |
  | `scene_state` (revision, actor q, attachments, constraints) | `arm_planner`, `base_dynamics`, `base_driver`, `scene_grafter` |
  | `scene_result` | `base_dynamics`, `base_driver`, `scene_grafter` |
  | `constraint_force` | `workcell_coordinator` |
  | `sim_qpos` | `sim_perception` |
  | `inhand_pose` | `arm_planner` |
  | `topology_state`, `model_revision`, `ee_wrench` | the viewer |

One MuJoCo world has one owner. If the supervisor is to be the single control authority over the arm in these graphs, it must own
the world, because the arm's plant is that world. Every other reader must then see the world through the supervisor, without being
able to write to it or to slow it down.

## 2. The design

**2.1 The supervisor hosts the world.** A `ScenePlant` link generalizes `SimLink`: same snapshot authority per tick, same per-ARM-epoch
plant release, same FAULT_PLANT and CMD_LOST semantics. It builds the world through the project's existing injected factories
(`WORKCELL_LOADER`, `WORKCELL_BACKEND_FACTORY`, `CHAIN_FACTORY`, `MATE_POLICY`), so nothing project-specific moves into the library.
The supervised arm is the scene actor named by the scene's `assembly.roles.assembler`. Its commands come only through the existing
admission path (submit, release, stream), exactly as for SimLink and hardware.

**2.2 Observation leaves the supervisor through `observe/1` (§3).** It is read-only, needs no ownership, and cannot block the plant.
A Dora adapter node subscribes and republishes the same Dora outputs as today (`scene_state`, `sim_qpos`, `inhand_pose`, ...).
Graph readers keep their inputs, and only the publisher changes. This is the migration seam.

**2.3 The other writers are the open decisions (§4).** The arm is not the only actuator in that world. Read-only observation solves
readers, not writers.

## 3. The observe/1 contract

Negotiated like `trace/1`: the supervisor advertises `observe: 1` in `hello.extensions` only when its link observes (a single-model
`SimLink` with `sim_observe` stated, step 1; a scene host, step 2). A
client that sees no `observe/1` gets `Refused` and must not fall back to anything else.

**Operation** `observe(after: int = 0, limit: int = 64) -> ObservationBatch`; `limit` is capped at 64 (rule 7). It takes no ownership and no claim; any session may
call it. It never waits on the plant and never waits for a sample. It returns what the ring holds after `after`, oldest first,
at most `limit` samples.

**ObservationBatch**

| Field | Meaning |
|---|---|
| `schema` | `"observe/1"` |
| `samples` | list of `SceneObservation`, oldest first |
| `latest_seq` | the cursor to pass as `after` next time |
| `dropped` | samples that fell off the ring between `after` and the first sample returned (0 on a first read with `after=0`) |
| `errors` | samples that could not be recorded since the supervisor started (cumulative) |
| `rate_hz` | the publication rate the supervisor is running: `state_hz / ceil(state_hz / requested)`, so 100 requested at 250 Hz runs 83.3 Hz |
| `capacity` | the ring's size in observations |

**SceneObservation** (one plant tick, decimated). A field that cannot be computed in that tick is `None`, never a stale copy.

| Field | Meaning | Replaces |
|---|---|---|
| `seq` | monotonic per supervisor process | — |
| `sim_time_s` | plant time of this tick | — |
| `observed_ns` | supervisor monotonic clock when sampled | — |
| `model_revision` | the compiled model's revision (bumps on graft or recompile) | `model_revision` |
| `scene_revision` | the SceneState revision (the optimistic-concurrency counter) | `scene_state.revision` |
| `actor_q` | `{actor: {joint: q}}`, named, every actor | `motor_state_*`, `sim_qpos` (actor part) |
| `actor_dq` | same shape, velocities | `motor_state_*` (velocity part) |
| `gripper` | `{positions, efforts}` of the finger joints (one gripper per world today; keyed by actor when D2 adds a second) | `gripper_state` |
| `attachments`, `constraints` | as in `scene_state` today | `scene_state` |
| `object_poses` | `{object: [x, y, z, qw, qx, qy, qz]}` in the world frame (None until step 2b) | `sim_qpos` (object part) |
| `inhand_pose` | the held module in the EE-body frame, `[x, y, z, qw, qx, qy, qz]` (plant truth) | `inhand_pose` |
| `qpos` | the raw plant `qpos`, valid with `model_revision` | `sim_qpos` |
| `constraint_force` | `{constraint: [fx, fy, fz, tx, ty, tz]}` | `constraint_force` |
| `ee_wrench` | `{frame, wrench: [fx, fy, fz, tx, ty, tz]}` (one EE per world today; keyed by actor when D2 adds a second) | `ee_wrench` |
| `topology` | the topology record as published today (without `schema`) | `topology_state` |

**Rules**
1. **The plant never waits for a reader.** The ring overwrites, and a slow reader sees `dropped`. This is the trace/1 rule.
2. **Decimation:** one observation every `ceil(plant_rate / rate_hz)` plant ticks. The default `rate_hz` is 100, stated in the
   scene-host config, never hardcoded. 100 Hz covers every current reader: perception samples at 1 Hz, and the planner and
   coordinator react at 10–100 Hz.
3. **Consistency:** every field of one observation comes from the same plant tick. The plant is touched only by the tick thread
   (step, then publish, then observe), so no other thread can change it in between. A scene host with other writers must keep that
   confinement, or take observations under the lock its writers take. No field is assembled across ticks.
4. **Read-only:** `observe/1` has no write operation. Scene mutations are not observations (§4, D1).
5. **No secrets of authority:** observations carry no session ids or execution ids. Authority is read with `authority()`, as today.
6. **Size bound:** an observation is bounded by the scene (actors × joints + objects + constraints). The ring holds `N` observations;
   `N` is stated in config (`ring_s`, at most 10 s), with a suggested 2 s at `rate_hz`.
7. **Read bound:** a read returns at most 64 observations. Building and serializing a batch runs in the supervisor's process and
   competes with the plant tick for the interpreter. Measured (review of step 1): four tight readers taking 1024 samples per read
   slowed the plant to 0.44x real time. At 64 it stays at 1.0x, and the bench holds that. So a reader must poll at least
   `rate_hz / 64` times a second, or it drops samples (1.3 Hz at 83.3 Hz; 15.6 Hz at 1 kHz). The Dora adapter's timer must respect it.
8. **Sequences are per process:** a reader whose session changes (a new hello) starts again from 0.

**Dora adapter.** A `scene_observer` node polls `observe(after=cursor)` at its timer, and republishes each sample to the outputs
the graph already wires, using the existing packers (`pack_scene_state`, ...). It forwards `dropped` as a counter output and never
interpolates. With it, `sim_perception`, `arm_planner`, `base_dynamics`, `base_driver`, `workcell_coordinator` and the viewer keep
their inputs unchanged.

## 4. Decisions needed before building (operator)

* **D1 — the other writers.** The base, the gripper and scene mutations also write to the world. Options:
  * (a) The supervisor admits them too. The base becomes a second supervised device in the same world (its own claim and ARM,
    through the SDK), and the gripper and scene mutations become negotiated actions (`scene_action/1`: graft, weld, release,
    gripper move), each with an owner lease and the existing optimistic `scene_revision` check. This gives one authority, with
    more work.
  * (b) The supervisor accepts them on an unowned command side-channel from the project's nodes, as `mujoco_interface` does today.
    That is quicker, but the world then has writers the supervisor does not admit.

  Recommendation: (a). The plan's rule is one control authority, and the base is a real device on hardware too (`base_driver`
  drives DM motors).
* **D2 — the dual simulated pair.** Two arms in one world cannot have two supervisors. Options:
  * (a) One supervisor hosts the world and runs two RobotRuntimes, one per arm, sharing the ScenePlant tick. This is
    `group/1`-capable in simulation.
  * (b) Keep the pair in the library's in-process `SharedSimulation`, outside the supervisor.

  Recommendation: decide after D1. It reuses whatever multi-device admission D1 builds.
* **D3 — the first graph to migrate.** `sim_bench` (one arm, Hand, perception, no base actuation: fewer writers) before
  `sim_workcell_add` (base, grafts, coordinator). Recommendation: `sim_bench`.

## 5. What is built first, once D1 is decided

1. `observe/1` over the existing single-model `SimLink`, with only the fields it has. This proves the ring, the cursor, `dropped`
   and the adapter offline, before any scene work. Bench: a slow reader cannot slow the plant; `dropped` is exact; every field of
   an observation comes from one tick.
2. `ScenePlant` through the project's injected factories, with the arm only. `sim_bench` runs with the Dora `scene_observer`
   replacing `mujoco_interface`'s outputs, and the arm under the supervisor (the `DoraArmBridge` already serves the controller
   contract).
3. D1's writers.
4. `sim_workcell_add`.

Each step is its own reviewed slice with a track-list row. Retiring the Dora `arm_controller` for any graph stays an operator stop.
