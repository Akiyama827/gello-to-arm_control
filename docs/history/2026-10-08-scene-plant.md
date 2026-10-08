# Scene hosting step 2b: the supervisor hosts a scene world (ScenePlant)

Design: 2026-10-08-scene-hosting-design.md (D1 (a), D3 `sim_bench` first). Step 2a moved the world logic into `SceneCore`; this
puts that world behind the supervisor.

**ScenePlant is a plant, not a second link.** `SimLink` already is the supervisor's simulated link (authority, ARM epochs, the
deadman, FAULT_PLANT, observe/1). `ScenePlant` (`plants/mujoco/scene_core.py`) gives it the supervised actor's slice of a
`SceneCore` world, built by the same `build_world` the Dora plant node uses (the project's injected factories, unchanged), so no
authority or world logic is copied:
* `step(command)` writes the actor's slice and steps the whole world once. `SimLink` writes every tick (a command, a hold or a
  release word), so that slice's own idle deadman never fires. A DISARM releases that slice only.
* The other actors (the base) keep the world's spawn hold until their writers exist (step 3, D1). The gripper keeps its spawn
  opening (gripper commands are D1 too).
* `observe()` adds the world's actors by name, so observe/1's `actor_q`/`actor_dq` are `{actor: {joint: q}}` for EVERY actor
  (one model: this instance, as before).
* `SimLink.from_config` builds it when the config describes a scene (`describes_scene`: `WORKCELL_SCENE` or `scene.arm`, the
  one rule `build_world` also uses). A scene plant needs `sim_actor` (which world actor this supervisor drives, checked against
  the arm's joint count) and `idle_timeout_sec`; neither is assumed.

**observe/1 `object_poses`** (promised for step 2b): `MuJoCoBackend.observe()` reports every free body by name as
`[x, y, z, qw, qx, qy, qz]` (its free joint's qpos, world frame, exact after the step); None in a world with none.

Checks (`check_sim_supervisor.py`, new `scene_plant_unit`, synthetic workcell: a 3-joint arm, a 1-joint peer, a free part): the
config refusals (no `sim_actor`, no `idle_timeout_sec`, unknown actor, joint-count mismatch); ARM and tracking through `SimLink`;
the peer's spawn hold untouched by commands and by DISARM; the arm slice released on DISARM; actor-keyed observations, object
poses, topology; a failing world step latches FAULT_PLANT. Mutants caught: no slice write, other actors zeroed, observations not
keyed by actor, no object poses. Related, all pass: `check_sim_supervisor`, `check_scene_core`, `check_dora_arm` (the
supervisor over a single-model sim plant), `python -m` scene_core, backend, sim_link.

On the real composed world (`configs/entries/sim_franka_bench.yaml` with the project's factories, scratch, not a check): actors
`assembler` (7) and `base` (2); the module's free body pose; in-hand pose, topology, gripper, EE wrench and constraint forces all
present. Holding the spawn pose with kp 200 and NO torque feedforward sags 0.12 rad in 1 s: the composed path bakes no plant
gravity compensation (the workcell adapter does, for the arm actor), so today's graph holds the arm with the controller's gravity
feedforward. For step 2c (open): the supervised path must carry the same feedforward, or the composed plant must compensate
the arm as the workcell path does. That is a plant-model / control decision for the operator, not made here.

Next (2c): `sim_bench` under the supervisor: its config gains `rt.instance_id`, `sim_servo`, `sim_observe`, `sim_actor:
assembler`; the Dora `scene_observer` replaces `mujoco_interface`'s outputs; `sim_bridge`'s gripper path waits for D1.
