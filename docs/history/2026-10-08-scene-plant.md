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
present. Gravity, corrected twice. My first note called the 0.12 rad sag (kp 200, zero feedforward) an open decision; my second called it
only a probe artifact because the controller ships gravity feedforward (`plant_gravity_comp` unset, 0 gravcomp bodies in the
composed world). The review showed the second is half right: feedforward reaches the arm only while the controller streams.
`SimLink`'s own words carry zero torque: the release word (before ARM, after DISARM), the ARM hold before the first command and
the deadman hold. Measured on the composed world, disarmed for 2 s after `open()`: the arm falls up to 3.72 rad from spawn
(the Dora node holds the spawn pose at kp 60 and sags 0.38 rad in the same 2 s), and ARM then latches the collapsed pose as its
hold. A real FR3 does not fall when disarmed. This is a plant-model decision and it GATES step 2c (no `sim_bench` under the
supervisor before it): compensate the arm actor in the plant (as the workcell path does; then the config must say
`plant_gravity_comp: true` so the controller stops shipping gravity, which changes today's Dora `sim_bench` too), or model the
FR3's brakes in the released state, or accept a falling disarmed twin. Asked of the operator.

`sim_actor` replaces the design's `assembly.roles.assembler` (§2.1): the library knows no roles; the project's config names the
actor.

Review round 1: ACCEPT (no claim falsified; threading, `describes_scene` equivalence and single-model observe/1 confirmed;
observe cost 55 -> 73 us per call, median 0.15 ms in-thread against a 4 ms tick). Taken:
* pins for what survived: the slice is written before the world steps (the step a sample reports ran under that sample's command;
  mutant "tick first" caught), object poses are this tick's world wxyz pose (checked against an independent kinematics pass;
  mutants "xyzw" and "spawn qpos0" caught);
* the actor's world joints must be the arm config's joints in order (suffix match; only the count was checked; mutant caught);
  `idle_timeout_sec` must be finite and positive; an unnamed free body is keyed `body<id>`.
Notes for 2c: scene `actor_q` keys are world names (`asm_fr3_joint1`) while the supervisor's state stream uses config names
(`fr3_joint1`); finger joints appear under `gripper`, not in the scene's `actor_q`.

Next (2c): `sim_bench` under the supervisor: its config gains `rt.instance_id`, `sim_servo`, `sim_observe`, `sim_actor:
assembler`; the Dora `scene_observer` replaces `mujoco_interface`'s outputs; `sim_bridge`'s gripper path waits for D1.
