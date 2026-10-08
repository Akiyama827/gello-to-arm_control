# The simulated world's per-tick logic, shared (scene hosting, step 2a)

Design: 2026-10-08-scene-hosting-design.md. To host the composed world in the supervisor (step 2b) without a second copy of the
plant, the world logic moves out of the Dora `mujoco_interface` node into a Dora-free module, `plants/mujoco/scene_core.py`. The
node and the coming supervisor scene link are its two front ends. The code moved with its comments; it was not rewritten.

* `build_world(cfg, period=...)` builds the world a config describes: a workcell scene (`WORKCELL_SCENE`, the project's injected
  loader), a composed plant scene (`scene.arm`), or one model. It loads it and opens the jaws.
* `SceneCore` holds the command vector and its rules:
  * per-actor slices, each with its own deadman: gains are zeroed after the idle timeout, the warning is said once per gap, and
    time the host spent blocked in its own work is credited back;
  * the single-model ARM gate, and the Soft-mode expiry;
  * gripper commands;
  * optimistic scene changes, refused on a stale revision;
  * stepping, and the sim-time-behind-wall accounting.
* The node keeps only Dora: event decoding, message packing, publication timers and the debug-force printout. Its `main` lost 248
  lines and gained 18.
* `MuJoCoBackend.observe()` adds the scene fields for observe/1: the raw `qpos` (valid with the model revision), the scene revision,
  attachments and constraints, the in-hand pose, the constraint reactions, the topology, the gripper and the EE wrench. A field the
  world cannot have is None.

Checked:
* `tools/bench/check_scene_core.py` (new, in the offline suite) checks the moved rules directly: a silent slice's gains are zeroed
  (and only that slice's); the warning is said once, even while another slice keeps a shared Cartesian term active; stall credit;
  the ARM gate; Soft expiry; a change to a world without a scene is refused. Mutants killed: no stall credit, a warning every tick,
  every slice zeroed, the ARM gate ignored.
* Live, base (1095b1c) against this branch:
  * `sim-add` (workcell path): the plant's log lines are identical (actuators, slices, jaws, both slices' applied gains), with no
    tracebacks.
  * `sim franka bench` (composed path): the same actuators and slices; all nodes ready; perception registers the module from the
    plant's published state (fit 1.00); no tracebacks. The graph needs about 40 s to start (mesh decomposition), so a shorter run
    shows a timeout-induced cascade at base and tip alike.
* `check_sim_supervisor.py` (SimLink over the same backend) and `python -m arm_control.plants.mujoco.scene_core` both run without
  importing Dora.
