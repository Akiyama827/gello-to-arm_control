# One world, several supervised actors; scene_action/1 (scene-hosting D1(a)/D2(a); sim-add on supervisors)

Operator 2026-10-09: sim-add moves onto supervisors. In it, the base's supervisor also commands the module joint above the base,
and the slice gets a sim-only e2e. The design is `2026-10-08-scene-hosting-design.md`:
* D1(a): the base is a second supervised device; scene edits go through `scene_action/1`.
* D2(a): one supervisor process hosts the world.

Library:
* `SharedWorld` (`plants/mujoco/scene_core.py`): one composed world and its `SceneCore` under one lock.
  - The HOST actor's servo thread steps the world. A co-hosted actor writes its slice and reads the state the host's last step
    left, so it sees one host tick of latency.
  - `ScenePlant.cohost(shared, cfg, joints)` builds a co-hosted actor's plant; its `load` and `close` leave the world alone.
  - The Hand's finger I/O runs under the same lock.
* `SimLink.from_config(cfg, cohost_of=shared)`. `SimLink.scene_world` is set only on the hosting link.
* Service `--cohost CONFIG=SOCKET` (repeatable, sim only):
  - each co-host is another `RobotRuntime` with its own config, claim, ARM and policy, and its own `Service` thread;
  - co-hosts start after the host, so the world is loaded first, and close before it.
* `launch.start_supervisor(cohosts={PREFIX: config})` adds `PREFIX_CONFIG` and `PREFIX_SOCKET` to the graph env.
* `scene_action/1` (extension `scene_action: 1`, on the hosting supervisor of a world with a scene state):
  - `scene_claim` takes a lease for one writing session. It is separate from the arm's claim and is released when the session
    ends.
  - `scene_apply(request_id, revision, state)` is the plant node's optimistic `scene_command` (a stale revision is refused). It
    runs on the caller's thread, not on the controller loop.
  - SDK: `ArmClient.scene_claim()`, `ArmClient.scene_apply()`.
* observe/1 samples carry the world's whole `scene_state` echo.
* `scene_observer` changes:
  - it republishes `scene_state` every 2 s and at once on a new revision, plus `constraint_force` at 20 Hz;
  - it applies a graph's `scene_command` through scene_action/1 and answers on `scene_result`, as the plant node did.
* Body-list arms: `arm.body_list` (a `BodyListDynamics` description) is the executor's construction model, with no URDF and no
  payload. `BodyListDynamics.joint_limits` comes from the description's optional `joint_limits`; where none is stated it is
  +/-inf.
* **The world clock** (`SharedWorld.clock`, `SimLink.clock`):
  - A scene change pauses the world while it recompiles: about 0.35 s for a storage-release toggle in isolation, measured. Every
    actor's servo waits on the lock meanwhile.
  - The supervisor's controller now runs on the link's clock when the link has one. For a hosted world, that clock is wall time
    minus every pause, so it stands still with the world. A sample's arrival AGE (wall) is carried onto that clock, never the
    wall stamp, and SimLink stamps a sample when it publishes it (after the step). SimLink's state freshness excuses a pause
    in progress, for at most `PAUSE_EXCUSE_MAX_S` (5 s): a longer recompile is a stalled plant.
  - Without this, the FR3 policy's 0.25 s state timeout faulted the lift leg ("motor state stale") at the storage release, the
    first supervised e2e run. The plant-clocked Dora controller never saw that pause.
  - The RT link has no clock, so the hardware path keeps wall time and is unchanged.

Project (sim-add):
* `dataflows/sim_workcell_add.yml` now has real-add's shape:
  - the FR3 runs through `supervised_arm` and `franka_hand` (the hosted sim Hand);
  - the base runs through `base_client`;
  - `scene_observer` replaces the plant node.
* `arm_controller`, `sim_bridge` and `mujoco_interface` are retired from the graph. `nodes/sim_bridge.py` and
  `assembly/runtime/sim_bridge.py` are deleted (no other user). `nodes/base_driver.py` and `BaseDriver.main` are deleted too.
* `tools/run.py sim-add`:
  - starts the FR3's supervisor over the sim plant, with the base's supervisor as a co-host (`stage_base_sim_config`);
  - the base co-host gets the world's `base` actor joints `[base__joint_1, base__fixed_1, row_01__module__P_to_M]`;
  - its construction model is the workcell's body list (`prepare_model`);
  - its kp 60 / kd 2 / 27 N.m and 0.6 rad/s / 1.5 rad/s² for every joint come from `configs/tasks/base_motion.yaml`, the old
    in-process sim profile.
* `configs/simulation/franka_workcell_add.yaml` changes:
  - the FR3 execution policy is on (the plant-time `execution_policy: null` is gone, with its speed copy);
  - `sim_actor: franka`;
  - `hosted_world.yaml` (observe/1, the slice deadman) is shared with `franka_bench.yaml`.
* `SupervisedBaseDriver` works in sim. With `joint_map=None` it uses the world's names and drives the module joint too
  (base_only off). As the in-process sim driver did, the sim base holds whatever the FR3's authority: the client ARMs it at start
  and ignores the FR3's ARM/DISARM, unless the profile sets `require_authority`.
* The base settle tolerances moved to `configs/robots/base/settle.yaml`, shared by `real_base.yaml` and `sim_base.yaml`. The
  values are the same.
* `check_pipeline_sim` changes:
  - its node whitelist is now per id;
  - it refuses any plant but `sim` and any remote `rt.host` for every supervisor;
  - the trace topics moved to the supervised nodes.

Checks:
* new `check_scene_cohost`, all passing:
  - two links in one world track their own targets;
  - a scene change pauses the world clock and is not staleness;
  - the co-host never steps or closes the world;
  - over the service, two sockets are each claimed, ARMed and moved to COMPLETED;
  - the scene lease works, a stale revision is refused, and an apply is accepted.
* `check_sim_supervisor`, `check_load_model`, the scene_observer self-check: passing.
* Project: `check_pipeline_sim sim-add` PASSED, all 12 gates in 75.9 s, against 76.8 s for the Dora baseline the same day.
* Project, also passing: `check_add_contracts`, `check_franka_ownership` (config duplicates), `check_base_supervisor`,
  `check_supervised_real`, `check_base_profile`, the base_driver self-check, `ruff`.

Review round 1 (opus): BLOCK, fixed:
* MUST: the controller's state age mixed clocks. Wall-time arrival stamps were written into a controller on the paused
  clock, so every age ran short by the total of all past pauses, and a dead plant hid behind it (the 0.25 s policy timeout
  fell back to the link's 0.5 s). Also, the first sample after a pause carried a pre-pause stamp. The runtime now carries
  the age across (`c._clock() - (monotonic - rx)`), and SimLink stamps on publish. The check pauses the world, kills the servo
  and asserts the stale error within the policy timeout.
* SHOULDs:
  - A co-host took `now` before the world lock, wrote a pre-pause time after the host had used the stall credit, and its slice
    was zeroed. `now` is now taken inside the lock; the check runs six pauses with no slice zeroed.
  - A co-host could not see a stopped world. It now faults (FAULT_PLANT) when the host has not stepped within the world's
    idle timeout of world time; the check makes the host's step fail.
  - A malformed scene request (KeyError/TypeError) reached the SDK as an internal error and killed scene_observer. It is now
    `Refused`, and the observer answers `ok=False` for any non-timeout supervisor error, as the plant node did.
  - The co-hosted base hung limp from supervisor start until its first ARM (the disarmed release word overrode the world's
    spawn hold). A co-host now writes its slice only once its supervisor first drives it; the check asserts the spawn hold.
* NITs taken:
  - the pause excuse is capped;
  - `supports_cartesian` is cached at open, so a recompile cannot race it;
  - a co-host refuses Cartesian commands;
  - the host closes the world under its lock;
  - a co-host whose start fails is closed before the world;
  - the stale sim_bridge pointer is fixed;
  - the sim base states MuJoCo's joint ranges (`prepare_model(joint_limits=True)`);
  - records corrected (below, the migrations doc, the track list).

OPEN:
* `BaseDriver`'s in-process executor and `base_command` path have no node any more. They stay as the self-checked decision core
  of `SupervisedBaseDriver`, and are retired once those checks move onto the client.
* The library's `arm_controller` and `mujoco_interface` nodes are still used by `sim_view_arm.yml`; that is the library-leftovers
  slice.
