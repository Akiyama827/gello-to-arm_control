"""The simulated world's per-tick logic, with no transport: one MuJoCo world, its command vector, and the rules that keep it safe.

Extracted from the Dora ``mujoco_interface`` node (2026-10-08-scene-hosting-design.md, step 2a) so that two front ends drive the
SAME world code: the Dora node (messages in, messages out) and the supervisor's scene link (admission in, ``observe/1`` out). The
behaviour is the node's, moved, not rewritten:

* per-actor command SLICES of one command vector (``arm_slices``), each with its own deadman: a slice whose commander has been
  silent longer than ``idle_timeout`` has its gains zeroed (said once per gap), counting only time this host was listening --
  time spent blocked in its own work (a model recompile) is credited back, because the physics was frozen too;
* the single-model world (no slices): an ARM gate from the commander (disarmed = zero effort) and a Soft (pose-hold) mode that
  expires with its commander;
* gripper commands and optimistic scene changes (graft, weld, release), refused on a stale revision;
* stepping, and the sim-time-behind-wall accounting the node reports.

What to publish, and when, stays with the front end.
"""
from __future__ import annotations

from functools import cached_property
import os
import threading
import time

import numpy as np

from arm_control.messages import scene_state_from_payload, scene_state_payload
from arm_control.node_utils import entry_point, scene_injections
from arm_control.plants.mujoco.composer import build_scene_backend, build_workcell_backend
from arm_control.plants.mujoco.single import build_single_model_backend, open_jaws


def describes_scene(cfg) -> bool:
    """Whether a config's plant is a scene (a workcell, ``WORKCELL_SCENE``, or a composed plant, ``scene.arm``) and not one model.
    ``build_world`` and the supervisor's sim plant branch on this one rule."""
    return bool(os.environ.get("WORKCELL_SCENE") or (cfg.get("scene") or {}).get("arm"))


def build_world(cfg, *, period: float):
    """The world a robot config describes: a workcell scene (``WORKCELL_SCENE``), a composed plant scene (``scene.arm``), or one
    model. Returns ``(backend, arm_slices, joint_names, n)``; the backend is loaded and its jaws opened."""
    # Injected by the GRAPH, resolved for BOTH scene paths. This used to sit
    # inside the `if scene_path:` branch below, so the composed-scene path
    # could inject nothing -- and once a chain_factory became mandatory for a
    # scene with objects (2026-09-03) that path raised at startup on every
    # run. Every composed-scene scenario was dead from that day.
    injections = scene_injections()

    scene_path = os.environ.get("WORKCELL_SCENE")
    scene_cfg = cfg.get("scene")
    if scene_path:
        loader = entry_point("WORKCELL_LOADER")
        # What an assembly chain IS belongs to the project with modules; this
        # plant grafts bodies and asks whether they landed seated, but defines
        # neither the topology nor the mate. An object-free scene needs neither.
        backend, arm_slices, joint_names, n = build_workcell_backend(
            scene_path,
            control_period=period,
            timestep=float(cfg.get("sim_timestep", 0.001)),
            launch_viewer=bool(cfg.get("sim_launch_viewer", False)),
            enable_self_collision=bool(cfg.get("sim_self_collision", False)),
            loader=loader,
            chain_factory=injections["chain_factory"],
        )
        print(
            f"[mujoco_interface] generic workcell: {n} actuators, slices={arm_slices}",
            flush=True,
        )
    elif describes_scene(cfg):
        # `scene.arm` is what makes a scene block a COMPOSED PLANT scene, and
        # the truthiness of `scene:` alone is not: a real-arm config carries a
        # partial `scene:` describing the docking base as a static fixture for
        # the visualizer and the planner's obstacle set, while its plant is one
        # model. Branching on `scene_cfg` alone sent that config down this path
        # and died with `KeyError: 'arm'` inside _model_spec -- which is how
        # `view.py sim franka motion` was broken.
        backend, arm_slices, joint_names, n = build_scene_backend(
            scene_cfg,
            control_period=period,
            launch_viewer=bool(cfg.get("sim_launch_viewer", False)),
            enable_self_collision=bool(cfg.get("sim_self_collision", False)),
            **injections,
        )
        if cfg.get("sim_gravcomp", False):
            # The robot's twin models a control box that holds the arm up whether or not a host streams (the FR3 sim profile),
            # as the single-model and workcell plants already do: the ARM's bodies, by its prefix, never the base or modules.
            # Set before the first compile -- gravcomp is baked into the spec. (Armature is NOT applied on this path: the
            # backend's ARM_ARMATURE loop is workcell-only and sim_armature is single-model only.)
            prefix = str((scene_cfg.get("arm") or {}).get("prefix") or "")
            if not prefix:
                raise ValueError("sim_gravcomp on a composed scene needs scene.arm.prefix (which bodies are the arm)")
            backend.gravcomp_prefixes = (prefix,)
        print(
            f"[mujoco_interface] composed scene: {n} actuators, slices={arm_slices}",
            flush=True,
        )
    else:
        if scene_cfg:
            # Say so out loud. A partial scene block is legitimate (see above)
            # but "the plant ignored part of your config" must never be silent.
            print(
                f"[mujoco_interface] scene block present but names no `arm` "
                f"({sorted(scene_cfg)}) — running SINGLE-MODEL. That block is "
                f"for other consumers (visualizer, planner obstacles); a "
                f"composed plant scene needs scene.arm.",
                flush=True,
            )
        # Single-model mode (view/motion graphs): one MJCF/URDF, plain
        # ``motor_command`` in / ``motor_state`` out, no welds, no ground.
        joint_names = list(cfg.joint_names)
        n = len(joint_names) if joint_names else cfg.num_motors
        arm_slices = {}
        backend = build_single_model_backend(cfg, control_period=period)
        print(f"[mujoco_interface] single model: {backend.model_path}", flush=True)
    backend.load()

    # THE PLANT DECLARES THE STARTING STATE. On the bench the arm is wherever
    # it is and the host reads it; in the twin the plant spawns it somewhere
    # defined. MuJoCo's implicit zero is not a declaration, it is an accident --
    # and for the FR3's Hand zero means SHUT, so every graph started with the
    # jaws closed around whatever they were parked over. Measured: at the bench
    # grasp pose the shut jaws bury 29.85 mm into the module, the contact
    # solver pushes back, and the approach settles ~0.039 rad short of target
    # forever (the planner never sees it -- it checks collisions at the grasp
    # profile's finger width, not the plant's actual one). Open to the arm's
    # own configured width, exactly as run_workcell_add_demo.py does.
    # NB the count comes from the PLANT, not gripper_joints(cfg): that list is
    # deliberately EMPTY for an arm whose gripper is its own device (the FR3's
    # Hand), while the composed scene still has the finger joints.
    open_jaws(backend, cfg)
    return backend, arm_slices, joint_names, n


class SceneCore:
    """One world's command vector and safety rules (see the module docstring). Not thread-safe: one thread drives it."""

    def __init__(self, backend, arm_slices: dict, n: int, *, period: float, idle_timeout: float, log=print) -> None:
        self.backend, self.arm_slices, self.n = backend, dict(arm_slices), int(n)
        self.period, self.idle_timeout, self._log = float(period), float(idle_timeout), log
        self.command = {
            # The SPAWN POSE, read back from the plant — not zeros. The comment
            # below has always said "hold at the spawn pose"; the code held at
            # q=0, which for any arm whose spawn pose is not the origin is not a
            # hold at all but a full-authority move to the origin. Measured on the
            # FR3 scene: the arm slams off its ready pose at up to 30 rad/s during
            # the pre-arm window, and whether it has wrecked the scene by the time
            # the operator arms is a race against sim_bridge's park taking over.
            # That is the "start pose is in collision per the plan world" the
            # planner intermittently refused on. Pre-existing, law-independent
            # (the old Python PD does it too) — found while A/B-ing the servo law.
            "position": backend.motor_state()["position"].copy(),
            "velocity": np.zeros(self.n),
            "torque": np.zeros(self.n),
            # Gentle hold at the spawn pose until real commands arrive: with zero
            # gains the arm free-falls during graph startup and settles into a
            # tangle downstream planners then refuse to start from.
            "kp": np.zeros(self.n) + 60.0,
            "kd": np.zeros(self.n) + 2.0,
            # Optional Cartesian-impedance target + task-frame K_c/D_c. None until
            # a command carries one; None = joint-space PD only, i.e. today.
            "cartesian": None,
        }
        self.last_cmd_time = {arm: 0.0 for arm in self.arm_slices}
        self._idle_warned = {arm: False for arm in self.arm_slices}
        self._last_single_command = 0.0
        self.single_armed: bool | None = None  # None: graphs that wire no authority
        self._stall = 0.0                      # seconds this host spent NOT listening, since the last step
        self.sim_steps = 0
        self._wall_t0 = 0.0
        self._slow_warned_at = 0.0
        self._slow_debt = 0.0

    # -- commands ----------------------------------------------------------------------------------------------
    def set_single_armed(self, armed: bool) -> None:
        """The single-model ARM gate: disarmed means no effort and no Soft mode."""
        self.single_armed = bool(armed)
        if not self.single_armed:
            self.command["pose_hold"] = None
            self.command["cartesian"] = None
            for key in ("kp", "kd", "torque", "velocity"):
                self.command[key] = np.zeros(self.n)

    def set_single(self, command: dict, now: float) -> bool:
        """A whole-vector command (no slices). Ignored while the ARM gate says disarmed; returns whether it was taken."""
        if self.single_armed is False:
            return False
        self.command = command
        self._last_single_command = now
        return True

    def set_slice(self, arm: str, sub: dict, now: float) -> bool:
        """One actor's slice of the command vector; returns False for an actor this world has no slice for."""
        info = self.arm_slices.get(arm)
        if info is None:
            return False
        s, m = info["start"], info["n"]
        self.last_cmd_time[arm] = now
        for key in ("position", "velocity", "torque", "kp", "kd"):
            self.command[key][s: s + m] = sub[key]
        # Cartesian impedance is an EE-level term: the arm that owns the configured EE body owns it (last writer wins).
        self.command["cartesian"] = sub["cartesian"]
        return True

    def apply_gripper(self, position, *, force_n: float) -> None:
        self.backend.apply_gripper_command(position, force_n=float(force_n))

    def apply_scene(self, request: dict) -> dict:
        """An optimistic scene change. Returns the scene_state payload after it; raises ValueError (stale revision) or the
        decoder's KeyError/TypeError for a malformed request. Time blocked in the recompile is credited to every deadman."""
        if not self.backend.scene_state:
            raise ValueError("this world has no scene state")
        if request["revision"] <= self.backend.scene_state.revision:
            raise ValueError("stale scene revision")
        next_state = scene_state_from_payload(self.backend.scene_state, request["state"], request["revision"])
        started = time.monotonic()
        self.backend.apply_scene_state(next_state)
        self.credit_stall(time.monotonic() - started)
        return scene_state_payload(self.backend.scene_state)

    def credit_stall(self, seconds: float) -> None:
        """Forgive time this host spent blocked inside its own work. The deadman's premise ("wall time passed and no command came,
        so the commander is dead") is false while the host is blocked: nobody was listening, and no sim time passed either. The
        clock restarts at the resume, so a commander that really died is still caught one timeout later."""
        if seconds > 0.0:
            self._stall += float(seconds)

    def zero_slice(self, arm: str) -> None:
        info = self.arm_slices[arm]
        s, m = info["start"], info["n"]
        for key in ("torque", "kp", "kd"):
            self.command[key][s: s + m] = 0.0
        self.command["cartesian"] = None
        self.command["pose_hold"] = None

    # -- the tick ----------------------------------------------------------------------------------------------
    def tick(self, now: float) -> dict:
        """Apply the deadmans, step the world once, account sim time against wall time. Returns the backend's state."""
        if self._stall > 0.0:
            # Push every idle deadline forward by the blocked time: "has the commander been quiet while we were LISTENING?"
            for arm in self.last_cmd_time:
                if self.last_cmd_time[arm] > 0.0:
                    self.last_cmd_time[arm] += self._stall
            self._stall = 0.0
        command = self.command
        for arm, info in self.arm_slices.items():
            if self.last_cmd_time[arm] > 0.0 and now - self.last_cmd_time[arm] > self.idle_timeout:
                span = slice(info["start"], info["start"] + info["n"])
                active = any(np.any(command[key][span]) for key in ("kp", "kd", "torque"))
                active = active or command.get("pose_hold") is not None or command.get("cartesian") is not None
                if not self._idle_warned[arm] and active:
                    # Said once per gap: a silent slice going limp looks downstream like a servo that cannot track.
                    self._idle_warned[arm] = True
                    self._log(f"[mujoco_interface] {arm}: no command for {now - self.last_cmd_time[arm]:.2f} s — zeroing gains "
                              f"(idle_timeout_sec={self.idle_timeout})", flush=True)
                self.zero_slice(arm)
            elif self.last_cmd_time[arm] > 0.0:
                self._idle_warned[arm] = False
        if (not self.arm_slices and command.get("pose_hold") is not None
                and now - self._last_single_command > self.idle_timeout):
            # A stopped commander cannot leave an unrefreshed Soft mode.
            command["pose_hold"] = None
            command["position"] = self.backend.data.qpos[self.backend._qadr].copy()
            command["velocity"] = np.zeros(self.n)
        state = self.backend.step(command)
        # Warn on CUMULATIVE drift, not single-step spikes (viewer syncs take ~14 ms 15x/s with ample physics headroom).
        self.sim_steps += 1
        if self._wall_t0 == 0.0:
            self._wall_t0 = now
        debt = (now - self._wall_t0) - self.sim_steps * self.period
        if debt > 0.5 and debt - self._slow_debt > 0.25 and now - self._slow_warned_at > 5.0:
            self._slow_warned_at, self._slow_debt = now, debt
            self._log(f"[mujoco_interface] sim time {debt:.1f} s behind wall clock and falling further (plant-clock pacing "
                      "keeps execution safe; motion plays in slow motion)", flush=True)
        return state

    def position_cmd(self, state: dict) -> np.ndarray:
        """The position command the backend echoes, padded to the vector width."""
        pos_cmd = np.zeros(self.n, dtype=np.float64)
        echoed = np.asarray(state.get("position_cmd", []), dtype=np.float64)
        pos_cmd[: min(self.n, echoed.size)] = echoed[: min(self.n, echoed.size)]
        return pos_cmd


#: the longest a scene change may pause a hosted world and still be excused (staleness, the controller clock); measured 0.35 s
#: for a storage-release recompile. ponytail: one bound for every world; per-world if a heavier scene ever needs more.
PAUSE_EXCUSE_MAX_S = 5.0


class SharedWorld:
    """One composed world and its ``SceneCore``, shared by every supervised actor this process hosts (scene-hosting D1(a)/D2(a),
    operator 2026-10-09: the FR3 and the base in one sim world, each on its own supervisor). The HOST actor's servo thread steps
    it; the others write their slice and read the state the last step left. Every write, step and read is under ``lock``."""

    def __init__(self, world, *, period: float, idle_timeout: float) -> None:
        backend, slices, names, n = world
        if not (np.isfinite(idle_timeout) and idle_timeout > 0):
            raise ValueError("idle_timeout_sec must be finite and positive")
        self.backend, self.slices, self.names = backend, dict(slices), list(names)
        self.core = SceneCore(backend, slices, n, period=period, idle_timeout=idle_timeout)
        self.lock = threading.Lock()
        self.last: dict | None = None  # the backend state after the last step
        self.host_steps = 0  # how many times the host has stepped the world
        # (seconds the world has been paused for scene changes, start of the change in progress or None): ONE attribute, so a
        # reader on another thread never sees a finished pause counted twice
        self._pause = (0.0, None)

    def apply_scene(self, request: dict) -> dict:
        """scene_action/1: an optimistic scene change, between steps (``SceneCore.apply_scene``). The world is paused while
        it is rebuilt (every actor's servo waits on the lock): that time is not plant time (``clock``)."""
        with self.lock:
            paused, _ = self._pause
            started = time.monotonic()
            self._pause = (paused, started)
            try:
                return self.core.apply_scene(request)
            finally:  # what clock() excused, no more: a pause past the cap must not turn the clock back when it ends
                self._pause = (paused + min(time.monotonic() - started, PAUSE_EXCUSE_MAX_S), None)

    def stalled(self, now: float) -> float:
        """Seconds the world has been paused by the scene change in progress (0 when none is), at most PAUSE_EXCUSE_MAX_S:
        a recompile that hangs longer is a stalled plant, and staleness and the controller clock resume."""
        _, started = self._pause
        return 0.0 if started is None else min(max(0.0, now - started), PAUSE_EXCUSE_MAX_S)

    def clock(self) -> float:
        """The world's time on the wall clock's scale: monotonic time less every pause for a scene change, so it stands still
        while the world does. A supervisor's controller deadlines run on it (a 0.35 s recompile is no stale state in a world
        that did not move), as SceneCore's slice deadmen are credited the same stall."""
        now = time.monotonic()
        paused, _ = self._pause
        return now - paused - self.stalled(now)


class ScenePlant:
    """One actor of a scene world, as the plant behind a ``SimLink`` (scene hosting, step 2b; 2026-10-08-scene-hosting-design.md).

    The supervisor hosts the world: ``SimLink`` keeps its authority, deadman and observe/1 rules unchanged, and this adapter gives
    it the supervised actor's slice of a ``SceneCore`` -- the same world code the Dora plant node runs. Each step writes the
    actor's slice (``SimLink`` writes every tick: a command, a hold or a release word, so the slice's own idle deadman never
    fires for it); the HOST actor's step also steps the whole world once, a co-hosted actor's (``cohost``) reads the state the
    host's last step left (one host tick of latency). Actors with no supervisor keep the world's spawn hold. The Hand
    (``sim_hand``) is served by ``SimLink`` for every sim plant, from the world's ``backend``. Every touch of the world is under
    the ``SharedWorld`` lock.
    """

    def __init__(self, world, actor: str, joints: list, *, period: float, idle_timeout: float, host: bool = True) -> None:
        """``world``: a ``build_world`` tuple (this actor hosts it) or a ``SharedWorld`` another actor hosts (``host=False``)."""
        shared = world if isinstance(world, SharedWorld) else SharedWorld(world, period=period, idle_timeout=idle_timeout)
        backend, slices, names = shared.backend, shared.slices, shared.names
        if actor not in slices:
            raise ValueError(f"sim_actor {actor!r} is not an actor of this world (actors: {sorted(slices)})")
        width = len(joints)
        if slices[actor]["n"] != width:
            raise ValueError(f"sim_actor {actor!r} has {slices[actor]['n']} joints; the arm config names {width}")
        start = slices[actor]["start"]
        # The world prefixes an actor's joints (``asm_fr3_joint1``); the order must still be the arm config's, or every command
        # would drive the wrong joint.
        world_joints = list(names)[start: start + width]
        if not all(w.endswith(j) for w, j in zip(world_joints, joints)):
            raise ValueError(f"sim_actor {actor!r} joints {world_joints} are not the arm config's {list(joints)} in order")
        self.actor, self.backend, self._slices = actor, backend, dict(slices)
        self._span = slice(start, start + width)
        self.shared, self.host = shared, bool(host)
        self._core = shared.core
        self._driven = False  # a co-host: its supervisor has written a non-release command
        # a co-host: the world's idle timeout in this actor's own steps (25 at 250 Hz and 0.1 s)
        self._max_unstepped = max(1, int(np.ceil(idle_timeout / period)))
        self._host_seen, self._unstepped = 0, 0

    @classmethod
    def from_config(cls, cfg, joints: list, *, period: float) -> "ScenePlant":
        actor = cfg.get("sim_actor")
        if not actor:
            raise ValueError("a scene plant needs sim_actor (the world's actor this supervisor drives): nothing is assumed")
        idle = cfg.get("idle_timeout_sec")
        if idle is None:
            raise ValueError("a scene plant needs idle_timeout_sec (the other actors' deadman): nothing is assumed")
        return cls(build_world(cfg, period=period), str(actor), list(joints), period=period, idle_timeout=float(idle))

    @classmethod
    def cohost(cls, shared: SharedWorld, cfg, joints: list, *, period: float) -> "ScenePlant":
        """Another actor (``sim_actor``) of a world this process already hosts: it writes its slice, the host steps the world."""
        actor = cfg.get("sim_actor")
        if not actor:
            raise ValueError("a co-hosted actor needs sim_actor (the world's actor its supervisor drives): nothing is assumed")
        return cls(shared, str(actor), list(joints), period=period, idle_timeout=shared.core.idle_timeout, host=False)

    @cached_property  # first read at SimLink.open, the world loaded: the EE chain never changes, and a recompile must not race it
    def supports_cartesian(self) -> bool:
        """The world's EE chain is this actor's: a Cartesian spec from it drives its own arm, never another actor's. A co-host
        takes joint commands only (refused at admission, not at the step)."""
        if not self.host:
            return False
        mine = set(list(self.backend.joint_names)[self._span])
        return bool(self.backend.supports_cartesian and set(self.backend.ee_chain_joints) <= mine)

    def load(self) -> None:
        if self.host:
            self.backend.load()

    def close(self) -> None:
        if self.host:
            with self.shared.lock:  # never under a scene change still running at shutdown
                self.backend.close()

    def _mine(self, state: dict) -> dict:
        return {"position": np.asarray(state["position"], float)[self._span].copy(),
                "velocity": np.asarray(state["velocity"], float)[self._span].copy()}

    def step(self, command: dict) -> dict:
        with self.shared.lock:
            now = time.monotonic()  # inside the lock: a step that waited out a pause must not write a pre-pause time
            if self.host:
                self._core.set_slice(self.actor, dict(command, cartesian=command.get("cartesian")), now)
                self.shared.last = self._core.tick(now)
                self.shared.host_steps += 1
                return self._mine(self.shared.last)
            if self.shared.host_steps == 0:  # the host has not stepped yet (start-up): nothing to fail, nothing to write
                return self._mine(self.backend.motor_state())
            if command.get("cartesian") is not None:
                raise ValueError("a co-hosted actor takes joint commands only (the world's Cartesian law is the host's)")
            # A co-host whose world has stopped stepping (the host failed or is gone) is a failed plant, not a fresh one. Counted
            # in this actor's OWN steps, not in time: a stall of the whole process (GC, descheduling) stops both servo threads
            # alike and must not fault anyone; this many co-host steps with no host step proves the host alone stopped.
            if self.shared.host_steps != self._host_seen:
                self._host_seen, self._unstepped = self.shared.host_steps, 0
            else:
                self._unstepped += 1
                if self._unstepped > self._max_unstepped:
                    raise RuntimeError(f"the hosting actor has not stepped the world in {self._unstepped} of this actor's steps")
            # Until its supervisor first drives it, a co-hosted actor keeps the world's spawn hold: SimLink's disarmed release
            # word would otherwise drop it limp from start-up to the first ARM (the base has no gravity compensation).
            released = not any(np.any(command[k]) for k in ("kp", "kd", "torque"))
            if not (released and not self._driven):
                self._driven = True
                self._core.set_slice(self.actor, dict(command, cartesian=None), now)
            return self._mine(self.shared.last)

    def motor_state(self) -> dict:
        with self.shared.lock:
            return self._mine(self.backend.motor_state())

    def apply_scene(self, request: dict) -> dict:
        return self.shared.apply_scene(request)

    def observe(self) -> dict:
        with self.shared.lock:
            world = self.backend.observe()
        names = list(self.backend.joint_names)
        world["actors"] = {a: names[s["start"]: s["start"] + s["n"]] for a, s in self._slices.items()}
        return world
