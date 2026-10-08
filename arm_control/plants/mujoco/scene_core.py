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

import os
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


class ScenePlant:
    """One actor of a scene world, as the plant behind a ``SimLink`` (scene hosting, step 2b; 2026-10-08-scene-hosting-design.md).

    The supervisor hosts the world: ``SimLink`` keeps its authority, deadman and observe/1 rules unchanged, and this adapter gives
    it the supervised actor's slice of a ``SceneCore`` -- the same world code the Dora plant node runs. Each step writes the
    actor's slice (``SimLink`` writes every tick: a command, a hold or a release word, so the slice's own idle deadman never
    fires for it) and steps the whole world once. The other actors keep the world's spawn hold until their own writers exist
    (step 3, D1); the gripper keeps its spawn opening. Driven by ``SimLink``'s one servo thread, as ``SceneCore`` requires.
    """

    def __init__(self, world, actor: str, width: int, *, period: float, idle_timeout: float) -> None:
        backend, slices, _names, n = world
        if actor not in slices:
            raise ValueError(f"sim_actor {actor!r} is not an actor of this world (actors: {sorted(slices)})")
        if slices[actor]["n"] != width:
            raise ValueError(f"sim_actor {actor!r} has {slices[actor]['n']} joints; the arm config names {width}")
        start = slices[actor]["start"]
        self.actor, self.backend, self._slices = actor, backend, dict(slices)
        self._span = slice(start, start + width)
        self._core = SceneCore(backend, slices, n, period=period, idle_timeout=idle_timeout)

    @classmethod
    def from_config(cls, cfg, width: int, *, period: float) -> "ScenePlant":
        actor = cfg.get("sim_actor")
        if not actor:
            raise ValueError("a scene plant needs sim_actor (the world's actor this supervisor drives): nothing is assumed")
        idle = cfg.get("idle_timeout_sec")
        if idle is None:
            raise ValueError("a scene plant needs idle_timeout_sec (the other actors' deadman): nothing is assumed")
        return cls(build_world(cfg, period=period), str(actor), width, period=period, idle_timeout=float(idle))

    def load(self) -> None:
        self.backend.load()

    def close(self) -> None:
        self.backend.close()

    def _mine(self, state: dict) -> dict:
        return {"position": np.asarray(state["position"], float)[self._span].copy(),
                "velocity": np.asarray(state["velocity"], float)[self._span].copy()}

    def step(self, command: dict) -> dict:
        now = time.monotonic()
        self._core.set_slice(self.actor, dict(command, cartesian=None), now)
        return self._mine(self._core.tick(now))

    def motor_state(self) -> dict:
        return self._mine(self.backend.motor_state())

    def observe(self) -> dict:
        world = self.backend.observe()
        names = list(self.backend.joint_names)
        world["actors"] = {a: names[s["start"]: s["start"] + s["n"]] for a, s in self._slices.items()}
        return world
