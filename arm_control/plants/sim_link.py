"""A simulated plant behind the supervisor's link surface (CTRL-053, simulation as an alternate plant).

``RobotRuntime`` talks to its plant through a *link* (``RtBackend`` for an ``arm_rt_server``). ``SimLink`` is the other
implementation: the same surface over an in-process physics plant, so the supervisor, the SDK and admission are one code path
whether the robot is a real arm or a simulation. It is the ros2_control picture: the supervisor is the controller manager,
the plant (hardware or simulation) is a hardware plugin.

What it models of the RT server, and nothing more:

* authority: ARM starts a new ARM epoch and holds the measured pose until commands flow (``FLAG_HOLDING``); DISARM bumps the
  stop epoch and clears a fault; ARM while a fault is latched is refused;
* commands: a command is accepted only while armed and not faulted, and ``last_cmd_seq`` is the sequence of the last accepted
  one, which is what the APPLIED rule reads;
* the deadman: after ``hold_ms`` without a command the plant holds the measured pose (with the last accepted command's gains, the
  configured hold gains before any), after ``fault_ms`` it latches ``FAULT_CMD_LOST`` and keeps holding until DISARM;
* a DISARM releases the plant (zero-gain word), and so does a plant failure (``FAULT_PLANT``).

What it does not: no native reference (v7) path, no group record, no pose-hold, no clock domain (a simulation has no RT host
clock), and no RT-side torque clamp or slew beyond what the plant itself applies. Every servo parameter comes from the config's
``sim_servo`` block with no defaults: ``state_hz``, ``hold_ms``, ``fault_ms``, ``hold_kp``, ``hold_kd``. A simulation
that is not told them does not invent them.
"""
from __future__ import annotations

from collections import deque
from itertools import islice
import secrets
import threading
import time

import numpy as np

from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.client import RtConfig, RtLinkError

_SERVO_KEYS = ("state_hz", "hold_ms", "fault_ms", "hold_kp", "hold_kd")


# observe/1 bounds. The ring is memory and a read's scan; reads are bounded by the service (OBSERVE_LIMIT_MAX) because building
# and serializing a batch runs in this process and competes with the tick for the interpreter (review: 4 tight readers of 1024
# samples slowed the plant to 0.44x real time; at 64 it stayed at 1.0x).
OBSERVE_RING_MAX_S = 10.0
OBSERVE_LIMIT_MAX = 64


class SimLink:
    def __init__(self, config: RtConfig, joint_names, plant, *, servo: dict, observe: dict | None = None) -> None:
        missing = [k for k in _SERVO_KEYS if k not in servo]
        if missing:
            raise ValueError(f"sim_servo needs {', '.join(_SERVO_KEYS)} (missing: {', '.join(missing)}): nothing is assumed")
        if not rtp.valid_id(config.instance_id):
            raise ValueError(f"instance_id {config.instance_id!r} is not a valid instance id")
        self.config = config
        self.joint_names = list(joint_names)
        self.n = len(self.joint_names)
        self._plant = plant
        try:
            hz, self.hold_ms, fault_ms, kp, kd = (float(servo[k]) for k in _SERVO_KEYS)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sim_servo values must be numbers: {exc}") from exc
        # The RT server's own launch checks: 0 < hold < fault strictly, a state rate it can run, finite non-negative gains.
        if not np.isfinite([hz, self.hold_ms, fault_ms, kp, kd]).all():
            raise ValueError("sim_servo values must be finite")
        if not 1 <= hz <= 1000:
            raise ValueError("sim_servo.state_hz must be 1..1000 (the RT server's range)")
        if not 0 < self.hold_ms < fault_ms:
            raise ValueError("sim_servo: 0 < hold_ms < fault_ms is required")
        if kp < 0 or kd < 0:
            raise ValueError("sim_servo: hold_kp and hold_kd must not be negative")
        self._period = 1.0 / hz
        self._fault_s = fault_ms / 1e3
        self._hold_kp = np.full(self.n, kp)
        self._hold_kd = np.full(self.n, kd)
        self.backend_name = "sim"
        self.rt_setup = "sim"
        self.clock_domain = None             # a simulation has no RT host clock to attest
        self.group_capable = False
        self.supports_pose_hold = False
        self.identity: rtp.Identity | None = None
        self.session_id = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: rtp.State | None = None
        self._state_rx_t = 0.0
        self._history: deque = deque(maxlen=32)
        self._cmd = None                     # the last accepted command dict
        self._cmd_t = 0.0
        self._cmd_seq = 0                    # last sequence handed out
        self._last_cmd_seq = 0               # last sequence accepted (what the state reports)
        self._armed = False
        self._arm_epoch = 0                  # the current ARM epoch (0 while disarmed)
        self._epochs = 0                     # ARM epochs handed out, ever: monotonic, never reused
        self._stop_epoch = 0
        self._arm_t = 0.0
        self._hold_q: np.ndarray | None = None
        self._in_hold = False
        self._plant_ok = True                # per ARM epoch, as in the RT: false after a failed step until the next ARM
        self._tick_snap = None               # the authority snapshot of the tick in flight (for its failure sample)
        self._pub_stop_epoch = 0             # the stop epoch the published samples show: advanced only by a RELEASED step
        self._cmd_gains = None               # the gains of the last accepted command this epoch (a hold keeps them, as the RT does)
        self._fault_code = 0
        self._latched_fault = ""
        self._state_seq = 0
        # observe/1 (2026-10-08-scene-hosting-design.md): one observation every ``_obs_every`` ticks, from the tick that stepped
        # the plant, into a bounded ring the tick never waits on. Off unless the config states ``sim_observe``.
        self.observing = observe is not None
        self.observe_rate_hz, self._obs_every, self._obs_tick = 0.0, 0, 0
        self._obs: deque = deque(maxlen=1)
        self._obs_seq = self._obs_errors = 0
        if observe is not None:
            try:
                rate, ring_s = float(observe["rate_hz"]), float(observe["ring_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"sim_observe needs numeric rate_hz and ring_s: {exc}") from exc
            if not (np.isfinite([rate, ring_s]).all() and 0 < rate <= hz and 0 < ring_s <= OBSERVE_RING_MAX_S):
                raise ValueError(f"sim_observe: 0 < rate_hz <= sim_servo.state_hz ({hz:g}) and "
                                 f"0 < ring_s <= {OBSERVE_RING_MAX_S:g} are required")
            self._obs_every = int(np.ceil(hz / rate))
            self.observe_rate_hz = hz / self._obs_every
            self._obs = deque(maxlen=max(1, int(np.ceil(ring_s * self.observe_rate_hz))))

    @classmethod
    def from_config(cls, cfg) -> "SimLink":
        """The simulated plant a robot config describes: its ``sim_*`` plant keys (``plants.mujoco.single``), its ``rt`` block for
        the instance id and freshness bounds, and the required ``sim_servo`` block."""
        from arm_control.config import arm_joints
        from arm_control.plants.mujoco.scene_core import ScenePlant, describes_scene
        from arm_control.plants.mujoco.single import build_single_model_backend, open_jaws

        servo = dict(cfg.get("sim_servo") or {})
        if "state_hz" not in servo:
            raise ValueError("config has no sim_servo.state_hz: nothing is assumed")
        period = 1.0 / float(servo["state_hz"])
        joints = arm_joints(cfg)
        if describes_scene(cfg):  # the supervisor hosts the whole world and drives one actor of it (``sim_actor``)
            plant = ScenePlant.from_config(cfg, joints, period=period)
        else:
            plant = build_single_model_backend(cfg, control_period=period)
            open_jaws(plant, cfg)  # the Hand spawns open, as in the Dora plant node (MuJoCo's zero is a shut hand)
        return cls(RtConfig.from_config(cfg), joints, plant, servo=servo, observe=cfg.get("sim_observe"))

    @property
    def num_motors(self) -> int:
        return self.n

    # -- lifecycle ---------------------------------------------------------------------------------------------
    def open(self) -> None:
        self._plant.load()
        joint_hash = rtp.joint_order_hash(self.joint_names)
        self.session_id = secrets.randbits(63) | 1
        self.identity = rtp.Identity(self.config.instance_id, self.config.task_model, self.session_id, joint_hash,
                                     rtp.identity_hash(self.config.instance_id, joint_hash), 0, self.n)
        self._stop.clear()
        self._publish(time.monotonic())
        self._thread = threading.Thread(target=self._run, name=f"sim-link-{self.config.instance_id}", daemon=True)
        self._thread.start()
        print(f"[sim_link] instance '{self.config.instance_id}', {self.n} joints, session {self.session_id:016x}", flush=True)

    def close(self) -> None:
        try:
            self.safe_stop()
        except Exception:  # noqa: BLE001 -- closing must finish
            pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():  # a step still running: closing the plant under it would rebuild it
                print("[sim_link] servo thread did not stop; the plant is left open", flush=True)
                return
            self._thread = None
        self._plant.close()

    # -- authority ---------------------------------------------------------------------------------------------
    def enable_all(self) -> None:
        with self._lock:
            if self._fault_code:
                raise RtLinkError(f"arm refused: {self._latched_fault or 'latched fault'}")
            if self._armed:
                return  # already armed: acknowledged without a new epoch, as the RT server does
            state = self._state
            self._hold_q = np.asarray(state.q if state else np.zeros(self.n), dtype=float).copy()
            self._cmd, self._cmd_t, self._in_hold, self._cmd_gains = None, 0.0, False, None
            self._plant_ok = True
            self._epochs += 1
            self._arm_epoch = self._epochs
            self._armed = True
            self._arm_t = time.monotonic()
            epoch = self._arm_epoch
        # The ack is receipt; the ARM is complete in a sample that was STEPPED under it (samples carry the authority they ran in).
        self._await_state(lambda s: s.armed and s.arm_epoch == epoch, "arm")

    def safe_stop(self) -> bool:
        with self._lock:
            self._armed = False
            self._fault_code, self._latched_fault = 0, ""
            self._cmd, self._hold_q, self._in_hold, self._cmd_gains = None, None, False, None
            self._stop_epoch += 1  # every DISARM, as the RT server counts them
            stop = self._stop_epoch
        if self._thread is None:
            return True
        # Confirmed only by a sample that was stepped with the servo already released (zero gains), not by the ack alone.
        return self._await_state(lambda s: not s.armed and s.stop_epoch >= stop, "disarm", raise_on_timeout=False)

    def _await_state(self, predicate, what, *, raise_on_timeout=True) -> bool:
        """The ack is receipt; completion is a state sample produced after it (as on the RT link)."""
        deadline = time.monotonic() + 0.5 + 4 * self._period
        while time.monotonic() < deadline:
            state, _ = self.latest_state()
            if state is not None and predicate(state):
                return True
            time.sleep(self._period / 2)
        if raise_on_timeout:
            raise RtLinkError(f"{what} not reflected in the state stream")
        return False

    # -- commands ----------------------------------------------------------------------------------------------
    def apply_command(self, command) -> int | None:
        if command.get("cartesian") is not None or command.get("pose_hold") is not None:
            raise RtLinkError("the simulated link supports joint commands only")
        n = self.n
        cmd = {key: np.asarray(command.get(key, np.zeros(n)), dtype=float).reshape(-1)[:n].copy()
               for key in ("position", "velocity", "torque", "kp", "kd")}
        with self._lock:
            self._cmd_seq = (self._cmd_seq + 1) & 0xFFFFFFFF
            if self._armed and not self._fault_code:  # a disarmed or faulted servo ignores commands
                self._cmd, self._cmd_t = cmd, time.monotonic()
                self._cmd_gains = (cmd["kp"], cmd["kd"])
                self._last_cmd_seq = self._cmd_seq
            return self._cmd_seq

    def negotiate_reference(self, model_token: str) -> bool:
        return False

    # -- state -------------------------------------------------------------------------------------------------
    def latest_state(self):
        with self._lock:
            return self._state, self._state_rx_t

    def recent_states(self) -> tuple:
        with self._lock:
            return tuple(self._history)

    def motor_state(self, *, state=None) -> dict:
        if state is None:
            state, _ = self.latest_state()
        zeros = np.zeros(self.n)
        if state is None:
            return {"position": zeros, "velocity": zeros, "position_cmd": zeros, "velocity_cmd": zeros,
                    "torque_cmd": zeros, "kp": zeros, "kd": zeros, "torque": zeros}
        return {"position": np.asarray(state.q), "velocity": np.asarray(state.dq), "position_cmd": np.asarray(state.q_cmd),
                "velocity_cmd": np.asarray(state.qd_cmd), "torque_cmd": np.asarray(state.tau_cmd), "kp": zeros,
                "kd": zeros, "torque": np.asarray(state.tau)}

    def motor_health(self) -> dict:
        state, rx_t = self.latest_state()
        now = time.monotonic()
        age = now - rx_t if rx_t > 0 else float("inf")
        stale = age > self.config.state_timeout_s
        fault = self._latched_fault or (f"sim state stream stale ({age:.2f}s)" if stale and rx_t > 0 else "")
        return {"armed": bool(state and state.armed), "state_fresh": not stale, "latched_fault": fault,
                "any_fault": bool(fault) or bool(state and state.faulted), "holding": bool(state and state.holding),
                "holding_q": list(state.q_cmd) if state and state.holding else None, "backend": self.backend_name,
                "supports_pose_hold": False, "state_age_s": age if rx_t > 0 else -1.0,
                "last_cmd_seq": state.last_cmd_seq if state else 0, "sent_cmd_seq": self._cmd_seq,
                "online_mask": state.online_mask if state else 0, "active_mask": (1 << self.n) - 1,
                "instance_id": self.config.instance_id, "session_id": f"{self.session_id:016x}",
                "arm_epoch": self._arm_epoch if self._armed else 0, "stop_epoch": state.stop_epoch if state else 0,
                "state_seq": state.state_seq if state else 0, "state_t_mono_ns": state.t_mono_ns if state else 0}

    # -- the servo thread --------------------------------------------------------------------------------------
    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                self._tick(now)
            except Exception as exc:  # noqa: BLE001 -- a plant that fails is a plant fault, latched, never a dead thread
                self._plant_failed(now, exc)
            next_t += self._period
            if now - next_t > self._period:
                next_t = now
            time.sleep(max(0.0, next_t - time.monotonic()))

    def _plant_failed(self, now: float, exc: Exception) -> None:
        """The failing cycle publishes ITS OWN sample, as the RT does: the authority the step ran under (``_tick_snap``), the
        fault, not HOLDING, the last good pose. An ARM or DISARM that landed during the step is therefore not confirmed by it, and
        a failure of a step that ran under authority that has since changed is not latched against the new authority."""
        with self._lock:
            # No snapshot means the failure came before this tick stored one: nothing ran, so the live authority is the truth.
            snap = dict(self._tick_snap or dict(armed=self._armed, arm_epoch=self._arm_epoch, stop_epoch=self._pub_stop_epoch,
                                                last_cmd_seq=self._last_cmd_seq, gen=(self._arm_epoch, self._stop_epoch)))
            current = (self._arm_epoch, self._stop_epoch) == snap.get("gen", (self._arm_epoch, self._stop_epoch))
            if current:
                self._plant_ok = False  # the servo releases the plant until the next ARM, whatever fault is already latched
                self._cmd = None
                if not self._fault_code:  # the first cause is the one latched and published; later ones do not overwrite it
                    self._fault_code, self._latched_fault = rtp.FAULT_PLANT, f"sim plant failed: {exc!r}"
            last = self._state
            motor = dict(position=np.asarray(last.q if last else np.zeros(self.n), float),
                         velocity=np.asarray(last.dq if last else np.zeros(self.n), float))
            snap.update(fault=self._fault_code or rtp.FAULT_PLANT, holding=False)
            snap.pop("gen", None)
        self._publish(now, motor=motor, snapshot=snap)

    def _tick(self, now: float) -> None:
        self._tick_snap = None  # a failure before this tick's snapshot is stored must not reuse the previous tick's authority
        with self._lock:
            command = None
            holding = False
            if self._armed and not self._plant_ok:
                # A failed plant is not driven: like the RT server (which writes nothing), the servo releases it.
                measured = self._state.q if self._state else np.zeros(self.n)
                command = dict(position=np.asarray(measured, float).copy(), velocity=np.zeros(self.n),
                               torque=np.zeros(self.n), kp=np.zeros(self.n), kd=np.zeros(self.n))
            elif self._armed:
                silent = now - max(self._cmd_t, self._arm_t)
                if self._cmd is not None and not self._fault_code and silent * 1e3 <= self.hold_ms:
                    command, self._in_hold = self._cmd, False
                else:
                    holding = True
                    if not self._in_hold:  # entering a hold (ARM, a lost deadman, a fault): hold where the arm IS
                        measured = self._state.q if self._state else np.zeros(self.n)
                        self._hold_q = np.asarray(measured, dtype=float).copy()
                        self._cmd, self._in_hold = None, True
                    kp, kd = self._cmd_gains if self._cmd_gains is not None else (self._hold_kp, self._hold_kd)
                    command = dict(position=self._hold_q, velocity=np.zeros(self.n), torque=np.zeros(self.n), kp=kp, kd=kd)
                    if silent >= self._fault_s and not self._fault_code:
                        self._fault_code, self._latched_fault = rtp.FAULT_CMD_LOST, "command stream lost"
            else:
                # Released: the plant is told so. MuJoCoBackend.step(None) would keep re-applying its last command, so a
                # disarmed servo writes an explicit zero-gain, zero-torque word (the plant's own passive support stays).
                measured = self._state.q if self._state else np.zeros(self.n)
                command = dict(position=np.asarray(measured, float).copy(), velocity=np.zeros(self.n),
                               torque=np.zeros(self.n), kp=np.zeros(self.n), kd=np.zeros(self.n))
            if not self._armed:  # only a released step publishes a DISARM (a DISARM then ARM inside one step shows neither)
                self._pub_stop_epoch = self._stop_epoch
            snapshot = dict(armed=self._armed, fault=self._fault_code, holding=holding,
                            arm_epoch=self._arm_epoch, stop_epoch=self._pub_stop_epoch, last_cmd_seq=self._last_cmd_seq)
            # kept under the lock BEFORE the step: if the step raises, the failure sample is the one this step ran under
            self._tick_snap = dict(snapshot, gen=(self._arm_epoch, self._stop_epoch))
        motor = self._plant.step(command)
        self._publish(now, motor=motor, command=command, snapshot=snapshot)
        if self.observing:
            self._observe()

    def _observe(self) -> None:
        """Record this tick's observation (every ``_obs_every`` ticks). Called right after the step, so every field is the world the
        step left; never raises into the tick."""
        self._obs_tick += 1
        if self._obs_tick % self._obs_every:
            return
        try:
            world = self._plant.observe()
            # A scene names its actors (every one, by its world joint names); one model is this instance's own actor.
            actors = world.get("actors") or {self.config.instance_id: list(world["joint_q"])}
            actor_q = {a: {j: world["joint_q"][j] for j in names} for a, names in actors.items()}
            actor_dq = {a: {j: world["joint_dq"][j] for j in names} for a, names in actors.items()}
            sample = dict(seq=self._obs_seq + 1, sim_time_s=world["sim_time_s"], observed_ns=time.monotonic_ns(),
                          model_revision=world["model_revision"], scene_revision=world.get("scene_revision"),
                          actor_q=actor_q, actor_dq=actor_dq,
                          gripper=world.get("gripper"), attachments=world.get("attachments"),
                          constraints=world.get("constraints"), object_poses=world.get("object_poses"),
                          inhand_pose=world.get("inhand_pose"),
                          constraint_force=world.get("constraint_force"), ee_wrench=world.get("ee_wrench"),
                          topology=world.get("topology"), qpos=world.get("qpos"))
            with self._lock:
                self._obs_seq += 1
                self._obs.append(sample)
        except Exception:  # noqa: BLE001 -- observation must not stop the plant
            with self._lock:
                self._obs_errors += 1

    def observations(self, after: int = 0, limit: int = 64) -> dict:
        """observe/1: the observations recorded after sequence ``after``, oldest first, at most ``limit``. ``latest_seq`` is the
        cursor for the next read (the last sample returned; with none, ``after`` clamped to the newest, so a cursor from an earlier
        process cannot stall a reader). ``dropped`` counts samples that fell off the ring before the reader came back (0 on a
        first read); the plant never waits for a reader. Sequences are per supervisor PROCESS: a reader starts again from 0 when
        its session changes (a new hello). ``limit`` is capped at OBSERVE_LIMIT_MAX. Samples are shared: do not mutate them."""
        limit = min(max(1, int(limit)), OBSERVE_LIMIT_MAX)
        with self._lock:  # sequences are contiguous, so the start index is arithmetic; islice walks to it (O(start + limit)),
            # and a caught-up reader -- the common case -- returns before walking at all
            caught_up = not self._obs or after >= self._obs[-1]["seq"]
            start = 0 if caught_up else max(0, after - self._obs[0]["seq"] + 1)
            ring = [] if caught_up else list(islice(self._obs, start, start + limit))
            latest, errors, capacity = self._obs_seq, self._obs_errors, self._obs.maxlen
        dropped = max(0, ring[0]["seq"] - after - 1) if ring and after > 0 else 0
        return dict(schema="observe/1", samples=ring, latest_seq=ring[-1]["seq"] if ring else min(max(after, 0), latest),
                    dropped=dropped, errors=errors, capacity=capacity, rate_hz=self.observe_rate_hz)

    def _publish(self, now: float, *, motor=None, command=None, snapshot=None) -> None:
        """Publish one state sample. ``snapshot`` is the authority the tick STEPPED the plant under, so a sample never reports a
        state its physics did not run in (an ARM or DISARM that lands mid-step shows in the next sample)."""
        zeros = np.zeros(self.n)
        with self._lock:
            if motor is None:
                motor = self._plant.motor_state()
            if snapshot is None:
                snapshot = dict(armed=self._armed, fault=self._fault_code, holding=False, arm_epoch=self._arm_epoch,
                                stop_epoch=self._pub_stop_epoch, last_cmd_seq=self._last_cmd_seq)
            flags = rtp.FLAG_ARMED if snapshot["armed"] else 0
            if snapshot["fault"]:
                flags |= rtp.FLAG_FAULTED
            if snapshot["holding"]:
                flags |= rtp.FLAG_HOLDING
            self._state_seq += 1
            q, dq = np.asarray(motor["position"], float), np.asarray(motor["velocity"], float)
            q_cmd = q if command is None else np.asarray(command["position"], float)
            kp = zeros if command is None else np.asarray(command["kp"], float)
            kd = zeros if command is None else np.asarray(command["kd"], float)
            qd_cmd = zeros if command is None else np.asarray(command["velocity"], float)
            tau_ff = zeros if command is None else np.asarray(command["torque"], float)
            # NOT the RT's numbers: this is the PD word the plant was given, before the compiled servo law's clamp and slew,
            # and ``tau`` is the same word, not a measured torque.
            tau_cmd = tau_ff + kp * (q_cmd - q) + kd * (qd_cmd - dq)
            state = rtp.State(
                n=self.n, state_seq=self._state_seq, last_cmd_seq=snapshot["last_cmd_seq"], t_mono_ns=time.monotonic_ns(),
                flags=flags, fault_code=snapshot["fault"], session_id=self.session_id,
                identity_hash=self.identity.identity_hash if self.identity else 0, arm_epoch=snapshot["arm_epoch"],
                stop_epoch=snapshot["stop_epoch"], q=q.tolist(), dq=dq.tolist(), tau=tau_cmd.tolist(),
                tau_cmd=tau_cmd.tolist(), q_cmd=q_cmd.tolist(), qd_cmd=qd_cmd.tolist(), wrench=[0.0] * 6)
            self._state, self._state_rx_t = state, now
            self._history.append((state, now))
