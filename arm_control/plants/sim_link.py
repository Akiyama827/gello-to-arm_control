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
* the deadman: after ``hold_ms`` without a command the plant holds the measured pose at the hold gains, after ``fault_ms`` it
  latches ``FAULT_CMD_LOST`` and keeps holding until DISARM.

What it does not: no native reference (v7) path, no group record, no pose-hold, no clock domain (a simulation has no RT host
clock), and no RT-side torque clamp or slew beyond what the plant itself applies. Every servo parameter comes from the config's
``sim_servo`` block with no defaults: ``state_hz``, ``hold_ms``, ``fault_ms``, ``hold_kp``, ``hold_kd``. A simulation
that is not told them does not invent them.
"""
from __future__ import annotations

from collections import deque
import secrets
import threading
import time

import numpy as np

from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.client import RtConfig, RtLinkError

_SERVO_KEYS = ("state_hz", "hold_ms", "fault_ms", "hold_kp", "hold_kd")


class SimLink:
    def __init__(self, config: RtConfig, joint_names, plant, *, servo: dict) -> None:
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
        self._cmd_gains = None               # the gains of the last accepted command this epoch (a hold keeps them, as the RT does)
        self._fault_code = 0
        self._latched_fault = ""
        self._state_seq = 0

    @classmethod
    def from_config(cls, cfg) -> "SimLink":
        """The simulated plant a robot config describes: its ``sim_*`` plant keys (``plants.mujoco.single``), its ``rt`` block for
        the instance id and freshness bounds, and the required ``sim_servo`` block."""
        from arm_control.config import arm_joints
        from arm_control.plants.mujoco.single import build_single_model_backend

        servo = dict(cfg.get("sim_servo") or {})
        if "state_hz" not in servo:
            raise ValueError("config has no sim_servo.state_hz: nothing is assumed")
        plant = build_single_model_backend(cfg, control_period=1.0 / float(servo["state_hz"]))
        return cls(RtConfig.from_config(cfg), arm_joints(cfg), plant, servo=servo)

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
                with self._lock:
                    self._fault_code, self._latched_fault = rtp.FAULT_PLANT, f"sim plant failed: {exc!r}"
                    self._cmd = None
            next_t += self._period
            if now - next_t > self._period:
                next_t = now
            time.sleep(max(0.0, next_t - time.monotonic()))

    def _tick(self, now: float) -> None:
        with self._lock:
            command = None
            holding = False
            if self._armed:
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
            snapshot = dict(armed=self._armed, fault=self._fault_code, holding=holding or bool(self._fault_code),
                            arm_epoch=self._arm_epoch, stop_epoch=self._stop_epoch, last_cmd_seq=self._last_cmd_seq)
        motor = self._plant.step(command)
        self._publish(now, motor=motor, command=command, snapshot=snapshot)

    def _publish(self, now: float, *, motor=None, command=None, snapshot=None) -> None:
        """Publish one state sample. ``snapshot`` is the authority the tick STEPPED the plant under, so a sample never reports a
        state its physics did not run in (an ARM or DISARM that lands mid-step shows in the next sample)."""
        zeros = np.zeros(self.n)
        with self._lock:
            if motor is None:
                motor = self._plant.motor_state()
            if snapshot is None:
                snapshot = dict(armed=self._armed, fault=self._fault_code, holding=False, arm_epoch=self._arm_epoch,
                                stop_epoch=self._stop_epoch, last_cmd_seq=self._last_cmd_seq)
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
