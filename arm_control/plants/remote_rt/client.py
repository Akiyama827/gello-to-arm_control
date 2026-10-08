"""RT-machine client — the plant backend that reaches an arm THROUGH the RT
server (``rt/``) instead of a device on this host.

Same method surface as ``dm_backend`` / ``franka_backend`` (``open`` /
``enable_all`` / ``apply_command`` / ``motor_state`` / ``motor_health`` /
``safe_stop`` / ``close``), so ``nodes/rt_interface.py`` stays a thin adapter
and the graph cannot tell which transport a plant is behind.

Split of responsibilities with the server:
- The SERVER owns safety: staleness->hold, fault latching, torque
  clamp + slew. A dead PC leaves the arm holding, not falling.
- This client owns REPORTING: it mirrors the server's flags into
  ``motor_health`` and refuses nothing except sending while closed. The
  DISARM->ARM cycle is the only way past a server-side latch, exactly like
  the bench bridges.

Config (the arm's hardware fragment)::

    rt:
      host: 172.16.1.2
      udp_port: 47800
      tcp_port: 47801
      instance_id: fr3_1        # must equal the server's --instance-id
      state_timeout_s: 0.5

Admission (protocol v5): HELLO, then the server's IDENTITY; this client checks
instance id, joint count and joint ORDER against its own config and CLAIMs
them. The server refuses ARM without an accepted claim and admits only
commands carrying this session id and the current ARM epoch. That is flow
binding against wiring/config mistakes, not authentication.

``RtObserver`` is the separate read-only path: a UDP lease, no TCP session,
nothing that can arm, disarm, command or steal the commander's stream.
"""
from __future__ import annotations

import queue
import hashlib
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from arm_control.plants.remote_rt import protocol as rtp


def parse_clock_domain(token: str) -> tuple[str, str, str] | None:
    """``<host12>:<boot12>`` (lowercase hex) -> ('mono', host, boot); anything else, including a
    token cut off by the HELLO size limit, is None (unattested), never a partial identity."""
    parts = token.split(':')
    if len(parts) == 2 and all(len(v) == 12 and all(c in '0123456789abcdef' for c in v) for v in parts):
        return ('mono', *parts)
    return None


class RtLinkError(RuntimeError):
    """Server unreachable, refused, or speaking a different protocol."""


class ReferenceRefused(RtLinkError):
    def __init__(self, operation: int, result: int, reply: dict | None = None):
        super().__init__(f"reference operation {operation} refused: {result}")
        self.operation, self.result = operation, result
        self.reply = reply or {}  # a group reply also carries the peer verdict and group state


class ReferenceCancelledBeforeDispatch(RtLinkError):
    """Local proof that a cancel arrived first and COMMIT was never sent."""


def epoch_reached(value: int, target: int) -> bool:
    """32-bit generation comparison with wraparound (value >= target)."""
    return ((int(value) - int(target)) & 0xFFFFFFFF) < 0x80000000


class RtObserver:
    """Read-only copy of one RT server's state stream.

    UDP lease only: it opens no control session, sends no command packet, and
    closing it sends nothing (the lease simply expires). States whose identity
    hash is not the expected instance/joint order are dropped and counted.
    """

    def __init__(self, host: str, udp_port: int, instance_id: str,
                 joint_names: list[str], *, lease_ms: int = 1000) -> None:
        if not rtp.valid_id(instance_id):
            raise ValueError(f"observer needs a provisioned instance id, got {instance_id!r}")
        self.host, self.udp_port, self.lease_ms = host, int(udp_port), int(lease_ms)
        self.instance_id = instance_id
        self.joint_names = list(joint_names)
        self.expected_identity = rtp.identity_hash(
            instance_id, rtp.joint_order_hash(self.joint_names))
        self.identity_mismatches = 0
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state: rtp.State | None = None
        self._rx_t = 0.0      # content (new state_seq) arrival
        self._link_t = 0.0    # any datagram arrival

    def open(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.connect((self.host, self.udp_port))
        self._sock.settimeout(0.05)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest_state(self) -> tuple[rtp.State | None, float]:
        with self._lock:
            return self._state, self._rx_t

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()
        self._sock = None

    def _loop(self) -> None:
        renew = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= renew:
                try:
                    self._sock.send(rtp.pack_observe(self.lease_ms))
                except OSError:
                    pass
                renew = now + self.lease_ms / 3000.0
            try:
                data = self._sock.recv(rtp.STATE_SIZE)
            except (socket.timeout, OSError):
                continue
            try:
                state = rtp.unpack_state(data)
            except ValueError:
                continue
            if state.identity_hash != self.expected_identity or state.n != len(self.joint_names):
                self.identity_mismatches += 1
                continue
            with self._lock:
                t = time.monotonic()
                self._link_t = t
                if self._state is None or state.state_seq != self._state.state_seq:
                    self._rx_t = t
                self._state = state


@dataclass(frozen=True)
class RtConfig:
    host: str = "127.0.0.1"
    udp_port: int = 47800
    tcp_port: int = 47801
    state_timeout_s: float = 0.5
    ack_timeout_s: float = 2.0
    # Refuse servers whose command deadman is slower than this (0 disables).
    # The handguide bench runs --fault-ms 3600000; a COMMANDER graph against
    # that server has no staleness reflex at all — the HELLO now carries the
    # server's launch flags exactly so this check can exist.
    max_fault_ms: float = 0.0
    # Provisioned instance this config is FOR. Required: an empty id cannot be
    # claimed, so a config without one cannot command anything.
    instance_id: str = ""
    # Expected RT task-space model id ("" = not relied on). Pose hold with a
    # calibrated PC model must pin this; see the capability gate in apply_command.
    task_model: str = ""

    @classmethod
    def from_config(cls, cfg) -> "RtConfig":
        raw = dict(cfg.get("rt") or {})
        return cls(
            host=str(raw.get("host", "127.0.0.1")),
            udp_port=int(raw.get("udp_port", 47800)),
            tcp_port=int(raw.get("tcp_port", 47801)),
            state_timeout_s=float(raw.get("state_timeout_s", 0.5)),
            ack_timeout_s=float(raw.get("ack_timeout_s", 2.0)),
            max_fault_ms=float(raw.get("max_fault_ms", 0.0)),
            instance_id=str(raw.get("instance_id", "")),
            task_model=str(raw.get("task_model", "")),
        )


class RtBackend:
    """One arm behind an ``arm_rt_server``."""

    def __init__(self, config: RtConfig, joint_names: list[str]) -> None:
        self.config = config
        self.joint_names = list(joint_names)
        self.n = len(joint_names)
        self.backend_name = ""
        # HELLO clk=<host>:<boot> (CTRL-041): the RT host's keyed clock identity, or None
        # (unattested: an older server, or one that could not read its host identity).
        self.clock_domain: tuple[str, str, str] | None = None
        self._group_committing = False  # a COMMIT_GROUP wait is in flight (the only exchange group_cancel may interrupt)
        # CTRL-040: the server was started with a group record (HELLO g=1); events are the server's
        # unsolicited CTL_GROUP_STATUS reports (a peer fault/cancel/loss retired this member).
        self.group_capable = False
        self.group_events: deque = deque(maxlen=16)
        self._tcp: socket.socket | None = None
        self._udp: socket.socket | None = None
        self._rx_thread: threading.Thread | None = None
        self._ctl_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._state: rtp.State | None = None
        self._state_history = deque(maxlen=32)
        self._state_rx_t = 0.0
        self._status_q: queue.Queue = queue.Queue()
        self._latched_fault = ""
        self._cmd_seq = 0
        self._last_send_t = 0.0
        self._last_send_err = 0.0
        self._link_rx_t = 0.0  # ANY state arrival (vs _state_rx_t = new content)
        self._send_lock = threading.Lock()  # node thread + _ctl_loop PONG
        self._active_mask = 0
        self.supports_pose_hold = False
        self.supports_cartesian = False
        self.identity: rtp.Identity | None = None
        self.session_id = 0
        self._arm_epoch = 0
        self._expected_identity = 0
        self._stream_fault = ""
        self._reference_mode = False
        self._reference_model = b""
        self._reference_seq = 0
        self._upload_id = 0
        self._upload_lock = threading.Lock()
        self._upload_cancel = threading.Event()
        self._reference_cv = threading.Condition()
        self._reference_wait_seq = None
        self._reference_reply = None
        # Armed-only mode selection (CAP_MODE_SELECT). The generation restarts
        # at 0 on every ARM; the client mirrors it so a retry after a lost reply
        # still carries the identical bytes RT can answer idempotently.
        self._mode_q: queue.Queue = queue.Queue()
        self._mode_generation = 0

    @classmethod
    def from_config(cls, cfg) -> "RtBackend":
        from arm_control.config import arm_joints

        return cls(RtConfig.from_config(cfg), arm_joints(cfg))

    @property
    def num_motors(self) -> int:
        return self.n

    # -- lifecycle ------------------------------------------------------------
    def open(self) -> None:
        self._reference_mode = False
        self._reference_model = b""
        self._reference_seq = self._upload_id = 0
        self._mode_generation = 0
        self._upload_cancel.set()
        cfg = self.config
        try:
            self._tcp = socket.create_connection(
                (cfg.host, cfg.tcp_port), timeout=cfg.ack_timeout_s
            )
        except OSError as exc:
            raise RtLinkError(
                f"cannot reach arm_rt_server at {cfg.host}:{cfg.tcp_port}: {exc}"
            ) from exc
        self._tcp.settimeout(0.2)
        try:
            self._admit()
        except (RtLinkError, ValueError) as exc:
            self._tcp.close()
            self._tcp = None
            if isinstance(exc, ValueError):
                raise RtLinkError(
                    f"RT protocol mismatch; client requires version {rtp.VERSION}: {exc}"
                ) from exc
            raise
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.connect((cfg.host, cfg.udp_port))
        self._udp.settimeout(0.2)
        self._stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        self._ctl_thread = threading.Thread(target=self._ctl_loop, daemon=True)
        self._ctl_thread.start()
        # Teach the state-stream address. Content is ignored: the server admits
        # a disarmed session's datagram ONLY as its return address.
        self._send_command_raw(np.zeros(self.n), np.zeros(self.n), np.zeros(self.n),
                               np.zeros(self.n), np.zeros(self.n))
        print(
            f"[rt_link] connected to {cfg.host} — instance '{self.identity.instance_id}', "
            f"backend '{self.backend_name}', {self.n} joints, session "
            f"{self.session_id:016x}",
            flush=True,
        )

    def _admit(self) -> None:
        """HELLO -> IDENTITY -> CLAIM. Raises before any UDP is sent."""
        cfg = self.config
        if not rtp.valid_id(cfg.instance_id):
            raise RtLinkError(
                f"rt.instance_id {cfg.instance_id!r} is not a provisioned instance id; "
                "refusing to claim an RT server without knowing which arm it must be"
            )
        hello = self._await_control({rtp.CTL_HELLO}, cfg.ack_timeout_s)
        if hello is None:
            raise RtLinkError("no HELLO from server (protocol/version mismatch?)")
        if hello.arg != self.n:
            raise RtLinkError(
                f"server drives {hello.arg} joints, config says {self.n} "
                f"({self.joint_names})"
            )
        # HELLO text: "<backend> hold_ms=<v> fault_ms=<v>" (older servers send
        # the bare name — the flags are then unknown and unchecked).
        parts = hello.text.split()
        self.backend_name = parts[0] if parts else hello.text
        flags = dict(
            p.split("=", 1) for p in parts[1:] if "=" in p
        )
        # The streamed executor uses the server's actual deadman threshold,
        # never a locally assumed launch flag. Legacy bare-name HELLOs leave
        # this unknown and retain their existing RT-holding feedback path.
        self.hold_ms = float(flags["hold_ms"]) if "hold_ms" in flags else None
        if self.hold_ms is not None and (not np.isfinite(self.hold_ms) or self.hold_ms <= 0):
            raise RtLinkError("server advertised an invalid hold_ms")
        self.supports_pose_hold = flags.get("pose_hold") == str(rtp.POSE_HOLD_VERSION)
        self._active_mask = int(flags.get("active", str((1 << self.n) - 1)), 0)
        # Process-local RT setup: "ok" or the failed parts; older servers omit it.
        self.rt_setup = flags.get("rt", "unknown")
        self.clock_domain = parse_clock_domain(flags.get("clk", ""))
        self.group_capable = flags.get("g") == "1"
        fault_ms = float(flags.get("fault_ms", 0) or 0)
        if self.config.max_fault_ms > 0 and fault_ms > self.config.max_fault_ms:
            raise RtLinkError(
                f"server --fault-ms {fault_ms:.0f} exceeds rt.max_fault_ms "
                f"{self.config.max_fault_ms:.0f} — this looks like a handguide "
                "server (no command deadman); restart arm_rt_server with its "
                "normal flags before running a commander graph"
            )
        frame = self._await_control({rtp.CTL_IDENTITY}, cfg.ack_timeout_s)
        if frame is None:
            raise RtLinkError("no IDENTITY from server")
        ident = rtp.unpack_identity(frame.raw)
        want_hash = rtp.joint_order_hash(self.joint_names)
        problems = []
        if ident.instance_id != cfg.instance_id:
            problems.append(f"server is instance '{ident.instance_id}', "
                            f"config is for '{cfg.instance_id}'")
        if ident.n != self.n:
            problems.append(f"server drives {ident.n} joints, config has {self.n}")
        if ident.joint_order_hash != want_hash:
            problems.append("server's provisioned joint order differs from "
                            f"{self.joint_names}")
        if ident.identity_hash != rtp.identity_hash(ident.instance_id, ident.joint_order_hash):
            problems.append("server identity hash is inconsistent")
        if cfg.task_model and ident.model_id != cfg.task_model:
            problems.append(f"server task model '{ident.model_id}', expected "
                            f"'{cfg.task_model}'")
        if problems:
            raise RtLinkError("RT identity refused: " + "; ".join(problems))
        claim = rtp.Identity(cfg.instance_id, cfg.task_model, ident.session_id, want_hash,
                             rtp.identity_hash(cfg.instance_id, want_hash), 0, self.n)
        self._send_control(rtp.CTL_CLAIM, text=rtp.pack_identity(claim))
        status = self._await_control({rtp.CTL_STATUS}, cfg.ack_timeout_s)
        if status is None or not status.arg & rtp.FLAG_CLAIMED:
            raise RtLinkError(f"RT claim refused: {status.text if status else 'no ack'}")
        self.identity = ident
        self.session_id = ident.session_id
        self._expected_identity = ident.identity_hash
        self.supports_pose_hold = bool(ident.capabilities & rtp.CAP_POSE_HOLD)
        self.supports_cartesian = bool(ident.capabilities & rtp.CAP_CARTESIAN_TRAJECTORY)

    def enable_all(self) -> None:
        """Arm. The server holds the current pose until commands flow."""
        # Clear BEFORE consenting: a CTL_FAULT that lands after the ack is
        # new evidence and must not be erased by this cycle's bookkeeping.
        self._latched_fault = ""
        self._seed_command_seq()
        status = self._control_roundtrip(rtp.CTL_ARM)
        # FAULTED in the ack is the refusal — ARMED alone is not consent: the
        # server keeps its ARMED flag while fault-HOLDING a latched arm.
        if status.arg & rtp.FLAG_FAULTED or not status.arg & rtp.FLAG_ARMED:
            raise RtLinkError(f"arm refused: {status.text or 'latched fault'}")
        if status.seq == 0:
            raise RtLinkError("arm ack carries no ARM epoch (pre-v5 server?)")
        self._arm_epoch = status.seq
        self._mode_generation = 0  # a new epoch seeds generation 0

    def _seed_command_seq(self) -> None:
        """Number this epoch's commands above the server's ``last_cmd_seq``.

        The server keeps that value across ARM generations and client
        sessions, while a new client restarts at 1. Seeded before ARM (no
        command of the new epoch exists yet), a stale value can then never
        pass for one of ours -- the execution contract's APPLIED rule
        (docs/ARCHITECTURE.md) depends on it.
        """
        deadline = time.monotonic() + self.config.ack_timeout_s
        state, _ = self.latest_state()
        while state is None and time.monotonic() < deadline:
            time.sleep(0.01)
            state, _ = self.latest_state()
        if state is None:
            raise RtLinkError("no state from server; refusing to arm without the state stream")
        if not epoch_reached(self._cmd_seq, state.last_cmd_seq):
            self._cmd_seq = state.last_cmd_seq

    def safe_stop(self) -> bool:
        """Disarm (also clears a server-side fault latch, matching the bench
        bridges' explicit DISARM->ARM recovery cycle).

        Returns True only when the drop of authority is CONFIRMED: the ack
        must show ARMED cleared, and the state stream must reflect it — the
        TCP ack alone is not proof, since the RT loop applies the flag at its
        next sample (which can be seconds away inside a blocking plant call).
        """
        if self._tcp is None:
            return True
        try:
            status = self._control_roundtrip(rtp.CTL_DISARM)
        except RtLinkError as exc:
            print(f"[rt_link] disarm: {exc}", flush=True)
            return False
        acked = time.monotonic()
        self._arm_epoch = 0  # no further command of the old epoch is admissible
        self._mode_generation = 0
        if status.arg & rtp.FLAG_ARMED:
            print("[rt_link] disarm ack still shows ARMED", flush=True)
            return False
        # The ack is RECEIPT. Completion is a state sample produced after it
        # whose stop epoch covers this request's DISARM generation.
        deadline = acked + 0.5
        while time.monotonic() < deadline:
            state, rx_t = self.latest_state()
            if (state is not None and rx_t > acked and not state.armed
                    and epoch_reached(state.stop_epoch, status.seq)):
                # Confirmed drop clears the client-side latch too — the server
                # cleared its own on DISARM, and a stale fault string here
                # keeps the panel demanding a recovery cycle the operator
                # already performed (trains them to ignore the fault badge).
                self._latched_fault = ""
                return True
            time.sleep(0.01)
        print("[rt_link] disarm not yet reflected in the state stream", flush=True)
        return False

    def set_active_mask(self, mask: int) -> None:
        """Activate fixed slots; removal is allowed only while disarmed."""
        mask = int(mask)
        configured = (1 << self.n) - 1
        if mask < 0 or mask & ~configured:
            raise ValueError(f"active mask must fit {self.n} configured slots")
        state, _ = self.latest_state()
        if state is not None and state.armed and self._active_mask & ~mask:
            raise RuntimeError("cannot remove an active slot while armed")
        status = self._control_roundtrip(rtp.CTL_SET_ACTIVE, arg=mask)
        if status.arg != mask:
            raise RtLinkError(status.text or "active mask refused")
        self._active_mask = mask

    def close(self) -> None:
        try:
            self.safe_stop()
        except Exception:
            pass
        self._stop.set()
        for sock in (self._udp, self._tcp):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._udp = self._tcp = None

    # -- command path ---------------------------------------------------------
    def apply_command(self, command: dict[str, Any]) -> int | None:
        """Stream one joint-servo word (the full bridge contract — the server
        applies torque clamp + slew; nothing is dropped on this side).
        Returns the wire seq sent, or None if nothing left (closed, send error)."""
        cartesian = command.get("cartesian")
        if cartesian is not None:
            # v8 (operator 2026-10-09): task frame + Kc/Dc in the robot BASE frame; the box computes the target as FK(q_des).
            # A world-frame spec (the in-process controller's pose/task_R/kc/dc) is refused, never sent.
            if not self.supports_cartesian:
                raise RtLinkError("RT backend does not advertise the v8 Cartesian law (CAP_CARTESIAN_TRAJECTORY)")
            if not isinstance(cartesian, dict) or cartesian.get("frame") != "base":
                raise RtLinkError("a cartesian command must be a base-frame spec (frame: base, task_R, kc, dc)")
        pose_hold = command.get("pose_hold")
        if pose_hold is not None and cartesian is not None:
            raise RtLinkError("a command carries a pose-hold tail or a cartesian tail, not both")
        if pose_hold is not None and not self.supports_pose_hold:
            raise RtLinkError(f"RT backend does not advertise pose_hold={rtp.POSE_HOLD_VERSION}")
        if self._udp is None:
            return None
        return self._send_command_raw(
            np.asarray(command["position"], dtype=float),
            np.asarray(command.get("velocity", np.zeros(self.n)), dtype=float),
            np.asarray(command.get("torque", np.zeros(self.n)), dtype=float),
            np.asarray(command.get("kp", np.zeros(self.n)), dtype=float),
            np.asarray(command.get("kd", np.zeros(self.n)), dtype=float),
            pose_hold=pose_hold, cartesian=cartesian,
        )

    def _send_command_raw(self, q, qd, tau, kp, kd, pose_hold=None, cartesian=None) -> int | None:
        self._cmd_seq = (self._cmd_seq + 1) & 0xFFFFFFFF
        pkt = rtp.pack_command(
            n=self.n,
            seq=self._cmd_seq,
            t_mono_ns=time.monotonic_ns(),
            session_id=self.session_id,
            arm_epoch=self._arm_epoch,
            q_des=q[: self.n],
            qd_des=qd[: self.n],
            tau_ff=tau[: self.n],
            kp=kp[: self.n],
            kd=kd[: self.n],
            pose_hold=pose_hold,
            cartesian=cartesian,
        )
        return self._send_command_packet(pkt)

    def _send_command_packet(self, pkt) -> int | None:
        """Shared UDP output diagnostics/failure semantics for v5/v6/v7."""
        if self._udp is None:
            return None
        now = time.monotonic()
        if self._last_send_t and now - self._last_send_t > 0.3:
            print(f"[rt_link] send gap {now - self._last_send_t:.2f}s "
                  f"(seq {self._cmd_seq})", flush=True)
        self._last_send_t = now
        try:
            self._udp.send(pkt)
        except OSError as exc:
            # NEVER silent: a persistently failing send is indistinguishable
            # from a healthy stream to every layer above (found hunting a
            # CMD_LOST latch whose packets vanished between bridge and server).
            if now - self._last_send_err > 1.0:
                self._last_send_err = now
                print(f"[rt_link] UDP send failed: {exc}", flush=True)
            return None
        return self._cmd_seq

    def negotiate_reference(self, model_token: str) -> bool:
        """Disarmed opt-in; an old server remains usable in streamed mode."""
        if self.identity is None or not self.identity.capabilities & rtp.CAP_REFERENCE_HORIZON:
            return False
        if not model_token:
            raise ValueError("native reference requires the bound execution model token")
        digest = hashlib.sha256(model_token.encode()).digest()
        if self._reference_mode:
            if digest != self._reference_model:
                raise RtLinkError("reconnect to change reference model binding")
            return True
        packet = rtp.pack_reference_control(
            ctl_type=rtp.CTL_REFERENCE_MODE, seq=0, session_id=self.session_id,
            joint_order_hash=self.identity.joint_order_hash, model_digest=digest, enable=1)
        while not self._status_q.empty():
            self._status_q.get_nowait()
        self._send_frame(packet)
        try:
            status = self._status_q.get(timeout=self.config.ack_timeout_s)
        except queue.Empty as exc:
            raise RtLinkError("no reference mode acknowledgement") from exc
        if status.text != "reference mode accepted":
            raise RtLinkError(status.text)
        self._reference_model = digest
        self._reference_mode = True
        return True

    def select_mode(self, desired_mode: int, *, model_token: str | None = None,
                    timeout_s: float | None = None) -> dict:
        """Armed-only bounded selection generation; "Soft" = MODE_STREAMED.

        Returns the server's ``unpack_mode_status`` reply. On
        ``MODE_SEL_OK`` the client advances its mirrored generation, so an
        identical retry after a lost reply carries the same bytes RT answers
        idempotently. MODE_REFERENCE requires ``CAP_MODE_SELECT`` and the
        disarmed opt-in with a matching bound digest.
        """
        if self.identity is None:
            raise RtLinkError("no identity: connect and claim first")
        if not self._arm_epoch:
            raise RtLinkError("mode selection is armed-only")
        digest = b""
        if desired_mode == rtp.MODE_REFERENCE:
            if not self.identity.capabilities & rtp.CAP_MODE_SELECT:
                raise RtLinkError("server does not offer mode selection")
            if not self._reference_mode:
                raise RtLinkError("MODE_REFERENCE requires the disarmed opt-in")
            if not model_token:
                raise RtLinkError("MODE_REFERENCE requires the bound model token")
            digest = hashlib.sha256(model_token.encode()).digest()
        elif desired_mode != rtp.MODE_STREAMED:
            raise ValueError("desired_mode must be MODE_STREAMED or MODE_REFERENCE")
        predecessor = self._mode_generation
        generation = predecessor + 1
        packet = rtp.pack_mode_select(
            session_id=self.session_id, generation=generation,
            expected_predecessor=predecessor, arm_epoch=self._arm_epoch,
            desired_mode=desired_mode, model_digest=digest)
        while not self._mode_q.empty():
            self._mode_q.get_nowait()
        self._send_frame(packet)
        budget = self.config.ack_timeout_s if timeout_s is None else timeout_s
        try:
            reply = self._mode_q.get(timeout=budget)
        except queue.Empty as exc:
            raise RtLinkError("no mode select acknowledgement") from exc
        if reply["generation"] == generation and reply["result"] == rtp.MODE_SEL_OK:
            self._mode_generation = generation
        return reply

    def _reference_exchange(self, operation: int, *, _scope=None, _socket=None,
                            _trace=None, **fields) -> dict:
        # One frame in flight, and no send lock held while awaiting a reply.
        # Lifecycle traffic/PONG can interleave even with a stalled upload peer.
        if _scope is not None:
            self._reference_scope(*_scope)
        connection = self._tcp if _socket is None else _socket
        self._reference_seq += 1
        if self._reference_seq > 0xFFFFFFFF:
            raise RtLinkError("reference request sequence exhausted; reconnect")
        seq = self._reference_seq
        packet = rtp.pack_reference_control(
            ctl_type=operation, seq=seq, t_mono_ns=time.monotonic_ns(),
            negotiated=self._reference_mode, **fields)
        with self._reference_cv:
            self._reference_wait_seq, self._reference_reply = seq, None
        try:
            for attempt in range(2):  # one exact retry; never a bulk enqueue
                if self._upload_cancel.is_set() or self._stop.is_set():
                    if attempt == 0 and operation == rtp.CTL_REFERENCE_COMMIT:
                        # Local proof the RT never saw COMMIT (unlike an
                        # interruption after the send, which is uncertain).
                        raise ReferenceCancelledBeforeDispatch(
                            "COMMIT cancelled before dispatch")
                    raise RtLinkError("reference upload interrupted by lifecycle")
                if _scope is not None:
                    self._reference_scope(*_scope)
                self._send_frame(packet, sock=connection)
                if _trace is not None:
                    _trace('commit_sent', time.monotonic(), attempt)
                deadline = time.monotonic() + self.config.ack_timeout_s
                with self._reference_cv:
                    while self._reference_reply is None:
                        if self._upload_cancel.is_set() or self._stop.is_set():
                            raise RtLinkError("reference upload interrupted by lifecycle")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._reference_cv.wait(min(.02, remaining))
                    reply = self._reference_reply
                if reply is not None:
                    if _trace is not None:
                        _trace('commit_reply', time.monotonic(), reply['result'])
                    if reply['operation'] != operation or reply['result'] != rtp.REF_OK:
                        raise ReferenceRefused(operation, reply["result"], reply)
                    return reply
            raise RtLinkError("no reference acknowledgement after exact retry")
        finally:
            with self._reference_cv:
                self._reference_wait_seq, self._reference_reply = None, None

    def upload_reference(self, trajectory, *, reference_id: int, revision: int = 1,
                         expected_revision: int = 0, terminal: bool = True,
                         session_id: int | None = None, arm_epoch: int | None = None) -> int:
        """Upload/admit an already PC-admitted trajectory; never select it.

        At most one caller and one in-flight 128-byte frame. No background
        frame queue exists; each CHUNK acknowledgement yields the socket to
        lifecycle/control traffic before the next CHUNK is produced.
        """
        session_id, arm_epoch = self._reference_scope(session_id, arm_epoch)
        connection = self._tcp
        scope = (session_id, arm_epoch)
        if not self._upload_lock.acquire(blocking=False):
            raise RtLinkError("one reference upload is already in progress")
        try:
            self._upload_cancel.clear()
            times = np.asarray(trajectory.times, dtype=float)
            q = np.asarray(trajectory.positions, dtype=float)
            velocity = trajectory.velocities
            k = len(times)
            if not 2 <= k <= rtp.MAX_REFERENCE_KNOTS or q.shape != (k, self.n):
                raise ValueError("reference requires 2..128 knots and the configured joint count")
            arrays = [times[:, None], q]
            if velocity is not None:
                velocity = np.asarray(velocity, dtype=float)
                if velocity.shape != q.shape:
                    raise ValueError("reference velocity shape mismatch")
                arrays.append(velocity)
            rows = np.concatenate(arrays, axis=1)
            if not np.isfinite(rows).all() or times[0] < 0 or not (np.diff(times) > 0).all():
                raise ValueError("reference needs finite values and increasing nonnegative times")
            body = rows.astype('<f8', copy=False).tobytes()
            self._upload_id += 1
            upload_id = self._upload_id
            self._reference_exchange(
                rtp.CTL_REFERENCE_BEGIN, _scope=scope, _socket=connection,
                session_id=session_id, upload_id=upload_id,
                reference_id=reference_id, joint_order_hash=self.identity.joint_order_hash,
                model_digest=self._reference_model, arm_epoch=arm_epoch,
                expected_revision=expected_revision, revision=revision, byte_count=len(body),
                knots=k, n=self.n, flags=(rtp.REFERENCE_HERMITE if velocity is not None else 0)
                | (rtp.REFERENCE_TERMINAL if terminal else 0))
            for offset in range(0, len(body), rtp.REFERENCE_CHUNK_BYTES):
                chunk = body[offset:offset + rtp.REFERENCE_CHUNK_BYTES]
                self._reference_exchange(rtp.CTL_REFERENCE_CHUNK, _scope=scope, _socket=connection,
                                         upload_id=upload_id,
                                         offset=offset, length=len(chunk), data=chunk)
            self._reference_exchange(rtp.CTL_REFERENCE_SEAL, _scope=scope, _socket=connection,
                                     upload_id=upload_id,
                                     sha256=hashlib.sha256(body).digest())
            return upload_id
        finally:
            self._upload_lock.release()

    def _reference_scope(self, session_id, arm_epoch):
        session_id = self.session_id if session_id is None else session_id
        arm_epoch = self._arm_epoch if arm_epoch is None else arm_epoch
        if (not self._reference_mode or not arm_epoch or session_id != self.session_id
                or arm_epoch != self._arm_epoch):
            raise RtLinkError("reference session/ARM epoch is no longer current")
        return session_id, arm_epoch

    def commit_reference(self, upload_id: int, *, reference_id: int, revision: int = 1,
                         expected_revision: int = 0, protected_until: float = 0.,
                         latest_phase: float = 0., session_id: int | None = None,
                         arm_epoch: int | None = None, selection_budget_ns: int = 0,
                         _trace=None) -> dict:
        session_id, arm_epoch = self._reference_scope(session_id, arm_epoch)
        # No clear here: this attempt's upload cleared the flag, so a set flag is
        # a cancel that arrived since. _reference_exchange sees it under this
        # lock, before COMMIT is sent.
        with self._upload_lock:
            return self._reference_exchange(
                rtp.CTL_REFERENCE_COMMIT, _scope=(session_id, arm_epoch), _trace=_trace,
                session_id=session_id, upload_id=upload_id,
                reference_id=reference_id, arm_epoch=arm_epoch, revision=revision,
                expected_revision=expected_revision, protected_until=protected_until,
                latest_phase=latest_phase, selection_budget_ns=selection_budget_ns)

    # -- CTRL-040 group execution ------------------------------------------------------------
    def group_join(self, group_id: int, member_index: int, member_count: int, *,
                   session_id: int | None = None, arm_epoch: int | None = None) -> dict:
        """Bind this member to ``group_id``. The reply carries the member's group state."""
        if not self.group_capable:
            raise RtLinkError("server was not started with a group record")
        session_id, arm_epoch = self._reference_scope(session_id, arm_epoch)
        with self._upload_lock:
            self._upload_cancel.clear()  # a fresh group starts a fresh upload sequence
            return self._reference_exchange(
                rtp.CTL_GROUP_JOIN, _scope=(session_id, arm_epoch), session_id=session_id,
                group_id=group_id, arm_epoch=arm_epoch, member_index=member_index,
                member_count=member_count)

    def group_commit(self, upload_id: int, *, reference_id: int, group_id: int, member_index: int,
                     budget_ns: int, session_id: int | None = None,
                     arm_epoch: int | None = None) -> dict:
        """COMMIT for a group. Returns when this member's reference is SELECTED (reply
        ``selected_ns``, ``deadline_ns`` on the RT host clock); refused with the peer verdict if a
        peer is bad, the group deadline passed, or the group was cancelled while waiting."""
        if not self.group_capable:
            raise RtLinkError("server was not started with a group record")
        session_id, arm_epoch = self._reference_scope(session_id, arm_epoch)
        self._group_committing = True
        try:
            with self._upload_lock:
                return self._reference_exchange(
                    rtp.CTL_GROUP_COMMIT, _scope=(session_id, arm_epoch), session_id=session_id,
                    group_id=group_id, upload_id=upload_id, reference_id=reference_id,
                    budget_ns=budget_ns, arm_epoch=arm_epoch, member_index=member_index)
        finally:
            self._group_committing = False

    def group_cancel(self, group_id: int, member_index: int, member_count: int, *,
                     session_id: int | None = None, arm_epoch: int | None = None) -> dict:
        if not self.group_capable:
            raise RtLinkError("server was not started with a group record")
        session_id, arm_epoch = self._reference_scope(session_id, arm_epoch)
        if self._group_committing:  # interrupt only a COMMIT_GROUP wait: never another upload or the member's own hold
            self._upload_cancel.set()
        with self._upload_lock:
            self._upload_cancel.clear()
            return self._reference_exchange(
                rtp.CTL_GROUP_CANCEL, _scope=(session_id, arm_epoch), session_id=session_id,
                group_id=group_id, arm_epoch=arm_epoch, member_index=member_index,
                member_count=member_count)

    def cancel_reference(self, reference_id: int, revision: int, *,
                         session_id: int | None = None, arm_epoch: int | None = None) -> None:
        session_id, arm_epoch = self._reference_scope(session_id, arm_epoch)
        self._upload_cancel.set()
        with self._upload_lock:
            self._upload_cancel.clear()
            self._reference_exchange(rtp.CTL_CANCEL_REFERENCE, _scope=(session_id, arm_epoch),
                                     session_id=session_id,
                                     id=reference_id, arm_epoch=arm_epoch, revision=revision)

    def apply_reference_envelope(self, command: dict, *, reference_id: int, revision: int,
                                 phase: float, sample_received_at: float,
                                 valid_for_s: float, phase_window_s: float,
                                 session_id: int | None = None, arm_epoch: int | None = None) -> int | None:
        """PC dynamics evaluated at the feedback reference sample, never a second clock."""
        session_id, arm_epoch = self._reference_scope(session_id, arm_epoch)
        self._cmd_seq = (self._cmd_seq + 1) & 0xFFFFFFFF
        age_us = max(0, int((time.monotonic() - sample_received_at)*1e6))
        if age_us > int(valid_for_s*1e6):
            return None  # stale computation: existing freshness fallback, no new authority
        packet = rtp.pack_reference_command(
            n=self.n, seq=self._cmd_seq, arm_epoch=arm_epoch,
            t_mono_ns=time.monotonic_ns(), session_id=session_id,
            reference_id=reference_id, revision=revision, phase=phase, age_us=age_us,
            valid_for_us=int(valid_for_s*1e6), phase_window_us=int(phase_window_s*1e6),
            tau_ff=command['torque'], kp=command['kp'], kd=command['kd'],
            negotiated=self._reference_mode)
        return self._send_command_packet(packet)

    def interrupt_reference_upload(self) -> None:
        """Interrupt the bounded sender; retirement remains an explicit control operation."""
        self._upload_cancel.set()

    def abort_reference_upload(self, upload_id: int | None = None, *,
                               session_id: int | None = None, arm_epoch: int | None = None) -> None:
        session_id, arm_epoch = self._reference_scope(session_id, arm_epoch)
        self._upload_cancel.set()
        with self._upload_lock:
            self._upload_cancel.clear()
            candidate = self._upload_id if upload_id is None else upload_id
            if not candidate:
                return
            self._reference_exchange(rtp.CTL_REFERENCE_ABORT, _scope=(session_id, arm_epoch),
                                     session_id=session_id, id=candidate, arm_epoch=arm_epoch, revision=0)

    # -- feedback -------------------------------------------------------------
    def latest_state(self) -> tuple[rtp.State | None, float]:
        with self._state_lock:
            return self._state, self._state_rx_t

    def recent_states(self) -> tuple:
        """Bounded received packets for timestamp alignment; no extrapolation."""
        with self._state_lock:
            return tuple(self._state_history)

    def motor_state(self, *, state=None) -> dict[str, np.ndarray]:
        if state is None:
            state, _ = self.latest_state()
        if state is None:
            zeros = np.zeros(self.n)
            return {
                "position": zeros, "velocity": zeros, "position_cmd": zeros,
                "velocity_cmd": zeros, "torque_cmd": zeros,
                "kp": zeros, "kd": zeros, "torque": zeros,
            }
        zeros = np.zeros(self.n)
        return {
            "position": np.asarray(state.q),
            "velocity": np.asarray(state.dq),
            "position_cmd": np.asarray(state.q_cmd),
            # RT-selected reference, including zero-velocity holds. This is
            # neither measured dq nor a local echo of the last sent packet.
            "velocity_cmd": np.asarray(state.qd_cmd),
            # The servo's own post-clamp post-slew output — the tau_J_d
            # analogue, and the honest number for tracking plots.
            "torque_cmd": np.asarray(state.tau_cmd),
            "kp": zeros,
            "kd": zeros,
            "torque": np.asarray(state.tau),
        }

    def motor_health(self) -> dict:
        state, rx_t = self.latest_state()
        now = time.monotonic()
        age = now - rx_t if rx_t > 0 else float("inf")
        stale = age > self.config.state_timeout_s
        link_age = now - self._link_rx_t if self._link_rx_t > 0 else float("inf")
        if stale and rx_t > 0:
            # Two distinct failures share the staleness symptom; say which.
            stale_msg = (
                f"rt servo frozen ({age:.2f}s, link alive)"
                if link_age <= self.config.state_timeout_s
                else f"rt state stream stale ({age:.2f}s)"
            )
        else:
            stale_msg = ""
        fault = self._latched_fault or self._stream_fault or stale_msg
        return {
            # The server's flag VERBATIM. "I cannot see the server" must
            # never be reported as "the arm is not armed" — an armed arm
            # holding 30 N.m/rad behind a stale stream is the dangerous
            # direction. Staleness is its own field (and rides any_fault).
            "armed": bool(state and state.armed),
            "state_fresh": not stale,
            "latched_fault": fault,
            "any_fault": bool(fault) or bool(state and state.faulted),
            "holding": bool(state and state.holding),
            "holding_q": list(state.q_cmd) if state and state.holding else None,
            "backend": self.backend_name,
            "supports_pose_hold": self.supports_pose_hold,
            "supports_cartesian": self.supports_cartesian,
            "state_age_s": age if rx_t > 0 else -1.0,
            "last_cmd_seq": state.last_cmd_seq if state else 0,
            "sent_cmd_seq": self._cmd_seq,
            "online_mask": state.online_mask if state else 0,
            "active_mask": self._active_mask,
            "instance_id": self.config.instance_id,
            "session_id": f"{self.session_id:016x}",
            "arm_epoch": self._arm_epoch,
            "stop_epoch": state.stop_epoch if state else 0,
            "state_seq": state.state_seq if state else 0,
            "state_t_mono_ns": state.t_mono_ns if state else 0,
        }

    # -- internals ------------------------------------------------------------
    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._udp.recv(rtp.REFERENCE_STATE_SIZE + 1)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                if self._reference_mode and len(data) == rtp.REFERENCE_STATE_SIZE:
                    state, state_reference = rtp.unpack_reference_state(data, negotiated=True)
                    state.reference = state_reference
                else:
                    state = rtp.unpack_state(data)
            except ValueError:
                continue
            if state.identity_hash != self._expected_identity or state.n != self.n:
                # Not the arm this session claimed: never let it reach the graph.
                self._stream_fault = "rt state stream identity mismatch"
                continue
            with self._state_lock:
                # CONTENT freshness, not arrival freshness: the server
                # deliberately re-sends the last packet at ~10 Hz while its
                # RT thread is inside a blocking plant call, precisely so the
                # client can tell "link alive, servo busy" from "link dead".
                # Stamping every arrival made a frozen servo read as healthy
                # forever (the client half of that contract was never
                # implemented — audit 2026-07-29).
                now = time.monotonic()
                self._link_rx_t = now
                if self._state is None or state.state_seq != self._state.state_seq:
                    self._state_rx_t = now
                    self._state_history.append((state, now))
                self._state = state

    def _ctl_loop(self) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._tcp.recv(rtp.CTL_SIZE - len(buf))
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                self._latched_fault = self._latched_fault or "rt control session closed"
                return
            buf += chunk
            if len(buf) < rtp.CTL_SIZE:
                continue
            frame, buf = buf[: rtp.CTL_SIZE], b""
            if self._reference_mode:
                try:
                    op, seq, reply = rtp.unpack_reference_control(frame, negotiated=True)
                except ValueError:
                    pass
                else:
                    if op == rtp.CTL_REFERENCE_STATUS or (
                            op == rtp.CTL_GROUP_STATUS and reply['operation']):
                        with self._reference_cv:
                            if seq == self._reference_wait_seq:
                                self._reference_reply = reply
                                self._reference_cv.notify_all()
                    elif op == rtp.CTL_GROUP_STATUS:
                        self.group_events.append(dict(reply, received_at=time.monotonic()))
                    continue
            try:
                msg = rtp.unpack_control(frame)
            except ValueError:
                continue
            if msg.ctl_type == rtp.CTL_FAULT:
                self._latched_fault = f"rt fault {msg.arg}: {msg.text}"
                print(f"[rt_link] FAULT from server — {self._latched_fault}", flush=True)
            elif msg.ctl_type == rtp.CTL_STATUS:
                self._status_q.put(msg)
            elif msg.ctl_type == rtp.CTL_MODE_STATUS:
                try:
                    self._mode_q.put(rtp.unpack_mode_status(frame))
                except ValueError:
                    pass
            elif msg.ctl_type == rtp.CTL_PING:
                try:
                    self._send_control(rtp.CTL_PONG)
                except RtLinkError:
                    # Session is gone; the server's deadman will latch. This
                    # thread exiting cleanly beats a traceback-killed daemon.
                    self._latched_fault = (
                        self._latched_fault or "rt control session closed"
                    )
                    return

    def _send_control(self, ctl_type: int, text: str | bytes = "", arg: int = 0) -> None:
        if ctl_type in (rtp.CTL_DISARM, rtp.CTL_ARM):
            self._upload_cancel.set()
        pkt = rtp.pack_control(
            ctl_type=ctl_type,
            seq=0,
            arg=arg,
            t_mono_ns=time.monotonic_ns(),
            text=text,
        )
        self._send_frame(pkt)

    def _send_frame(self, pkt: bytes, *, sock=None) -> None:
        try:
            # Writers share this socket (node thread + the PONG reply in
            # _ctl_loop); interleaved partial sends would corrupt the framing.
            with self._send_lock:
                (self._tcp if sock is None else sock).sendall(pkt)
        except OSError as exc:
            # socket.timeout / BrokenPipeError are OSError, NOT RtLinkError —
            # unconverted they sailed past every caller's except clause and
            # killed the plant node on the DISARM path of all places (audit
            # 2026-07-29). One guard here fixes every caller.
            raise RtLinkError(f"control link down: {exc}") from exc

    def _control_roundtrip(self, ctl_type: int, *, arg: int = 0) -> rtp.Control:
        while not self._status_q.empty():  # drop stale acks
            self._status_q.get_nowait()
        self._send_control(ctl_type, arg=arg)
        try:
            return self._status_q.get(timeout=self.config.ack_timeout_s)
        except queue.Empty as exc:
            raise RtLinkError("no STATUS ack from server") from exc

    def _await_control(self, wanted: set[int], deadline_s: float) -> rtp.Control | None:
        """Pre-thread handshake read: answers PINGs, returns the first wanted frame."""
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            msg = self._recv_control(deadline_s=max(0.0, deadline - time.monotonic()))
            if msg is None:
                return None
            if msg.ctl_type == rtp.CTL_PING:
                self._send_control(rtp.CTL_PONG)
            elif msg.ctl_type in wanted:
                return msg
        return None

    def _recv_control(self, deadline_s: float) -> rtp.Control | None:
        deadline = time.monotonic() + deadline_s
        buf = b""
        while time.monotonic() < deadline and len(buf) < rtp.CTL_SIZE:
            try:
                chunk = self._tcp.recv(rtp.CTL_SIZE - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                return None
            buf += chunk
        return rtp.unpack_control(buf) if len(buf) == rtp.CTL_SIZE else None


def _demo() -> None:
    """The LOOPBACK RUNG: protocol parity + a live session against the fake
    server, exercising exactly the paths that are dangerous to discover on
    hardware — staleness hold, fault latch, latch-refuses-arm, DISARM+ARM
    recovery. Skips (loudly, rc 0) if the server binary is not built."""
    import dataclasses
    import os
    import shutil
    import struct
    import subprocess
    import sys
    from pathlib import Path

    # ARM_RT_DEMO_HOST points the SAME sequence at a remote server (rung 1:
    # the RT box running --backend fake --n 3) instead of spawning one here.
    remote = os.environ.get("ARM_RT_DEMO_HOST")
    repo = Path(__file__).resolve().parents[3]
    binary = os.environ.get("ARM_RT_SERVER_BIN") or str(repo / "rt" / "build" / "arm_rt_server")
    selfcheck = str(Path(binary).parent / "protocol_selfcheck")
    if remote is None and not Path(binary).exists():
        print(
            "rt_backend: SKIPPED live loopback — build the server first:\n"
            "  cmake -B rt/build rt && cmake --build rt/build",
        )
        return

    # 1. Wire parity: the C++ golden hex must equal ours, byte for byte.
    # (Remote rung: run the box's protocol_selfcheck over ssh and diff instead.)
    if remote is None and Path(selfcheck).exists():
        theirs = subprocess.run(
            [selfcheck], capture_output=True, text=True, check=True
        ).stdout.strip().splitlines()
        from arm_control.plants.remote_rt.protocol import golden_lines

        assert theirs == golden_lines(), "C++/Python protocol drift — fix both, bump VERSION"
        print("rt_backend: protocol parity ok")

    # 2. Live session against the fake plant (local spawn, or the RT box).
    server = None
    if remote is None:
        server = subprocess.Popen(
            [binary, "--backend", "fake", "--n", "3", "--udp-port", "48810",
             "--tcp-port", "48811", "--hold-ms", "100", "--fault-ms", "600",
             "--slew", "5"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        host, udp, tcp = "127.0.0.1", 48810, 48811
    else:
        # Remote servers run default ports and default thresholds (100/1000 ms
        # — the fault-latch wait below covers both 600 and 1000 ms).
        host, udp, tcp = remote, 47800, 47801
    joints = ["j0", "j1", "j2"]

    def config(instance="fake"):
        return RtConfig(host=host, udp_port=udp, tcp_port=tcp, instance_id=instance)

    backend = RtBackend(config(), joints)
    try:
        time.sleep(0.3)
        # Admission refuses the wrong arm and the wrong joint order before any
        # UDP is sent -- a swapped deployment config cannot claim this server.
        for bad, names, why in ((config("fr3_2"), joints, "instance"),
                                (config(), ["j1", "j0", "j2"], "joint order"),
                                (config(""), joints, "provisioned")):
            try:
                RtBackend(bad, names).open()
            except RtLinkError as exc:
                assert why in str(exc), exc
            else:
                raise AssertionError(f"{why} mismatch was admitted")
        # The SERVER enforces the claim too: a client that skips its own checks
        # (or predates them) gets neither a claim nor ARM.
        raw = RtBackend(config(), joints)
        raw._tcp = socket.create_connection((host, tcp), timeout=2.0)
        raw._tcp.settimeout(0.2)
        raw._await_control({rtp.CTL_HELLO}, 2.0)
        ident = rtp.unpack_identity(raw._await_control({rtp.CTL_IDENTITY}, 2.0).raw)
        for field, value, reason in (
                ("instance_id", "fr3_2", "instance id"),
                ("joint_order_hash", ident.joint_order_hash ^ 1, "joint order"),
                ("session_id", ident.session_id ^ 1, "session"),
                ("capabilities", rtp.CAP_CALIBRATED_TASK_SPACE, "capability")):
            wrong = dataclasses.replace(ident, **{field: value})
            raw._send_control(rtp.CTL_CLAIM, text=rtp.pack_identity(wrong))
            ack = raw._await_control({rtp.CTL_STATUS}, 2.0)
            assert not ack.arg & rtp.FLAG_CLAIMED and reason in ack.text, (field, ack)
        raw._send_control(rtp.CTL_ARM)
        ack = raw._await_control({rtp.CTL_STATUS}, 2.0)
        assert not ack.arg & rtp.FLAG_ARMED and "CLAIM" in ack.text, ack
        raw._tcp.close()
        time.sleep(0.3)  # let the server retire that session

        backend.open()
        assert backend.session_id and backend.identity.instance_id == "fake"
        assert backend.backend_name == "fake"
        assert not backend.supports_pose_hold
        # v8: the fake has no geometry, so it does not advertise the Cartesian law, and the link refuses to send one
        assert not backend.supports_cartesian
        base_spec = {"frame": "base", "task_R": np.eye(3).tolist(), "kc": [100.0] * 3 + [10.0] * 3, "dc": [10.0] * 6}
        for spec in (base_spec, {"pose": [0] * 7, "task_R": np.eye(3).tolist(), "kc": [0] * 6, "dc": [0] * 6}):
            try:
                backend.apply_command({"position": np.zeros(3), "cartesian": spec})
                raise AssertionError("a cartesian command reached a server that does not advertise it")
            except RtLinkError as exc:
                assert "CAP_CARTESIAN_TRAJECTORY" in str(exc), exc
        backend.supports_cartesian = True  # even when advertised, a world-frame spec is never sent
        try:
            backend.apply_command({"position": np.zeros(3), "cartesian": {"pose": [0] * 7}})
            raise AssertionError("a world-frame cartesian spec was sent")
        except RtLinkError as exc:
            assert "base-frame" in str(exc), exc
        try:
            backend.apply_command({"position": np.zeros(3), "cartesian": base_spec, "pose_hold": {}})
            raise AssertionError("a command with both tails was sent")
        except RtLinkError as exc:
            assert "not both" in str(exc), exc
        backend.supports_cartesian = False
        assert not backend._control_roundtrip(rtp.CTL_STATUS).arg & rtp.FLAG_ARMED
        backend.set_active_mask(0b011)
        backend.enable_all()
        backend.set_active_mask(0b111)
        assert backend._control_roundtrip(rtp.CTL_STATUS).arg & rtp.FLAG_ARMED
        try:
            backend.set_active_mask(0b011)
            raise AssertionError("active slots must not be removed while armed")
        except RuntimeError:
            pass
        time.sleep(0.05)
        assert backend.motor_health()["online_mask"] == 0b111

        # Track a step target; the fake integrator must actually converge.
        target = np.array([0.5, -0.3, 0.2])
        kp, kd = np.full(3, 60.0), np.full(3, 10.0)
        for _ in range(150):  # 1.5 s at 100 Hz
            backend.apply_command(
                {"position": target, "velocity": np.zeros(3),
                 "torque": np.zeros(3), "kp": kp, "kd": kd}
            )
            time.sleep(0.01)
        state, _ = backend.latest_state()
        err = float(np.max(np.abs(np.asarray(state.q) - target)))
        assert state.armed and not state.holding, backend.motor_health()
        assert err < 0.05, f"fake plant not tracking (err {err:.3f} rad)"
        # Nonzero RT echo must survive the Python backend mapping. The hold
        # below must then report zero despite retaining this incoming packet.
        desired_velocity = np.array([.125, -.25, .375])
        for _ in range(10):
            backend.apply_command(dict(position=target, velocity=desired_velocity,
                                       torque=np.zeros(3), kp=kp, kd=kd))
            time.sleep(.01)
        assert np.array_equal(backend.motor_state()['velocity_cmd'], desired_velocity)

        # Staleness -> HOLD (stop commanding past hold-ms, before fault-ms).
        # Unsupported pose hold, oversized joint commands and replays must neither
        # change the accepted command nor refresh its deadman.
        bound = dict(n=3, t_mono_ns=time.monotonic_ns(), q_des=target,
                     qd_des=np.zeros(3), tau_ff=np.zeros(3), kp=kp, kd=kd,
                     session_id=backend.session_id, arm_epoch=backend._arm_epoch)
        replay = rtp.pack_command(seq=backend._cmd_seq, **bound)
        unsupported = rtp.pack_command(seq=backend._cmd_seq + 1, **bound,
            pose_hold=dict(id=1, kc=[100.]*6, dc=[10.]*6,
                           nullspace_kp=0., nullspace_kd=3.))
        # Fresh sequence numbers, but another session or a stale ARM epoch.
        foreign = rtp.pack_command(seq=backend._cmd_seq + 2,
                                   **(bound | dict(session_id=backend.session_id ^ 1)))
        stale_epoch = rtp.pack_command(seq=backend._cmd_seq + 3,
                                       **(bound | dict(arm_epoch=backend._arm_epoch - 1)))
        nan = bytearray(rtp.pack_command(seq=backend._cmd_seq + 4, **bound))
        struct.pack_into("<d", nan, 32, float("nan"))
        rejected = (replay, replay + b"invalid", unsupported, foreign, stale_epoch, bytes(nan))
        for i in range(36):
            backend._udp.send(rejected[i % len(rejected)])
            time.sleep(.01)
        state, _ = backend.latest_state()
        assert state.holding and not state.faulted, backend.motor_health()
        assert np.array_equal(backend.motor_state()['velocity_cmd'], np.zeros(3))
        assert state.last_cmd_seq == backend._cmd_seq
        held = np.asarray(state.q)

        # A passive observer sees the same arm and changes nothing; one that
        # expects another instance gets no state at all.
        observer = RtObserver(host, udp, "fake", joints, lease_ms=500)
        stranger = RtObserver(host, udp, "fr3_2", joints, lease_ms=500)
        observer.open()
        stranger.open()
        track = {"position": target, "velocity": np.zeros(3),
                 "torque": np.zeros(3), "kp": kp, "kd": kd}
        for _ in range(30):
            backend.apply_command(track)
            time.sleep(0.01)
        seen, _ = observer.latest_state()
        assert seen is not None and seen.armed and seen.session_id == backend.session_id
        assert stranger.latest_state()[0] is None and stranger.identity_mismatches > 0
        observer.close()
        stranger.close()
        for _ in range(10):
            backend.apply_command(track)
            time.sleep(0.01)
        state, _ = backend.latest_state()
        assert state.armed and not state.faulted and not state.holding
        assert state.last_cmd_seq == backend._cmd_seq, backend.motor_health()

        # Resume -> tracking again.
        for _ in range(30):
            backend.apply_command(
                {"position": target, "velocity": np.zeros(3),
                 "torque": np.zeros(3), "kp": kp, "kd": kd}
            )
            time.sleep(0.01)
        state, _ = backend.latest_state()
        assert not state.holding and state.armed

        # Prolonged silence -> FAULT latch; commands are ignored; ARM refused.
        time.sleep(0.9 if remote is None else 1.4)
        state, _ = backend.latest_state()
        assert state.faulted and state.fault_code == rtp.FAULT_CMD_LOST
        assert np.array_equal(backend.motor_state()['velocity_cmd'], np.zeros(3))
        try:
            backend.enable_all()
            raise AssertionError("ARM must be refused while latched")
        except RtLinkError:
            pass
        # DISARM+ARM clears — the one recovery path.
        backend.safe_stop()
        backend.enable_all()
        time.sleep(0.05)  # let the next state datagram reflect the cleared latch
        state, _ = backend.latest_state()
        # After re-arm with no fresh commands the server holds (by design).
        assert state is not None and state.armed and not state.faulted

        # A BACK-TO-BACK DISARM->ARM (can complete between two 1 kHz samples)
        # must still be a full epoch — gains reset, plant retry, hold pose
        # recapture — and tracking must resume afterwards. Guards the
        # arm-generation counter: sampling the armed LEVEL misses this edge.
        # Raw roundtrip on purpose: the verified safe_stop WAITS for the RT
        # loop to observe the disarm, which would guarantee the edge is seen
        # and never exercise the sub-tick pair.
        backend._control_roundtrip(rtp.CTL_DISARM)
        backend.enable_all()
        old_epoch = backend._arm_epoch
        target2 = np.array([0.1, 0.2, -0.2])
        for _ in range(120):
            backend.apply_command(
                {"position": target2, "velocity": np.zeros(3),
                 "torque": np.zeros(3), "kp": kp, "kd": kd}
            )
            time.sleep(0.01)
        state, _ = backend.latest_state()
        err2 = float(np.max(np.abs(np.asarray(state.q) - target2)))
        assert state.armed and not state.faulted and not state.holding
        assert err2 < 0.05, f"no tracking after fast re-arm (err {err2:.3f} rad)"

        # Session restart isolation: after a confirmed stop and reconnect, the
        # old session's packets (right epoch number, fresh seq) are not admitted.
        old_session = backend.session_id
        assert backend.safe_stop(), "disarm not confirmed by stop epoch"
        backend.close()
        backend = RtBackend(config(), joints)
        backend.open()
        assert backend.session_id != old_session
        backend.enable_all()
        # A new client numbers above the server's surviving last_cmd_seq, so
        # that stale value can never read as acceptance of its commands.
        prior, _ = backend.latest_state()
        assert epoch_reached(backend._cmd_seq, prior.last_cmd_seq) and backend._cmd_seq > 0
        time.sleep(0.05)
        before, _ = backend.latest_state()
        ghost = rtp.pack_command(n=3, seq=10**6, t_mono_ns=0, q_des=target,
                                 qd_des=np.zeros(3), tau_ff=np.zeros(3), kp=kp, kd=kd,
                                 session_id=old_session, arm_epoch=backend._arm_epoch)
        for _ in range(20):
            backend._udp.send(ghost)
            time.sleep(0.01)
        after, _ = backend.latest_state()
        assert after.last_cmd_seq == before.last_cmd_seq and after.holding, backend.motor_health()
        assert old_epoch != backend._arm_epoch
        print(
            f"rt_backend: ok (tracked to {err * 1e3:.1f} mrad, hold at "
            f"{np.round(held, 3).tolist()}, fault latch + DISARM/ARM recovery)"
        )
    finally:
        backend.close()
        log = ""
        if server is not None:
            server.terminate()
            try:
                log = server.communicate(timeout=3)[0]
            except subprocess.TimeoutExpired:
                server.kill()
    if server is not None:
        # Each rejection must have happened for ITS reason, not a neighbour's.
        for reason in ("claim refused: instance id mismatch",
                       "claim refused: joint order mismatch",
                       "claim refused: wrong session",
                       "is not the claimed control session",
                       "from ARM epoch"):
            assert reason in log, f"server never logged {reason!r}"
    _ = shutil, sys  # keep imports honest if asserts are stripped


if __name__ == "__main__":
    _demo()
