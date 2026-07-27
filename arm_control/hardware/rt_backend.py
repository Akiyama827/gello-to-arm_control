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
      state_timeout_s: 0.5
"""
from __future__ import annotations

import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from arm_control import rt_protocol as rtp


class RtLinkError(RuntimeError):
    """Server unreachable, refused, or speaking a different protocol."""


@dataclass(frozen=True)
class RtConfig:
    host: str = "127.0.0.1"
    udp_port: int = 47800
    tcp_port: int = 47801
    state_timeout_s: float = 0.5
    ack_timeout_s: float = 2.0

    @classmethod
    def from_config(cls, cfg) -> "RtConfig":
        raw = dict(cfg.get("rt") or {})
        return cls(
            host=str(raw.get("host", "127.0.0.1")),
            udp_port=int(raw.get("udp_port", 47800)),
            tcp_port=int(raw.get("tcp_port", 47801)),
            state_timeout_s=float(raw.get("state_timeout_s", 0.5)),
            ack_timeout_s=float(raw.get("ack_timeout_s", 2.0)),
        )


class RtBackend:
    """One arm behind an ``arm_rt_server``."""

    def __init__(self, config: RtConfig, joint_names: list[str]) -> None:
        self.config = config
        self.joint_names = list(joint_names)
        self.n = len(joint_names)
        self.backend_name = ""
        self._tcp: socket.socket | None = None
        self._udp: socket.socket | None = None
        self._rx_thread: threading.Thread | None = None
        self._ctl_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._state: rtp.State | None = None
        self._state_rx_t = 0.0
        self._status_q: queue.Queue = queue.Queue()
        self._latched_fault = ""
        self._cmd_seq = 0

    @classmethod
    def from_config(cls, cfg) -> "RtBackend":
        from arm_control.config import arm_joints

        return cls(RtConfig.from_config(cfg), arm_joints(cfg))

    @property
    def num_motors(self) -> int:
        return self.n

    # -- lifecycle ------------------------------------------------------------
    def open(self) -> None:
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
        hello = self._recv_control(deadline_s=cfg.ack_timeout_s)
        if hello is None or hello.ctl_type != rtp.CTL_HELLO:
            raise RtLinkError("no HELLO from server (protocol/version mismatch?)")
        if hello.arg != self.n:
            raise RtLinkError(
                f"server drives {hello.arg} joints, config says {self.n} "
                f"({self.joint_names})"
            )
        self.backend_name = hello.text
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.connect((cfg.host, cfg.udp_port))
        self._udp.settimeout(0.2)
        self._stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        self._ctl_thread = threading.Thread(target=self._ctl_loop, daemon=True)
        self._ctl_thread.start()
        # Prime the state stream: the server replies to the source address of
        # the last command datagram, so send one zero-authority packet now
        # (content is ignored while disarmed; the ADDRESS is the payload).
        self._send_command_raw(np.zeros(self.n), np.zeros(self.n), np.zeros(self.n),
                               np.zeros(self.n), np.zeros(self.n))
        print(
            f"[rt_link] connected to {cfg.host} — backend '{self.backend_name}', "
            f"{self.n} joints",
            flush=True,
        )

    def enable_all(self) -> None:
        """Arm. The server holds the current pose until commands flow."""
        status = self._control_roundtrip(rtp.CTL_ARM)
        # FAULTED in the ack is the refusal — ARMED alone is not consent: the
        # server keeps its ARMED flag while fault-HOLDING a latched arm.
        if status.arg & rtp.FLAG_FAULTED or not status.arg & rtp.FLAG_ARMED:
            raise RtLinkError(f"arm refused: {status.text or 'latched fault'}")
        self._latched_fault = ""

    def safe_stop(self) -> None:
        """Disarm (also clears a server-side fault latch, matching the bench
        bridges' explicit DISARM->ARM recovery cycle)."""
        if self._tcp is None:
            return
        try:
            self._control_roundtrip(rtp.CTL_DISARM)
        except RtLinkError as exc:
            print(f"[rt_link] disarm: {exc}", flush=True)

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
    def apply_command(self, command: dict[str, Any]) -> None:
        """Stream one joint-servo word (the full bridge contract — the server
        applies torque clamp + slew; nothing is dropped on this side)."""
        if self._udp is None:
            return
        self._send_command_raw(
            np.asarray(command["position"], dtype=float),
            np.asarray(command.get("velocity", np.zeros(self.n)), dtype=float),
            np.asarray(command.get("torque", np.zeros(self.n)), dtype=float),
            np.asarray(command.get("kp", np.zeros(self.n)), dtype=float),
            np.asarray(command.get("kd", np.zeros(self.n)), dtype=float),
        )

    def _send_command_raw(self, q, qd, tau, kp, kd) -> None:
        self._cmd_seq += 1
        pkt = rtp.pack_command(
            n=self.n,
            seq=self._cmd_seq,
            t_mono_ns=time.monotonic_ns(),
            q_des=q[: self.n],
            qd_des=qd[: self.n],
            tau_ff=tau[: self.n],
            kp=kp[: self.n],
            kd=kd[: self.n],
        )
        try:
            self._udp.send(pkt)
        except OSError:
            pass  # transient; staleness accounting reports it

    # -- feedback -------------------------------------------------------------
    def latest_state(self) -> tuple[rtp.State | None, float]:
        with self._state_lock:
            return self._state, self._state_rx_t

    def motor_state(self) -> dict[str, np.ndarray]:
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
            "velocity_cmd": zeros,
            # The servo's own post-clamp post-slew output — the tau_J_d
            # analogue, and the honest number for tracking plots.
            "torque_cmd": np.asarray(state.tau_cmd),
            "kp": zeros,
            "kd": zeros,
            "torque": np.asarray(state.tau),
        }

    def motor_health(self) -> dict:
        state, rx_t = self.latest_state()
        age = time.monotonic() - rx_t if rx_t > 0 else float("inf")
        stale = age > self.config.state_timeout_s
        armed = bool(state and state.armed) and not stale
        fault = self._latched_fault or (
            f"rt state stream stale ({age:.2f}s)" if stale and rx_t > 0 else ""
        )
        return {
            "armed": armed,
            "latched_fault": fault,
            "any_fault": bool(fault) or bool(state and state.faulted),
            "holding": bool(state and state.holding),
            "backend": self.backend_name,
            "state_age_s": age if rx_t > 0 else -1.0,
            "last_cmd_seq": state.last_cmd_seq if state else 0,
            "sent_cmd_seq": self._cmd_seq,
        }

    # -- internals ------------------------------------------------------------
    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._udp.recv(rtp.STATE_SIZE)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                state = rtp.unpack_state(data)
            except ValueError:
                continue
            with self._state_lock:
                self._state = state
                self._state_rx_t = time.monotonic()

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
            try:
                msg = rtp.unpack_control(frame)
            except ValueError:
                continue
            if msg.ctl_type == rtp.CTL_FAULT:
                self._latched_fault = f"rt fault {msg.arg}: {msg.text}"
                print(f"[rt_link] FAULT from server — {self._latched_fault}", flush=True)
            elif msg.ctl_type == rtp.CTL_STATUS:
                self._status_q.put(msg)
            elif msg.ctl_type == rtp.CTL_PING:
                self._send_control(rtp.CTL_PONG)

    def _send_control(self, ctl_type: int, text: str = "") -> None:
        pkt = rtp.pack_control(
            ctl_type=ctl_type, seq=0, arg=0, t_mono_ns=time.monotonic_ns(), text=text
        )
        self._tcp.sendall(pkt)

    def _control_roundtrip(self, ctl_type: int) -> rtp.Control:
        while not self._status_q.empty():  # drop stale acks
            self._status_q.get_nowait()
        self._send_control(ctl_type)
        try:
            return self._status_q.get(timeout=self.config.ack_timeout_s)
        except queue.Empty as exc:
            raise RtLinkError("no STATUS ack from server") from exc

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
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    binary = os.environ.get("ARM_RT_SERVER_BIN") or str(repo / "rt" / "build" / "arm_rt_server")
    selfcheck = str(Path(binary).parent / "protocol_selfcheck")
    if not Path(binary).exists():
        print(
            "rt_backend: SKIPPED live loopback — build the server first:\n"
            "  cmake -B rt/build rt && cmake --build rt/build",
        )
        return

    # 1. Wire parity: the C++ golden hex must equal ours, byte for byte.
    if Path(selfcheck).exists():
        theirs = subprocess.run(
            [selfcheck], capture_output=True, text=True, check=True
        ).stdout.strip().splitlines()
        from arm_control.rt_protocol import golden_lines

        assert theirs == golden_lines(), "C++/Python protocol drift — fix both, bump VERSION"
        print("rt_backend: protocol parity ok")

    # 2. Live loopback against the fake plant on high ports.
    server = subprocess.Popen(
        [binary, "--backend", "fake", "--n", "3", "--udp-port", "48810",
         "--tcp-port", "48811", "--hold-ms", "100", "--fault-ms", "600",
         "--slew", "5"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    backend = RtBackend(
        RtConfig(host="127.0.0.1", udp_port=48810, tcp_port=48811), ["j0", "j1", "j2"]
    )
    try:
        time.sleep(0.3)
        backend.open()
        assert backend.backend_name == "fake"
        backend.enable_all()

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

        # Staleness -> HOLD (stop commanding past hold-ms, before fault-ms).
        time.sleep(0.3)
        state, _ = backend.latest_state()
        assert state.holding and not state.faulted, backend.motor_health()
        held = np.asarray(state.q)

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
        time.sleep(0.9)
        state, _ = backend.latest_state()
        assert state.faulted and state.fault_code == rtp.FAULT_CMD_LOST
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
        print(
            f"rt_backend: ok (tracked to {err * 1e3:.1f} mrad, hold at "
            f"{np.round(held, 3).tolist()}, fault latch + DISARM/ARM recovery)"
        )
    finally:
        backend.close()
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
    _ = shutil, sys  # keep imports honest if asserts are stripped


if __name__ == "__main__":
    _demo()
