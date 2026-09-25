"""Minimal application SDK (CTRL-022): one robot through its headless supervisor.

    from arm_control.sdk import ArmClient
    with ArmClient("fr3_1") as arm:
        arm.claim(); arm.arm()
        status = arm.execute(request)            # a contract TrajectoryRequest
        arm.wait(status.execution_id, timeout_s=10)

Stdlib, numpy and the execution contract only: importing this loads no Dora,
Arrow, MuJoCo, planner or supervisor code. ``ArmClient`` speaks the
supervisor's named operations through a channel; the local Unix socket is the
one transport today. Connecting has no side effect on the robot: nothing is
claimed, armed or moved until the caller asks.
"""
from __future__ import annotations

import itertools
import json
import os
import socket
import tempfile
import threading
from pathlib import Path

from arm_control.execution import codec
from arm_control.execution.contract import (
    CONTRACT_VERSION, Authority, Clock, ExecutionStatus, Receipt, RobotState, Stamp,
    TrajectoryRequest,
)

#: Upper bound on one message; a long 7-DOF trajectory is well under 1 MB.
MAX_MESSAGE = 16 << 20


class SupervisorError(RuntimeError):
    """The supervisor could not be reached or answered with an error."""


class Refused(SupervisorError):
    """The supervisor declined the operation; nothing changed."""


def default_socket_path(instance_id: str) -> Path:
    """Per-user socket for one robot's supervisor (directory mode 0700)."""
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.join(
        tempfile.gettempdir(), f"arm_control-{os.getuid()}")
    return Path(base) / "arm_control" / f"{instance_id}.sock"


class SocketChannel:
    """Newline-delimited JSON request/response over the supervisor's socket.
    One outstanding request at a time; a timeout closes the channel, since
    the reply framing can no longer be trusted."""

    def __init__(self, path: str | os.PathLike, *, timeout_s: float = 5.0) -> None:
        self.timeout_s = float(timeout_s)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout_s)
        try:
            self._sock.connect(os.fspath(path))
        except OSError as exc:
            self._sock.close()
            raise SupervisorError(f"no supervisor at {path}: {exc}") from exc
        self._rx = self._sock.makefile("rb")

    def call(self, op: str, wait_s: float = 0.0, **args):
        with self._lock:
            if self._sock is None:
                raise SupervisorError("channel is closed")
            rid = next(self._ids)
            line = json.dumps({"id": rid, "op": op, "args": args}).encode() + b"\n"
            try:
                self._sock.settimeout(self.timeout_s + wait_s)
                self._sock.sendall(line)
                reply = self._rx.readline(MAX_MESSAGE + 1)
            except OSError as exc:
                self._close()
                raise SupervisorError(f"{op}: {exc}") from exc
            if not reply.endswith(b"\n"):
                self._close()
                raise SupervisorError(f"{op}: supervisor closed the connection")
        msg = json.loads(reply)
        if msg.get("id") != rid:
            raise SupervisorError(f"{op}: reply out of order")
        if msg["ok"]:
            return msg["result"]
        error = {"refused": Refused, "timeout": TimeoutError}.get(msg.get("kind"),
                                                                  SupervisorError)
        raise error(msg["error"])

    def _close(self) -> None:
        if self._sock is not None:
            self._rx.close()
            self._sock.close()
            self._sock = None

    def close(self) -> None:
        with self._lock:
            self._close()


class ArmClient:
    """One robot, by instance id, through its supervisor."""

    def __init__(self, instance_id: str, *, socket_path: str | os.PathLike | None = None,
                 timeout_s: float = 5.0, channel=None) -> None:
        self.instance_id = instance_id
        self.timeout_s = float(timeout_s)
        self._channel = channel or SocketChannel(
            socket_path or default_socket_path(instance_id), timeout_s=timeout_s)
        hello = self._channel.call("hello")
        if hello["contract_version"] != CONTRACT_VERSION:
            self.close()
            raise SupervisorError(f"supervisor implements contract "
                                  f"v{hello['contract_version']}, SDK v{CONTRACT_VERSION}")
        self.identity = codec.identity_from(hello["identity"])
        if self.identity.instance_id != instance_id:
            self.close()
            raise SupervisorError(f"supervisor serves {self.identity.instance_id!r}, "
                                  f"not {instance_id!r}")
        self.capabilities = codec.capabilities_from(hello["capabilities"])
        self.session: str = hello["session"]

    # -- observation and time --------------------------------------------------
    def now(self) -> Stamp:
        """The supervisor's clock, the one ``admit_by`` must use."""
        return codec.stamp_from(self._channel.call("now"))

    def deadline(self, seconds: float) -> Stamp:
        now = self.now()
        return Stamp(now.ns + int(seconds * 1e9), Clock.PC, now.domain)

    def state(self) -> RobotState:
        return codec.state_from(self._channel.call("state"))

    def authority(self) -> Authority:
        return codec.authority_from(self._channel.call("authority"))

    # -- ownership and authority -----------------------------------------------
    def claim(self) -> Authority:
        return codec.authority_from(self._channel.call("claim"))

    def unclaim(self) -> Authority:
        return codec.authority_from(self._channel.call("unclaim"))

    def arm(self, timeout_s: float = 5.0) -> Authority:
        """ARM and wait (bounded) until the controller is ready to execute."""
        return codec.authority_from(
            self._channel.call("arm", wait_s=timeout_s, timeout_s=timeout_s))

    def disarm(self, timeout_s: float = 5.0) -> Authority:
        """DISARM and wait (bounded) for the stop to be confirmed."""
        return codec.authority_from(
            self._channel.call("disarm", wait_s=timeout_s, timeout_s=timeout_s))

    # -- execution ---------------------------------------------------------------
    def execute(self, request: TrajectoryRequest) -> ExecutionStatus:
        """Submit; returns once admitted or refused (``status.phase``). Motion
        proceeds in the supervisor; follow it with ``status``/``wait``."""
        return codec.status_from(self._channel.call("submit", request=codec.encode(request)))

    def release(self, execution_id: str) -> ExecutionStatus:
        return codec.status_from(self._channel.call("release", execution_id=execution_id))

    def status(self, execution_id: str) -> ExecutionStatus:
        return codec.status_from(self._channel.call("status", execution_id=execution_id))

    def wait(self, execution_id: str, timeout_s: float) -> ExecutionStatus:
        """Block until terminal, at most ``timeout_s``; TimeoutError otherwise.
        Never cancels: a timed-out wait leaves the execution running."""
        status = codec.status_from(self._channel.call(
            "status", wait_s=timeout_s, execution_id=execution_id, wait_for_s=timeout_s))
        if not status.terminal:
            raise TimeoutError(f"{execution_id} still {status.phase.value} after {timeout_s}s")
        return status

    def cancel(self, execution_id: str, reason: str = "cancelled",
               timeout_s: float = 2.0) -> ExecutionStatus:
        """Cancel one execution and wait (bounded) for its terminal status."""
        self._channel.call("cancel", execution_id=execution_id, reason=reason)
        return self.wait(execution_id, timeout_s)

    def receipts(self, execution_id: str) -> tuple[Receipt, ...]:
        return tuple(codec.receipt_from(r) for r in
                     self._channel.call("receipts", execution_id=execution_id))

    # -- lifetime ----------------------------------------------------------------
    def close(self) -> None:
        """Ends this session: ownership, if held, is released by the supervisor
        (its live executions are cancelled; the robot holds)."""
        self._channel.close()

    def __enter__(self) -> ArmClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
