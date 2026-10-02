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
import time
from pathlib import Path

from arm_control.execution import codec
from arm_control.execution.contract import (
    CONTRACT_VERSION, Authority, Clock, ExecutionStatus, Receipt, RobotState, Stamp,
    TrajectoryRequest,
)

#: Upper bound on one message; a long 7-DOF trajectory is well under 1 MB.
MAX_MESSAGE = 16 << 20
#: Waits are repeated short requests, so the channel is free between them
#: (another thread can cancel or DISARM) and the supervisor notices a dropped
#: client promptly.
WAIT_SLICE_S = 0.25


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
            # The request has been sent, so a reply that cannot be read leaves its outcome UNKNOWN to the caller: report
            # it as a SupervisorError (the same class as a lost reply), never a ValueError a caller could take for
            # "refused", and drop the channel (its framing can no longer be trusted).
            try:
                msg = json.loads(reply)
                reply_id, ok = msg.get("id"), msg["ok"]
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self._close()
                raise SupervisorError(f"{op}: malformed reply ({type(exc).__name__}); outcome unknown") from exc
            if reply_id != rid:
                self._close()
                raise SupervisorError(f"{op}: reply out of order; outcome unknown")
        if ok:
            try:
                return msg["result"]
            except KeyError as exc:
                self._close()
                raise SupervisorError(f"{op}: reply without a result; outcome unknown") from exc
        error = {"refused": Refused, "timeout": TimeoutError}.get(msg.get("kind"), SupervisorError)
        raise error(msg.get("error", "unspecified error"))

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
        self.extensions = dict(hello.get("extensions") or {})

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
        return self._await_authority(self._channel.call("arm")["gen"], True, timeout_s)

    def disarm(self, timeout_s: float = 5.0) -> Authority:
        """DISARM and wait (bounded) for the stop to be confirmed."""
        return self._await_authority(self._channel.call("disarm")["gen"], False, timeout_s)

    def _await_authority(self, gen: int, armed: bool, timeout_s: float) -> Authority:
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            got = self._channel.call("await_authority", wait_s=WAIT_SLICE_S, gen=gen,
                                     armed=armed, wait_for_s=min(WAIT_SLICE_S, max(left, 0.0)))
            if got["settled"]:
                return codec.authority_from(got["authority"])
            if left <= 0:
                raise TimeoutError(f"{'ARM' if armed else 'DISARM'} not confirmed within "
                                   f"{timeout_s}s; check authority()")

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
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            status = codec.status_from(self._channel.call(
                "status", wait_s=WAIT_SLICE_S, execution_id=execution_id,
                wait_for_s=min(WAIT_SLICE_S, max(left, 0.0))))
            if status.terminal:
                return status
            if left <= 0:
                raise TimeoutError(
                    f"{execution_id} still {status.phase.value} after {timeout_s}s")

    def cancel(self, execution_id: str, reason: str = "cancelled",
               timeout_s: float = 2.0) -> ExecutionStatus:
        """Cancel one execution and wait (bounded) for its terminal status."""
        self._channel.call("cancel", execution_id=execution_id, reason=reason)
        return self.wait(execution_id, timeout_s)

    def receipts(self, execution_id: str) -> tuple[Receipt, ...]:
        return tuple(codec.receipt_from(r) for r in
                     self._channel.call("receipts", execution_id=execution_id))

    # -- negotiated controller/1 -----------------------------------------------
    def _controller_call(self, op, **args):
        if self.extensions.get('controller') != 1:
            raise Refused('supervisor does not advertise controller/1')
        return self._channel.call(op, **args)

    def controller_presets(self):
        return tuple(codec.preset_from(p) for p in self._controller_call('controller_presets'))

    def controller_state(self):
        return codec.controller_state_from(self._controller_call('controller_state'))

    def select_control_preset(self, preset_id: str):
        return codec.controller_state_from(self._controller_call(
            'controller_select', preset_id=preset_id))

    def set_payload(self, payload):
        return codec.payload_state_from(self._controller_call(
            'controller_payload', payload=codec.encode(payload)))

    # -- negotiated stream/1 ----------------------------------------------------
    def _stream_call(self, op, **args):
        if self.extensions.get('stream') != 1:
            raise Refused('supervisor does not advertise stream/1; no raw-jog fallback')
        return self._channel.call(op, **args)

    def open_stream(self, request):
        """Returns REQUESTED; trusted admission continues asynchronously."""
        return codec.stream_view_from(self._stream_call('stream_open', request=codec.encode(request)))

    def stream_state(self, execution_id):
        """Pure observation; never acquires a splice challenge."""
        return codec.stream_view_from(self._stream_call('stream_state', execution_id=execution_id))

    def stream_anchor(self, execution_id):
        return codec.anchor_from(self._stream_call('stream_anchor', execution_id=execution_id))

    def revise_stream(self, request):
        return codec.revision_status_from(self._stream_call('stream_revise', request=codec.encode(request)))

    def stream_revision(self, execution_id, request_id):
        return codec.revision_status_from(self._stream_call(
            'stream_revision', execution_id=execution_id, request_id=request_id))

    def abort_stream_revision(self, execution_id, request_id):
        return codec.revision_status_from(self._stream_call(
            'stream_abort', execution_id=execution_id, request_id=request_id))

    # -- lifetime ----------------------------------------------------------------
    def close(self) -> None:
        """Ends this session: ownership, if held, is released by the supervisor
        (its live executions are cancelled; the robot holds)."""
        self._channel.close()

    def __enter__(self) -> ArmClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
