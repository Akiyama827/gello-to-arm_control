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


def _dec(decoder, value):
    """Decode a supervisor reply. The request has already been sent, so a reply that parsed as JSON but does not fit the
    contract (unknown enum, missing or extra field) leaves the outcome UNKNOWN: SupervisorError, never the ValueError,
    KeyError or TypeError a caller could take for 'refused / nothing sent'."""
    try:
        return decoder(value)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SupervisorError(f"unreadable reply ({type(exc).__name__}: {exc}); outcome unknown") from exc


def _items(decoder, values):
    if not isinstance(values, list):
        raise ValueError(f"expected a list, got {type(values).__name__}")
    return tuple(decoder(v) for v in values)


def _mapping(value):
    if not isinstance(value, dict):
        raise ValueError(f"expected an object, got {type(value).__name__}")
    return value


def _mappings(value):
    return [_mapping(v) for v in _items(lambda v: v, value)]


def _need(mapping, *keys):
    for key in keys:
        mapping[key]
    return mapping


_HISTORY_SAMPLE = ("t_mono_ns", "state_seq", "age_ns", "session_id", "arm_epoch", "stop_epoch", "armed", "faulted", "q", "dq")


def _history(value):
    value = _need(_mapping(value), "clock_domain", "now_ns", "authority", "samples")
    if value["clock_domain"] is not None and not isinstance(value["clock_domain"], str):
        raise ValueError("clock_domain is neither text nor null")
    _need(_mapping(value["authority"]), "session_id", "arm_epoch")
    for sample in _mappings(value["samples"]):
        _need(sample, *_HISTORY_SAMPLE)
        if not isinstance(sample["q"], list) or not isinstance(sample["dq"], list):
            raise ValueError("sample joints are not lists")
    return value


def _trace(value):
    value = _need(_mapping(value), "samples", "latest_seq", "dropped", "errors", "capacity")
    for key in ("latest_seq", "dropped", "errors", "capacity"):
        if type(value[key]) is not int:
            raise ValueError(f"{key} is not an integer")
    for sample in _mappings(value["samples"]):
        if type(_need(sample, "seq", "pc_monotonic", "plan_id", "q_rt")["seq"]) is not int:
            raise ValueError("sample seq is not an integer")
    return value


def _observation(value):
    value = _need(_mapping(value), "schema", "samples", "latest_seq", "dropped", "errors", "capacity", "rate_hz")
    if value["schema"] != "observe/1":
        raise ValueError(f"unexpected observation schema {value['schema']!r}")
    for key in ("latest_seq", "dropped", "errors", "capacity"):
        if type(value[key]) is not int:
            raise ValueError(f"{key} is not an integer")
    if isinstance(value["rate_hz"], bool) or not isinstance(value["rate_hz"], (int, float)):
        raise ValueError("rate_hz is not a number")
    for sample in _mappings(value["samples"]):
        sample = _need(sample, "seq", "sim_time_s", "observed_ns", "model_revision", "actor_q")
        if type(sample["seq"]) is not int:
            raise ValueError("observation seq is not an integer")
        _mapping(sample["actor_q"])
    return value


def _generation(reply):
    gen = reply["gen"]
    if type(gen) is not int:
        raise ValueError(f"request generation {gen!r} is not an integer")
    return gen


def _settled(reply):
    settled = reply["settled"]
    if type(settled) is not bool:
        raise ValueError(f"settled {settled!r} is not a boolean")
    return settled, (codec.authority_from(reply["authority"]) if settled else None)


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
                raise SupervisorError(f"{op}: supervisor closed the connection"
                                      + (" (or sent an oversize reply)" if len(reply) > MAX_MESSAGE else ""))
            # The request has been sent, so a reply that cannot be read leaves its outcome UNKNOWN to the caller: report
            # it as a SupervisorError (the same class as a lost reply), never a ValueError a caller could take for
            # "refused", and drop the channel (its framing can no longer be trusted).
            try:
                msg = json.loads(reply)
                reply_id, ok = msg.get("id"), msg["ok"]
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
                self._close()
                raise SupervisorError(f"{op}: malformed reply ({type(exc).__name__}); outcome unknown") from exc
            if reply_id != rid or type(reply_id) is not int:
                self._close()
                raise SupervisorError(f"{op}: reply out of order; outcome unknown")
            if ok is True:
                if "result" not in msg:
                    self._close()
                    raise SupervisorError(f"{op}: reply without a result; outcome unknown")
                return msg["result"]
            if ok is not False:
                self._close()
                raise SupervisorError(f"{op}: malformed reply (ok={ok!r}); outcome unknown")
            kind, text = msg.get("kind"), msg.get("error", "unspecified error")
            # `kind` may be any JSON value (a list is unhashable): compare, never hash it
            error = Refused if kind == "refused" else TimeoutError if kind == "timeout" else SupervisorError
        raise error(text if isinstance(text, str) else repr(text))

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
        try:
            hello = self._channel.call("hello")
            if hello["contract_version"] != CONTRACT_VERSION:
                raise SupervisorError(f"supervisor implements contract "
                                      f"v{hello['contract_version']}, SDK v{CONTRACT_VERSION}")
            self.identity = _dec(codec.identity_from, hello["identity"])
            if self.identity.instance_id != instance_id:
                raise SupervisorError(f"supervisor serves {self.identity.instance_id!r}, "
                                      f"not {instance_id!r}")
            self.capabilities = _dec(codec.capabilities_from, hello["capabilities"])
            self.session: str = hello["session"]
            self.extensions = dict(hello.get("extensions") or {})
        except (SupervisorError, KeyError, TypeError, AttributeError, ValueError) as exc:
            self.close()  # never leak the socket from a failed handshake
            if isinstance(exc, SupervisorError):
                raise
            raise SupervisorError(f"unreadable hello ({type(exc).__name__}: {exc})") from exc

    # -- observation and time --------------------------------------------------
    def now(self) -> Stamp:
        """The supervisor's clock, the one ``admit_by`` must use."""
        return _dec(codec.stamp_from, self._channel.call("now"))

    def deadline(self, seconds: float) -> Stamp:
        now = self.now()
        return Stamp(now.ns + int(seconds * 1e9), Clock.PC, now.domain)

    def state(self) -> RobotState:
        return _dec(codec.state_from, self._channel.call("state"))

    def authority(self) -> Authority:
        return _dec(codec.authority_from, self._channel.call("authority"))

    # -- ownership and authority -----------------------------------------------
    def claim(self) -> Authority:
        return _dec(codec.authority_from, self._channel.call("claim"))

    def unclaim(self) -> Authority:
        return _dec(codec.authority_from, self._channel.call("unclaim"))

    def arm(self, timeout_s: float = 5.0) -> Authority:
        """ARM and wait (bounded) until the controller is ready to execute."""
        return self._await_authority(_dec(_generation, self._channel.call("arm")), True, timeout_s)

    def disarm(self, timeout_s: float = 5.0) -> Authority:
        """DISARM and wait (bounded) for the stop to be confirmed."""
        return self._await_authority(_dec(_generation, self._channel.call("disarm")), False, timeout_s)

    def _await_authority(self, gen: int, armed: bool, timeout_s: float) -> Authority:
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            got = self._channel.call("await_authority", wait_s=WAIT_SLICE_S, gen=gen,
                                     armed=armed, wait_for_s=min(WAIT_SLICE_S, max(left, 0.0)))
            settled, authority = _dec(_settled, got)
            if settled:
                return authority
            if left <= 0:
                raise TimeoutError(f"{'ARM' if armed else 'DISARM'} not confirmed within "
                                   f"{timeout_s}s; check authority()")

    # -- execution ---------------------------------------------------------------
    def execute(self, request: TrajectoryRequest) -> ExecutionStatus:
        """Submit; returns once admitted or refused (``status.phase``). Motion
        proceeds in the supervisor; follow it with ``status``/``wait``."""
        return _dec(codec.status_from, self._channel.call("submit", request=codec.encode(request)))

    def release(self, execution_id: str) -> ExecutionStatus:
        return _dec(codec.status_from, self._channel.call("release", execution_id=execution_id))

    def status(self, execution_id: str) -> ExecutionStatus:
        return _dec(codec.status_from, self._channel.call("status", execution_id=execution_id))

    def wait(self, execution_id: str, timeout_s: float) -> ExecutionStatus:
        """Block until terminal, at most ``timeout_s``; TimeoutError otherwise.
        Never cancels: a timed-out wait leaves the execution running."""
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            status = _dec(codec.status_from, self._channel.call(
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
        return _dec(lambda rs: _items(codec.receipt_from, rs),
                    self._channel.call("receipts", execution_id=execution_id))

    # -- negotiated setpoint/1 ------------------------------------------------
    def setpoint(self, q) -> float:
        """Hold the arm at joint target ``q`` (owner, ARMed, no live execution). Re-send within the returned timeout (s) or the
        arm holds its measured pose: not sending is the stop. Refused (nothing changed) with the supervisor's reason."""
        if self.extensions.get('setpoint') != 1:
            raise Refused('supervisor does not advertise setpoint/1')
        return _dec(lambda r: float(_need(_mapping(r), 'timeout_s')['timeout_s']),
                    self._channel.call('setpoint', q=[float(v) for v in q]))

    # -- negotiated controller/1 -----------------------------------------------
    def _controller_call(self, op, **args):
        if self.extensions.get('controller') != 1:
            raise Refused('supervisor does not advertise controller/1')
        return self._channel.call(op, **args)

    def controller_presets(self):
        return _dec(lambda ps: _items(codec.preset_from, ps), self._controller_call('controller_presets'))

    def controller_state(self):
        return _dec(codec.controller_state_from, self._controller_call('controller_state'))

    def select_control_preset(self, preset_id: str):
        return _dec(codec.controller_state_from, self._controller_call(
            'controller_select', preset_id=preset_id))

    def hold(self, reason: str = 'hold'):
        """Stop whatever runs (executions, a setpoint) and hold the measured pose, still ARMed; clears the hold a failed settle
        freezes in. Returns the controller state."""
        return _dec(codec.controller_state_from, self._controller_call('controller_hold', reason=reason))

    def set_payload(self, payload):
        return _dec(codec.payload_state_from, self._controller_call(
            'controller_payload', payload=codec.encode(payload)))

    def set_load_model(self, revision: int, model: dict) -> int:
        """Install a revisioned rigid-body load model (``arm_control.dynamics.BodyListDynamics``'s description, over this arm's
        joints in order) as the controller's RNEA. Refused while an execution is live, for a revision not above the installed
        one, and for gravity above the policy's torque limits. Returns the installed revision."""
        if self.extensions.get('load_model') != 1:
            raise Refused('supervisor does not advertise load_model/1')
        return int(self._channel.call('load_model_set', revision=int(revision), model=model))

    # -- negotiated scene_action/1 ----------------------------------------------
    def scene_claim(self) -> bool:
        """Take the hosted world's scene lease (one writing session, released when this connection ends)."""
        if self.extensions.get('scene_action') != 1:
            raise Refused('supervisor does not advertise scene_action/1')
        return bool(self._channel.call('scene_claim'))

    def scene_apply(self, request_id: str, revision: int, state: dict) -> dict:
        """An optimistic scene change (``scene_state`` payload at ``revision``, which must be above the world's). Returns
        ``request_id`` and the world's ``scene_state`` after it; a stale revision is Refused."""
        if self.extensions.get('scene_action') != 1:
            raise Refused('supervisor does not advertise scene_action/1')
        return self._channel.call('scene_apply', request_id=str(request_id), revision=int(revision), state=state)

    # -- negotiated stream/1 ----------------------------------------------------
    def _stream_call(self, op, **args):
        if self.extensions.get('stream') != 1:
            raise Refused('supervisor does not advertise stream/1 (joint targets on the streamed path: setpoint/1)')
        return self._channel.call(op, **args)

    def open_stream(self, request):
        """Returns REQUESTED; trusted admission continues asynchronously."""
        return _dec(codec.stream_view_from, self._stream_call('stream_open', request=codec.encode(request)))

    def stream_state(self, execution_id):
        """Pure observation; never acquires a splice challenge."""
        return _dec(codec.stream_view_from, self._stream_call('stream_state', execution_id=execution_id))

    def stream_anchor(self, execution_id):
        return _dec(codec.anchor_from, self._stream_call('stream_anchor', execution_id=execution_id))

    def revise_stream(self, request):
        return _dec(codec.revision_status_from, self._stream_call('stream_revise', request=codec.encode(request)))

    def stream_revision(self, execution_id, request_id):
        return _dec(codec.revision_status_from, self._stream_call(
            'stream_revision', execution_id=execution_id, request_id=request_id))

    def abort_stream_revision(self, execution_id, request_id):
        return _dec(codec.revision_status_from, self._stream_call(
            'stream_abort', execution_id=execution_id, request_id=request_id))

    # -- negotiated group/1 (CTRL-040 RT group execution through this arm's supervisor) -------------------------
    def _group_call(self, op, **args):
        if self.extensions.get('group') != 1:
            raise Refused('supervisor does not advertise group/1')
        return self._channel.call(op, **args)

    def group_join(self, group_id: int, member_index: int, member_count: int) -> dict:
        """Bind this arm to the RT group; the reply names the RT clock domain (None: unattested)."""
        return _dec(_mapping, self._group_call('group_join', group_id=group_id, member_index=member_index,
                                               member_count=member_count))

    def group_release(self, execution_id: str, group_id: int, budget_ns: int):
        """Release a staged ON_RELEASE execution through COMMIT_GROUP; selection is observed asynchronously
        (``group_report``). All members commit within ``budget_ns`` of each other or the RT refuses the group."""
        return _dec(codec.status_from, self._group_call('group_release', execution_id=execution_id,
                                                        group_id=group_id, budget_ns=budget_ns))

    def group_report(self, execution_id: str) -> dict:
        """``selected_ns``/``deadline_ns``/``reference_id`` once the RT selected it, ``error`` if it refused, ``{}``
        while pending. Times are on the RT host clock."""
        return _dec(_mapping, self._group_call('group_report', execution_id=execution_id))

    def group_cancel(self, group_id: int) -> dict:
        return _dec(_mapping, self._group_call('group_cancel', group_id=group_id))

    def group_events(self) -> list:
        return _dec(_mappings, self._group_call('group_events'))

    # -- negotiated trace/1 -----------------------------------------------------
    def trace_samples(self, after: int = 0) -> dict:
        """The supervisor's controller ticks recorded after sequence ``after`` (``seq`` on each sample), oldest first, from
        a bounded ring: ``dropped`` counts ticks that fell off it before this read (the reader's gap), ``errors`` ticks that
        could not be recorded, ``latest_seq`` where to read from next. Each sample is the controller diagnostic the direct
        hardware runtime keeps (``*_pc`` the PC's command terms, ``*_rt`` the RT packet it consumed)."""
        if self.extensions.get('trace') != 1:
            raise Refused('supervisor does not advertise trace/1')
        return _dec(_trace, self._channel.call('trace_samples', after=int(after)))

    # -- negotiated observe/1 ---------------------------------------------------
    def observe(self, after: int = 0, limit: int = 64) -> dict:
        """The plant's world, read-only, no ownership: observations after sequence ``after`` (oldest first, at most ``limit``),
        ``latest_seq`` the cursor for the next read, ``dropped`` the samples that fell off the ring before this read, ``rate_hz``
        the publication rate. Each observation is one plant tick: ``sim_time_s``, ``model_revision``, ``actor_q``/``actor_dq``
        by actor and joint name; fields a plant cannot provide are None, never a stale copy."""
        if self.extensions.get('observe') != 1:
            raise Refused('supervisor does not advertise observe/1')
        return _dec(_observation, self._channel.call('observe', after=int(after), limit=int(limit)))

    # -- negotiated state_history/1 ---------------------------------------------
    def state_history(self) -> dict:
        """The supervisor's recent RT packets: ``samples`` (oldest first; RT stamp ``t_mono_ns`` on ``clock_domain``,
        ``age_ns`` on the supervisor's clock at ``now_ns``), ``authority`` and ``clock_domain`` (None: unattested)."""
        if self.extensions.get('state_history') != 1:
            raise Refused('supervisor does not advertise state_history/1')
        return _dec(_history, self._channel.call('state_history'))

    # -- lifetime ----------------------------------------------------------------
    def close(self) -> None:
        """Ends this session: ownership, if held, is released by the supervisor
        (its live executions are cancelled; the robot holds)."""
        self._channel.close()

    def __enter__(self) -> ArmClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
