"""One robot behind the headless supervisor (CTRL-022).

The existing ``ArmController`` -- still the only executor -- over the RT link,
stepped at the execution policy's 100 Hz in this process, plus the execution
ledger that turns controller events and RT evidence into contract receipts
(``arm_control.execution.contract``).

Threading: every controller call and ledger change happens on the loop thread
while it holds ``_cv``. Client threads enqueue operations and read snapshots
under the same lock, so a stalled or slow client never delays a tick.
Authority RPCs (ARM/DISARM round trips) run on their own worker, as in
``multi_arm.hardware.RemoteHardware``, whose single-arm pump this mirrors.

Ownership: at most one client session owns the robot (``claim``). Losing the
owner cancels its live executions and holds, still armed; nobody reacquires
control automatically. The runtime never arms on its own: a fresh start is
disarmed with no owner.
"""
from __future__ import annotations

import itertools
import queue
import socket
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.config import arm_joints
from arm_control.control.arm_controller import ArmController
from arm_control.control.factory import build_arm_controller
from arm_control.execution import codec
from arm_control.execution.contract import (
    CONTRACT_VERSION, Authority, Capabilities, Clock, ExecutionStatus, Lifecycle, Phase,
    Receipt, RobotIdentity, RobotState, Stamp, Start, TrajectoryRequest, Validity,
    admission_refusal, selection_refusal, supersession_refusal,
)
from arm_control.messages import (
    pack_control_update, pack_json_message, pack_motor_state_dict, pack_plan,
    unpack_controller_event, unpack_json_message, unpack_motor_command,
)
from arm_control.control.inprocess import QueueControllerNode
from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.client import RtBackend, epoch_reached
from arm_control.plants.remote_rt.reference import ReferenceHorizon
from arm_control.plants.remote_rt.observation import AppliedTracker, authority, robot_state

#: Executions kept for status queries and request_id idempotency.
#: ponytail: fixed window; persist the ledger if replay needs more (CTRL-050).
RETAINED = 512


class Refused(Exception):
    """An operation the supervisor declines; no state changed."""


def monotonic_domain() -> str:
    """This host's CLOCK_MONOTONIC identity: one boot of one machine."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f"pc:{f.read().strip()}"
    except OSError:  # ponytail: non-Linux has no boot id; unique per process instead
        return f"pc:{socket.gethostname()}:{uuid.uuid4().hex[:8]}"


class _Node(QueueControllerNode):
    """The controller's outputs; its ARM/DISARM requests go to the runtime."""

    def __init__(self, runtime: RobotRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    def send_output(self, topic, value):
        if topic == "arm":
            self.runtime._request_authority(bool(unpack_json_message(value)["armed"]))
        else:
            super().send_output(topic, value)


@dataclass
class _Execution:
    request: TrajectoryRequest
    status: ExecutionStatus
    receipts: list[Receipt] = field(default_factory=list)
    first_seq: int | None = None
    epoch: int = 0
    cancel_reason: str | None = None
    completed_at: float | None = None  # feedback completion awaiting APPLIED


class RobotRuntime:
    def __init__(self, cfg, *, link=None) -> None:
        self.cfg = cfg
        self.joints = arm_joints(cfg)
        self.instance_id = str((cfg.get("rt") or {}).get("instance_id", ""))
        self.link = link if link is not None else RtBackend.from_config(cfg)
        self.domain = monotonic_domain()
        # How long a completion observed on feedback may wait for acceptance
        # evidence before the execution is FAULTED instead of COMPLETED. That
        # evidence is the next state packet, which the link's own freshness
        # bound already promises; past it the link is stale anyway.
        self.applied_grace_s = float(self.link.config.state_timeout_s)
        self._native = None
        self._native_pending = None
        self._native_epoch = 0
        self._native_failed = False
        self.controller = self._build_controller()
        self.period = self.controller.execution_policy.period
        self._profile_gains = (self.controller.executor.kp.copy(),
                               self.controller.executor.kd.copy())
        self.identity: RobotIdentity | None = None
        self.capabilities: Capabilities | None = None
        self._cv = threading.Condition()
        self._ops: queue.Queue = queue.Queue()
        self._executions: dict[str, _Execution] = {}
        self._by_request: dict[str, str] = {}
        self._ids = itertools.count(1)
        self._owner: str | None = None
        self._closed_sessions: set[str] = set()
        self._tracker = AppliedTracker()
        self._fed_rx = 0.0
        self._seen_seq: int | None = None
        self._rpc = ThreadPoolExecutor(max_workers=1, thread_name_prefix="authority")
        self._pending: list[tuple[Future, bool, int]] = []
        self._authority_gen = 0       # last ARM/DISARM request
        self._authority_done = 0      # last one completed
        self._authority_failed = (0, "")  # last failed one: (generation, reason)
        self._authority_fault = ""    # cleared by the next success
        self._state = RobotState(self.instance_id or "unknown", Validity.MISSING, "not connected")
        self._authority = Authority(Lifecycle.DISCONNECTED)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- construction ----------------------------------------------------------
    def _build_controller(self) -> ArmController:
        return build_arm_controller(
            self.cfg, _Node(self), arm_id=self.instance_id, require_policy=True,
            clock=time.monotonic, plant_reports_health=True,
            reference_source=(None if self._native is None else lambda: self._native.point))

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        """Claim the RT server (never ARM) and start the 100 Hz loop."""
        self.link.open()
        if (self.controller.model_token and (self.cfg.get("rt") or {}).get("native_reference", True)
                and self.link.negotiate_reference(self.controller.model_token)):
            self._native = ReferenceHorizon(self.link)
            self.controller.bind_reference_source(lambda: self._native.point)
        ident = self.link.identity
        self.identity = RobotIdentity(self.instance_id, tuple(self.joints),
                                      f"rt:{self.link.backend_name}", ident.model_id)
        self.capabilities = Capabilities(
            CONTRACT_VERSION, ("joint_impedance",), protocol_version=rtp.VERSION,
            pose_hold=bool(self.link.supports_pose_hold),
            rt_setup=str(getattr(self.link, "rt_setup", "unknown")))
        self._thread = threading.Thread(target=self._loop, name="supervisor", daemon=True)
        self._thread.start()

    def close(self, timeout_s: float = 5.0) -> None:
        """Bounded: stop the loop, DISARM through the link, release the socket."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._tracker.disarm_sent()
        if self._native is not None:
            self._native.close()
        for fut, _, _ in self._pending:
            fut.cancel()  # queued ARM/DISARM work; link.close() disarms itself
        closing = self._rpc.submit(self.link.close)
        try:
            closing.result(timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 -- reported, shutdown continues
            print(f"[supervisor] link close: {exc}", flush=True)
        self._rpc.shutdown(wait=False, cancel_futures=True)

    def now(self) -> Stamp:
        return Stamp(time.monotonic_ns(), Clock.PC, self.domain)

    # -- client API (any thread) -------------------------------------------------
    def _call(self, fn, *args, timeout_s: float = 2.0):
        fut: Future = Future()
        self._ops.put((fn, args, fut))
        return fut.result(timeout=timeout_s)

    def close_session(self, session: str) -> None:
        """A client connection ended: its later requests are refused at once,
        and its ownership is released on the loop."""
        with self._cv:
            self._closed_sessions.add(session)
        self._ops.put((self._owner_lost, (session,), Future()))

    def claim(self, session: str) -> Authority:
        return self._call(self._claim, session)

    def unclaim(self, session: str) -> Authority:
        return self._call(self._unclaim, session)

    def arm(self, session: str) -> int:
        """Request ARM; returns the request generation for ``await_authority``."""
        return self._call(self._arm, session)

    def disarm(self, session: str) -> int:
        """Request DISARM; returns the request generation for ``await_authority``."""
        return self._call(self._disarm, session)

    def submit(self, session: str, request: TrajectoryRequest) -> ExecutionStatus:
        return self._call(self._submit, session, request)

    def release(self, session: str, execution_id: str) -> ExecutionStatus:
        return self._call(self._release, session, execution_id)

    def cancel(self, session: str, execution_id: str, reason: str = "cancelled"
               ) -> ExecutionStatus:
        return self._call(self._cancel, session, execution_id, reason)

    def status(self, execution_id: str, *, wait_s: float = 0.0) -> ExecutionStatus:
        """Current status; with ``wait_s`` waits (bounded) for a terminal one."""
        with self._cv:
            ex = self._get(execution_id)
            self._cv.wait_for(lambda: ex.status.terminal, timeout=max(0.0, wait_s))
            return ex.status

    def receipts(self, execution_id: str) -> tuple[Receipt, ...]:
        with self._cv:
            return tuple(self._get(execution_id).receipts)

    def state(self) -> RobotState:
        with self._cv:
            return self._state

    def authority(self) -> Authority:
        with self._cv:
            return self._authority

    def await_authority(self, gen: int, *, armed: bool, wait_s: float) -> Authority | None:
        """The authority once request ``gen`` has settled (ARMED and ready, or
        the DISARM confirmed), None if not within ``wait_s``; Refused if the
        request failed."""
        def settled():
            if self._authority_failed[0] >= gen:
                return True
            if armed:
                return (self._authority.lifecycle is Lifecycle.ARMED
                        and self.controller.runtime_state["ready"])
            return (self._authority_done >= gen
                    and self._authority.lifecycle is not Lifecycle.ARMED)
        with self._cv:
            if not self._cv.wait_for(settled, timeout=max(0.0, wait_s)):
                return None
            failed_gen, reason = self._authority_failed
            if failed_gen >= gen:
                raise Refused(reason)
            return self._authority

    # -- operations (loop thread, under _cv) -------------------------------------
    def _get(self, execution_id: str) -> _Execution:
        try:
            return self._executions[execution_id]
        except KeyError:
            raise Refused(f"unknown execution {execution_id!r}") from None

    def _owns(self, session: str) -> None:
        if session in self._closed_sessions:
            raise Refused("this client session is closed")
        if self._owner != session:
            raise Refused(f"{self.instance_id} is owned by another client" if self._owner
                          else f"claim {self.instance_id} first")

    def _claim(self, session: str) -> Authority:
        if session in self._closed_sessions:
            raise Refused("this client session is closed")
        if self._owner not in (None, session):
            raise Refused(f"{self.instance_id} is already owned by another client")
        self._owner = session
        self._publish()
        return self._authority

    def _unclaim(self, session: str) -> Authority:
        self._owns(session)
        self._abandon("owner released the robot")
        return self._authority

    def _owner_lost(self, session: str) -> None:
        if self._owner == session:
            self._abandon("owner disconnected")
        # Every op this session queued ran before this one (FIFO), and its
        # connection has ended: nothing can name it again.
        self._closed_sessions.discard(session)

    def _abandon(self, reason: str) -> None:
        """Cancel the owner's live executions and hold; stay armed."""
        for ex in list(self._executions.values()):
            if not ex.status.terminal and ex.completed_at is None:
                self._cancel_execution(ex, reason)
        self._owner = None
        self._publish()

    def _arm(self, session: str) -> int:
        self._owns(session)
        if any(not want for _, want, _ in self._pending):
            raise Refused("a DISARM is still being confirmed; ARM after it completes")
        if not self.controller.armed:
            self.controller.on_control(pack_control_update(arm=True))
            self._pump_controller()
        return self._authority_gen

    def _disarm(self, session: str) -> int:
        self._owns(session)
        for ex in self._executions.values():
            if not ex.status.terminal and ex.completed_at is None:
                ex.cancel_reason = ex.cancel_reason or "disarmed"
        # The controller's own stop path: cancel the leg, then drop authority.
        # Always emits the controller's ``arm`` output, so a DISARM is always sent.
        self.controller.on_control(pack_control_update(cancel=True, arm=False, reason="disarm"))
        self._pump_controller()
        return self._authority_gen

    def _submit(self, session: str, request: TrajectoryRequest) -> ExecutionStatus:
        self._owns(session)
        known = self._by_request.get(request.request_id)
        if known is not None:  # idempotency: the same request_id is one execution
            ex = self._executions[known]
            if codec.encode(ex.request) != codec.encode(request):
                raise Refused(f"request_id {request.request_id!r} was already used "
                              "with different content")
            return ex.status
        current = self._current()
        eid = f"{self.instance_id}.{next(self._ids)}.{uuid.uuid4().hex[:8]}"
        first = Receipt(eid, request.request_id, self.instance_id, Phase.REQUESTED, self.now())
        ex = _Execution(request, ExecutionStatus.start(first), [first])
        self._executions[eid] = ex
        self._by_request[request.request_id] = eid
        refusal = (
            admission_refusal(request, self.identity, self.capabilities, self._authority,
                              now=self.now(), model_token=self.controller.model_token)
            or supersession_refusal(request, current)
            or ("reference selection is in flight; cancel before a new admission"
                if request.supersedes and self._native is not None
                and self._native.key == request.supersedes
                and self._native.selection_requested else "")
            or ("the joint_impedance profile needs waypoint velocities"
                if request.trajectory.velocities is None else ""))
        if refusal:
            self._receipt(ex, Phase.REJECTED, refusal)
            return ex.status
        kp, kd = ((request.gains.kp, request.gains.kd) if request.gains is not None
                  else self._profile_gains)
        completion = None if request.completion is None else dict(
            vel_tol=request.completion.vel_tol, goal_q=request.completion.goal_q,
            residual_max=request.completion.residual_max)
        traj = request.trajectory
        try:
            # Always staged gated: selection (expiry, ARM) is the supervisor's
            # decision for both starts, taken just before the release.
            plan = pack_plan(plan_id=eid, phase=request.request_id, gated=True, times=traj.times,
                             positions=traj.positions, velocities=traj.velocities,
                             kp=kp, kd=kd, completion=completion,
                             model_token=request.model_token)
            self.controller.on_plan(plan)
        except (TypeError, ValueError) as exc:
            self._receipt(ex, Phase.REJECTED, f"controller refused the plan: {exc}")
            return ex.status
        self._pump_controller()
        if ex.status.phase is Phase.REQUESTED:
            self._receipt(ex, Phase.REJECTED, "the controller did not acknowledge the plan")
        elif ex.status.phase is Phase.ADMITTED and request.start is Start.ON_ADMIT:
            self._select(ex)
        return ex.status

    def _release(self, session: str, execution_id: str) -> ExecutionStatus:
        self._owns(session)
        ex = self._get(execution_id)
        if ex.status.phase is not Phase.ADMITTED:
            return ex.status  # already selected or ended: release is idempotent
        if ex.request.start is not Start.ON_RELEASE:
            raise Refused(f"{execution_id} starts on admission, not on release")
        self._select(ex)
        return ex.status

    def _select(self, ex: _Execution) -> None:
        """Expiry, then ARM, then the controller's own release checks."""
        if self._expire(ex):
            return
        refusal = selection_refusal(self._authority)
        if refusal:
            self._receipt(ex, Phase.REJECTED, refusal)
            self._drop_staged(ex, refusal)
            return
        if self._native is not None:
            if self._native_pending == ex.status.execution_id:
                return
            refusal = self.controller.release_error(ex.status.execution_id)
            if refusal:
                self._receipt(ex, Phase.REJECTED, refusal)
                self._drop_staged(ex, refusal)
                return
            self._native_pending = ex.status.execution_id
            return  # upload and selection are observed asynchronously on ticks
        self.controller.on_control(pack_control_update(execute=ex.status.execution_id))
        self._pump_controller()

    def _cancel(self, session: str, execution_id: str, reason: str) -> ExecutionStatus:
        self._owns(session)
        ex = self._get(execution_id)
        if not ex.status.terminal and ex.completed_at is None:
            self._cancel_execution(ex, reason or "cancelled")
        return ex.status

    def _cancel_execution(self, ex: _Execution, reason: str) -> None:
        ex.cancel_reason = reason
        self._retire_native(ex.status.execution_id)
        if self.controller.current_plan_id == ex.status.execution_id:
            self.controller.on_control(pack_control_update(
                cancel=True, reason=reason, cancel_plan_id=ex.status.execution_id))
        self._pump_controller()

    def _drop_staged(self, ex: _Execution, reason: str) -> None:
        self._retire_native(ex.status.execution_id)
        if self.controller.current_plan_id == ex.status.execution_id:
            self.controller.on_control(pack_control_update(
                cancel=True, reason=reason, cancel_plan_id=ex.status.execution_id))
            self._pump_controller(reconcile=False)

    # -- the ledger --------------------------------------------------------------
    def _current(self) -> ExecutionStatus | None:
        """The robot's live execution, if any (at most one is live)."""
        live = [ex for ex in self._executions.values() if not ex.status.terminal]
        return live[-1].status if live else None

    def _receipt(self, ex: _Execution, phase: Phase, reason: str = "", **evidence) -> None:
        r = Receipt(ex.status.execution_id, ex.status.request_id, self.instance_id, phase,
                    self.now(), reason, evidence)
        ex.status = ex.status.apply(r)
        ex.receipts.append(r)
        self._cv.notify_all()

    def _prune(self) -> None:
        """Forget the oldest terminal executions beyond RETAINED."""
        excess = len(self._executions) - RETAINED
        if excess > 0:
            for eid in [e for e, x in self._executions.items() if x.status.terminal][:excess]:
                self._by_request.pop(self._executions.pop(eid).request.request_id, None)

    def _pump_controller(self, *, sent_for: str | None = None, reconcile: bool = True) -> None:
        """Forward the controller's commands to the plant and turn its events
        into receipts. ``sent_for`` names the execution the forwarded commands
        were stepped from (the one still executing after the tick)."""
        c = self.controller
        sent, events = [], []
        for topic, value in c.node.sent:
            if topic == "motor_command":
                seq = self._send_command(unpack_motor_command(value, c.n_arm))
                if seq is not None:
                    sent.append(seq)
            elif topic == "controller_event":
                events.append(unpack_controller_event(value))
        c.node.sent = []
        for event in events:
            self._on_event(event)
        rs = c.runtime_state
        if rs["executing"]:
            ex = self._executions.get(rs["plan_id"])
            if ex is not None and ex.status.phase is Phase.ADMITTED and self._native is None:
                self._receipt(ex, Phase.SELECTED, arm_epoch=self._authority.arm_epoch)
        if sent_for is not None and sent:
            ex = self._executions.get(sent_for)
            if ex is not None and ex.status.phase is Phase.SELECTED and ex.first_seq is None:
                ex.first_seq, ex.epoch = sent[0], int(self.link.motor_health()["arm_epoch"])
                self._receipt(ex, Phase.SENT, command_seq=sent[0], arm_epoch=ex.epoch)
        if reconcile:
            self._reconcile()

    def _on_event(self, event: dict) -> None:
        ex = self._executions.get(event.get("plan_id") or "")
        kind = event["kind"]
        if ex is None or ex.status.terminal:
            return
        if kind == "plan_staged":
            if not event["ok"]:
                self._receipt(ex, Phase.REJECTED, event.get("reason") or "controller refused")
                return
            self._receipt(ex, Phase.ADMITTED)
            old = ex.request.supersedes and self._executions.get(ex.request.supersedes)
            if old and not old.status.terminal:
                self._retire_native(old.status.execution_id)
                self._receipt(old, Phase.SUPERSEDED, f"superseded by {ex.status.execution_id}",
                              superseded_by=ex.status.execution_id)
        elif kind == "leg_result":
            reason = event.get("reason") or ""
            if event["ok"]:
                ex.completed_at = time.monotonic()
            elif ex.cancel_reason is not None:
                self._receipt(ex, Phase.CANCELLED, ex.cancel_reason)
            elif ex.status.phase is Phase.ADMITTED:
                self._receipt(ex, Phase.REJECTED, reason or "refused at selection")
            else:
                self._receipt(ex, Phase.FAULTED, reason or "controller ended the leg")

    def _reconcile(self) -> None:
        """Executions the controller no longer holds, with no event to say why."""
        now = time.monotonic()
        current = self.controller.current_plan_id
        for eid, ex in list(self._executions.items()):
            phase = ex.status.phase
            if ex.status.terminal:
                continue
            if ex.completed_at is not None:
                if phase is Phase.APPLIED:
                    self._receipt(ex, Phase.COMPLETED)
                elif now - ex.completed_at > self.applied_grace_s:
                    self._receipt(ex, Phase.FAULTED, "completed on feedback without plant "
                                                     "acceptance evidence")
                continue
            if phase in (Phase.REQUESTED,) or current == eid:
                continue
            if ex.cancel_reason is not None:
                self._receipt(ex, Phase.CANCELLED, ex.cancel_reason)
            elif phase is Phase.ADMITTED:
                self._receipt(ex, Phase.CANCELLED, "discarded by the controller "
                                                   "(disarm, fault or stale observation)")
            else:
                self._receipt(ex, Phase.FAULTED, "the controller dropped the running execution")

    def _expire(self, ex: _Execution) -> bool:
        # COMMIT carries a conservative RT-local deadline budget. Once sent,
        # wait for its selection/refusal evidence; PC receipt time is not the
        # selection time and cannot honestly prove EXPIRED.
        committing = (self._native is not None and self._native.key == ex.status.execution_id
                      and self._native.selection_requested)
        if ex.status.phase is Phase.ADMITTED and not committing and ex.request.expired(self.now()):
            self._receipt(ex, Phase.EXPIRED, "admit_by passed before selection")
            self._drop_staged(ex, "expired")
            return True
        return False

    # -- negotiated RT reference -------------------------------------------------
    def _retire_native(self, execution_id: str) -> None:
        if self._native_pending == execution_id:
            self._native_pending = None
        if self._native is not None and self._native.key == execution_id:
            self._native.cancel()

    def _native_step(self, sample) -> None:
        h, c = self._native, self.controller
        state, rx = sample
        h.poll(state, rx)
        if state is None or not state.armed:
            return
        if state.arm_epoch != self._native_epoch:
            self._native_epoch = state.arm_epoch
            self._native_failed = False
        ex = self._executions.get(h.key)
        if (h.error and ex is not None and not ex.status.terminal
                and (not h.selected or h.error == 'REFERENCE_UNDERFLOW')):
            # Reference failure is distinct from a transport/plant fault. In
            # particular, never heal UNDERFLOW with an implicit hold admission.
            self._native_failed = True
            if ex.status.phase is Phase.ADMITTED and h.error == 'REFERENCE_UNDERFLOW':
                self._receipt(ex, Phase.SELECTED, reference_id=h.id,
                              reference_revision=h.revision, arm_epoch=h.arm_epoch)
            failure = Phase.FAULTED
            if ex.status.phase is Phase.ADMITTED:
                failure = Phase.EXPIRED if ex.request.expired(self.now()) else Phase.REJECTED
            self._receipt(ex, failure, h.error,
                          reference_id=h.id, reference_revision=h.revision)
            self._drop_staged(ex, h.error)
        if h.key is None and h.error:
            self._native_failed = True
        if h.key is None and h.ready and self._native_pending is None and not self._native_failed:
            h.approve()
        eid = self._native_pending
        ex = self._executions.get(eid)
        if ex is None or ex.status.terminal or c.current_plan_id != eid:
            if eid is not None:
                self._retire_native(eid)
            return
        if h.key != eid:
            # Cancellation/hold upload finishes before another candidate may
            # enter this same bounded session slot.
            if h.busy:
                if h.key is None:
                    h.cancel()
                return
            observed = h.observed
            if observed and observed['status'] == rtp.REFERENCE_RUNNING:
                h.cancel()
                return
            try:
                deadline = ex.request.admit_by
                h.stage(eid, c.staged_trajectory(eid),
                        selection_deadline_ns=None if deadline is None else deadline.ns)
                self._native_failed = False  # a new, explicit admission
            except (ValueError, RuntimeError) as exc:
                self._receipt(ex, Phase.REJECTED, str(exc))
                self._drop_staged(ex, str(exc))
            return
        if h.selected:
            if ex.status.phase is Phase.ADMITTED:
                self._receipt(ex, Phase.SELECTED, reference_id=h.id, reference_revision=h.revision,
                              reference_phase=h.point.time, arm_epoch=h.arm_epoch)
            c.on_control(pack_control_update(execute=eid))
            self._native_pending = None
            if not c.runtime_state['executing']:
                h.cancel()
            self._pump_controller()
        elif h.ready:
            refusal = selection_refusal(self._authority) or c.release_error(eid)
            if refusal:
                self._receipt(ex, Phase.REJECTED, refusal)
                self._drop_staged(ex, refusal)
            else:
                h.approve()

    def _send_command(self, command):
        h = self._native
        if h is None:
            return self.link.apply_command(command)
        state, _ = self.link.latest_state()
        if state is None or not state.armed:
            return None
        ref, point = h.observed, h.observed_point
        if (state.session_id, state.arm_epoch) != (h.session_id, h.arm_epoch):
            ref = point = None
        # Before this ARM generation has ever selected a reference, the
        # existing compensated idle hold remains the v5 command path.
        if state.reference is None or not state.reference['reference_id']:
            return self.link.apply_command(command)
        if ref is not None and point is not None:
            matches = (np.array_equal(command['position'], point.position)
                       and np.array_equal(command['velocity'], point.velocity))
            if matches:
                return self.link.apply_reference_envelope(
                    command, reference_id=ref['reference_id'], revision=ref['revision'],
                    phase=point.time, sample_received_at=h.received_at,
                    valid_for_s=.05, phase_window_s=.05,
                    session_id=h.session_id, arm_epoch=h.arm_epoch)
        # The controller's existing cancel/contact/relatch path owns the hold
        # target and its compensation. Admit that fixed target explicitly;
        # never mislabel its dynamics as an envelope for the old moving curve.
        if (self._native_failed or self._native_pending is not None
                or self.controller.runtime_state['executing'] or h.busy):
            return None
        if ref and ref['status'] == rtp.REFERENCE_UNDERFLOW:
            self._native_failed = True
            return None
        if ref and ref['status'] == rtp.REFERENCE_RUNNING:
            h.cancel()
            return None
        if not h.busy:
            q = command['position']
            hold = JointTrajectory([0., self.period], np.stack([q, q]), np.zeros((2, len(q))))
            h.stage(None, hold)
        return None

    # -- authority RPCs ------------------------------------------------------------
    def _request_authority(self, armed: bool) -> None:
        """From the controller's ``arm`` output. One worker, bounded by the
        link's ACK timeouts; a DISARM drops queued ARMs (RemoteHardware's rule)."""
        if not armed:
            self._native_pending = None
            if self._native is not None:
                self._native.cancel()
            self._tracker.disarm_sent()
            if any(not want and not fut.done() for fut, want, _ in self._pending):
                return  # one DISARM in flight covers this one
            for fut, want, _ in self._pending:
                if want:
                    fut.cancel()
        self._authority_gen += 1
        gen = self._authority_gen
        work = self.link.enable_all if armed else self.link.safe_stop
        self._pending.append((self._rpc.submit(work), armed, gen))

    def _drain_authority(self) -> None:
        while self._pending and self._pending[0][0].done():
            fut, armed, gen = self._pending.pop(0)
            if fut.cancelled():
                continue
            try:
                ok = fut.result() is not False
                reason = "" if ok else "DISARM not confirmed by the state stream"
            except Exception as exc:  # noqa: BLE001 -- the link's refusal text
                ok, reason = False, str(exc)
            self._authority_done = gen
            self._authority_fault = "" if ok else reason
            if not ok:
                self._authority_failed = (gen, reason)
            if armed and not ok and self.controller.armed:
                self.controller.on_control(pack_control_update(arm=False))
            self._cv.notify_all()

    # -- the loop --------------------------------------------------------------------
    def _loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._cv:
                while True:
                    try:
                        fn, args, fut = self._ops.get_nowait()
                    except queue.Empty:
                        break
                    if not fut.set_running_or_notify_cancel():
                        continue
                    try:
                        fut.set_result(fn(*args))
                    except Exception as exc:  # noqa: BLE001 -- returned to the caller
                        fut.set_exception(exc)
                try:
                    self._step()
                except Exception as exc:  # noqa: BLE001
                    # A tick that raises must not kill the loop silently; the
                    # controller's own stop path takes authority down.
                    print(f"[supervisor] step failed: {exc!r}", flush=True)
                    self.controller.on_control(pack_control_update(stop=True, reason=repr(exc)))
                self._cv.notify_all()
            next_t += self.period
            now = time.monotonic()
            if now - next_t > self.period:
                next_t = now  # after a stall: one tick, never a catch-up burst
            time.sleep(max(0.0, next_t - now))

    def _step(self) -> None:
        c, link = self.controller, self.link
        self._drain_authority()
        health = link.motor_health()
        if self._authority_fault:
            health = dict(health, any_fault=True,
                          latched_fault=health["latched_fault"] or self._authority_fault)
        c.on_motor_health(pack_json_message("motor_health", health))
        history = link.recent_states()
        for state, rx in history:
            if rx > self._fed_rx:
                self._fed_rx = rx
                self._tracker.feed(state)
        if history and health["state_fresh"]:
            state, rx = history[-1]
            if state.state_seq != self._seen_seq:
                self._seen_seq = state.state_seq
                c.on_motor_state(pack_motor_state_dict(link.motor_state(state=state)))
                # Arrival freshness survives a delayed tick (RemoteHardware's seam).
                if c._last_state_at is not None:
                    c._last_state_at = rx
        self._publish()
        for ex in list(self._executions.values()):
            self._expire(ex)
        if self._native is not None:
            self._native_step(history[-1] if history else (None, 0.))
        c.tick()
        rs = c.runtime_state
        self._pump_controller(sent_for=rs["plan_id"] if rs["executing"] else None)
        for ex in list(self._executions.values()):
            if (ex.first_seq is not None and not ex.status.terminal
                    and self._tracker.fault_seen(ex.epoch)):
                # The frozen APPLIED rule: a FAULTED state in the execution's
                # ARM generation ends it FAULTED; no later clean state undoes it.
                self._receipt(ex, Phase.FAULTED, "RT fault latched in this ARM generation")
            elif ex.status.phase is Phase.SENT:
                confirmed = self._tracker.confirmed(ex.epoch)
                if confirmed is not None and epoch_reached(confirmed, ex.first_seq):
                    self._receipt(ex, Phase.APPLIED, command_seq=confirmed, arm_epoch=ex.epoch)
        self._reconcile()
        if c.stopped:
            # A stopped ArmController is terminal (its Dora node exits). Here a
            # fresh one replaces it: disarmed, nothing staged, new admission.
            print("[supervisor] controller stopped; replaced with a fresh, disarmed one",
                  flush=True)
            self.controller = self._build_controller()
            self._reconcile()
        self._prune()
        self._publish()

    def _publish(self) -> None:
        """Snapshot state and authority as contract values for readers."""
        link = self.link
        state, rx = link.latest_state()
        health = link.motor_health()
        now = time.monotonic()
        rt_identity = RobotIdentity(self.instance_id, tuple(self.joints), "rt")
        self._state = robot_state(
            state, rt_identity, received_ns=int(rx * 1e9), now_ns=int(now * 1e9),
            max_age_s=float(link.config.state_timeout_s), pc_domain=self.domain,
            rt_domain=f"rt:{self.instance_id}")
        # ``session`` stays the RT control session (the frozen contract); the
        # client that owns the robot shows only as CLAIMED vs CONNECTED.
        if state is None or not health["state_fresh"]:
            self._authority = Authority(Lifecycle.DISCONNECTED)
            return
        rt = authority(state, claimed_session=link.session_id,
                       fault_text=health["latched_fault"])
        if rt.lifecycle is Lifecycle.CLAIMED and self._owner is None:
            rt = replace(rt, lifecycle=Lifecycle.CONNECTED)  # supervisor's claim, no owner
        elif rt.lifecycle is Lifecycle.CONNECTED:  # another RT session's packet
            rt = Authority(Lifecycle.DISCONNECTED)
        self._authority = rt
