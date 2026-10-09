"""One robot behind the headless supervisor (CTRL-022).

The existing ``ArmController`` -- still the only executor -- over the RT link,
stepped at the execution policy's 100 Hz in this process, plus the execution
ledger that turns controller events and RT evidence into contract receipts
(``arm_control.execution.contract``).

Threading: every controller call and ledger change happens on the loop thread
while it holds ``_cv``. Client threads enqueue operations and read snapshots
under the same lock, so a stalled or slow client never delays a tick.
Authority RPCs (ARM/DISARM round trips) run on their own worker (the pump of
the retired direct dual runtime, which this replaced).

Ownership: at most one client session owns the robot (``claim``). Losing the
owner cancels its live executions and holds, still armed; nobody reacquires
control automatically. The runtime never arms on its own: a fresh start is
disarmed with no owner.
"""
from __future__ import annotations

import collections
import itertools
import queue
import socket
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from arm_control.control.execution_policy import ENVELOPE_TOL
from arm_control.motion import JointState, JointTrajectory
from arm_control.config import arm_joints
from arm_control.control.arm_controller import ArmController
from arm_control.control.diagnostic import controller_diagnostic
from arm_control.control.factory import build_arm_controller
from arm_control.control.presets import resolve_presets
from arm_control.execution.controller import ControllerState, PayloadState, PayloadValue
from arm_control.execution import codec
from arm_control.execution.stream import (StreamOpen, StreamRevision, StreamView,
                                          SpliceAnchor, RevisionStatus, SelectionState)
from arm_control.supervisor.admission import StreamAdmission, StreamFacts
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
from arm_control.multi_arm.diagnostics import SupervisorFaultTrace
from arm_control.multi_arm.group import GROUP_BUDGET_S, GROUP_MIN_LEAD_MS
from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.client import ReferenceRefused, RtBackend, RtLinkError, epoch_reached
from arm_control.plants.remote_rt.reference import ReferenceHorizon, SelectionUnresolved
from arm_control.plants.remote_rt.observation import AppliedTracker, authority, robot_state

#: Executions kept for status queries and request_id idempotency.
#: ponytail: fixed window; persist the ledger if replay needs more (CTRL-050).
RETAINED = 512
#: trace/1 ring: the newest controller ticks kept for a reader (about 0.64 s at the 100 Hz tick); older ones are overwritten
#: and reported as a gap, never waited for
TRACE_TICKS = 64


def next_tick(previous: float, now: float, period: float) -> float:
    """The servo loop's next deadline, taken after a tick at ``now``: one period after the previous deadline, or, when that
    has already passed (a late wake or a slow step), a full period after ``now`` -- the late tick was the one tick, never a
    catch-up burst nor a second tick right behind it (the retired Dora loop's rule). Steps persistently longer than a
    period therefore run at 1/(step + period), not back to back."""
    due = previous + period
    return now + period if now > due else due


def offered_profiles(link, *, native: bool, cartesian_kp_min: float | None) -> tuple[str, ...]:
    """The controller profiles a supervisor admits. ``cartesian_impedance`` only when the plant runs the law (the RT box's v8,
    the twin's), legs are STREAMED (the native reference plays joints only, operator 2026-10-09) and the robot's policy states
    the joint-kp floor such a leg must clear (without one every Cartesian request would be refused: never advertise that)."""
    cartesian = bool(getattr(link, "supports_cartesian", False)) and not native and cartesian_kp_min is not None
    return ("joint_impedance",) + (("cartesian_impedance",) if cartesian else ())


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
    stream: _Stream | None = field(default=None, kw_only=True)
    payload_revision: int = field(default=0, kw_only=True)
    status: ExecutionStatus
    receipts: list[Receipt] = field(default_factory=list)
    first_seq: int | None = None
    epoch: int = 0
    cancel_reason: str | None = None
    drop: tuple | None = None  # (reason, RT reference id, session, ARM epoch): cancel awaiting RT confirmation
    completed_at: float | None = None  # feedback completion awaiting APPLIED


@dataclass
class _Attempt:
    request: StreamOpen | StreamRevision
    status: RevisionStatus
    curve: JointTrajectory
    facts: StreamFacts
    deadline_ns: int
    cut: float = 0.
    terminal: bool = False
    sent_seq: int | None = None
    upload_started: bool = False


@dataclass
class _Stream:
    owner: str
    opened: StreamOpen
    facts: StreamFacts
    epoch: tuple
    gains: tuple
    payload_revision: int
    anchor: SpliceAnchor | None = None
    pending: str | None = None
    selected: str | None = None



class RobotRuntime:
    def __init__(self, cfg, *, link=None, stream_admission: StreamAdmission | None = None) -> None:
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
        self._stream_admission = stream_admission
        self._stream_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stream-admission')
        self._stream_job = None
        self._stream_marks = {}  # live service-session marks, independent of claim
        self._stream_attempts = {}  # (service session, request id) -> bounded record
        self._native = None
        self._native_pending = None
        self._native_epoch = 0
        self._native_failed = False
        self._trace = collections.deque(maxlen=TRACE_TICKS)  # trace/1: the newest controller ticks, a bounded ring
        self._trace_seq = 0           # sequence of the last tick recorded (gaps = ticks that fell off the ring)
        self._trace_errors = 0        # ticks whose record could not be built (never allowed to stop the loop)
        self._last_packet = (None, 0.)  # (RT state packet, receive time) the controller last consumed
        self._group = None            # group/1: the joined RT group (group_id, member_index, member_count)
        self._group_release = {}      # execution id -> (group_id, member_index, budget_ns), consumed at COMMIT
        self._group_reports = {}      # execution id -> the RT's COMMIT_GROUP reply or refusal (bounded)
        self._group_cancelling = 0.0  # monotonic deadline: no measured hold is staged while a group CANCEL is on the wire
        self._group_hold_grace = 0.0  # monotonic: after it, the RT's group retire may have retired this arm's standing hold
        self.controller = self._build_controller()
        self._payload = PayloadState(0, PayloadValue(0.))
        self._load_model_revision = -1  # controller_load_model: none installed (the construction model's RNEA)
        self._scene_owner = None  # scene_action/1: the one session that may change the hosted world's scene
        self._presets = resolve_presets(cfg.get('control_mode') or {}, self.joints,
                                       self.controller.executor.kp, self.controller.executor.kd)
        self._soft_ids = itertools.count(1)
        self._selected_soft = None
        self._mode_feedback = dict(law='joint', kp=self.controller.executor.kp.copy(),
                                   kd=self.controller.executor.kd.copy())
        self.period = self.controller.execution_policy.period
        # CTRL-051: this supervisor's own fault record (the ticks 10 s before a fault to 2 s after), written off the loop
        self.fault_trace = SupervisorFaultTrace(self.instance_id, self.period, context=dict(
            joints=list(self.joints), rt_host=getattr(self.link.config, 'host', None)))
        self._latched_seen = False    # a latched fault was present last tick: only its rising edge opens a fault window (the
                                      # text can carry an age that changes every tick, e.g. "rt state stream stale (0.42s)")
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

    @property
    def extensions(self):
        result = {'controller': 1, 'load_model': 1}
        if self._native is not None and self._stream_admission is not None:
            result['stream'] = 1
        if self._native is not None and getattr(self.link, 'group_capable', False):
            result['group'] = 1
        if callable(getattr(self.link, 'recent_states', None)):
            result['state_history'] = 1
            result['trace'] = 1
        if getattr(self.link, 'observing', False):
            result['observe'] = 1
        world = getattr(self.link, 'scene_world', None)
        if world is not None and world.backend.scene_state:
            result['scene_action'] = 1
        if self._native is None:  # setpoints are words on the streamed path; a native reference plays only its own curve
            result['setpoint'] = 1
            result['scheduled_release'] = 1  # this supervisor's controller owns the trajectory clock (a native one: the RT)
        return result

    def observe(self, after: int = 0, limit: int = 64) -> dict:
        """observe/1: the plant's world, read-only and without ownership (2026-10-08-scene-hosting-design.md)."""
        return self.link.observations(after, limit)

    def scene_claim(self, session: str) -> bool:
        """scene_action/1: take the hosted world's scene lease (one writing session; released when that session ends). Separate
        from the arm's claim: the scene writer (a grafter) is not the arm's owner."""
        with self._cv:
            if session in self._closed_sessions:
                raise Refused('session closed')
            if self._scene_owner not in (None, session):
                raise Refused('the scene lease is held by another session')
            self._scene_owner = session
        return True

    def scene_apply(self, session: str, request_id: str, revision: int, state: dict) -> dict:
        """scene_action/1: an optimistic scene change on the hosted world (``SceneCore.apply_scene``: stale revision refused).
        Runs on the caller's thread under the world's lock, never on this loop: a recompile stalls the world's servo threads (the
        stall is credited to the slice deadmans), not the controller."""
        with self._cv:
            if self._scene_owner != session:
                raise Refused('scene_apply needs the scene lease (scene_claim)')
        try:
            scene = self.link.scene_world.apply_scene(dict(revision=int(revision), state=state))
        except (KeyError, TypeError, ValueError) as exc:  # stale, or a malformed state (the plant node answered ok=False)
            raise Refused(f'scene change refused: {exc!r}') from None
        return dict(request_id=str(request_id), scene_state=scene)

    def trace_samples(self, after: int = 0) -> dict:
        """trace/1: controller ticks recorded after sequence ``after``, oldest first (read-only, no ownership needed).
        ``dropped`` counts the ticks in between that fell off the ring before the reader came back (0 when ``after`` is 0:
        a first read has no gap to measure); ``errors`` counts ticks that could not be recorded. The loop never waits for
        a reader: the ring overwrites, and the gap is reported, not hidden."""
        with self._cv:
            ring = [s for s in self._trace if s['seq'] > after]
            latest, errors = self._trace_seq, self._trace_errors
        dropped = max(0, ring[0]['seq'] - after - 1) if ring and after > 0 else 0
        return dict(samples=ring, latest_seq=latest, dropped=dropped, errors=errors, capacity=TRACE_TICKS)

    def _record_trace(self, health) -> None:
        """One per-arm controller diagnostic per tick (`control.diagnostic`), stamped with a
        sequence, the PC time, the authority lifecycle and the native reference. Never raises into the loop."""
        try:
            packet, received = self._last_packet
            sample = controller_diagnostic(self.link, self.controller, packet, received, health,
                                            reference_time=self.controller.exec_time)
            native = self._native
            # The controller runs an EXECUTION; the identity a client requested (and a pair joins its lineage on) is the
            # request id: ``plan_id`` carries that, ``execution_id`` the controller's own.
            execution = self._executions.get(sample['plan_id'])
            sample.update(execution_id=sample['plan_id'],
                          plan_id=sample['plan_id'] if execution is None else execution.request.request_id)
            sample.update(seq=self._trace_seq + 1, pc_monotonic=time.monotonic(),
                          lifecycle=self._authority.lifecycle.value, executing=bool(self.controller.runtime_state['executing']),
                          reference_id=getattr(native, 'id', None), reference_revision=getattr(native, 'revision', None))
            self._trace_seq += 1
            self._trace.append(sample)
            self.fault_trace.record(sample)
        except Exception:  # noqa: BLE001 -- diagnostics must not stop the control loop
            self._trace_errors += 1

    def state_history(self) -> dict:
        """state_history/1: the link's recent RT packets with their RT-host stamps, read-only (no ownership needed, like
        ``state``). ``age_ns`` is each packet's age on THIS host's monotonic clock at ``now_ns``; the RT stamp
        ``t_mono_ns`` is on the clock named by ``clock_domain`` (None: unattested)."""
        now = time.monotonic()
        samples = [dict(t_mono_ns=st.t_mono_ns, state_seq=st.state_seq, age_ns=int(max(0.0, now - rx) * 1e9),
                        session_id=st.session_id, arm_epoch=st.arm_epoch, stop_epoch=st.stop_epoch,
                        armed=st.armed, faulted=st.faulted, q=list(st.q), dq=list(st.dq))
                   for st, rx in self.link.recent_states()]
        health = self.link.motor_health()
        domain = getattr(self.link, 'clock_domain', None)
        return dict(clock_domain=None if domain is None else ':'.join(domain), now_ns=int(now * 1e9),
                    authority=dict(session_id=health['session_id'], arm_epoch=health['arm_epoch']),
                    samples=samples)

    # -- controller/1: runtime consequences, existing controller law -------------
    def controller_presets(self):
        return tuple(p.choice for p in self._presets.values())

    def controller_state(self):
        with self._cv:
            return self._controller_state()

    def _controller_state(self):
        mode = self._mode_feedback
        preset_id = None
        for name, preset in self._presets.items():
            if preset.choice.law != mode['law']:
                continue
            if mode['law'] == 'soft':
                if name == self._selected_soft:
                    preset_id = name
            elif (np.array_equal(preset.kp, mode['kp'])
                  and np.array_equal(preset.kd, mode['kd'])):
                preset_id = name
                break
        return ControllerState(mode['law'], preset_id, tuple(mode['kp']),
                               tuple(mode['kd']), self._payload)

    def select_control_preset(self, session, preset_id):
        return self._call(self._select_control_preset, session, preset_id)

    def _select_control_preset(self, session, preset_id):
        self._owns(session)
        preset = self._presets.get(preset_id)
        if preset is None:
            raise Refused(f'unknown controller preset {preset_id!r}')
        c = self.controller
        spec = None
        if preset.pose_hold is not None:
            state, _ = self.link.latest_state()
            if (self._native is not None and state is not None and state.reference
                    and state.reference['reference_id']):
                raise Refused('Soft unavailable after native reference use in this ARM epoch: '
                              'RT retains reference ownership after cancellation')
            try:
                spec = c.validate_pose_hold(dict(preset.pose_hold, id=next(self._soft_ids)))
            except (TypeError, ValueError) as exc:
                raise Refused(str(exc)) from exc
        if self._controller_state().preset_id == preset_id:
            return self._controller_state()
        for ex in list(self._executions.values()):
            if not ex.status.terminal:
                self._cancel_execution(ex, f'controller preset changed to {preset_id}')
        # The shared controller still owns resets, hold capture and Soft law.
        if spec is not None:
            self._selected_soft = preset_id
            c.on_control(pack_control_update(pose_hold=spec))
        else:
            self._selected_soft = None
            c.on_control(pack_control_update(gains=dict(kp=preset.kp.tolist(), kd=preset.kd.tolist())))
        self._pump_controller()
        state = self._controller_state()
        if state.law != preset.choice.law:
            raise Refused('controller did not confirm the requested law')
        return state

    def hold(self, session, reason='hold'):
        """Stop whatever runs and hold the MEASURED pose, still ARMed: live executions are cancelled, a setpoint is dropped, and
        the controller's ``frozen`` hold after a failed settle is cleared (the in-process controller's cancel/Stop). Returns
        the controller state. DISARM is the other way out of ``frozen``; this keeps the arm powered."""
        return self._call(self._hold, session, str(reason))

    def _hold(self, session, reason):
        self._owns(session)
        for ex in list(self._executions.values()):
            if not ex.status.terminal and (ex.stream is not None or ex.completed_at is None):
                self._cancel_execution(ex, reason)
        self.controller.on_control(pack_control_update(cancel=True, reason=reason))
        self._pump_controller()
        return self._controller_state()

    def set_payload(self, session, payload):
        return self._call(self._set_payload, session, payload)

    def _set_payload(self, session, payload):
        self._owns(session)
        payload = codec.payload_from(codec.encode(payload))
        if self._current() is not None:
            raise Refused('cancel and retire the arm execution before replacing payload')
        # Install the validated complete value once; the executor applies its
        # compensation, while this revision owns admission identity.
        try:
            self.controller.executor.set_payload(payload.mass_kg, com_offset=payload.com_ee)
        except ValueError as exc:
            raise Refused(str(exc)) from None
        self._payload = PayloadState(self._payload.revision + 1, payload)
        return self._payload

    def set_load_model(self, session, revision, model):
        return self._call(self._set_load_model, session, revision, model)

    def _set_load_model(self, session, revision, model):
        """Install a revisioned rigid-body load model (operator 2026-10-09, B1): this arm's RNEA from a body list, e.g. the base
        with the modules grafted onto it. Refused while an execution is live, for a revision not above the installed one, for
        another joint order, on a native-reference supervisor (the RT plays the reference), and when gravity at the measured
        pose is non-finite or above the policy's torque limits. Returns the installed revision."""
        from arm_control.dynamics import BodyListDynamics

        self._owns(session)
        if type(revision) is not int or revision <= self._load_model_revision:
            raise Refused(f'load model revision must be an integer above {self._load_model_revision}')
        if self._current() is not None:
            raise Refused('cancel and retire the arm execution before replacing the load model')
        if self._native is not None:
            raise Refused('a native-reference supervisor plays the reference on the RT box: no PC load model')
        if not isinstance(model, dict) or list(model.get('joints') or []) != list(self.joints):
            raise Refused(f'load model joints must be this arm\'s, in order: {list(self.joints)}')
        c = self.controller
        timeout = c.execution_policy.state_timeout_sec if c.execution_policy is not None else 0.5
        if c._last_state_at is None or c._clock() - c._last_state_at > timeout:
            raise Refused('no fresh measured pose to check the load model at')
        try:
            dynamics = BodyListDynamics(model)
            tau = dynamics.gravity(np.asarray(c.last_state.position, float))
        except (ValueError, KeyError, TypeError) as exc:
            raise Refused(f'load model refused: {exc}') from None
        limits = c.execution_policy.torque_limits if c.execution_policy is not None else None
        if not np.isfinite(tau).all() or (limits is not None and np.any(np.abs(tau) > limits)):
            raise Refused(f'load model gravity {np.round(tau, 3).tolist()} N.m at the measured pose exceeds the torque limits')
        try:
            c.executor.set_dynamics(dynamics)
        except ValueError as exc:
            raise Refused(str(exc)) from None
        self._load_model_revision = revision
        return revision

    # -- setpoint/1: a live joint target with a deadman (console jog, policy actions) -------
    def setpoint(self, session, q):
        """Hold the arm at ``q`` (arm joints, rad) on the joint law until the next setpoint, or until none arrives within the
        controller's jog timeout: then it holds the MEASURED pose (not the last target). Refused while an execution is live,
        for a target that moves any joint further past its position limit than the pose the arm ARMed at (never further out,
        operator 2026-10-09), and for one further from
        the measured pose than the policy's abort_pos_err_rad (the controller would cancel it on the next tick, after telling
        the caller it was taken). Returns the timeout, which is the rate a caller must keep up. Within that, the step size is the
        caller's envelope (the console's JogLimits); the supervisor checks only what holds for every caller."""
        return self._call(self._setpoint, session, q)

    def _setpoint(self, session, q):
        self._owns(session)
        if self._native is not None:
            raise Refused('setpoints are on the streamed path; this supervisor plays the native reference (use stream/1)')
        q = np.asarray(q, dtype=float).ravel()
        if q.shape != (len(self.joints),) or not np.isfinite(q).all():
            raise Refused(f'a setpoint is {len(self.joints)} finite joint positions')
        if self._current() is not None:
            raise Refused('an execution is live: cancel it before streaming setpoints')
        c = self.controller
        refusal = c.jog_refusal()  # the controller's own state first: its reason is the real one
        if refusal is not None:
            raise Refused(f'setpoint refused: {refusal}')
        policy, measured = c.execution_policy, np.asarray(c.last_state.position, float)
        band = c.envelope_band()  # the envelope, widened to where the arm has been since ARM: never further out
        if band is not None:
            worse = np.flatnonzero((q < band[0] - ENVELOPE_TOL) | (q > band[1] + ENVELOPE_TOL))
            if worse.size:
                raise Refused(f'setpoint moves {self.joints[worse[0]]} past its position limit')
        step = float(np.max(np.abs(q - measured)))
        if step > policy.abort_pos_err_rad:
            raise Refused(f'setpoint is {step:.3f} rad from the measured pose; the controller aborts beyond '
                          f'{policy.abort_pos_err_rad:g} rad (abort_pos_err_rad): send nearer targets')
        refusal = c.on_jog(pack_json_message('jog', {'q': q.tolist(), 'reason': 'setpoint'}))
        if refusal is not None:
            raise Refused(f'setpoint refused: {refusal}')
        return {'timeout_s': c.jog_timeout_s}

    # -- stream/1: same execution ledger, reference owner and control loop -------
    def open_stream(self, session, request):
        return self._call(self._open_stream, session, request)

    def revise_stream(self, session, request):
        return self._call(self._revise_stream, session, request)

    def stream_anchor(self, session, execution_id):
        return self._call(self._stream_anchor, session, execution_id)

    def abort_stream_revision(self, session, execution_id, request_id):
        return self._call(self._stream_abort, session, execution_id, request_id)

    def _stream_get(self, session, execution_id):
        ex = self._get(execution_id)
        if ex.stream is None or ex.stream.owner != session:
            raise Refused('stream is not owned by this service session')
        return ex

    def stream_state(self, session, execution_id):
        with self._cv:
            return self._stream_view(self._stream_get(session, execution_id))

    def stream_revision(self, session, execution_id, request_id):
        with self._cv:
            self._stream_get(session, execution_id)
            attempt = self._stream_attempts.get((session, request_id))
            if attempt is None or attempt.status.execution_id != execution_id:
                raise Refused('revision history unavailable')
            return attempt.status

    def _stream_view(self, ex):
        s, h = ex.stream, self._native
        selected = h is not None and h.key == ex.status.execution_id and s.selected is not None
        ref = h.feedback if selected else None
        pending = self._stream_attempts.get((s.owner, s.pending))
        return StreamView(ex.status, s.opened.model_token, s.opened.scene_token,
                          s.opened.stream_policy_id,
                          h.id if ref else None, ref['revision'] if ref else None,
                          ref['phase'] if ref else None,
                          h.protected_until if ref else None,
                          float(h.trajectory.times[-1]) if ref else None,
                          h.terminal if ref else None, s.pending,
                          pending.status.revision if pending else None, s.anchor)

    def _stream_retry(self, session, request):
        self._owns(session)
        if self.extensions.get('stream') != 1:
            raise Refused('stream/1 requires native reference and trusted admission')
        old = self._stream_attempts.get((session, request.request_id))
        if old is not None and codec.encode(old.request) != codec.encode(request):
            raise Refused('request_id already used with different content')
        return old

    def _stream_slot(self, session, sequence, kind):
        marks = self._stream_marks.setdefault(session, [0, 0])
        if sequence <= marks[kind]:
            raise Refused('stale sequence or evicted history')
        if self._stream_job is not None:
            raise Refused('admission worker busy; no queued intents')
        return marks

    def _stream_facts_error(self, ex, facts):
        s = ex.stream
        if self.controller.mode_law != 'joint':
            return 'stream requires joint-impedance mode'
        if self._payload.value.mass_kg != 0. or s.payload_revision != self._payload.revision:
            return 'transient PC compensation payload must be zero for stream/1'
        if facts != s.facts:
            return 'model/scene/policy snapshot changed'
        if facts.model_token != self.controller.model_token or facts.model_token != s.opened.model_token:
            return 'stream model differs from bound execution model'
        if facts.scene_token != s.opened.scene_token or facts.policy.stream_policy_id != s.opened.stream_policy_id:
            return 'stream scene/policy differs from trusted composition'
        if s.epoch != (self._owner, self.link.session_id, self._authority.arm_epoch):
            return 'stream owner/session/ARM epoch changed'
        kp, kd = s.gains
        if (not np.array_equal(kp, self.controller.executor.kp)
                or not np.array_equal(kd, self.controller.executor.kd)):
            return 'stream gains changed'
        return ''

    def _open_stream(self, session, request):
        request = codec.stream_open_from(codec.encode(request))
        old = self._stream_retry(session, request)
        if old is not None:
            return self._stream_view(self._get(old.status.execution_id))
        marks = self._stream_slot(session, request.open_seq, 0)
        if self._current() is not None:
            raise Refused('execution busy; cancel explicitly before opening another stream')
        facts = self._stream_admission.snapshot()
        eid = f'{self.instance_id}.{next(self._ids)}.{uuid.uuid4().hex[:8]}'
        at = self.now()
        receipt = Receipt(eid, request.request_id, self.instance_id, Phase.REQUESTED, at)
        stream = _Stream(session, request, facts,
                         (session, self.link.session_id, self._authority.arm_epoch),
                         tuple(x.copy() for x in self._profile_gains), self._payload.revision,
                         pending=request.request_id)
        ex = _Execution(request.as_request(), ExecutionStatus.start(receipt), [receipt], stream=stream,
                        payload_revision=self._payload.revision)
        attempt = self._new_attempt(ex, request, request.initial, at)
        self._executions[eid] = ex
        self._stream_attempts[session, request.request_id] = attempt
        marks[0] = request.open_seq
        self._queue_stream_check(ex, attempt)
        return self._stream_view(ex)  # worker results only applied on a later loop tick

    def _new_attempt(self, ex, request, curve, at, *, cut=0., terminal=False):
        deadline = at.ns + int(ex.stream.facts.policy.max_candidate_age_s * 1e9)
        if request.admit_by is not None:
            deadline = min(deadline, request.admit_by.ns)
        expected = request.expected_revision if isinstance(request, StreamRevision) else None
        status = RevisionStatus(ex.status.execution_id, request.request_id, expected,
                                1 if expected is None else expected+1, Phase.REQUESTED,
                                SelectionState.NOT_REQUESTED, at)
        return _Attempt(request, status, curve, ex.stream.facts, deadline, cut, terminal)

    def _stream_anchor(self, session, execution_id):
        self._owns(session)
        ex = self._stream_get(session, execution_id)
        s, h, now = ex.stream, self._native, self.now()
        if ex.status.terminal or ex.drop is not None or s.pending is not None or h.busy or h.key != execution_id:
            raise Refused('stream candidate slot unavailable')
        if not h.selected or h.terminal or self._state.validity is not Validity.VALID:
            raise Refused('stream is stale, unselected or terminal')
        if s.anchor is not None and s.anchor.expires_at.ns > now.ns:
            return s.anchor
        policy, curve = s.facts.policy, h.trajectory
        minimum = h.point.time + policy.min_commit_lead_s + policy.max_candidate_age_s
        eligible = np.flatnonzero((curve.times >= h.protected_until) & (curve.times > minimum))
        if not len(eligible):
            raise Refused('no eligible whole-segment cut in selected reference')
        i = int(eligible[0])
        s.anchor = SpliceAnchor(uuid.uuid4().hex, h.revision, float(curve.times[i]),
                                tuple(curve.positions[i]), tuple(curve.velocities[i]), now,
                                Stamp(now.ns + int(policy.max_candidate_age_s*1e9), Clock.PC, self.domain))
        return s.anchor

    def _revise_stream(self, session, request):
        request = codec.stream_revision_from(codec.encode(request))
        old = self._stream_retry(session, request)
        if old is not None:
            return old.status
        ex = self._stream_get(session, request.execution_id)
        marks = self._stream_slot(session, request.proposal_seq, 1)
        s, h, at = ex.stream, self._native, self.now()
        if ex.status.terminal or ex.drop is not None or s.pending is not None or h.busy or h.key != request.execution_id:
            raise Refused('stream candidate slot unavailable')
        anchor = s.anchor
        if (anchor is None or anchor.anchor_id != request.anchor_id or anchor.expires_at.ns <= at.ns
                or request.expected_revision != anchor.expected_revision or h.revision != request.expected_revision):
            raise Refused('anchor is missing, consumed, expired or belongs to another revision')
        # Curve validity is admission work: accepted intent consumes its challenge
        # and sequence even when splice/scene/limit checks subsequently refuse it.
        attempt = self._new_attempt(ex, request, request.suffix, at,
                                    cut=anchor.cut_phase, terminal=request.terminal)
        attempt.deadline_ns = min(attempt.deadline_ns, anchor.expires_at.ns)
        self._stream_attempts[session, request.request_id] = attempt
        s.anchor, s.pending = None, request.request_id
        marks[1] = request.proposal_seq
        try:
            attempt.curve = h.splice(request.suffix, cut=anchor.cut_phase,
                                     expected_revision=request.expected_revision)
        except ValueError as exc:
            self._stream_fail(ex, attempt, str(exc))
        else:
            self._queue_stream_check(ex, attempt)
        return attempt.status

    def _queue_stream_check(self, ex, attempt):
        request, state = attempt.request, self._state
        reason = self._stream_facts_error(ex, self._stream_admission.snapshot())
        reason = reason or admission_refusal(replace(ex.request, admit_by=request.admit_by), self.identity, self.capabilities,
                                             self._authority, now=self.now(),
                                             model_token=self.controller.model_token)
        if isinstance(request, StreamRevision) and request.scene_token != ex.stream.facts.scene_token:
            reason = reason or 'revision scene differs from stream'
        if request.admit_by is not None and request.admit_by.domain != self.domain:
            reason = reason or 'deadline belongs to another PC clock'
        if state.validity is not Validity.VALID or state.q is None or state.dq is None:
            reason = reason or 'fresh finite position/velocity required'
        curve, policy = attempt.curve, attempt.facts.policy
        bounds = self.controller.execution_policy
        if any(getattr(bounds, name) is None for name in
               ('position_lower', 'position_upper', 'velocity_limits', 'acceleration_limits')):
            reason = reason or 'stream requires explicit complete motion limits'
        phase = 0. if isinstance(request, StreamOpen) else self._native.point.time
        if curve.times[-1] - phase > policy.max_future_s:
            reason = reason or 'candidate exceeds stream future budget'
        if not reason:
            kp, kd = ex.stream.gains
            plan = dict(times=curve.times, positions=curve.positions, velocities=curve.velocities, kp=kp, kd=kd)
            start = state.q if isinstance(request, StreamOpen) else curve.positions[0]
            reason = self.controller.execution_policy.plan_error(plan, start, len(self.joints))
            if isinstance(request, StreamOpen):
                tolerance = np.asarray(policy.start_velocity_tolerance_rad_s)
                if tolerance.shape != state.dq.shape or np.any(np.abs(state.dq-curve.velocities[0]) > tolerance):
                    reason = reason or 'stream start velocity mismatch'
        # Even immediate admission refusal is completed asynchronously so the
        # bounded open reply always represents REQUESTED allocation.
        def check():
            return reason or self._stream_admission.check(curve, attempt.facts)
        self._stream_job = (ex.status.execution_id, request.request_id,
                            self._stream_worker.submit(check))

    def _stream_fail(self, ex, attempt, reason, *, refused=False, expired=False):
        phase = Phase.EXPIRED if expired and attempt.status.phase is Phase.ADMITTED else Phase.REJECTED
        if attempt.upload_started and ex.stream.selected is not None:
            # A refused COMMIT can leave the sealed upload in the non-RT slot.
            # Clear that unselected candidate before allowing another revision.
            self._native.abort_candidate()
        attempt.status = replace(attempt.status, phase=phase, reason=reason, at=self.now(),
                                 selection_state=SelectionState.REFUSED if refused else attempt.status.selection_state)
        ex.stream.pending = None
        self._native_pending = None
        if ex.stream.selected is None and not ex.status.terminal:
            self._receipt(ex, phase, reason)
            if self.controller.current_plan_id == ex.status.execution_id:
                self._drop_staged(ex, reason)

    def _stream_abort(self, session, execution_id, request_id):
        self._owns(session)
        ex = self._stream_get(session, execution_id)
        attempt = self._stream_attempts.get((session, request_id))
        if attempt is None or attempt.status.execution_id != execution_id:
            raise Refused('revision history unavailable')
        if attempt.status.selection_state is not SelectionState.NOT_REQUESTED:
            return attempt.status  # PENDING remains structurally uncertain; no mutation
        if ex.stream.pending != request_id:
            return attempt.status
        if attempt.upload_started:
            self._native.abort_candidate()
        phase = Phase.REJECTED if attempt.status.phase is Phase.REQUESTED else Phase.CANCELLED
        attempt.status = replace(attempt.status, phase=phase, at=self.now(), reason='candidate aborted')
        ex.stream.pending = None
        if ex.stream.selected is None and not ex.status.terminal:
            self._receipt(ex, phase, 'candidate aborted')
            self._drop_staged(ex, 'candidate aborted')
        return attempt.status

    def _stream_select(self, ex):
        # No COMMIT here: upload and final snapshot recheck progress on ticks.
        self._native_pending = ex.status.execution_id

    def _stream_step(self):
        h = self._native
        if self._stream_job is not None and self._stream_job[2].done():
            eid, rid, job = self._stream_job
            self._stream_job = None
            ex = self._executions.get(eid)
            if ex is not None and not ex.status.terminal and ex.stream.pending == rid:
                attempt = self._stream_attempts[ex.stream.owner, rid]
                try:
                    reason = job.result() or self._stream_facts_error(ex, self._stream_admission.snapshot())
                except Exception as exc:
                    reason = f'trusted admission failed: {exc}'
                if reason or self.now().ns >= attempt.deadline_ns:
                    self._stream_fail(ex, attempt, reason or 'admission deadline expired')
                else:
                    if isinstance(attempt.request, StreamOpen):
                        q = attempt.curve
                        kp, kd = ex.stream.gains
                        self.controller.on_plan(pack_plan(
                            plan_id=eid, phase=rid, gated=True, times=q.times, positions=q.positions,
                            velocities=q.velocities, kp=kp, kd=kd, model_token=ex.request.model_token))
                        self._pump_controller()
                        if ex.status.phase is not Phase.ADMITTED:
                            self._stream_fail(ex, attempt, ex.status.reason or 'controller refused')
                            return True
                    attempt.status = replace(attempt.status, phase=Phase.ADMITTED, at=self.now())
                    if isinstance(attempt.request, StreamRevision) or ex.request.start is Start.ON_ADMIT:
                        self._stream_select(ex)
        current = self._current()
        ex = self._executions.get(current.execution_id) if current else None
        if ex is None or ex.stream is None:
            return False
        s, eid = ex.stream, ex.status.execution_id
        if s.anchor is not None and s.anchor.expires_at.ns <= self.now().ns:
            s.anchor = None  # lifecycle maintenance, never the state read
        reason = self._stream_facts_error(ex, self._stream_admission.snapshot())
        if reason:
            self._cancel_execution(ex, reason)
            return True
        if h.key == eid and h.error in ('REFERENCE_UNDERFLOW', 'REFERENCE_EXPIRED'):
            if ex.status.phase is Phase.ADMITTED:
                self._receipt(ex, Phase.SELECTED, reference_id=h.id, reference_revision=h.revision)
            self._receipt(ex, Phase.FAULTED, h.error)
            for rid in (s.selected, s.pending):
                failed = self._stream_attempts.get((s.owner, rid))
                if failed is not None:
                    selected = failed.status.revision == h.revision
                    failed.status = replace(failed.status,
                        phase=Phase.FAULTED if selected else Phase.EXPIRED,
                        selection_state=SelectionState.SELECTED if selected else SelectionState.REFUSED,
                        at=self.now(), reason=h.error)
            self._native_failed = True
            self._drop_staged(ex, h.error)
            s.pending = s.anchor = None
            return True
        attempt = self._stream_attempts.get((s.owner, s.pending))
        if attempt is not None:
            pending = attempt.status.selection_state is SelectionState.PENDING
            if not pending and self.now().ns >= attempt.deadline_ns:
                self._stream_fail(ex, attempt, 'candidate deadline expired', expired=True)
                return True
            if attempt.status.phase is Phase.ADMITTED and self._native_pending == eid:
                if not attempt.upload_started:
                    if h.busy:
                        if h.key != eid:
                            h.cancel()
                        return True
                    try:
                        if s.selected is None:
                            h.stage(eid, attempt.curve, stream=True, terminal=False,
                                    selection_deadline_ns=attempt.deadline_ns)
                        else:
                            h.stage_suffix(attempt.curve, cut=attempt.cut,
                                           expected_revision=attempt.status.expected_revision,
                                           terminal=attempt.terminal,
                                           latest_phase=attempt.cut-s.facts.policy.min_commit_lead_s,
                                           selection_deadline_ns=attempt.deadline_ns)
                        attempt.upload_started = True
                        self._native_failed = False
                    except (ValueError, RuntimeError) as exc:
                        self._stream_fail(ex, attempt, str(exc))
                elif h.candidate_selection == 'refused':
                    self._stream_fail(ex, attempt, h.candidate_error, refused=True,
                                      expired=h.candidate_result == rtp.REF_LATE)
                elif h.candidate_error and h.candidate_selection == 'not_requested':
                    self._stream_fail(ex, attempt, h.candidate_error)
                elif h.ready:
                    reason = selection_refusal(self._authority) or self._stream_facts_error(ex, self._stream_admission.snapshot())
                    if self._state.validity is not Validity.VALID:
                        reason = reason or 'observation became stale before selection'
                    if s.selected is None:
                        reason = reason or self.controller.release_error(eid)
                        if self._state.dq is None or np.any(np.abs(self._state.dq-attempt.curve.velocities[0]) > np.asarray(s.facts.policy.start_velocity_tolerance_rad_s)):
                            reason = reason or 'start velocity changed before selection'
                    if reason:
                        self._stream_fail(ex, attempt, reason)
                    else:
                        attempt.status = replace(attempt.status, selection_state=SelectionState.PENDING, at=self.now())
                        h.approve()
                elif (h.candidate_selection == 'selected' and h.key == eid
                      and h.revision == attempt.status.revision):
                    attempt.status = replace(attempt.status, phase=Phase.SELECTED,
                                             selection_state=SelectionState.SELECTED, at=self.now())
                    s.selected, s.pending = attempt.request.request_id, None
                    self._native_pending = None
                    if ex.status.phase is Phase.ADMITTED:
                        self._receipt(ex, Phase.SELECTED, reference_id=h.id, reference_revision=h.revision)
                        self.controller.on_control(pack_control_update(execute=eid))
                        self._pump_controller()
                    self.controller.select_reference_window(eid, h.trajectory, terminal=h.terminal)
        selected = self._stream_attempts.get((s.owner, s.selected))
        if selected is not None and selected.status.phase is Phase.SENT:
            confirmed = self._tracker.confirmed(h.arm_epoch)
            if confirmed is not None and epoch_reached(confirmed, selected.sent_seq):
                selected.status = replace(selected.status, phase=Phase.APPLIED, at=self.now())
        return True


    # -- construction ----------------------------------------------------------
    def _build_controller(self) -> ArmController:
        return build_arm_controller(
            self.cfg, _Node(self), arm_id=self.instance_id, require_policy=True,
            # the plant's clock: wall time for a real arm; a sim world's own (it stands still while the world is paused for a
            # scene change, as plant time did for the plant-clocked controller this supervisor replaced)
            clock=getattr(self.link, 'clock', time.monotonic), plant_reports_health=True,
            reference_source=(None if self._native is None else lambda: self._native.point))

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        """Claim the RT server (never ARM) and start the 100 Hz loop."""
        self.link.open()
        # OPT-IN, like the server's --enable-reference: the v7 native path is
        # not qualified for any deployment. The registry owns the per-instance
        # fact and stages it into rt.native_reference, so both sides flip
        # together; a config that never declares it keeps the streamed path.
        if (self.controller.model_token and (self.cfg.get("rt") or {}).get("native_reference", False)
                and self.link.negotiate_reference(self.controller.model_token)):
            self._native = ReferenceHorizon(self.link)
            self.controller.bind_reference_source(lambda: self._native.point)
        elif self.link.hold_ms is not None:
            self.controller.set_stream_hold_deadline(self.link.hold_ms)
        ident = self.link.identity
        self.identity = RobotIdentity(self.instance_id, tuple(self.joints),
                                      f"rt:{self.link.backend_name}", ident.model_id)
        self.capabilities = Capabilities(
            CONTRACT_VERSION, offered_profiles(self.link, native=self._native is not None,
                                               cartesian_kp_min=self.controller.execution_policy.cartesian_kp_min),
            protocol_version=rtp.VERSION,
            pose_hold=bool(self.link.supports_pose_hold),
            rt_setup=str(getattr(self.link, "rt_setup", "unknown")))
        self._thread = threading.Thread(target=self._loop, name="supervisor", daemon=True)
        self._thread.start()

    def close(self, timeout_s: float = 5.0) -> None:
        """Bounded: stop the loop, DISARM through the link, release the socket."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._stream_worker.shutdown(wait=False, cancel_futures=True)
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
        # After the DISARM: an open fault window is written with what it has; it never delays the stop (never raises).
        self.fault_trace.close(timeout_s, record_errors=self._trace_errors)

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
            if self._scene_owner == session:
                self._scene_owner = None
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

    def release(self, session: str, execution_id: str, start_at: Stamp | None = None) -> ExecutionStatus:
        """``start_at`` (scheduled_release/1): anchor the trajectory at this PC-monotonic stamp, not at the release (a pair
        released through two supervisors starts both members at one stamp). It must lie between the CTRL-040 minimum lead
        and budget ahead of this supervisor's clock."""
        return self._call(self._release, session, execution_id, start_at)

    def cancel(self, session: str, execution_id: str, reason: str = "cancelled"
               ) -> ExecutionStatus:
        return self._call(self._cancel, session, execution_id, reason)

    # -- group/1: CTRL-040 RT group execution, through this supervisor's one RT session ----------------
    def group_join(self, session: str, group_id: int, member_index: int, member_count: int) -> dict:
        # The RT exchange runs on the CALLER's thread, never on the 100 Hz loop (it can wait for an ack); the loop only
        # checks ownership and records the result.
        self._call(self._group_check, session)
        # Recorded BEFORE the exchange: if its outcome is uncertain the RT may have joined, and group_cancel must still
        # reach it. A clear refusal (the RT answered) forgets it again.
        epoch, previous = self._call(self._group_set, session, (group_id, member_index, member_count))
        try:
            reply = self.link.group_join(group_id, member_index, member_count)
        except ReferenceRefused as exc:
            self._call(self._group_unset, session, previous)  # the RT answered: restore what it is still joined to
            raise Refused(f'group join refused: {exc} {getattr(exc, "reply", None) or ""}') from exc
        except RtLinkError as exc:
            raise Refused(f'group join outcome unknown ({exc}); cancel the group before joining again') from exc
        domain = getattr(self.link, 'clock_domain', None)
        return dict(clock_domain=None if domain is None else ':'.join(domain), state=reply.get('state'),
                    arm_epoch=epoch, reference_limit=rtp.MAX_REFERENCE_KNOTS)

    def group_release(self, session: str, execution_id: str, group_id: int, budget_ns: int) -> ExecutionStatus:
        return self._call(self._group_release_op, session, execution_id, group_id, budget_ns)

    def group_cancel(self, session: str, group_id: int) -> dict:
        # Order matters: (1) the loop forgets the group and cancels its members' executions and holds off staging a measured
        # hold; (2) here, wait for the member-cancel jobs to leave the link worker, so no other exchange sets the shared
        # interrupt flag under the GROUP_CANCEL; (3) the RT exchange; (4) the loop may stage the hold again. Otherwise the
        # RT's retire hits a hold that was just selected, or the cancel interrupts it, and the arm goes silent and latches.
        joined = self._call(self._group_take, session, group_id)
        if joined is None:
            return dict(cancelled=False, reason='not joined')
        try:
            h = self._native
            end = time.monotonic() + 2.0
            while h is not None and h.busy and time.monotonic() < end:
                time.sleep(0.005)
            self.link.group_cancel(group_id, joined[1], joined[2])  # on the caller's thread, see group_join
        except ReferenceRefused as exc:
            if exc.result == rtp.REF_GROUP_REFUSED:  # the RT answered that it is not in this group: nothing to retry
                return dict(cancelled=False, reason='the RT is not in this group')
            self._call(self._group_restore, session, joined)
            raise Refused(f'group cancel not confirmed: {exc} {getattr(exc, "reply", None) or ""}') from exc
        except RtLinkError as exc:
            self._call(self._group_restore, session, joined)  # not confirmed: a retry must reach the RT again
            raise Refused(f'group cancel not confirmed: {exc} {getattr(exc, "reply", None) or ""}') from exc
        finally:
            self._call(self._group_cancel_done)
        return dict(cancelled=True)

    def group_report(self, execution_id: str) -> dict:
        """The RT's group COMMIT reply for this execution (``selected_ns``, ``deadline_ns``, ``reference_id``), its
        refusal (``error``), or ``{}`` while it is still pending."""
        with self._cv:
            return dict(self._group_reports.get(execution_id, {}))

    def group_events(self, session: str) -> list:
        """Unsolicited RT group reports (a peer fault/cancel/loss retired this member), drained. Owner only: another
        client must not take the owner's events."""
        self._call(self._group_check, session)
        out = []
        events = getattr(self.link, 'group_events', None)
        while events:
            out.append(dict(events.popleft()))
        return out

    def _group_check(self, session):
        self._owns(session)
        if self._native is None or not getattr(self.link, 'group_capable', False):
            raise Refused('the RT server has no group record (group/1 is unavailable)')

    def _group_set(self, session, joined):
        self._owns(session)
        if self._group is not None and self._group[0] != joined[0] and self._group[3] == self._authority.arm_epoch:
            # The RT holds one group record: with an uncertain exchange a second group would leave it unknown which one
            # the RT is in. Cancel the first. (A group from an earlier ARM epoch is stale: it cannot be released any more.)
            raise Refused(f'group join refused: already joined to group {self._group[0]}: cancel it before joining another')
        previous, self._group = self._group, (*joined, self._authority.arm_epoch)
        return self._authority.arm_epoch, previous

    def _group_release_op(self, session, execution_id, group_id, budget_ns):
        self._owns(session)
        if self._group is None or self._group[0] != group_id:
            raise Refused('join the group before releasing a member')
        ack_ns = int(float(self.link.config.ack_timeout_s) * 1e9)
        if not 0 < budget_ns <= min(rtp.GROUP_MAX_BUDGET_NS, ack_ns // 2):
            raise Refused('budget must be positive and at most half the RT acknowledgement timeout (the COMMIT_GROUP wait)')
        if self._group[3] != self._authority.arm_epoch:
            self._group = None
            raise Refused('the ARM epoch changed since the group was joined: join again')
        ex = self._get(execution_id)
        if ex.stream is not None or ex.request.start is not Start.ON_RELEASE:
            raise Refused('a group member is a planned ON_RELEASE execution')
        self._group_release[execution_id] = (group_id, self._group[1], int(budget_ns))
        while len(self._group_release) > 16:
            self._group_release.pop(next(iter(self._group_release)))
        self._group_reports.pop(execution_id, None)
        try:
            return self._release(session, execution_id)
        except BaseException:
            self._group_release.pop(execution_id, None)
            raise

    def _group_take(self, session, group_id, cancel_members=True):
        self._owns(session)
        if self._group is None or self._group[0] != group_id:
            return None
        joined, self._group = self._group, None
        if cancel_members:
            # Members released for this group, or committing through it right now, must not fall back to a plain COMMIT
            # and start alone: cancel their executions with the group.
            h = self._native
            eids = list(self._group_release)
            if h is not None and h.group_used is not None and h.group_used[0] == group_id and h.key is not None:
                eids.append(h.key)
            for eid in dict.fromkeys(eids):
                ex = self._executions.get(eid)
                if ex is not None and not ex.status.terminal and ex.completed_at is None:
                    self._cancel_execution(ex, 'group cancelled')
            # No measured hold is staged until the RT CANCEL has been sent (group_cancel clears this; the deadline is only
            # a bound if it never gets to).
            now = time.monotonic()
            self._group_cancelling = now + 2.0 + 2 * float(self.link.config.ack_timeout_s)  # the busy wait plus the exchange
            # The re-stage grace starts here, not after the RT reply: a caller-thread delay after the reply must not
            # let the RT's retire of a standing hold be taken for a native failure.
            self._group_hold_grace = self._group_cancelling + 1.0
        self._group_release.clear()
        return joined

    def _group_restore(self, session, joined):
        self._owns(session)
        if self._group is None:
            self._group = joined

    def _group_cancel_done(self):
        # The RT's GROUP_CANCEL retires whatever reference this member has selected, including a standing measured hold
        # (an RT-side scoping question, CTRL-042). That retire is the answer to OUR cancel, so the hold is re-staged for a
        # short grace period instead of ending the command stream as a native failure.
        self._group_cancelling = 0.0
        self._group_hold_grace = time.monotonic() + 1.0

    def _group_unset(self, session, previous):
        self._owns(session)
        self._group = previous

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
        self._stream_marks.pop(session, None)

    def _abandon(self, reason: str) -> None:
        """Cancel the owner's live executions and its live setpoint, and hold; stay armed."""
        for ex in list(self._executions.values()):
            if not ex.status.terminal and ex.completed_at is None:
                self._cancel_execution(ex, reason)
        self.controller.clear_jog()
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
        for ex in list(self._executions.values()):
            if not ex.status.terminal and (ex.stream is not None or ex.completed_at is None):
                # Also planned native executions: their CANCELLED waits for the RT to show
                # the DISARM (rt_disarmed / rt_epoch), not for the request being sent.
                self._cancel_execution(ex, "disarmed")
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
        ex = _Execution(request, ExecutionStatus.start(first), [first],
                        payload_revision=self._payload.revision)
        self._executions[eid] = ex
        self._by_request[request.request_id] = eid
        refusal = (
            # Operator 2026-10-09, enforced here as well as by not offering the profile: the native reference uploads joints
            # only, so a Cartesian leg on it would run as joints. A distinct reason, checked first.
            ("Cartesian legs are streamed only; this supervisor plays the native reference"
             if request.cartesian is not None and self._native is not None else "")
            or admission_refusal(request, self.identity, self.capabilities, self._authority,
                              now=self.now(), model_token=self.controller.model_token)
            or supersession_refusal(request, current)
            or ("reference selection is in flight; cancel before a new admission"
                if request.supersedes and self._native is not None
                and self._native.key == request.supersedes
                and self._native.selection_requested else "")
            or (f"the {request.profile} profile needs waypoint velocities"
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
                             cartesian=None if request.cartesian is None else request.cartesian.spec(),
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

    def _release(self, session: str, execution_id: str, start_at: Stamp | None = None) -> ExecutionStatus:
        self._owns(session)
        ex = self._get(execution_id)
        if start_at is not None:
            if self._native is not None or ex.stream is not None:
                raise Refused('a scheduled release needs the supervisor\'s own trajectory clock (a planned leg, not native)')
            if start_at.clock is not Clock.PC or not self.domain or start_at.domain != self.domain:
                raise Refused(f'start_at is not on this supervisor\'s clock ({self.domain or "unattested"})')
            lead = (start_at.ns - time.monotonic_ns()) / 1e9
            if not GROUP_MIN_LEAD_MS / 1e3 <= lead <= GROUP_BUDGET_S:
                raise Refused(f'start_at is {lead * 1e3:.1f} ms ahead: it must be {GROUP_MIN_LEAD_MS} to '
                              f'{GROUP_BUDGET_S * 1e3:.0f} ms ahead')
        if ex.stream is not None:
            if ex.status.phase is not Phase.ADMITTED:
                raise Refused('stream release requires ADMITTED')
            if ex.request.start is not Start.ON_RELEASE:
                raise Refused('stream starts on admission')
            self._stream_select(ex)
            return ex.status
        if ex.status.phase is not Phase.ADMITTED:
            return ex.status  # already selected or ended: release is idempotent
        if ex.request.start is not Start.ON_RELEASE:
            raise Refused(f"{execution_id} starts on admission, not on release")
        self._select(ex, start_at)
        return ex.status

    def _select(self, ex: _Execution, start_at: Stamp | None = None) -> None:
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
        if start_at is None:
            self.controller.on_control(pack_control_update(execute=ex.status.execution_id))
        else:  # the stamp on the controller's own clock (a sim world's runs paused with the world)
            c = self.controller
            c._release(ex.status.execution_id, start=c.exec_time + (start_at.ns - time.monotonic_ns()) / 1e9)
        self._pump_controller()

    def _cancel(self, session: str, execution_id: str, reason: str) -> ExecutionStatus:
        self._owns(session)
        ex = self._get(execution_id)
        if not ex.status.terminal and ex.completed_at is None:
            self._cancel_execution(ex, reason or "cancelled")
        return ex.status

    def _cancel_execution(self, ex: _Execution, reason: str) -> None:
        if ex.drop is not None:
            return  # already cancelled, waiting for the RT to confirm the drop
        ex.cancel_reason = reason
        ex.completed_at = None
        if ex.stream is not None:
            ex.stream.anchor = None
            h = self._native
            if ex.status.phase is Phase.REQUESTED:
                self._receipt(ex, Phase.REJECTED, reason)
                self._settle_stream_attempts(ex, reason)
            elif not ex.status.terminal:
                if h is not None and h.key == ex.status.execution_id and h.selection_requested:
                    # The RT may hold this reference: CANCELLED means the drop was
                    # confirmed, so the receipt waits for the CANCEL/ABORT ack or for
                    # the RT itself to show the reference retired (_confirm_drops).
                    ex.drop = (reason, h.id, h._session, h._epoch)
                else:
                    self._receipt(ex, Phase.CANCELLED, reason)
                    self._settle_stream_attempts(ex, reason)
            ex.stream.pending = None
        elif (not ex.status.terminal and ex.status.phase is not Phase.REQUESTED
                and self._native is not None and self._native.key == ex.status.execution_id
                and self._native.selection_requested):
            # A planned native execution: the controller's own leg cancel is not the RT's
            # confirmation either (see above).
            ex.drop = (reason, self._native.id, self._native._session, self._native._epoch)
        self._retire_native(ex.status.execution_id)
        if self.controller.current_plan_id == ex.status.execution_id:
            self.controller.on_control(pack_control_update(
                cancel=True, reason=reason, cancel_plan_id=ex.status.execution_id))
        self._pump_controller()

    def _settle_stream_attempts(self, ex: _Execution, reason: str) -> None:
        for (owner, _), attempt in self._stream_attempts.items():
            if owner == ex.stream.owner and attempt.status.execution_id == ex.status.execution_id:
                phase = Phase.REJECTED if attempt.status.phase is Phase.REQUESTED else Phase.CANCELLED
                attempt.status = replace(attempt.status, phase=phase, at=self.now(), reason=reason)

    def _confirm_drops(self, state) -> None:
        """Write the CANCELLED receipt of executions whose RT drop is now confirmed:
        the CANCEL/ABORT was acknowledged (or COMMIT provably never selected), or the
        RT's own state shows the reference retired or replaced. An unconfirmed drop
        stays non-terminal; nothing is claimed."""
        h = self._native
        for ex in list(self._executions.values()):
            if ex.drop is None or ex.status.terminal:
                continue
            reason, ref_id, session, epoch = ex.drop
            how = h.dropped.get(ex.status.execution_id) if h is not None else None
            if how in ('unconfirmed', None) and state is not None:
                ref = state.reference
                if state.session_id != session or state.arm_epoch != epoch:
                    how = 'rt_epoch'  # every RT reference dies with its session/ARM epoch
                elif not state.armed:
                    how = 'rt_disarmed'  # a disarmed RT holds no reference
                elif ref and ref['reference_id'] and (ref['reference_id'] > ref_id or (
                        ref['reference_id'] == ref_id
                        and ref['status'] not in (rtp.REFERENCE_RUNNING, rtp.REFERENCE_END))):
                    how = 'rt_observed'
                else:
                    how = None
            elif how == 'unconfirmed':
                how = None
            if how is None:
                continue
            ex.drop = None
            if h is not None:
                h.dropped.pop(ex.status.execution_id, None)
            self._receipt(ex, Phase.CANCELLED, reason, drop_confirmed=how)
            if ex.stream is not None:
                self._settle_stream_attempts(ex, reason)

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
        if phase is Phase.FAULTED:
            self.fault_trace.trigger(f'{ex.status.execution_id} FAULTED: {reason}', time.monotonic())
        if phase is Phase.ADMITTED:
            evidence["payload_revision"] = ex.payload_revision
        r = Receipt(ex.status.execution_id, ex.status.request_id, self.instance_id, phase,
                    self.now(), reason, evidence)
        ex.status = ex.status.apply(r)
        ex.receipts.append(r)
        if phase.terminal:
            ex.drop = None
        self._cv.notify_all()

    def _prune(self) -> None:
        """One bounded history budget across executions and stream attempts."""
        def excess():
            return len(self._executions) + len(self._stream_attempts) > RETAINED
        protected = {(ex.stream.owner, rid) for ex in self._executions.values()
                     if ex.stream is not None and not ex.status.terminal
                     for rid in (ex.stream.pending, ex.stream.selected, ex.request.request_id)}
        for key, attempt in list(self._stream_attempts.items()):
            if excess() and key not in protected and attempt.status.phase is not Phase.REQUESTED:
                self._stream_attempts.pop(key)
        for eid, ex in list(self._executions.items()):
            if excess() and ex.status.terminal:
                self._executions.pop(eid)
                self._by_request.pop(ex.request.request_id, None)
                for key, attempt in list(self._stream_attempts.items()):
                    if attempt.status.execution_id == eid:
                        self._stream_attempts.pop(key)

    def _pump_controller(self, *, sent_for: str | None = None, reconcile: bool = True) -> None:
        """Forward the controller's commands to the plant and turn its events
        into receipts. ``sent_for`` names the execution the forwarded commands
        were stepped from (the one still executing after the tick)."""
        c = self.controller
        sent, events = [], []
        for topic, value in c.node.sent:
            if topic == "motor_command":
                command = unpack_motor_command(value, c.n_arm)
                seq = self._send_command(command)
                if seq is not None:
                    sent.append(seq)
                    if sent_for is not None and self._native is None:
                        c.streamed_command_sent(sent_for, command['position'], c._clock())
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
        if sent_for is not None and sent:
            ex = self._executions.get(sent_for)
            if ex is not None and ex.stream is not None and ex.stream.selected:
                attempt = self._stream_attempts.get((ex.stream.owner, ex.stream.selected))
                if attempt is not None and attempt.status.phase is Phase.SELECTED:
                    attempt.sent_seq = sent[0]
                    attempt.status = replace(attempt.status, phase=Phase.SENT, at=self.now())
        if reconcile:
            self._reconcile()

    def _on_event(self, event: dict) -> None:
        ex = self._executions.get(event.get("plan_id") or "")
        kind = event["kind"]
        if kind == 'mode' and event.get('ok') and event.get('mode_state'):
            self._mode_feedback = event['mode_state']
        if ex is None or ex.status.terminal:
            return
        if kind == "plan_staged":
            if not event["ok"]:
                self._receipt(ex, Phase.REJECTED, event.get("reason") or "controller refused")
                return
            self._receipt(ex, Phase.ADMITTED)
            if ex.stream is not None:
                return  # asynchronous stream admission owns release
            old = ex.request.supersedes and self._executions.get(ex.request.supersedes)
            if old and not old.status.terminal:
                self._retire_native(old.status.execution_id)
                self._receipt(old, Phase.SUPERSEDED, f"superseded by {ex.status.execution_id}",
                              superseded_by=ex.status.execution_id)
        elif kind == "leg_result":
            reason = event.get("reason") or ""
            if event["ok"]:
                ex.completed_at = time.monotonic()
            elif ex.drop is not None:
                pass  # the controller's own cancel is not the RT's confirmation
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
                applied = phase is Phase.APPLIED
                if ex.stream is not None:
                    attempt = self._stream_attempts.get((ex.stream.owner, ex.stream.selected))
                    applied = applied and attempt is not None and attempt.terminal and attempt.status.phase is Phase.APPLIED
                if applied:
                    self._receipt(ex, Phase.COMPLETED)
                elif now - ex.completed_at > self.applied_grace_s:
                    self._receipt(ex, Phase.FAULTED, "completed on feedback without plant "
                                                     "acceptance evidence")
                continue
            if phase in (Phase.REQUESTED,) or current == eid or ex.drop is not None:
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
        if ex.stream is not None:
            return False  # stream attempt deadlines and uncertainty are reconciled together
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

    def _selection_unresolved(self, h, reason: str) -> None:
        """A candidate stayed pending too long: fail it, retire it explicitly, then hold.

        The RT may or may not have selected it, so the execution is failed (never
        assumed selected) and CANCEL/ABORT is sent through the horizon's retire path.
        _native_failed stays clear: once the slot is free, the ordinary tail stages
        the measured hold, so commands resume instead of starving to CMD_LOST.
        """
        ex = self._executions.get(h.key)
        if ex is not None and ex.stream is not None and ex.drop is not None and not ex.status.terminal:
            # An operator cancel is already waiting for the RT's confirmation: free the slot, but leave
            # the receipt (and the operator's reason) to _confirm_drops.
            ex.stream.pending = ex.stream.anchor = None
            self._native_pending = None
            return
        if ex is not None and ex.stream is not None and not ex.status.terminal:
            s = ex.stream
            # REJECTED means "never moved"; the RT may have selected this candidate, so an
            # ADMITTED execution is CANCELLED (dropped; holds if it had been selected),
            # and one already SELECTED is FAULTED. The reason carries the cause.
            failure = Phase.CANCELLED if ex.status.phase is Phase.ADMITTED else Phase.FAULTED
            for rid in (s.selected, s.pending):
                attempt = self._stream_attempts.get((s.owner, rid)) if rid is not None else None
                if attempt is not None and not attempt.status.phase.terminal:
                    # selection_state stays as is: structurally uncertain, never claimed refused
                    phase = failure if attempt.status.request_id == s.selected else Phase.CANCELLED
                    attempt.status = replace(attempt.status, phase=phase, at=self.now(), reason=reason)
            self._receipt(ex, failure, reason, reference_id=h.id, reference_revision=h.revision)
            s.pending = s.anchor = None
            self._native_pending = None
            self._drop_staged(ex, reason)  # explicit CANCEL/ABORT + controller plan cancel
        else:
            h.cancel()

    def _native_step(self, sample) -> None:
        h, c = self._native, self.controller
        state, rx = sample
        try:
            h.poll(state, rx)
        except SelectionUnresolved as exc:
            self._selection_unresolved(h, str(exc))
        self._confirm_drops(state)
        old_end = (h.observed is not None and h.observed_point is not None
                   and h.observed['status'] == rtp.REFERENCE_END
                   and h.observed['reference_id'] != h.id)
        if (old_end and state is not None and np.isfinite(state.q).all()
                and np.isfinite(state.dq).all()):
            # A new candidate does not own the already selected END reference.
            # Keep its stationary endpoint and deadman alive during upload and
            # abort, using a fresh compensation decision from this sample.
            self._publish_measured_reference_hold(
                state, rx, h.observed, h.observed_point)
        if (h.selected and h._selection_rx == rx and h._first_tracking_at is None
                and h.key is not None and state is not None
                and np.isfinite(state.q).all() and np.isfinite(state.dq).all()):
            # RT has selected, but the controller has not made its first
            # tracking decision yet. Publish a measured-position compensation
            # decision from THIS state sample, with its original sample time.
            # The sender may repeat it but never make its age look younger.
            self._publish_measured_reference_hold(state, rx, h.feedback, h.point)
        if h.group_reply is not None and h.key is not None and h.group_reply.get('key') == h.key:
            # the RT's COMMIT_GROUP answer for THIS execution, whatever happens next
            self._group_reports[h.key] = {k: v for k, v in h.group_reply.items() if k != 'key'} | dict(reference_id=h.id)
            for stale in list(self._group_reports)[:-32]:
                del self._group_reports[stale]
        if self._stream_step():
            return
        if state is None or not state.armed:
            return
        if state.arm_epoch != self._native_epoch:
            self._native_epoch = state.arm_epoch
            self._native_failed = False
        ex = self._executions.get(h.key)
        if (h.error and ex is not None and not ex.status.terminal
                and (not h.selected
                     or h.error in ('REFERENCE_UNDERFLOW', 'REFERENCE_EXPIRED'))):
            # Reference failure is distinct from a transport/plant fault. In
            # particular, never heal UNDERFLOW with an implicit hold admission.
            self._native_failed = True
            if (ex.status.phase is Phase.ADMITTED
                    and h.error in ('REFERENCE_UNDERFLOW', 'REFERENCE_EXPIRED')):
                self._receipt(ex, Phase.SELECTED, reference_id=h.id,
                              reference_revision=h.revision, arm_epoch=h.arm_epoch)
            failure = Phase.FAULTED
            if ex.status.phase is Phase.ADMITTED:
                failure = Phase.EXPIRED if ex.request.expired(self.now()) else Phase.REJECTED
            self._receipt(ex, failure, h.error,
                          reference_id=h.id, reference_revision=h.revision)
            self._drop_staged(ex, h.error)
        if h.key is None and h.error:
            if time.monotonic() < self._group_hold_grace:
                h.error = ''  # the hold was retired by our own GROUP_CANCEL: the tail stages a new one
            else:
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
                h.group = self._group_release.pop(eid, None)  # a group member commits through COMMIT_GROUP
                h.approve()

    def _publish_measured_reference_hold(self, state, rx, ref, point) -> None:
        c = self.controller
        hold = c.executor.hold_command(
            JointState(np.asarray(state.q), np.asarray(state.dq)),
            integral_bias=c.integral.bias)
        self._native.publish(dict(position=hold.q_des, velocity=hold.qd_des,
                                  torque=hold.tau_ff, kp=hold.kp, kd=hold.kd),
                             reference_id=ref['reference_id'], revision=ref['revision'],
                             phase=point.time, sample_received_at=rx, tracking=False)

    def _send_command(self, command):
        h = self._native
        if h is None:
            return self.link.apply_command(command)
        state, _ = self.link.latest_state()
        if state is None or not state.armed:
            h.unpublish()
            return None
        ref, point = h.observed, h.observed_point
        if (state.session_id, state.arm_epoch) != (h.session_id, h.arm_epoch):
            ref = point = None
        if (ref is not None and ref['status'] == rtp.REFERENCE_END
                and ref['reference_id'] != h.id):
            # The candidate has no authority over the selected old endpoint.
            # _native_step published this sample's compensated END decision.
            return None
        # Before this ARM generation has ever selected a reference, the
        # existing compensated idle hold remains the v5 command path.
        if state.reference is None or not state.reference['reference_id']:
            h.unpublish()
            return self.link.apply_command(command)
        if ref is not None and point is not None:
            matches = (np.array_equal(command['position'], point.position)
                       and np.array_equal(command['velocity'], point.velocity))
            if matches:
                # CTRL-034 B1: the fixed-rate sender repeats this decision at its
                # own rate but never re-stamps it — the envelope always carries
                # this decision's sample time and phase, so a stalled tick makes
                # the decision age and the RT expire the reference.
                h.publish(command, reference_id=ref['reference_id'],
                          revision=ref['revision'], phase=point.time,
                          sample_received_at=h.received_at)
                return h.last_seq()
        # Any other command path is not tracking the selected reference: stop
        # the stream so the RT's own freshness rule owns the hold.
        h.unpublish()
        # The controller's existing cancel/contact/relatch path owns the hold
        # target and its compensation. Admit that fixed target explicitly;
        # never mislabel its dynamics as an envelope for the old moving curve.
        if (self._native_failed or self._native_pending is not None
                or self.controller.runtime_state['executing'] or h.busy
                or time.monotonic() < self._group_cancelling):
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
        link's ACK timeouts; a DISARM drops queued ARMs."""
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
            now = time.monotonic()
            next_t = next_tick(next_t, now, self.period)
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
                self._last_packet = (state, rx)
                c.on_motor_state(pack_motor_state_dict(link.motor_state(state=state)))
                # Arrival freshness survives a delayed tick.
                if c._last_state_at is not None:
                    # rx is wall time; the controller may run on a sim world's clock (paused with the world): carry the
                    # sample's AGE across, never the wall stamp itself, or every pause would shorten every later age
                    c._last_state_at = c._clock() - (time.monotonic() - rx)
        self._publish()
        for ex in list(self._executions.values()):
            self._expire(ex)
        if self._native is not None:
            self._native_step(history[-1] if history else (None, 0.))
        c.tick()
        rs = c.runtime_state
        self._pump_controller(sent_for=rs["plan_id"] if rs["executing"] else None)
        self._record_trace(health)
        latched = str(health["latched_fault"] or "")
        if latched and not self._latched_seen:
            self.fault_trace.trigger(f'latched fault: {latched}', time.monotonic())
        self._latched_seen = bool(latched)
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
            self.fault_trace.trigger(f'controller stopped: {getattr(c, "stop_reason", "")}', time.monotonic())
            # A stopped ArmController is terminal (its Dora node exits). Here a
            # fresh one replaces it: disarmed, nothing staged, new admission.
            print("[supervisor] controller stopped; replaced with a fresh, disarmed one",
                  flush=True)
            self.controller = self._build_controller()
            self._reconcile()
        self._prune()
        self._publish()
        self.fault_trace.poll(time.monotonic(), record_errors=self._trace_errors)

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
