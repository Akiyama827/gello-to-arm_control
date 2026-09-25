"""Transport-independent execution contract (CTRL-020).

The values an application, the supervisor and every transport adapter agree
on: robot identity, time stamps, observation, ownership/lifecycle, capability,
execution requests, receipts and cancellation. Values and pure checks only --
stdlib, numpy and ``arm_control.motion``. Nothing here imports Dora, Arrow,
Rerun, MuJoCo, a planner or a device SDK, and nothing here is a wire format:
the RT UDP/TCP protocol, the Dora ``plan``/``control``/``controller_event``
messages and in-process controllers ADAPT to these values.

Every concept below already existed somewhere; the mapping from the old names
is in docs/ARCHITECTURE.md ("Execution contract").

Conventions (normative):

* Joint vectors are in the robot's canonical joint order,
  ``RobotIdentity.joint_names`` -- the same order whose hash the RT server
  provisions. A request names its order and is refused on mismatch; nothing
  reorders silently.
* Units are SI (``UNITS``). Revolute joints in rad; gains kp N*m/rad,
  kd N*m*s/rad.
* Poses are ``[x, y, z, qw, qx, qy, qz]`` in metres with a unit quaternion,
  SCALAR FIRST (``QUATERNION``). ``q`` and ``-q`` are the same rotation.
  Every pose names its frame; ``"world"`` is the workcell board frame and
  ``"base:<instance_id>"`` a robot's base frame.
* Wrench is ``[fx, fy, fz, tx, ty, tz]`` (``WRENCH``), N and N*m, expressed in
  the robot base frame about the vendor's stiffness-frame origin; POSITIVE is
  the wrench the robot applies to the environment (libfranka
  ``O_F_ext_hat_K``). Present only when the plant reports it valid.
* Time: a ``Stamp`` carries its clock and that clock's identity. Stamps from
  different clocks are never subtracted (``elapsed`` refuses): RT-host
  monotonic, PC monotonic, robot time and execution time are separate
  domains, and a cross-host mapping needs a measured offset and uncertainty.
* Unknown is ``None``, never a fabricated zero: a missing velocity is not a
  stationary arm.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as _rt

#: Bumped when a public field or phase changes meaning. Adapters report the
#: version they implement in ``Capabilities.contract_version``.
CONTRACT_VERSION = 1

UNITS = {
    "joint_position": "rad",
    "joint_velocity": "rad/s",
    "joint_acceleration": "rad/s^2",
    "joint_torque": "N*m",
    "kp": "N*m/rad",
    "kd": "N*m*s/rad",
    "cartesian_position": "m",
    "force": "N",
    "moment": "N*m",
    "duration": "s",
    "stamp": "ns",
}
QUATERNION = "wxyz"  # pose vectors [x, y, z, qw, qx, qy, qz], unit norm
WRENCH = {
    "order": ("fx", "fy", "fz", "tx", "ty", "tz"),
    "expressed_in": "base",
    "reference_point": "vendor stiffness frame origin",
    "sign": "positive = applied by the robot to the environment",
}

_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be 1-64 chars of [A-Za-z0-9_.:-], got {value!r}")
    return value


def _vector(name: str, value, n: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.array(value, dtype=float, copy=True).ravel()
    if n is not None and arr.shape != (n,):
        raise ValueError(f"{name} must have {n} values, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    arr.setflags(write=False)
    return arr


# -- time ---------------------------------------------------------------------

class Clock(str, Enum):
    RT_HOST = "rt_host_monotonic"  # CLOCK_MONOTONIC on the RT host: execution authority
    PC = "pc_monotonic"            # the application/supervisor host's monotonic clock
    ROBOT = "robot"                # the plant's own time (e.g. libfranka Duration)
    EXECUTION = "execution"        # trajectory time, 0 at the execution's selection


@dataclass(frozen=True)
class Stamp:
    """A time in ns on one named clock. ``domain`` identifies the clock
    instance (host and boot); ``""`` means unattested, which is not the same
    as equal: two unattested stamps are still not comparable."""

    ns: int
    clock: Clock
    domain: str = ""

    def __post_init__(self) -> None:
        if type(self.ns) is not int or self.ns < 0:
            raise ValueError("Stamp.ns must be a non-negative int")
        if not isinstance(self.clock, Clock):
            raise ValueError("Stamp.clock must be a Clock")


def elapsed(later: Stamp, earlier: Stamp) -> float:
    """Seconds between two stamps of the SAME attested clock; else ValueError."""
    if later.clock is not earlier.clock or not later.domain or later.domain != earlier.domain:
        raise ValueError(f"stamps on different or unattested clocks: {earlier} vs {later}")
    return (later.ns - earlier.ns) * 1e-9


# -- identity and capability --------------------------------------------------

@dataclass(frozen=True)
class RobotIdentity:
    """Which robot: the deployment instance, its canonical joint order, the
    plant kind and the task model the plant computes kinematics with."""

    instance_id: str
    joint_names: tuple[str, ...]
    backend: str
    model_id: str = ""

    def __post_init__(self) -> None:
        if not _rt.valid_id(self.instance_id):
            raise ValueError(f"instance_id {self.instance_id!r} is not a provisioned id")
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names) or not all(
                isinstance(j, str) and j for j in names):
            raise ValueError("joint_names must be distinct non-empty names")
        object.__setattr__(self, "joint_names", names)

    @property
    def n(self) -> int:
        return len(self.joint_names)

    @property
    def identity_hash(self) -> int:
        """The RT server's identity hash for this instance and joint order."""
        return _rt.identity_hash(self.instance_id, _rt.joint_order_hash(self.joint_names))


@dataclass(frozen=True)
class Capabilities:
    """What the backend offers. Qualification is recorded separately and is
    never inferred from what a device advertises."""

    contract_version: int
    profiles: tuple[str, ...]           # controller profiles it can admit
    protocol_version: int | None = None  # RT wire version, when on RT
    pose_hold: bool = False
    observer: bool = False
    rt_setup: str = "unknown"           # "ok", the failed parts, or "unknown"
    qualification: Mapping[str, str] = field(default_factory=dict)

    def refusal(self, profile: str) -> str:
        """Unknown capability is refused before actuation."""
        if self.contract_version != CONTRACT_VERSION:
            return (f"backend implements contract v{self.contract_version}, "
                    f"application v{CONTRACT_VERSION}")
        if profile not in self.profiles:
            return f"controller profile {profile!r} not offered ({', '.join(self.profiles)})"
        return ""


# -- observation --------------------------------------------------------------

class Validity(str, Enum):
    VALID = "valid"
    STALE = "stale"      # well-formed, older than the consumer's freshness bound
    MISSING = "missing"  # nothing received for this robot yet
    INVALID = "invalid"  # received but unusable (identity, shape, non-finite)


@dataclass(frozen=True)
class RobotState:
    """One robot's observation as an application may use it.

    ``q``/``dq``/``tau`` are measured. ``q_ref``/``qd_ref``/``tau_ref`` are
    what the RT servo actually SELECTED and SENT on that tick (after holds,
    clamps and slew) -- not what the application requested. Arrays are
    ``None`` unless ``validity`` is VALID; wrench also needs the plant to
    report it valid. ``source`` is the sample's RT-host (or plant) time,
    ``received`` the time it became available to the reader.
    """

    instance_id: str
    validity: Validity
    reason: str = ""
    q: np.ndarray | None = None
    dq: np.ndarray | None = None
    tau: np.ndarray | None = None
    q_ref: np.ndarray | None = None
    qd_ref: np.ndarray | None = None
    tau_ref: np.ndarray | None = None
    wrench: np.ndarray | None = None
    source: Stamp | None = None
    received: Stamp | None = None
    state_seq: int | None = None
    last_command_seq: int | None = None  # freshest command the servo consumed

    def __post_init__(self) -> None:
        arrays = ("q", "dq", "tau", "q_ref", "qd_ref", "tau_ref")
        if self.validity is not Validity.VALID:
            if any(getattr(self, a) is not None for a in (*arrays, "wrench")):
                raise ValueError(f"a {self.validity.value} observation carries no values")
            if not self.reason:
                raise ValueError(f"a {self.validity.value} observation needs a reason")
            return
        if self.q is None or self.source is None or self.received is None:
            raise ValueError("a valid observation needs q, source and received stamps")
        q = _vector("q", self.q)
        object.__setattr__(self, "q", q)
        for name in arrays[1:]:
            object.__setattr__(self, name, _vector(name, getattr(self, name), q.size))
        object.__setattr__(self, "wrench", _vector("wrench", self.wrench, 6))


# -- ownership and lifecycle --------------------------------------------------

class Lifecycle(str, Enum):
    DISCONNECTED = "disconnected"
    OBSERVING = "observing"    # read-only lease; can never command
    CONNECTED = "connected"    # control session open, not yet claimed
    CLAIMED = "claimed"        # owner of this robot, disarmed
    ARMED = "armed"            # servo holds authority for arm_epoch
    FAULTED = "faulted"        # latched: still holding, commands ignored; DISARM clears


@dataclass(frozen=True)
class Fault:
    """A latched first cause. ``code`` is stable ("cmd_lost", "ctl_lost",
    "plant", "controller", ...); ``text`` is the bounded first-cause text."""

    code: str
    text: str
    source: str               # "rt", "controller", "hand", "supervisor"
    arm_epoch: int | None = None  # the authority generation it arose in


@dataclass(frozen=True)
class Authority:
    """Ownership and lifecycle of one robot as its owner sees it.

    ``arm_epoch`` is the RT ARM generation commands must carry (0 = none).
    DISARM completes when ``stop_epoch`` reaches the DISARM's generation --
    the servo stopped the plant -- not when the request was acknowledged.
    """

    lifecycle: Lifecycle
    session: str | None = None
    arm_epoch: int = 0
    stop_epoch: int = 0
    fault: Fault | None = None

    def __post_init__(self) -> None:
        if (self.lifecycle is Lifecycle.FAULTED) != (self.fault is not None):
            raise ValueError("FAULTED and only FAULTED carries a fault")
        if self.lifecycle in (Lifecycle.ARMED, Lifecycle.FAULTED) and self.arm_epoch <= 0:
            raise ValueError("an armed or faulted robot has an ARM generation")


# -- execution requests -------------------------------------------------------

@dataclass(frozen=True)
class JointGains:
    kp: np.ndarray
    kd: np.ndarray

    def __post_init__(self) -> None:
        kp, kd = _vector("kp", self.kp), _vector("kd", self.kd)
        if kd.shape != kp.shape or (kp < 0).any() or (kd < 0).any():
            raise ValueError("kp/kd must be matching non-negative vectors")
        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "kd", kd)


@dataclass(frozen=True)
class Completion:
    """When a trajectory counts as COMPLETED, judged on observed feedback: it
    played out, joint speed fell below ``vel_tol``, and -- when given -- the
    arm stopped within ``residual_max`` of ``goal_q``. ``None`` on a request
    means the backend's default: played out and at the final sample."""

    vel_tol: float
    goal_q: np.ndarray | None = None
    residual_max: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.vel_tol) and self.vel_tol > 0):
            raise ValueError("vel_tol must be positive")
        if (self.goal_q is None) != (self.residual_max is None):
            raise ValueError("goal_q and residual_max come together")
        object.__setattr__(self, "goal_q", _vector("goal_q", self.goal_q))


class Start(str, Enum):
    ON_ADMIT = "on_admit"      # selected as soon as it is admitted
    ON_RELEASE = "on_release"  # admitted for review; selected only by release(execution_id)
    ON_COMMIT = "on_commit"    # a group member: selected by the group's commit (CTRL-040)


@dataclass(frozen=True)
class TrajectoryRequest:
    """Execute one planned joint trajectory.

    ``request_id`` is the caller's idempotency key: submitting the same id
    again returns the existing execution, never a second one. ``times`` are
    execution time: the trajectory's first sample plays at selection, never at
    request time (review/gate time is not charged against the motion).
    ``admit_by`` bounds how long an admitted, unselected request stays
    selectable; past it the request EXPIRES instead of starting late. Once
    selected, a planned trajectory runs to completion, cancellation or fault;
    command freshness below that is the RT deadman's, not this TTL.
    """

    request_id: str
    instance_id: str
    joint_names: tuple[str, ...]
    trajectory: JointTrajectory
    profile: str = "joint_impedance"
    gains: JointGains | None = None   # None: the profile's configured gains
    model_token: str | None = None    # model + scene the plan was made against
    start: Start = Start.ON_ADMIT
    admit_by: Stamp | None = None
    completion: Completion | None = None
    group_id: str | None = None

    def __post_init__(self) -> None:
        _identifier("request_id", self.request_id)
        if self.group_id is not None:
            _identifier("group_id", self.group_id)
        if (self.start is Start.ON_COMMIT) != (self.group_id is not None):
            raise ValueError("a group member starts ON_COMMIT, and only a group member does")
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        if len(self.joint_names) != self.trajectory.num_joints:
            raise ValueError("joint_names must name every trajectory column")
        for arr in (self.trajectory.positions, self.trajectory.velocities):
            if arr is not None and not np.isfinite(arr).all():
                raise ValueError("trajectory must be finite")
        if self.gains is not None and self.gains.kp.size != self.trajectory.num_joints:
            raise ValueError("gains must cover every joint")


def admission_refusal(request: TrajectoryRequest, identity: RobotIdentity,
                      capabilities: Capabilities, authority: Authority,
                      *, model_token: str | None = None) -> str:
    """The generic admission checks every backend applies, in order; ``""``
    admits. Project admission (workcell, collision, Hand) is composed ABOVE
    this, never folded into it."""
    if request.instance_id != identity.instance_id:
        return f"request is for {request.instance_id!r}, robot is {identity.instance_id!r}"
    if request.joint_names != identity.joint_names:
        return "joint order differs from the robot's canonical order"
    refused = capabilities.refusal(request.profile)
    if refused:
        return refused
    if model_token is not None and request.model_token != model_token:
        return f"planned against model {request.model_token!r}, robot runs {model_token!r}"
    if authority.lifecycle is not Lifecycle.ARMED:
        return f"robot is {authority.lifecycle.value}, not armed"
    return ""


# -- execution status ---------------------------------------------------------

class Phase(str, Enum):
    """An execution's progress. The first five are the forward chain; the
    rest are terminal outcomes.

    REQUESTED  created by the application; nothing has answered yet
    ADMITTED   passed admission and staged; NOT executable authority yet
    SELECTED   the executor is sampling it as the active reference
    SENT       a command derived from it was handed to the plant transport
    APPLIED    the plant consumed such a command (RT: last_command_seq reached
               it) -- first evidence the robot is acting on it
    COMPLETED  its completion rule held on OBSERVED feedback
    REJECTED   refused at admission or at selection; it never moved
    SUPERSEDED an admitted, unselected request replaced by a newer admission
    EXPIRED    not selected before ``admit_by``
    CANCELLED  cancellation confirmed: the executor dropped it and holds
    FAULTED    stopped by a fault while it owned the reference
    """

    REQUESTED = "requested"
    ADMITTED = "admitted"
    SELECTED = "selected"
    SENT = "sent"
    APPLIED = "applied"
    COMPLETED = "completed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAULTED = "faulted"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL


_CHAIN = (Phase.REQUESTED, Phase.ADMITTED, Phase.SELECTED, Phase.SENT, Phase.APPLIED)
_TERMINAL = frozenset({Phase.COMPLETED, Phase.REJECTED, Phase.SUPERSEDED,
                       Phase.EXPIRED, Phase.CANCELLED, Phase.FAULTED})
# Which non-terminal phases each terminal outcome may follow.
_ENDS = {
    Phase.REJECTED: {Phase.REQUESTED, Phase.ADMITTED},
    Phase.SUPERSEDED: {Phase.ADMITTED},
    Phase.EXPIRED: {Phase.ADMITTED},
    Phase.CANCELLED: {Phase.ADMITTED, Phase.SELECTED, Phase.SENT, Phase.APPLIED},
    Phase.FAULTED: {Phase.SELECTED, Phase.SENT, Phase.APPLIED},
    Phase.COMPLETED: {Phase.APPLIED},  # completion needs evidence the plant acted
}
_EVIDENCE_KEYS = 16


class IllegalTransition(ValueError):
    pass


@dataclass(frozen=True)
class Receipt:
    """One step of one execution, as the layer that took it reports it.

    ``execution_id`` is assigned by the admitting layer and never reused;
    ``request_id`` is the caller's key. ``evidence`` is small and bounded:
    e.g. ``arm_epoch``, ``command_seq``, ``state_seq``, completion residuals.
    """

    execution_id: str
    request_id: str
    instance_id: str
    phase: Phase
    at: Stamp
    reason: str = ""
    evidence: Mapping[str, int | float | str] = field(default_factory=dict)
    group_id: str | None = None

    def __post_init__(self) -> None:
        _identifier("execution_id", self.execution_id)
        _identifier("request_id", self.request_id)
        if len(self.evidence) > _EVIDENCE_KEYS:
            raise ValueError(f"evidence is bounded to {_EVIDENCE_KEYS} entries")
        failed = self.phase in _ENDS and self.phase is not Phase.COMPLETED
        if failed and not self.reason:
            raise ValueError(f"a {self.phase.value} receipt needs a reason")


@dataclass(frozen=True)
class ExecutionStatus:
    """The fold of an execution's receipts. ``reached`` keeps the stamp of
    every phase actually reported; a phase the transport cannot observe may be
    skipped forward, but never back, and a terminal phase is final."""

    execution_id: str
    request_id: str
    instance_id: str
    phase: Phase
    reached: Mapping[Phase, Stamp]
    reason: str = ""
    group_id: str | None = None

    @property
    def terminal(self) -> bool:
        return self.phase.terminal

    @property
    def ok(self) -> bool | None:
        """True only for COMPLETED; None while in flight."""
        return (self.phase is Phase.COMPLETED) if self.terminal else None

    @classmethod
    def start(cls, receipt: Receipt) -> ExecutionStatus:
        if receipt.phase not in (Phase.REQUESTED, Phase.ADMITTED, Phase.REJECTED):
            raise IllegalTransition(f"an execution begins requested/admitted/rejected, "
                                    f"not {receipt.phase.value}")
        return cls(receipt.execution_id, receipt.request_id, receipt.instance_id,
                   receipt.phase, {receipt.phase: receipt.at}, receipt.reason,
                   receipt.group_id)

    def apply(self, receipt: Receipt) -> ExecutionStatus:
        if (receipt.execution_id, receipt.request_id, receipt.instance_id) != (
                self.execution_id, self.request_id, self.instance_id):
            raise IllegalTransition("receipt belongs to another execution")
        if receipt.phase is self.phase:
            return self  # duplicate delivery: idempotent
        if self.terminal:
            raise IllegalTransition(f"{self.execution_id} is already {self.phase.value}")
        if receipt.phase in _ENDS:
            if self.phase not in _ENDS[receipt.phase]:
                raise IllegalTransition(f"{receipt.phase.value} cannot follow {self.phase.value}")
        elif _CHAIN.index(receipt.phase) <= _CHAIN.index(self.phase):
            raise IllegalTransition(f"{receipt.phase.value} cannot follow {self.phase.value}")
        return replace(self, phase=receipt.phase, reason=receipt.reason,
                       reached={**self.reached, receipt.phase: receipt.at})


@dataclass(frozen=True)
class CancelRequest:
    """Cancel one execution. Completion is the CANCELLED receipt (the executor
    dropped it and holds); cancelling a terminal execution is a no-op that
    reports its terminal status. A cancel never disarms and never cancels a
    different execution than the one named."""

    execution_id: str
    reason: str = "cancelled"

    def __post_init__(self) -> None:
        _identifier("execution_id", self.execution_id)


def _self_check() -> None:
    now = [0]

    def at(clock=Clock.PC, domain="pc:boot-a"):
        now[0] += 1_000_000
        return Stamp(now[0], clock, domain)

    # Time: only one attested clock may be subtracted from itself.
    a, b = at(), at()
    assert abs(elapsed(b, a) - 1e-3) < 1e-12
    for bad in (Stamp(2, Clock.RT_HOST, "pc:boot-a"), Stamp(2, Clock.PC, "pc:boot-b"),
                Stamp(2, Clock.PC)):
        try:
            elapsed(bad, a)
            raise AssertionError("cross-clock subtraction accepted")
        except ValueError:
            pass

    fr3 = RobotIdentity("fr3_1", tuple(f"fr3_joint{i}" for i in range(1, 8)), "franka",
                        "libfranka-vendor")
    assert fr3.identity_hash == _rt.identity_hash("fr3_1", _rt.joint_order_hash(fr3.joint_names))
    caps = Capabilities(CONTRACT_VERSION, ("joint_impedance",), protocol_version=_rt.VERSION)
    armed = Authority(Lifecycle.ARMED, session="s1", arm_epoch=3)
    traj = JointTrajectory([0.0, 1.0], np.zeros((2, 7)), np.zeros((2, 7)))
    req = TrajectoryRequest("r1", "fr3_1", fr3.joint_names, traj, model_token="m1")
    assert admission_refusal(req, fr3, caps, armed, model_token="m1") == ""
    swapped = replace(req, joint_names=fr3.joint_names[::-1])
    for request, authority, token, why in (
            (swapped, armed, "m1", "joint order"),
            (replace(req, profile="cartesian_impedance"), armed, "m1", "not offered"),
            (req, armed, "m2", "planned against"),
            (req, Authority(Lifecycle.CLAIMED, session="s1"), "m1", "not armed")):
        assert why in admission_refusal(request, fr3, caps, authority, model_token=token)
    assert "contract v" in replace(caps, contract_version=0).refusal("joint_impedance")
    for bad in (dict(start=Start.ON_COMMIT), dict(group_id="g1"), dict(request_id="has space"),
                dict(joint_names=fr3.joint_names[:6])):
        try:
            replace(req, **bad)
            raise AssertionError(f"invalid request accepted: {bad}")
        except ValueError:
            pass
    TrajectoryRequest("r2", "fr3_1", fr3.joint_names, traj, start=Start.ON_COMMIT, group_id="g1")

    # Observation: unknown is None, and a valid sample is complete and finite.
    assert RobotState("fr3_1", Validity.MISSING, "no state yet").q is None
    for bad in (dict(validity=Validity.STALE, q=np.zeros(7)),
                dict(validity=Validity.VALID, q=np.zeros(7)),
                dict(validity=Validity.VALID, q=np.full(7, np.nan), source=a, received=b),
                dict(validity=Validity.VALID, q=np.zeros(7), dq=np.zeros(6), source=a, received=b)):
        try:
            RobotState("fr3_1", reason="x", **bad)
            raise AssertionError(f"invalid observation accepted: {bad}")
        except ValueError:
            pass
    try:
        Authority(Lifecycle.ARMED)
        raise AssertionError("armed without a generation")
    except ValueError:
        pass

    # Execution: the forward chain, skips forward only, terminals final.
    def receipt(phase, reason=""):
        return Receipt("e1", "r1", "fr3_1", phase, at(), reason)

    status = ExecutionStatus.start(receipt(Phase.REQUESTED))
    for phase in (Phase.ADMITTED, Phase.SELECTED, Phase.APPLIED):  # SENT unobserved
        status = status.apply(receipt(phase))
    assert status.apply(receipt(Phase.APPLIED)) is status  # duplicate is idempotent
    for illegal in (Phase.SENT, Phase.SELECTED, Phase.EXPIRED, Phase.SUPERSEDED):
        try:
            status.apply(receipt(illegal, "x"))
            raise AssertionError(f"{illegal} accepted after applied")
        except IllegalTransition:
            pass
    done = status.apply(receipt(Phase.COMPLETED))
    assert done.terminal and done.ok and set(done.reached) == {
        Phase.REQUESTED, Phase.ADMITTED, Phase.SELECTED, Phase.APPLIED, Phase.COMPLETED}
    for late in (receipt(Phase.CANCELLED, "late"), receipt(Phase.SENT), receipt(Phase.ADMITTED)):
        try:
            done.apply(late)
            raise AssertionError(f"terminal execution changed by {late.phase.value}")
        except IllegalTransition:
            pass
    # Completion needs evidence the plant acted; a selected-only one cannot complete.
    selected = ExecutionStatus.start(receipt(Phase.ADMITTED)).apply(receipt(Phase.SELECTED))
    try:
        selected.apply(receipt(Phase.COMPLETED))
        raise AssertionError("completed without application evidence")
    except IllegalTransition:
        pass
    cancelled = selected.apply(receipt(Phase.CANCELLED, "operator"))
    assert cancelled.terminal and cancelled.ok is False and cancelled.reason == "operator"
    try:
        receipt(Phase.REJECTED)  # a refusal must say why
        raise AssertionError("reasonless rejection accepted")
    except ValueError:
        pass
    try:
        Receipt("e1", "r1", "fr3_1", Phase.SENT, at(),
                evidence={f"k{i}": i for i in range(_EVIDENCE_KEYS + 1)})
        raise AssertionError("unbounded evidence accepted")
    except ValueError:
        pass
    try:
        status.apply(Receipt("e2", "r1", "fr3_1", Phase.SENT, at()))
        raise AssertionError("foreign receipt accepted")
    except IllegalTransition:
        pass
    print("execution contract self-check: PASS (clock domains, identity hash, admission "
          "order, observation validity, phase chain/skips/terminals, bounded receipts)")


if __name__ == "__main__":
    _self_check()
