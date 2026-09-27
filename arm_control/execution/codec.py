"""Contract values <-> JSON-able dicts, for the SDK/supervisor socket.

Stdlib + numpy only, like the contract. ``encode`` handles every contract
value generically; each ``*_from`` rebuilds one type, so decoding runs the
same validation as constructing the value directly (a malformed message is a
ValueError, never a half-built value).
"""
from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np

from arm_control.execution.contract import (
    Authority, Capabilities, Clock, Completion, ExecutionStatus, Fault, JointGains,
    Lifecycle, Phase, Receipt, RobotIdentity, RobotState, Stamp, Start,
    TrajectoryRequest, Validity,
)
from arm_control.motion import JointTrajectory


def encode(value):
    """Any contract value (or container of them) as plain JSON types."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict) or hasattr(value, "items"):
        return {encode(k) if isinstance(k, Enum) else str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def stamp_from(d) -> Stamp | None:
    return None if d is None else Stamp(int(d["ns"]), Clock(d["clock"]), str(d["domain"]))


def identity_from(d) -> RobotIdentity:
    return RobotIdentity(d["instance_id"], tuple(d["joint_names"]), d["backend"], d["model_id"])


def capabilities_from(d) -> Capabilities:
    return Capabilities(int(d["contract_version"]), tuple(d["profiles"]), d["protocol_version"],
                        bool(d["pose_hold"]), bool(d["observer"]), str(d["rt_setup"]))


def state_from(d) -> RobotState:
    return RobotState(**{**d, "validity": Validity(d["validity"]),
                         "source": stamp_from(d["source"]),
                         "received": stamp_from(d["received"])})


def authority_from(d) -> Authority:
    fault = None if d["fault"] is None else Fault(**d["fault"])
    return Authority(Lifecycle(d["lifecycle"]), d["session"], int(d["arm_epoch"]),
                     int(d["stop_epoch"]), fault, d["holding"])


def request_from(d) -> TrajectoryRequest:
    t = d["trajectory"]
    gains = None if d["gains"] is None else JointGains(**d["gains"])
    completion = None if d["completion"] is None else Completion(**d["completion"])
    return TrajectoryRequest(
        d["request_id"], d["instance_id"], tuple(d["joint_names"]),
        JointTrajectory(t["times"], t["positions"], t["velocities"]),
        d["profile"], gains, d["model_token"], Start(d["start"]),
        stamp_from(d["admit_by"]), completion, d["supersedes"])


def status_from(d) -> ExecutionStatus:
    return ExecutionStatus(d["execution_id"], d["request_id"], d["instance_id"],
                           Phase(d["phase"]),
                           {Phase(k): stamp_from(v) for k, v in d["reached"].items()},
                           d["reason"])


def receipt_from(d) -> Receipt:
    return Receipt(d["execution_id"], d["request_id"], d["instance_id"], Phase(d["phase"]),
                   stamp_from(d["at"]), d["reason"], dict(d["evidence"]))


def stream_open_from(d):
    from .stream import StreamOpen
    t = d['initial']
    return StreamOpen(d['request_id'], d['open_seq'], d['instance_id'],
                      tuple(d['joint_names']), d['model_token'], d['scene_token'],
                      d['stream_policy_id'], JointTrajectory(**t), Start(d['start']),
                      stamp_from(d['admit_by']))


def stream_revision_from(d):
    from .stream import StreamRevision
    return StreamRevision(**{**d, 'suffix': JointTrajectory(**d['suffix']),
                             'admit_by': stamp_from(d['admit_by'])})


def anchor_from(d):
    from .stream import SpliceAnchor
    return None if d is None else SpliceAnchor(
        **{**d, 'issued_at': stamp_from(d['issued_at']), 'expires_at': stamp_from(d['expires_at'])})


def revision_status_from(d):
    from .stream import RevisionStatus, SelectionState
    return RevisionStatus(**{**d, 'phase': Phase(d['phase']),
                             'selection_state': SelectionState(d['selection_state']),
                             'at': stamp_from(d['at'])})


def stream_view_from(d):
    from .stream import StreamView
    return StreamView(**{**d, 'execution': status_from(d['execution']),
                         'splice': anchor_from(d['splice'])})


def _self_check() -> None:
    import json

    at = Stamp(5, Clock.PC, "pc:boot")
    ident = RobotIdentity("fr3_1", ("a", "b"), "rt", "m")
    caps = Capabilities(1, ("joint_impedance",), 5, rt_setup="ok")
    traj = JointTrajectory([0.0, 1.0], [[0, 0], [0.1, 0.2]], [[0, 0], [0, 0]])
    req = TrajectoryRequest("r1", "fr3_1", ("a", "b"), traj, gains=JointGains([1, 2], [3, 4]),
                            start=Start.ON_RELEASE, admit_by=at,
                            completion=Completion(0.01, np.zeros(2), 0.003), supersedes="e0")
    state = RobotState("fr3_1", Validity.VALID, q=[1, 2], dq=[0, 0], source=at, received=at,
                       state_seq=3)
    auth = Authority(Lifecycle.FAULTED, "s", 2, 1, Fault("plant", "x", "rt", 2), False)
    status = ExecutionStatus.start(Receipt("e1", "r1", "fr3_1", Phase.ADMITTED, at))
    receipt = Receipt("e1", "r1", "fr3_1", Phase.SENT, at, evidence={"command_seq": 4})
    wire = lambda v: json.loads(json.dumps(encode(v)))  # noqa: E731
    assert identity_from(wire(ident)) == ident and capabilities_from(wire(caps)) == caps
    assert authority_from(wire(auth)) == auth and receipt_from(wire(receipt)) == receipt
    assert status_from(wire(status)) == status
    back = request_from(wire(req))
    assert back.trajectory.positions.tolist() == traj.positions.tolist()
    assert (back.gains.kd.tolist(), back.completion.residual_max, back.admit_by, back.start,
            back.supersedes) == ([3, 4], 0.003, at, Start.ON_RELEASE, "e0")
    s2 = state_from(wire(state))
    assert s2.q.tolist() == [1, 2] and s2.dq.tolist() == [0, 0] and s2.tau is None
    bad = wire(req)
    bad["trajectory"]["times"] = [0.0, float("nan")]
    try:
        request_from(bad)
        raise AssertionError("non-finite times decoded")
    except ValueError:
        pass
    print("execution codec self-check: PASS (identity, capabilities, state, authority, "
          "request, status, receipt round-trip through JSON; decoding validates)")


if __name__ == "__main__":
    _self_check()


def payload_from(d):
    from .controller import PayloadValue
    return PayloadValue(**d)


def payload_state_from(d):
    from .controller import PayloadState
    return PayloadState(d['revision'], payload_from(d['value']))


def preset_from(d):
    from .controller import ControllerPreset
    return ControllerPreset(**d)


def controller_state_from(d):
    from .controller import ControllerState
    return ControllerState(**{**d, 'payload': payload_state_from(d['payload'])})
