"""RT state packets as execution-contract values (``arm_control.execution``).

The one place RT wire flags become ``RobotState``/``Authority``. Pure: the
caller supplies the decoded ``protocol.State``, its PC receive time and the
freshness bound, so the same mapping serves the commanding client and a
read-only observer.
"""
from __future__ import annotations

from arm_control.execution.contract import (
    Authority, Clock, Fault, Lifecycle, RobotIdentity, RobotState, Stamp, Validity,
)
from arm_control.plants.remote_rt import protocol as rtp

FAULT_CODES = {rtp.FAULT_CMD_LOST: "cmd_lost", rtp.FAULT_CTL_LOST: "ctl_lost",
               rtp.FAULT_PLANT: "plant"}


def identity_hash(identity: RobotIdentity) -> int:
    """The RT server's identity hash for this instance and joint order.
    ValueError if the instance id does not fit the RT id field."""
    if not rtp.valid_id(identity.instance_id):
        raise ValueError(f"instance_id {identity.instance_id!r} is not a valid RT id")
    return rtp.identity_hash(identity.instance_id, rtp.joint_order_hash(identity.joint_names))


def _writing(state: rtp.State) -> bool:
    """Whether the servo selected a reference and wrote the plant that tick.
    Disarmed: nothing. Armed and FAULTED without HOLDING: a plant fault, no
    writes -- the robot's own safety has the arm. ponytail: a cause latched by
    another thread in the same tick can show FAULTED on a tick that still
    wrote; reported as not writing (conservative, one tick)."""
    armed = bool(state.flags & rtp.FLAG_ARMED)
    faulted = bool(state.flags & rtp.FLAG_FAULTED)
    return armed and (not faulted or bool(state.flags & rtp.FLAG_HOLDING))


def robot_state(state: rtp.State | None, identity: RobotIdentity, *, received_ns: int,
                now_ns: int, max_age_s: float, pc_domain: str = "",
                rt_domain: str = "") -> RobotState:
    """``received_ns``/``now_ns`` are PC monotonic; ``state.t_mono_ns`` is the
    RT host's clock and stays in that domain. Freshness is judged on the PC
    clock alone (receive age), never by subtracting across hosts."""
    iid = identity.instance_id
    if state is None:
        return RobotState(iid, Validity.MISSING, "no state received")
    if state.identity_hash != identity_hash(identity) or state.n != identity.n:
        return RobotState(iid, Validity.INVALID, "state is from another instance or joint order")
    age = (now_ns - received_ns) * 1e-9
    if age > max_age_s:
        return RobotState(iid, Validity.STALE, f"last state {age:.3f}s old (> {max_age_s:.3f}s)")
    wrench = state.wrench if state.flags & rtp.FLAG_WRENCH_VALID else None
    writing = _writing(state)  # the wire zero-fills these otherwise: unknown, not 0
    try:
        return RobotState(
            iid, Validity.VALID, q=state.q, dq=state.dq, tau=state.tau,
            selected_q=state.q_cmd if writing else None,
            selected_qd=state.qd_cmd if writing else None,
            output_tau=state.tau_cmd if writing else None, wrench=wrench,
            source=Stamp(state.t_mono_ns, Clock.RT_HOST, rt_domain),
            received=Stamp(received_ns, Clock.PC, pc_domain),
            state_seq=state.state_seq, last_command_seq=state.last_cmd_seq)
    except ValueError as exc:  # non-finite values on the wire
        return RobotState(iid, Validity.INVALID, str(exc))


def authority(state: rtp.State, *, fault_text: str = "") -> Authority:
    """The robot's lifecycle as its state packet shows it. ``session`` is the
    owning control session (hex), whoever owns it; ``holding`` is
    ``FLAG_HOLDING`` (False after a plant fault: nothing is written)."""
    session = f"{state.session_id:016x}" if state.session_id else None
    holding = bool(state.flags & rtp.FLAG_HOLDING)
    if state.flags & rtp.FLAG_FAULTED:
        fault = Fault(FAULT_CODES.get(state.fault_code, f"code_{state.fault_code}"),
                      fault_text, "rt", state.arm_epoch or None)
        return Authority(Lifecycle.FAULTED, session, state.arm_epoch, state.stop_epoch, fault,
                         holding)
    if state.flags & rtp.FLAG_ARMED:
        return Authority(Lifecycle.ARMED, session, state.arm_epoch, state.stop_epoch,
                         holding=holding)
    if state.flags & rtp.FLAG_CLAIMED:
        return Authority(Lifecycle.CLAIMED, session, 0, state.stop_epoch, holding=False)
    if session is not None:
        return Authority(Lifecycle.CONNECTED, session, 0, state.stop_epoch, holding=False)
    return Authority(Lifecycle.OBSERVING, None, 0, state.stop_epoch, holding=False)


def _self_check() -> None:
    golden = next(line.split()[1] for line in rtp.golden_lines() if line.startswith("STATE "))
    st = rtp.unpack_state(bytes.fromhex(golden))
    ident = RobotIdentity("fr3_2", rtp._GOLDEN_JOINTS, "franka")
    ok = robot_state(st, ident, received_ns=10_000, now_ns=20_000, max_age_s=0.1)
    assert ok.validity is Validity.VALID and ok.q.tolist() == st.q
    assert ok.source.clock is Clock.RT_HOST and ok.received.clock is Clock.PC
    assert ok.last_command_seq == st.last_cmd_seq == 42
    assert st.flags & rtp.FLAG_ARMED and not st.flags & rtp.FLAG_FAULTED
    assert ok.selected_q.tolist() == st.q_cmd and ok.selected_qd.tolist() == st.qd_cmd
    assert ok.output_tau.tolist() == st.tau_cmd
    assert identity_hash(ident) == st.identity_hash, "adapter hash differs from the golden"
    try:
        identity_hash(RobotIdentity("x" * 40, rtp._GOLDEN_JOINTS, "franka"))
        raise AssertionError("over-long RT id hashed")
    except ValueError:
        pass
    flags = st.flags & ~(rtp.FLAG_HOLDING | rtp.FLAG_ARMED)
    for off in (flags, flags | rtp.FLAG_ARMED | rtp.FLAG_FAULTED):  # disarmed / plant fault
        s2 = robot_state(rtp.State(**{**st.__dict__, "flags": off}), ident,
                         received_ns=0, now_ns=0, max_age_s=1)
        assert s2.q is not None and s2.selected_q is s2.selected_qd is s2.output_tau is None
    held = rtp.State(**{**st.__dict__,
                        "flags": flags | rtp.FLAG_ARMED | rtp.FLAG_FAULTED | rtp.FLAG_HOLDING})
    assert robot_state(held, ident, received_ns=0, now_ns=0, max_age_s=1).output_tau is not None
    assert (ok.wrench is not None) == bool(st.flags & rtp.FLAG_WRENCH_VALID)
    stale = robot_state(st, ident, received_ns=0, now_ns=200_000_000, max_age_s=0.1)
    assert stale.validity is Validity.STALE and stale.q is None
    other = RobotIdentity("fr3_1", rtp._GOLDEN_JOINTS, "franka")
    assert robot_state(st, other, received_ns=0, now_ns=0, max_age_s=1).validity is Validity.INVALID
    assert robot_state(None, ident, received_ns=0, now_ns=0, max_age_s=1).validity is Validity.MISSING
    nan = rtp.State(**{**st.__dict__, "q": [float("nan")] * 7})
    assert robot_state(nan, ident, received_ns=0, now_ns=0, max_age_s=1).validity is Validity.INVALID

    auth = authority(st)  # golden: armed, claimed session
    assert auth.lifecycle is Lifecycle.ARMED and auth.arm_epoch == st.arm_epoch
    faulted = rtp.State(**{**st.__dict__, "flags": st.flags | rtp.FLAG_FAULTED | rtp.FLAG_HOLDING,
                           "fault_code": rtp.FAULT_CMD_LOST})
    f = authority(faulted, fault_text="command stream lost")
    assert f.lifecycle is Lifecycle.FAULTED and f.fault.code == "cmd_lost" and f.holding
    plant = rtp.State(**{**st.__dict__, "flags": (st.flags | rtp.FLAG_FAULTED) & ~rtp.FLAG_HOLDING,
                         "fault_code": rtp.FAULT_PLANT})
    p = authority(plant)
    assert p.lifecycle is Lifecycle.FAULTED and p.fault.code == "plant" and p.holding is False
    idle = rtp.State(**{**st.__dict__, "flags": 0, "session_id": 0, "arm_epoch": 0})
    assert authority(idle).lifecycle is Lifecycle.OBSERVING
    print("RT observation adapter self-check: PASS (golden state -> valid RobotState, "
          "stale/foreign/missing/non-finite, selected/output only while writing, identity hash, "
          "ARMED/FAULTED holding vs plant-fault/OBSERVING authority)")


if __name__ == "__main__":
    _self_check()
