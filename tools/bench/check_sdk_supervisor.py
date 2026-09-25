"""SDK -> supervisor -> C++ fake RT server, end to end, device-free (CTRL-022).

    python tools/bench/check_sdk_supervisor.py --server <rt build>/arm_rt_server

Spawns the fake RT server and the supervisor process on loopback with a
generated 3-joint model, then drives them only through ``arm_control.sdk``.
This process imports the SDK and nothing heavier; the last check asserts it.
No robot address is used.

Installed-package form: run this file with a clean environment's Python that
has only ``pip install <arm_control>`` (numpy), and point the supervisor at a
full environment with ``--supervisor-python``/``--supervisor-path``.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from arm_control.execution.contract import (
    Clock, Lifecycle, Phase, Stamp, Start, TrajectoryRequest,
)
from arm_control.motion import JointTrajectory
from arm_control.sdk import ArmClient, Refused, SupervisorError

JOINTS = ("j1", "j2", "j3")
URDF = """<robot name="three">
  <link name="base"/>
  {links}
</robot>"""
LINK = """<link name="l{i}"><inertial><origin xyz="0 0 0.1"/><mass value="1"/>
  <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="j{i}" type="revolute"><parent link="{parent}"/><child link="l{i}"/>
  <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
  <limit lower="-2.9" upper="2.9" effort="50" velocity="2"/></joint>"""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_config(tmp: Path, udp: int, tcp: int) -> Path:
    links = "\n  ".join(LINK.format(i=i, parent="base" if i == 1 else f"l{i - 1}")
                        for i in (1, 2, 3))
    (tmp / "three.urdf").write_text(URDF.format(links=links))
    cfg = tmp / "robot.yaml"
    cfg.write_text(f"""arm:
  joints: [j1, j2, j3]
  urdf: {tmp / 'three.urdf'}
  ee_frame: l3
  kp: [100, 100, 100]
  kd: [20, 20, 20]
  max_tau: [50, 50, 50]
  done_pos_tol: 0.02
  done_vel_tol: 0.05
  settle_timeout_s: 3.0
  settle_dwell_s: 0.1
num_motors: 3
motor_names: [j1, j2, j3]
joint_names: [j1, j2, j3]
rt:
  host: 127.0.0.1
  udp_port: {udp}
  tcp_port: {tcp}
  instance_id: sdk_arm
  state_timeout_s: 0.5
execution_policy:
  command_rate_hz: 100
  state_timeout_sec: 0.25
  health_timeout_sec: 1.0
  start_pos_tol_rad: 0.1
  abort_pos_err_rad: 0.35
  hold_relatch_rad: 0.3
  velocity_limits: [1.0, 1.0, 1.0]
  acceleration_limits: [2.0, 2.0, 2.0]
""")
    return cfg


def move(rid: str, q0, q1, duration: float, **kw) -> TrajectoryRequest:
    """A rest-to-rest move (cubic via zero end velocities, midpoint speed)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    mid_v = 1.5 * (q1 - q0) / duration
    traj = JointTrajectory([0.0, duration / 2, duration],
                           np.stack([q0, (q0 + q1) / 2, q1]),
                           np.stack([0 * q0, mid_v, 0 * q0]))
    return TrajectoryRequest(rid, "sdk_arm", JOINTS, traj, **kw)


def phases(arm: ArmClient, eid: str) -> list[str]:
    return [r.phase.value for r in arm.receipts(eid)]


def completed_chain(arm: ArmClient, eid: str) -> None:
    """COMPLETED only after SENT and APPLIED, with acceptance at or past the
    execution's own first command, in the same ARM generation."""
    got = {r.phase: r for r in arm.receipts(eid)}
    assert arm.status(eid).phase is Phase.COMPLETED, phases(arm, eid)
    sent, applied = got[Phase.SENT].evidence, got[Phase.APPLIED].evidence
    assert applied["arm_epoch"] == sent["arm_epoch"] > 0, (sent, applied)
    assert applied["command_seq"] >= sent["command_seq"] > 0, (sent, applied)
    assert got[Phase.SENT].at.ns <= got[Phase.APPLIED].at.ns <= got[Phase.COMPLETED].at.ns


def expect_refused(fn, why: str) -> None:
    try:
        fn()
    except Refused as exc:
        assert why in str(exc), exc
        return
    raise AssertionError(f"not refused ({why})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="locally built arm_rt_server")
    parser.add_argument("--supervisor-python", default=sys.executable,
                        help="interpreter with the supervisor extra (default: this one)")
    parser.add_argument("--supervisor-path", default="",
                        help="PYTHONPATH for the supervisor process only")
    args = parser.parse_args()
    import arm_control
    print(f"  SDK from {Path(arm_control.__file__).parent}")
    tmp = Path(tempfile.mkdtemp(prefix="sdk_supervisor_"))
    udp, tcp = free_port(), free_port()
    sock = tmp / "sdk_arm.sock"
    rt = subprocess.Popen(
        [args.server, "--backend", "fake", "--n", "3", "--instance-id", "sdk_arm",
         "--joint-names", ",".join(JOINTS), "--udp-port", str(udp), "--tcp-port", str(tcp),
         "--hold-ms", "100", "--fault-ms", "600"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.3)
    env = dict(os.environ)
    if args.supervisor_path:
        env["PYTHONPATH"] = args.supervisor_path
    sup = subprocess.Popen(
        [args.supervisor_python, "-m", "arm_control.supervisor.service",
         "--config", str(write_config(tmp, udp, tcp)), "--socket", str(sock)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        deadline = time.monotonic() + 30
        while not sock.exists():
            assert sup.poll() is None, sup.stdout.read()
            assert time.monotonic() < deadline, "supervisor did not come up"
            time.sleep(0.05)
        scenario(sock, rt, sup)
    except BaseException:
        for name, proc in (("supervisor", sup), ("rt server", rt)):
            if proc.poll() is None:
                proc.kill()
            print(f"--- {name} log (tail)\n" + "".join(proc.stdout.readlines()[-40:]))
        raise
    finally:
        for proc in (sup, rt):
            if proc.poll() is None:
                proc.kill()
    print("check_sdk_supervisor: PASS")


def scenario(sock: Path, rt: subprocess.Popen, sup: subprocess.Popen) -> None:
    zero, target = np.zeros(3), np.array([0.3, -0.2, 0.25])
    arm = ArmClient("sdk_arm", socket_path=sock)
    # Connecting is observation only: nothing claimed, armed or moved.
    time.sleep(0.3)
    auth = arm.authority()
    assert auth.lifecycle is Lifecycle.CONNECTED and auth.session is None, auth
    state = arm.state()
    assert state.validity.value == "valid" and state.selected_q is None, state
    assert arm.capabilities.rt_setup != "unknown" and "joint_impedance" in arm.capabilities.profiles
    try:
        ArmClient("fr3_1", socket_path=sock)
        raise AssertionError("wrong instance accepted")
    except SupervisorError as exc:
        assert "serves 'sdk_arm'" in str(exc)
    expect_refused(lambda: arm.execute(move("r0", zero, target, 1.0)), "claim sdk_arm first")

    # One owner.
    assert arm.claim().lifecycle is Lifecycle.CLAIMED
    other = ArmClient("sdk_arm", socket_path=sock)
    expect_refused(other.claim, "already owned")
    expect_refused(lambda: other.execute(move("r0", zero, target, 1.0)), "owned by another")

    # ON_ADMIT while disarmed: admitted (ownership suffices), refused at selection.
    s = arm.execute(move("r1", zero, target, 1.0))
    assert s.phase is Phase.REJECTED and "not armed" in s.reason, s
    assert phases(arm, s.execution_id) == ["requested", "admitted", "rejected"]
    # ON_RELEASE stages while disarmed, then runs after ARM.
    staged = arm.execute(move("r2", zero, target, 1.0, start=Start.ON_RELEASE))
    assert staged.phase is Phase.ADMITTED, staged
    auth = arm.arm()
    assert auth.lifecycle is Lifecycle.ARMED and auth.arm_epoch > 0, auth
    arm.release(staged.execution_id)
    done = arm.wait(staged.execution_id, timeout_s=10)
    assert done.phase is Phase.COMPLETED, done
    got = arm.receipts(staged.execution_id)
    assert [r.phase.value for r in got] == [
        "requested", "admitted", "selected", "sent", "applied", "completed"], got
    completed_chain(arm, staged.execution_id)
    assert got[3].evidence["arm_epoch"] == auth.arm_epoch
    q = arm.state().q
    assert np.max(np.abs(q - target)) < 0.03, q

    # Idempotency: same request_id -> same execution; different content refused.
    again = arm.execute(move("r2", zero, target, 1.0, start=Start.ON_RELEASE))
    assert again.execution_id == staged.execution_id and again.phase is Phase.COMPLETED
    expect_refused(lambda: arm.execute(move("r2", target, zero, 1.0)), "different content")

    # Supersession: explicit, admitted-only.
    a = arm.execute(move("r3", target, zero, 1.0, start=Start.ON_RELEASE))
    b = arm.execute(move("r4", target, zero, 1.0, start=Start.ON_RELEASE))
    assert b.phase is Phase.REJECTED and "supersedes" in b.reason, b
    c = arm.execute(move("r5", target, zero, 1.0, start=Start.ON_RELEASE,
                         supersedes=a.execution_id))
    assert c.phase is Phase.ADMITTED, c
    a = arm.status(a.execution_id)
    assert a.phase is Phase.SUPERSEDED, a
    assert arm.receipts(a.execution_id)[-1].evidence["superseded_by"] == c.execution_id
    assert arm.release(a.execution_id).phase is Phase.SUPERSEDED  # never runs
    arm.release(c.execution_id)
    assert arm.wait(c.execution_id, 10).phase is Phase.COMPLETED
    completed_chain(arm, c.execution_id)  # a later execution: no stale acceptance

    # Expiry on the supervisor's clock; a foreign clock domain is refused.
    d = arm.execute(move("r6", zero, target, 1.0, start=Start.ON_RELEASE,
                         admit_by=arm.deadline(0.3)))
    assert d.phase is Phase.ADMITTED
    time.sleep(0.5)
    assert arm.status(d.execution_id).phase is Phase.EXPIRED
    assert arm.release(d.execution_id).phase is Phase.EXPIRED
    foreign = Stamp(arm.now().ns + 10**9, Clock.PC, "pc:another-host")
    f = arm.execute(move("r7", zero, target, 1.0, start=Start.ON_RELEASE, admit_by=foreign))
    assert f.phase is Phase.REJECTED and "this supervisor runs" in f.reason, f

    # Cancel mid-run completes with CANCELLED; the robot holds, still armed.
    long = arm.execute(move("r8", zero, target, 3.0))
    assert long.phase is Phase.SELECTED, long
    wait_phase(arm, long.execution_id, Phase.APPLIED)
    # SELECTED or later is never replaced.
    x = arm.execute(move("r9", zero, target, 1.0, supersedes=long.execution_id))
    assert x.phase is Phase.REJECTED and "only admitted" in x.reason, x
    # A long wait on one thread does not hold the channel against a cancel.
    waiter = threading.Thread(target=lambda: arm.wait(long.execution_id, 20), daemon=True)
    waiter.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    cancelled = arm.cancel(long.execution_id, "operator stop")
    assert time.monotonic() - t0 < 1.0, "cancel waited behind another thread's wait"
    waiter.join(timeout=2)
    assert not waiter.is_alive()
    assert cancelled.phase is Phase.CANCELLED and cancelled.reason == "operator stop"
    time.sleep(0.5)
    auth, state = arm.authority(), arm.state()
    assert auth.lifecycle is Lifecycle.ARMED and np.max(np.abs(state.dq)) < 0.05, (auth, state)
    held = state.q
    assert arm.cancel(long.execution_id).phase is Phase.CANCELLED  # terminal: no-op

    # Stalled clients block nobody: a half-sent frame on another connection
    # and a GIL-hogging thread here, while the owner's execution runs.
    stall = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stall.connect(str(sock))
    stall.sendall(b'{"id": 1, "op": "sta')
    stop_hog = threading.Event()

    def hog():
        while not stop_hog.is_set():
            sum(i * i for i in range(10000))
    threading.Thread(target=hog, daemon=True).start()
    g = arm.execute(move("r10", held, zero, 1.5))
    time.sleep(2.0)  # the client says nothing while the supervisor executes
    stop_hog.set()
    assert arm.wait(g.execution_id, 10).phase is Phase.COMPLETED
    completed_chain(arm, g.execution_id)
    stall.close()

    # Disarm cancels staged work and is confirmed by the stop epoch.
    staged = arm.execute(move("r11", zero, target, 1.0, start=Start.ON_RELEASE))
    auth = arm.disarm()
    assert auth.lifecycle is Lifecycle.CLAIMED and auth.stop_epoch > 0, auth
    s = arm.status(staged.execution_id)
    assert s.phase is Phase.CANCELLED and s.reason == "disarmed", s

    # Losing the owner cancels its live execution and holds; no auto-reacquire.
    arm.arm()
    h = arm.execute(move("r12", arm.state().q, target, 3.0))
    wait_phase(arm, h.execution_id, Phase.APPLIED)
    arm.close()
    try:
        arm.status(h.execution_id)
        raise AssertionError("closed client still answered")
    except SupervisorError:
        pass
    time.sleep(0.3)
    s = other.status(h.execution_id)
    assert s.phase is Phase.CANCELLED and s.reason == "owner disconnected", s
    auth = other.authority()
    assert auth.lifecycle is Lifecycle.ARMED and auth.session is None, auth
    assert other.claim().session == other.session

    # An owner PROCESS that dies inside a long wait is noticed within ~1 s.
    other.unclaim()
    owner = subprocess.Popen([sys.executable, "-c", OWNER_SCRIPT, str(sock)],
                             stdout=subprocess.PIPE, text=True)
    eid = owner.stdout.readline().strip()
    wait_phase(other, eid, Phase.APPLIED)
    owner.kill()
    t0 = time.monotonic()
    while not other.status(eid).terminal:
        assert time.monotonic() - t0 < 1.5, "owner death not noticed"
        time.sleep(0.02)
    s = other.status(eid)
    assert s.phase is Phase.CANCELLED and s.reason == "owner disconnected", s
    print(f"  owner process killed mid-wait: cancelled after {time.monotonic() - t0:.2f}s")
    assert other.claim().session == other.session

    # A plant that disappears mid-run faults the execution; nothing hangs.
    k = other.execute(move("r13", other.state().q, zero, 3.0))
    wait_phase(other, k.execution_id, Phase.APPLIED)
    rt.send_signal(signal.SIGKILL)
    s = other.wait(k.execution_id, 5)
    assert s.phase is Phase.FAULTED, s
    other.close()

    # Bounded shutdown.
    t0 = time.monotonic()
    sup.send_signal(signal.SIGTERM)
    sup.wait(timeout=10)
    print(f"  supervisor stopped in {time.monotonic() - t0:.2f}s (rc {sup.returncode})")

    heavy = sorted(m for m in sys.modules if m.split(".")[0] in {
        "pyarrow", "dora", "mujoco", "rerun", "pinocchio", "yaml", "torch"}
        or m.startswith(("arm_control.control", "arm_control.supervisor",
                         "arm_control.messages", "arm_control.contracts")))
    assert not heavy, f"the SDK process imported {heavy}"
    print("  receipts, ownership, idempotency, supersession, expiry, cancel, stalls, "
          "disarm, owner loss, plant loss: ok; SDK process imported nothing heavy")


OWNER_SCRIPT = """
import sys
import numpy as np
from arm_control.execution.contract import TrajectoryRequest
from arm_control.motion import JointTrajectory
from arm_control.sdk import ArmClient
arm = ArmClient("sdk_arm", socket_path=sys.argv[1])
arm.claim()
q0 = arm.state().q
q1 = np.array([0.3, -0.2, 0.25])
traj = JointTrajectory([0.0, 1.5, 3.0], np.stack([q0, (q0 + q1) / 2, q1]),
                       np.stack([0 * q0, (q1 - q0) / 2, 0 * q0]))
status = arm.execute(TrajectoryRequest("owner-proc", "sdk_arm", ("j1", "j2", "j3"), traj))
print(status.execution_id, flush=True)
arm.wait(status.execution_id, 30)
"""


def wait_phase(arm: ArmClient, eid: str, phase: Phase, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while phase not in arm.status(eid).reached:
        assert time.monotonic() < deadline, (eid, phase, phases(arm, eid))
        time.sleep(0.02)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
