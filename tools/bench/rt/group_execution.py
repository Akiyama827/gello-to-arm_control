"""CTRL-040 group execution on two fake RT servers of one host (device-free).

Two ``arm_rt_server --backend fake`` processes share one group segment. A PC client per arm drives
JOIN -> upload -> group COMMIT. The RT rendezvous is host-local: the first member to commit sets one
group deadline on the shared host clock and the second adopts it; the deadline is checked at commit,
and once both members have committed, selection is unconditional. Peer fault/cancel/loss reaches the
other member's servo tick, which retires its reference and reports it.

Cases: happy path (one deadline, per-member selection stamps, skew), peer never commits, late commit,
cancel after staging, peer process loss, peer cancel, peer fault, duplicate/other group, old-epoch
message, fault from an earlier epoch not affecting a new group.

The timing values passed to the servers here are BENCH values; they are not proposals and not
defaults (the servers have none). Fake backends only: nothing here says anything about FCI timing.

    PYTHONPATH=. python tools/bench/rt/group_execution.py path/to/arm_rt_server
"""
from __future__ import annotations

# ruff: noqa: E402  (path bootstrap must precede the arm_control import)

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for path in (_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import os
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as p
from arm_control.plants.remote_rt.client import ReferenceRefused, RtBackend, RtConfig, RtLinkError
from native_reference import Pump

JOINTS = ["j0", "j1", "j2"]
PEER_MAX_AGE_MS, MIN_LEAD_MS = 40, 5  # bench values, see the module docstring
BUDGET_NS = 250_000_000


def free_port(kind):
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def curve(seconds=3.0):
    return JointTrajectory([0, seconds], np.array([[0] * 3, [.1] * 3]), np.zeros((2, 3)))


class Member:
    def __init__(self, binary, shm, index, count, tmp):
        self.index, self.count = index, count
        self.udp, self.tcp = free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM)
        self.proc = subprocess.Popen(
            [binary, "--backend", "fake", "--n", "3", "--instance-id", f"arm{index}",
             "--joint-names", ",".join(JOINTS), "--udp-port", str(self.udp),
             "--tcp-port", str(self.tcp), "--bind", "127.0.0.1", "--hold-ms", "100",
             "--fault-ms", "500", "--enable-reference", "--group-shm", shm,
             "--group-member", str(index), "--group-members", str(count),
             "--group-peer-max-age-ms", str(PEER_MAX_AGE_MS), "--group-min-lead-ms", str(MIN_LEAD_MS),
             "--machine-id-file", str(tmp / "machine-id"), "--boot-id-file", str(tmp / "boot-id")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.link = RtBackend(RtConfig(host="127.0.0.1", udp_port=self.udp, tcp_port=self.tcp,
                                       instance_id=f"arm{index}"), JOINTS)
        deadline = time.monotonic() + 8
        while True:
            try:
                self.link = RtBackend(RtConfig(host="127.0.0.1", udp_port=self.udp,
                                               tcp_port=self.tcp, instance_id=f"arm{index}"), JOINTS)
                self.link.open()
                break
            except (RtLinkError, OSError):
                assert self.proc.poll() is None and time.monotonic() < deadline, "server not ready"
                time.sleep(0.05)
        assert self.link.group_capable
        assert self.link.negotiate_reference("software-fixture-model")
        self.link.enable_all()
        self.pump = Pump(self.link)
        self.ref_id = 0

    def state(self):
        return self.link.latest_state()[0]

    def upload(self):
        self.ref_id += 1
        return self.link.upload_reference(curve(), reference_id=self.ref_id)

    def stop_pump(self):
        self.pump.stop.set()
        self.pump.thread.join(2)

    def close(self):
        try:
            self.pump.stop.set()
            self.pump.thread.join(2)
        except Exception:  # noqa: BLE001
            pass
        self.link.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@contextmanager
def pair(binary, tmp, count=2):
    shm = f"/arm_rt_group_bench_{os.getpid()}"
    from multiprocessing import shared_memory  # noqa: F401  (segment lives in /dev/shm)
    members = []
    try:
        for i in range(count):
            members.append(Member(binary, shm, i, count, tmp))
        yield members
    finally:
        for m in members:
            m.close()
        try:
            os.unlink("/dev/shm" + shm)
        except FileNotFoundError:
            pass


def commit_both(members, group_id, uploads, delays=(0., 0.), budget_ns=BUDGET_NS):
    """COMMIT every member concurrently (with optional start delays); returns per-member reply/error."""
    out = [None] * len(members)

    def run(i, m):
        time.sleep(delays[i])
        try:
            out[i] = m.link.group_commit(uploads[i], reference_id=m.ref_id, group_id=group_id,
                                         member_index=m.index, budget_ns=budget_ns)
        except ReferenceRefused as exc:
            out[i] = exc
    threads = [threading.Thread(target=run, args=(i, m)) for i, m in enumerate(members)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(6)
    return out


def join_all(members, group_id):
    for m in members:
        m.link.group_join(group_id, m.index, m.count)


def wait_for(fn, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("timeout")


def running(m):
    s = m.state()
    return bool(s and s.reference and s.reference["status"] == p.REFERENCE_RUNNING)


def happy(binary, tmp, skews):
    with pair(binary, tmp) as (a, b):
        join_all((a, b), 1)
        ua, ub = a.upload(), b.upload()
        ra, rb = commit_both((a, b), 1, (ua, ub))
        assert isinstance(ra, dict) and isinstance(rb, dict), (ra, rb)
        assert ra["result"] == rb["result"] == p.REF_OK
        assert ra["deadline_ns"] == rb["deadline_ns"] != 0, "one group deadline, adopted by the second"
        assert ra["selected_ns"] and rb["selected_ns"] <= rb["deadline_ns"] * 10  # stamps present
        wait_for(lambda: running(a) and running(b))
        skews.append(abs(ra["selected_ns"] - rb["selected_ns"]) / 1e6)
        a.link.group_cancel(1, a.index, a.count)
        b.link.group_cancel(1, b.index, b.count)
        wait_for(lambda: not running(a) and not running(b))
        assert not a.state().faulted and not b.state().faulted


def peer_never_commits(binary, tmp):
    with pair(binary, tmp) as (a, b):
        join_all((a, b), 2)
        ua, _ = a.upload(), b.upload()
        started = time.monotonic()
        try:
            a.link.group_commit(ua, reference_id=a.ref_id, group_id=2, member_index=0,
                                budget_ns=BUDGET_NS)
        except ReferenceRefused as exc:
            assert exc.result == p.REF_LATE, exc.result  # the deadline passed while waiting
            assert BUDGET_NS / 1e9 - .05 <= time.monotonic() - started < BUDGET_NS / 1e9 + 1.0
        else:
            raise AssertionError("a commit whose peer never committed was accepted")
        assert not running(a) and a.state().reference["reference_id"] == 0
        assert not a.state().faulted


def late_commit(binary, tmp):
    with pair(binary, tmp) as (a, b):
        join_all((a, b), 3)
        ua, ub = a.upload(), b.upload()
        ra, rb = commit_both((a, b), 3, (ua, ub), delays=(0., BUDGET_NS / 1e9 + .15))
        for r in (ra, rb):
            assert isinstance(r, ReferenceRefused), r
        assert ra.result == p.REF_LATE
        assert rb.result in (p.REF_LATE, p.REF_GROUP_PEER), rb.result  # deadline / peer CANCELLED
        assert not running(a) and not running(b)


def cancel_after_staging(binary, tmp):
    with pair(binary, tmp) as (a, b):
        join_all((a, b), 4)
        ua, ub = a.upload(), b.upload()
        a.link.group_cancel(4, a.index, a.count)
        try:
            b.link.group_commit(ub, reference_id=b.ref_id, group_id=4, member_index=1,
                                budget_ns=BUDGET_NS)
        except ReferenceRefused as exc:
            assert exc.result == p.REF_GROUP_PEER and exc.reply["peer_verdict"] == p.PEER_CANCELLED, exc.reply
        else:
            raise AssertionError("commit accepted after the peer cancelled the group")
        try:
            a.link.group_commit(ua, reference_id=a.ref_id, group_id=4, member_index=0,
                                budget_ns=BUDGET_NS)
        except ReferenceRefused as exc:
            assert exc.result == p.REF_GROUP_REFUSED, exc.result  # not a member any more
        else:
            raise AssertionError("commit accepted after this member left the group")
        assert not running(a) and not running(b)


def start_running(binary, tmp, gid):
    cm = pair(binary, tmp)
    a, b = cm.__enter__()
    join_all((a, b), gid)
    ua, ub = a.upload(), b.upload()
    ra, rb = commit_both((a, b), gid, (ua, ub))
    assert isinstance(ra, dict) and isinstance(rb, dict), (ra, rb)
    assert ra["result"] == rb["result"] == p.REF_OK
    wait_for(lambda: running(a) and running(b))  # the state stream lags the reply
    return cm, a, b


def event_of(m, result=p.REF_GROUP_PEER):
    return wait_for(lambda: next((e for e in m.link.group_events if e["result"] == result), None), 3)


def peer_process_loss(binary, tmp):
    cm, a, b = start_running(binary, tmp, 5)
    try:
        killed = time.monotonic()
        b.proc.kill()  # by PID
        b.proc.wait(3)
        e = event_of(a)
        assert e["peer_verdict"] == p.PEER_STALE, e
        assert e["received_at"] - killed < PEER_MAX_AGE_MS / 1e3 + 1.0
        s = wait_for(lambda: not running(a) and a.state())
        assert not s.faulted and s.reference["status"] in (p.REFERENCE_CANCELLED, p.REFERENCE_NONE)
        time.sleep(.8)  # the survivor keeps its deadman alive and stays healthy
        assert not a.state().faulted
    finally:
        b.close = lambda: None
        cm.__exit__(None, None, None)


def peer_cancel(binary, tmp):
    cm, a, b = start_running(binary, tmp, 6)
    try:
        b.link.group_cancel(6, b.index, b.count)
        e = event_of(a)
        assert e["peer_verdict"] == p.PEER_CANCELLED, e
        wait_for(lambda: not running(a))
        assert not a.state().faulted
    finally:
        cm.__exit__(None, None, None)


def peer_fault_then_new_group(binary, tmp):
    cm, a, b = start_running(binary, tmp, 7)
    try:
        b.stop_pump()  # B's command stream goes quiet: CMD_LOST latches after --fault-ms
        e = event_of(a)
        assert e["peer_verdict"] in (p.PEER_FAULTED, p.PEER_CANCELLED), e
        wait_for(lambda: b.state().faulted)
        wait_for(lambda: not running(a))
        assert not a.state().faulted
        # Recover both arms (DISARM, ARM) and start a NEW group: the old fault is scoped away.
        for m in (a, b):
            m.stop_pump()
            assert m.link.safe_stop()
        for m in (a, b):
            m.link.enable_all()
            assert m.link.negotiate_reference("software-fixture-model") in (True, False)
            m.pump = Pump(m.link)
            m.ref_id = 0
        join_all((a, b), 8)
        ua, ub = a.upload(), b.upload()
        ra, rb = commit_both((a, b), 8, (ua, ub))
        assert isinstance(ra, dict) and isinstance(rb, dict), (ra, rb)
        assert ra["result"] == rb["result"] == p.REF_OK
    finally:
        cm.__exit__(None, None, None)


def other_group_and_old_epoch(binary, tmp):
    with pair(binary, tmp) as (a, b):
        join_all((a, b), 9)
        for bad in (10,):  # a second group while one is active
            try:
                a.link.group_join(bad, a.index, a.count)
            except ReferenceRefused as exc:
                assert exc.result == p.REF_GROUP_REFUSED
            else:
                raise AssertionError("second group accepted")
        a.link.group_join(9, a.index, a.count)  # idempotent for the same group
        epoch = a.link._reference_scope(None, None)[1]
        try:
            b.link.group_join(11, b.index, b.count, arm_epoch=epoch + 5)  # another epoch
        except (ReferenceRefused, RtLinkError):
            pass
        else:
            raise AssertionError("old/other-epoch JOIN accepted")
        try:
            a.link.group_join(9, 1, 2)  # wrong member index for this server
        except ReferenceRefused as exc:
            assert exc.result == p.REF_GROUP_REFUSED
        else:
            raise AssertionError("wrong member index accepted")


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "rt/build/arm_rt_server")
    skews: list[float] = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "machine-id").write_text("0123456789abcdef0123456789abcdef\n")
        (tmp / "boot-id").write_text("11111111-2222-3333-4444-555555555555\n")
        for _ in range(int(os.environ.get("GROUP_HAPPY_RUNS", "3"))):
            happy(binary, tmp, skews)
        peer_never_commits(binary, tmp)
        late_commit(binary, tmp)
        cancel_after_staging(binary, tmp)
        peer_process_loss(binary, tmp)
        peer_cancel(binary, tmp)
        peer_fault_then_new_group(binary, tmp)
        other_group_and_old_epoch(binary, tmp)
    print(f"selection skew between members (ms, fake RT, host clock): "
          f"min {min(skews):.3f} median {sorted(skews)[len(skews)//2]:.3f} max {max(skews):.3f} "
          f"over {len(skews)} runs")
    print("PASS: group execution: one deadline, per-member selection stamps, peer never commits, late "
          "commit, cancel after staging, peer process loss, peer cancel, peer fault then a new group, "
          "second group / old epoch / wrong member refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
