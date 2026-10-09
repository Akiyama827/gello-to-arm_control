"""Supervisor ledger rules the C++ fake server cannot provoke (CTRL-022).

    python tools/check/check_supervisor_ledger.py

The real ``RobotRuntime`` and ``ArmController`` over an in-process stub link
whose acceptance evidence and DISARM latency are controlled: a feedback
completion with no plant acceptance ends FAULTED, ARM is refused while a
DISARM is unconfirmed, and the ledger is bounded. The fake-server end to end
is check_sdk_supervisor.py; this complements it and opens no socket.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arm_control.config import load_robot_config
from arm_control.execution.contract import Lifecycle, Phase, RobotIdentity
from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.observation import identity_hash
from arm_control.supervisor import runtime as rt_mod
from arm_control.supervisor.runtime import Refused, RobotRuntime
from check_sdk_supervisor import JOINTS, move, write_config

SESSION = 0x5E55


class StubLink:
    """A perfectly tracking plant whose command acceptance is a switch."""

    def __init__(self, *, accept: bool, stop_delay_s: float = 0.0) -> None:
        self.accept, self.stop_delay_s = accept, stop_delay_s
        self.config = SimpleNamespace(state_timeout_s=0.5, host="stub")
        self.backend_name, self.supports_pose_hold, self.rt_setup = "stub", False, "ok"
        self.hold_ms = None  # the link contract: an RT HELLO without hold_ms (no streamed-gap deadline)
        self.identity, self.session_id = SimpleNamespace(model_id=""), SESSION
        self._hash = identity_hash(RobotIdentity("sdk_arm", JOINTS, "rt"))
        self.q = np.zeros(3)
        self.armed, self.epoch, self.stop_epoch = False, 0, 0
        self.seq = self.last_cmd = self.state_seq = 0
        self.history: deque = deque(maxlen=32)
        self._lock = threading.Lock()
        self.fault = ""  # the RT's latched fault text, switchable by a check

    def open(self) -> None:
        pass

    def _sample(self) -> None:
        self.state_seq += 1
        z = [0.0] * 3
        st = rtp.State(n=3, state_seq=self.state_seq, last_cmd_seq=self.last_cmd,
                       t_mono_ns=time.monotonic_ns(),
                       flags=rtp.FLAG_ARMED if self.armed else 0, session_id=SESSION,
                       identity_hash=self._hash, arm_epoch=self.epoch,
                       stop_epoch=self.stop_epoch, q=self.q.tolist(), dq=z, tau=z,
                       tau_cmd=z, q_cmd=self.q.tolist(), qd_cmd=z, wrench=[0.0] * 6)
        self.history.append((st, time.monotonic()))

    def recent_states(self):
        with self._lock:
            self._sample()
            return tuple(self.history)

    def latest_state(self):
        with self._lock:
            return self.history[-1] if self.history else (None, 0.0)

    def motor_health(self) -> dict:
        return dict(armed=self.armed, state_fresh=bool(self.history), latched_fault=self.fault,
                    any_fault=False, holding=False, arm_epoch=self.epoch,
                    supports_pose_hold=False)

    def motor_state(self, *, state=None) -> dict:
        z = np.zeros(3)
        q = np.asarray(state.q) if state is not None else self.q
        return dict(position=q, velocity=z, position_cmd=q, velocity_cmd=z, torque_cmd=z,
                    kp=z, kd=z, torque=z)

    def apply_command(self, command) -> int:
        with self._lock:
            self.seq += 1
            if self.armed:
                self.q = np.asarray(command["position"], float)[:3].copy()
                if self.accept:
                    self.last_cmd = self.seq
            return self.seq

    def enable_all(self) -> None:
        with self._lock:
            self.armed, self.epoch = True, self.epoch + 1

    def safe_stop(self) -> bool:
        time.sleep(self.stop_delay_s)
        with self._lock:
            self.armed, self.stop_epoch = False, self.epoch
        return True

    def close(self) -> None:
        self.safe_stop()


def runtime(link: StubLink) -> RobotRuntime:
    tmp = Path(tempfile.mkdtemp(prefix="supervisor_ledger_"))
    rt = RobotRuntime(load_robot_config(write_config(tmp, 1, 2)), link=link)
    rt.start()
    rt.claim("owner")
    return rt


def armed(rt: RobotRuntime) -> None:
    gen = rt.arm("owner")
    assert rt.await_authority(gen, armed=True, wait_s=3) is not None, rt.authority()


def settle(rt: RobotRuntime, eid: str, timeout_s: float = 6.0):
    return rt.status(eid, wait_s=timeout_s)


def check_fault_trace() -> None:
    """CTRL-051: the supervisor writes its own fault trace off the loop: the ticks pre_s before the first fault to post_s
    after it (later faults join the window), sha256-verified; a busy writer skips (counted), a failed write is counted, and
    neither stalls the servo loop; closing with a window open writes it, marked incomplete."""
    from unittest.mock import patch

    from arm_control.multi_arm import diagnostics, trace_export

    tmp = Path(tempfile.mkdtemp(prefix="supervisor_fault_"))
    link = StubLink(accept=True)
    rt = RobotRuntime(load_robot_config(write_config(tmp, 1, 2)), link=link)
    pre, post = .5, .3
    rt.fault_trace = diagnostics.SupervisorFaultTrace("sdk_arm", rt.period, root=tmp / "faults", pre_s=pre, post_s=post)
    rt.start()
    try:
        rt.claim("owner")
        armed(rt)
        time.sleep(pre + .3)
        t0 = time.monotonic()
        link.fault = "test fault"
        time.sleep(.1)
        link.fault = "a second fault"   # inside the window: the same trace
        deadline = time.monotonic() + 3
        while not rt.fault_trace.written and time.monotonic() < deadline:
            time.sleep(.02)
        assert rt.fault_trace.written, "no fault trace written"
        path = rt.fault_trace.written[0]
        report = json.loads(path.read_text())
        assert trace_export.verify(path)["ok"], trace_export.verify(path)
        assert report["schema"] == "arm_control.supervisor_fault/v1" and report["post_window_complete"]
        assert [f["reason"] for f in report["faults"]] == ["latched fault: test fault", "latched fault: a second fault"], report["faults"]
        start = report["fault_at_pc_monotonic"]
        stamps = [x["pc_monotonic"] for x in report["samples"]]
        assert abs(start - t0) < .05 and stamps[0] >= start - pre - 1e-9 and stamps[0] < start - pre + .05, (stamps[0], start)
        assert stamps[-1] >= start + post - .02 and np.diff(stamps).max() < .05, (stamps[-1], np.diff(stamps).max())
        assert len(stamps) >= .8 * (pre + post) / rt.period, len(stamps)
        # A busy writer: the next trace is SKIPPED (counted), and the loop keeps its cadence while the writer sleeps.
        real_write = diagnostics.write_trace

        def slow_write(*a, **k):
            time.sleep(1.)
            return real_write(*a, **k)
        link.fault = ""
        time.sleep(.05)
        with patch.object(diagnostics, "write_trace", slow_write):
            link.fault = "third"
            time.sleep(post + .1)       # window 3 flushed; its writer sleeps
            seq0 = rt.trace_samples()["latest_seq"]
            link.fault = "fourth"
            time.sleep(post + .1)       # window 4 flushes while the writer is busy
            assert rt.fault_trace.skipped == 1, rt.fault_trace.skipped
            ticks = rt.trace_samples()["latest_seq"] - seq0
            assert ticks >= .8 * (post + .1) / rt.period, f"the loop stalled behind the writer: {ticks} ticks"
            time.sleep(1.)
        # A failed write (disk full) is counted, never raised into the loop.
        link.fault = ""
        time.sleep(.05)
        with patch.object(diagnostics, "write_trace", side_effect=OSError(28, "No space left on device")):
            link.fault = "fifth"
            time.sleep(post + .2)
        assert rt.fault_trace.errors == 1 and rt.state().validity.value != "missing", rt.fault_trace.errors
        # Closing with a window open writes what it has.
        link.fault = ""
        time.sleep(.05)
        link.fault = "sixth"
        time.sleep(.05)
    finally:
        rt.close()
    last = json.loads(rt.fault_trace.written[-1].read_text())
    assert last["reason"] == "latched fault: sixth" and last["post_window_complete"] is False, last["reason"]
    assert len(list((tmp / "faults").glob("*.json"))) == 3, sorted((tmp / "faults").iterdir())
    print(f"  supervisor fault trace: {len(stamps)} ticks [-{pre}, +{post}] s, faults joined, busy writer skipped, "
          f"write failure counted, open window written on close")


def check_stall_schedule() -> None:
    """The servo loop's schedule (rt_mod.next_tick), modelled as the loop runs it: tick, take the next deadline, sleep to it.
    One late wake or one slow step, swept from just over a period to 40 ms: no two ticks closer than a period, and exactly one
    long gap per stall (one late tick, no burst)."""
    period = .01
    for stall in (.0105, .015, .019, .025, .04):
        for kind in ("wake", "step"):
            deadline, now, ticks = 0., 0., []
            for step in range(60):
                ticks.append(now)
                now += stall if (kind == "step" and step == 30) else .001       # the step itself
                deadline = rt_mod.next_tick(deadline, now, period)
                now = max(now, deadline) + (stall if (kind == "wake" and step == 30) else 0.)   # sleep
            gaps = np.diff(ticks)
            assert gaps.min() >= period - 1e-9, (stall, kind, gaps.min())
            assert (gaps > period + .002).sum() == 1, (stall, kind, gaps)
    assert np.isclose(rt_mod.next_tick(0., .005, .01), .01) and np.isclose(rt_mod.next_tick(0., .015, .01), .025)


def main() -> None:
    check_stall_schedule()
    check_fault_trace()
    # Completion observed on feedback, but the plant never accepted a command.
    rt = runtime(StubLink(accept=False))
    try:
        # The shared controller builder carried this config's values, not defaults.
        c = rt.controller
        assert (c._settling.timeout_s, c._settling.dwell_s) == (3.0, 0.1), c._settling
        assert c.executor.kp.tolist() == [100.0] * 3 and c.execution_policy is not None
        armed(rt)
        s = rt.submit("owner", move("unaccepted", np.zeros(3), [0.05, 0.05, 0.05], 1.0))
        s = settle(rt, s.execution_id)
        assert s.phase is Phase.FAULTED and "without plant acceptance" in s.reason, s
        assert Phase.APPLIED not in s.reached and Phase.SENT in s.reached, s.reached
        print(f"  feedback completion without acceptance: {s.phase.value} ({s.reason})")
    finally:
        rt.close()

    # The same plant accepting: COMPLETED after APPLIED.
    rt = runtime(StubLink(accept=True, stop_delay_s=0.5))
    try:
        armed(rt)
        s = settle(rt, rt.submit("owner", move("accepted", np.zeros(3), [0.05, 0.05, 0.05],
                                               1.0)).execution_id)
        assert s.phase is Phase.COMPLETED and Phase.APPLIED in s.reached, s
        # ARM is refused while a slow DISARM is still unconfirmed.
        gen = rt.disarm("owner")
        try:
            rt.arm("owner")
            raise AssertionError("ARM accepted during a pending DISARM")
        except Refused as exc:
            assert "DISARM is still being confirmed" in str(exc)
        assert rt.await_authority(gen, armed=False, wait_s=3).lifecycle is Lifecycle.CLAIMED
        # The ledger is bounded; forgotten request_ids are not idempotent keys.
        rt_mod.RETAINED = 3
        # Disarmed ON_ADMIT requests end REJECTED at selection: cheap terminals.
        ids = [rt.submit("owner", move(f"old{i}", np.zeros(3), np.zeros(3), 1.0)).execution_id
               for i in range(6)]
        time.sleep(0.1)  # one loop tick prunes
        try:
            rt.status(ids[0])
            raise AssertionError("ledger not pruned")
        except Refused:
            pass
        assert rt.status(ids[-1]).terminal
        print("  ARM during pending DISARM refused; ledger bounded at RETAINED")
    finally:
        rt.close()
    print("check_supervisor_ledger: PASS")


if __name__ == "__main__":
    main()
