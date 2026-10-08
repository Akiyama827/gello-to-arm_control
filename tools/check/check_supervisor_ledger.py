"""Supervisor ledger rules the C++ fake server cannot provoke (CTRL-022).

    python tools/check/check_supervisor_ledger.py

The real ``RobotRuntime`` and ``ArmController`` over an in-process stub link
whose acceptance evidence and DISARM latency are controlled: a feedback
completion with no plant acceptance ends FAULTED, ARM is refused while a
DISARM is unconfirmed, and the ledger is bounded. The fake-server end to end
is check_sdk_supervisor.py; this complements it and opens no socket.
"""
from __future__ import annotations

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
        self.config = SimpleNamespace(state_timeout_s=0.5)
        self.backend_name, self.supports_pose_hold, self.rt_setup = "stub", False, "ok"
        self.hold_ms = None  # the link contract: an RT HELLO without hold_ms (no streamed-gap deadline)
        self.identity, self.session_id = SimpleNamespace(model_id=""), SESSION
        self._hash = identity_hash(RobotIdentity("sdk_arm", JOINTS, "rt"))
        self.q = np.zeros(3)
        self.armed, self.epoch, self.stop_epoch = False, 0, 0
        self.seq = self.last_cmd = self.state_seq = 0
        self.history: deque = deque(maxlen=32)
        self._lock = threading.Lock()

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
        return dict(armed=self.armed, state_fresh=bool(self.history), latched_fault="",
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


def main() -> None:
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
