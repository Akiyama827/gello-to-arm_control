"""Bridges ``grasp_request`` -> ``GraspGate`` -> ``grasp_result``."""
from __future__ import annotations

import numpy as np

from arm_control.bridge.grasp_gate import GraspGate, GraspStatus


class GraspController:
    """Bridges ``grasp_request`` -> ``GraspGate`` -> ``grasp_result``.

    Owns the gripper motor slot from a ``close`` request through the lift
    (holding the grasp) until a ``release``.  Each tick it reads the gripper
    motor's live torque+position, steps the gate, and overrides the gripper slot
    of the outgoing command.  It defers to the safety layer: it never drives
    while disarmed and abandons the grasp on a latched fault (the safe-stop owns
    the motors).  It only *reads* safety state — the node forwards commands, so
    the safety layer stays the single command owner.
    """

    def __init__(self, gate: GraspGate, gripper_index: int) -> None:
        self.gate = gate
        self.gi = int(gripper_index)
        self._active = False
        self._request_id = ""
        self._target_id = ""
        self._result_sent = False

    @property
    def active(self) -> bool:
        return self._active

    def request(self, payload: dict) -> dict | None:
        """Handle a grasp_request; return an immediate grasp_result or None."""
        self._request_id = str(payload.get("request_id", ""))
        self._target_id = str(payload.get("target_id", ""))
        mode = str(payload.get("mode", "close"))
        if mode == "release":
            self.gate.release()
            self._active = True  # keep driving the gripper open each tick
            self._result_sent = True  # unsensed: ack the open immediately
            return self._result(True, "released")
        self.gate.close()
        self._active = True
        self._result_sent = False
        return None

    def step(
        self, state: dict, base_command: dict, armed: bool, faulted: bool
    ) -> tuple[dict | None, dict | None]:
        """One tick: return (merged 7-motor command | None, grasp_result | None)."""
        if not self._active:
            return None, None
        if faulted:
            self._active = False  # fault mid-grasp defers to the safety safe-stop
            return None, None
        if not armed:
            return None, None  # do not drive the gripper while disarmed
        pos = float(state["position"][self.gi])
        tau = float(state["torque"][self.gi])
        status = self.gate.step(pos, tau)
        cmd = self._merge(base_command, self.gate.command())
        result = None
        if status is GraspStatus.LOST:
            # Drop event AFTER a reported grasp: emit a second, failed result so
            # the orchestrator can freeze; relax and release the gripper slot.
            result = self._result(False, "object lost")
            self._active = False
        elif not self._result_sent:
            if status is GraspStatus.GRASPED:
                result = self._result(True, "grasped")
                self._result_sent = True  # latched: keep holding through the lift
            elif status is GraspStatus.MISSED:
                result = self._result(False, "no object")
                self._result_sent = True
                self._active = False  # nothing to hold; release the slot
        return cmd, result

    def _merge(self, base_command: dict, g) -> dict:
        out = {
            key: np.asarray(base_command[key], dtype=np.float64).copy()
            for key in ("position", "velocity", "torque", "kp", "kd")
        }
        out["position"][self.gi] = g.position
        out["velocity"][self.gi] = g.velocity
        out["torque"][self.gi] = g.torque
        out["kp"][self.gi] = g.kp
        out["kd"][self.gi] = g.kd
        return out

    def _result(self, ok: bool, reason: str) -> dict:
        return {
            "request_id": self._request_id,
            "target_id": self._target_id,
            "ok": bool(ok),
            "reason": reason,
        }
