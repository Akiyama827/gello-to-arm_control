"""Operator-console state for a self-sensing hand (Franka Hand).

Split out of ``arm_console`` on 2026-09-10. Everything here is about ONE
question -- may this hand act right now, and what is it holding -- and none of
it is about sliders, plans, jogging or HTTP, which is why it was worth its own
file. A console driving any hand that reports its own grasp verdict can use it.

THREADING: every method assumes THE CALLER ALREADY HOLDS the console lock.
This object deliberately owns no lock of its own -- the console snapshots hand
state and arm state together in one ``_state()`` under one non-reentrant lock,
and a second lock here would either deadlock that or tear the snapshot.

The ARM GATE is passed in rather than read: a hand does not know how arming
works, and making ``armed``/``fault`` arguments keeps that true.
"""
from __future__ import annotations

import time
import uuid

from arm_control.end_effectors.franka_hand import (
    GRASP_TIMEOUT_S,
    MAX_FORCE_N,
    MIN_FORCE_N,
    resolve_grasp_parameters,
)

# A hand observation older than this cannot admit an action: the operator would
# be acting on a picture of the jaws, not the jaws.
HAND_STALE_S = 0.6


class HandPanel:
    """Request gating, result interpretation and payload declaration."""

    def __init__(
        self,
        *,
        gripper_cfg: dict | None,
        payload_table: dict | None = None,
        width_max_m: float = 0.0,
        log=None,
    ) -> None:
        self._cfg = dict(gripper_cfg or {})
        self.defaults = resolve_grasp_parameters(self._cfg, {}) if self._cfg else None
        self._width_max_m = float(width_max_m)
        self._log = log or (lambda _message: None)
        # Manual teleop has no planner to declare what the hand is carrying, and
        # the FR3's own model cannot know either: Desk's end effector and the RT
        # box's --ee-mass are both FIXED startup values. So the operator names
        # the module, and its mass rides the controller's tau_ff exactly as the
        # planner's does. Keys are module ids from the config's `module_grasps`.
        self._payload_table = {
            str(k): v for k, v in (payload_table or {}).items() if isinstance(v, dict)
        }
        self._payload_id = ""
        self._payload_pending: dict | None = None
        self._state: dict = {}
        self._seq = None
        self._sample_at = float("-inf")
        self._pending: dict | None = None
        self._request_id = ""
        self._waiting = False
        self._sent_at = 0.0
        self._submit_seq = None
        self._status = "Idle"
        self._disarm_requested = False

    # -- plant observations ---------------------------------------------------
    def set_state(self, state: dict) -> None:
        if not state.get("available"):
            self._pending = None
            self._request_id = ""
            self._waiting = False
            self._status = "Unavailable: Hand disconnected"
        self._state = dict(state)
        seq = state.get("sample_seq")
        if seq is not None and seq != self._seq and state.get("measured"):
            self._seq = seq
            self._sample_at = time.monotonic()

    # -- the arm gate ---------------------------------------------------------
    def disarm_requested(self) -> None:
        """An operator DISARM: drop anything queued and refuse until re-armed."""
        self._disarm_requested = True
        self.cancel_pending()

    def rearmed(self) -> None:
        self._disarm_requested = False
        self.cancel_pending()

    def cancel_pending(self) -> None:
        if self._pending is not None:
            self._pending = None
            self._waiting = False

    # -- admission ------------------------------------------------------------
    def refusal(self, *, armed, fault, consuming: bool = False) -> str:
        """Why this hand may not act, or "" if it may. Both admission points."""
        if self.defaults is None or not self._state.get("force_grasp"):
            return "Force grasp unavailable: plant or Hand bridge does not support it"
        if (not self._state.get("available") or not self._state.get("measured")
                or time.monotonic() - self._sample_at > HAND_STALE_S):
            return "Hand feedback unavailable or stale"
        if armed is not True or fault or self._disarm_requested:
            return "Hand motion requires confirmed ARM and no fault"
        if self._state.get("busy"):
            return "Hand busy: wait for the current action"
        if not consuming and (self._waiting or self._submit_seq == self._seq):
            return "Hand action pending: wait for a fresh result and observation"
        return ""

    def busy(self) -> bool:
        return (self._waiting or bool(self._state.get("busy"))
                or (self._submit_seq is not None and self._submit_seq == self._seq))

    def owns_jaws(self) -> bool:
        """The hand has the jaws: a slider command would fight an action.

        Busy, or waiting on a result, or DISARM asked for everything to stop.
        Both slider gates ask exactly this, so they ask it once, here.
        """
        return self.busy() or self._disarm_requested

    def snapshot(self, *, armed, fault) -> dict:
        if self._waiting and time.monotonic() - self._sent_at > GRASP_TIMEOUT_S + 1:
            self._waiting = False
            self._pending = None
            self._request_id = ""
            self._status = "Unavailable: action result timed out"
        reason = self.refusal(armed=armed, fault=fault)
        fresh = (bool(self._state.get("available"))
                 and time.monotonic() - self._sample_at <= HAND_STALE_S)
        width = self._state.get("width")
        if (self._status.startswith("Open accepted") and fresh
                and self._state.get("measured") and not self._state.get("busy")
                and self._seq != self._submit_seq and width is not None
                and abs(float(width) - self.defaults["open_width_m"]) <= .002):
            self._status = "Open"
        return {
            "enabled": not reason,
            "reason": reason,
            "status": self._status,
            "busy": self.busy(),
            "defaults": self.defaults,
            "force_min_n": MIN_FORCE_N, "force_max_n": MAX_FORCE_N,
            "width_max_mm": self._width_max_m * 2000,
            "measured_width_mm": (
                float(width) * 1000
                if fresh and self._state.get("measured") and width is not None
                else None
            ),
            "payloads": [
                {"id": k, "mass_kg": float(v.get("mass_kg", 0.0))}
                for k, v in sorted(self._payload_table.items())
            ],
            "payload_id": self._payload_id,
        }

    # -- operator actions -----------------------------------------------------
    def request(self, payload: dict, *, armed, fault) -> None:
        if not isinstance(payload, dict) or set(payload) - {
            "mode", "width_m", "force_n", "payload_id"
        }:
            raise ValueError("Hand action accepts mode, width_m, force_n and payload_id only")
        if payload.get("mode") == "release" and set(payload) != {"mode"}:
            raise ValueError("Open uses the configured open width and speed")
        payload_id = str(payload.get("payload_id", "") or "")
        if payload_id and payload_id not in self._payload_table:
            raise ValueError(f"unknown payload {payload_id!r}")
        payload = {k: v for k, v in payload.items() if k != "payload_id"}
        reason = self.refusal(armed=armed, fault=fault)
        if reason:
            raise ValueError(reason)
        mode = payload.get("mode", "close")
        params = resolve_grasp_parameters(self._cfg, payload)
        self._request_id = uuid.uuid4().hex
        self._pending = {
            "request_id": self._request_id, "target_id": "hand", "mode": mode,
            "width_m": params["width_m"], "force_n": params["force_n"],
        }
        self._submit_seq = self._seq
        self._waiting = True
        self._sent_at = time.monotonic()
        self._status = "Grasp pending" if mode == "close" else "Open pending"
        self._payload_id = "" if mode == "release" else payload_id

    def pop_request(self, *, armed, fault) -> dict | None:
        """The grasp_request owed to the plant, re-gated at dispatch."""
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        reason = self.refusal(armed=armed, fault=fault, consuming=True)
        if time.monotonic() - self._sent_at > HAND_STALE_S:
            reason = "Hand request expired before dispatch"
        if reason:
            self._waiting = False
            self._status = reason
            self._log(f"REFUSED: {reason}")
            return None
        return pending

    def pop_payload(self) -> dict | None:
        """The payload declaration owed to the controller, at most once each."""
        pending, self._payload_pending = self._payload_pending, None
        return pending

    def set_result(self, result: dict) -> None:
        if not self._request_id or result.get("request_id") != self._request_id:
            return
        self._waiting = False
        reason = str(result.get("reason", ""))
        if result.get("ok"):
            self._status = (
                "Open accepted; waiting for finger feedback"
                if reason == "released" else "Held"
            )
        else:
            self._status = (
                "Lost" if reason == "object lost"
                else "Missed" if reason == "no object"
                else f"Unavailable: {reason}"
            )
        self._log(f"Hand: {self._status}")
        # A held module's weight is feedforward the controller cannot guess.
        # Declare on a confirmed hold; retract on anything else -- an open, a
        # miss, a drop -- because feeding forward a mass that is NOT in the hand
        # pushes the arm up just as hard as an undeclared one sags.
        held = bool(result.get("ok")) and reason != "released"
        entry = self._payload_table.get(self._payload_id) if held else None
        if not held:
            self._payload_id = ""
        self._payload_pending = {
            "mass_kg": float(entry.get("mass_kg", 0.0)) if entry else 0.0,
            "com_ee": list(entry.get("com_offset_ee") or ()) or None if entry else None,
        }
        if entry:
            self._log(f"Payload: {self._payload_id} {self._payload_pending['mass_kg']:.4f} kg")


def _self_check() -> None:
    """Hold declares the payload; open, miss and drop all retract it.

    Feeding forward a mass that is NOT in the hand pushes the arm up as hard as
    an undeclared one sags, so every non-hold outcome must retract -- including
    the ones that are not failures (a released module is gone too).
    """
    def panel():
        p = HandPanel(
            gripper_cfg={"force_n": 40.0, "open_width_m": .075, "grasp_width_m": .045},
            payload_table={"row_module": {"mass_kg": .5, "com_offset_ee": [0., 0., .06]}},
            width_max_m=.04,
        )
        # A live, idle, measured hand: what the plant reports between actions.
        p.set_state({"available": True, "measured": True, "force_grasp": True,
                     "busy": False, "width": .075, "sample_seq": 1})
        return p

    def result(p, ok, reason="", rid="r1"):
        p._request_id = rid
        p.set_result({"request_id": rid, "ok": ok, "reason": reason})
        return p.pop_payload()

    p = panel()
    p._payload_id = "row_module"
    assert result(p, True) == {"mass_kg": .5, "com_ee": [0., 0., .06]}
    assert p.pop_payload() is None, "a declaration must be sent once, not resent"

    for ok, reason in ((True, "released"), (False, "no object"), (False, "object lost")):
        p = panel()
        p._payload_id = "row_module"
        assert result(p, ok, reason) == {"mass_kg": 0.0, "com_ee": None}, (ok, reason)
        assert not p._payload_id, "a gone module must not stay selected"

    p = panel()          # nothing named: declare zero, never the last one
    assert result(p, True) == {"mass_kg": 0.0, "com_ee": None}

    p = panel()
    for bad in ({"mode": "close", "payload_id": "nope"},
                {"mode": "release", "payload_id": "row_module"},
                {"mode": "close", "spin": 1}):
        try:
            p.request(bad, armed=True, fault="")
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

    # The arm gate is the caller's to supply, and it is not advisory.
    p = panel()
    for armed, fault in ((False, ""), (None, ""), (True, "joint_limit")):
        assert p.refusal(armed=armed, fault=fault), (armed, fault)
    assert not p.refusal(armed=True, fault="")

    # A stale observation refuses even a perfectly armed hand.
    p = panel()
    p._sample_at = time.monotonic() - HAND_STALE_S - .1
    assert "stale" in p.refusal(armed=True, fault="")

    # DISARM takes the jaws away from the slider and the queue with it.
    p = panel()
    p.request({"mode": "close"}, armed=True, fault="")
    assert p.owns_jaws()
    p.disarm_requested()
    assert p.owns_jaws() and p.pop_request(armed=True, fault="") is None
    p.rearmed()
    assert not p._disarm_requested

    # Admitted at request time, re-gated at dispatch: a request that went stale
    # on the queue must not reach the plant.
    p = panel()
    p.request({"mode": "close"}, armed=True, fault="")
    p._sent_at = time.monotonic() - HAND_STALE_S - .1
    assert p.pop_request(armed=True, fault="") is None, "expired request dispatched"
    print("hand_panel: payload on hold only, arm gate and dispatch re-gate hold")


if __name__ == "__main__":
    _self_check()
