"""Grasp request/result state machine for self-sensing hands (Franka Hand).

The DM path senses grasps bridge-side (``GraspGate`` thresholds the gripper
motor torque). A Franka Hand states its own verdict: ``grasp()`` returns the
width-band-under-force check, and ``is_grasped`` tracks it afterwards. This
FSM turns that verdict stream into the orchestrator's grasp contract:

- ``close``  -> caller sends the hand a grasp action; the eventual verdict
  (``gdone``) becomes the GRASPED/MISSED ``grasp_result``.
- ``release`` -> caller opens the jaws; acked immediately (unsensed, same as
  the DM release path).
- after a held grasp, ``is_grasped`` falling is the drop event: a second,
  failed result under the close request_id — the LOST semantics the
  orchestrator already freezes on.

Pure over (request, gdone counter, state sample, clock), so the real node
(``nodes/franka_gripper.py`` against ``hand_bridge`` on the RT box) and the
sim bridge's hand emulation share it — one policy, two plants.
"""
from __future__ import annotations

# Full 80 mm stroke at the configured 0.05 m/s is 1.6 s plus the force phase;
# 8 s means a missing verdict (GSTOP kill, bridge reconnect mid-grasp)
# resolves from the freshest is_grasped sample instead of hanging the
# orchestrator.
GRASP_TIMEOUT_S = 8.0


def _result(rid: str, mid: str, ok: bool, reason: str) -> dict:
    return {"request_id": rid, "module_id": mid, "ok": ok, "reason": reason}


class HandGraspFsm:
    """grasp_request -> hand action + verdict -> grasp_result."""

    def __init__(self, gripper_cfg: dict) -> None:
        g = gripper_cfg
        self.grasp_width_m = float(g.get("grasp_width_m", 0.045))
        self.speed_mps = float(g.get("speed_mps", 0.05))
        self.force_n = float(g.get("force_n", 40.0))
        self.epsilon_inner_m = float(g.get("epsilon_inner_m", 0.02))
        self.epsilon_outer_m = float(g.get("epsilon_outer_m", 0.02))
        self.open_width_m = float(g.get("open_width_m", 0.075))
        self._pending: dict | None = None  # close in flight, awaiting verdict
        self._held: dict | None = None     # confirmed grasp being watched
        self._held_seen_grasped = False    # is_grasped observed True post-grasp

    def on_request(self, payload: dict, gdone_count: int, now: float):
        """-> (action: 'grasp' | 'open', immediate grasp_result | None)."""
        rid = str(payload.get("request_id", ""))
        mid = str(payload.get("module_id", ""))
        if str(payload.get("mode", "close")) == "release":
            # Unsensed, like the DM release: ack now, jaws travel after.
            self._pending = None
            self._held = None
            return "open", _result(rid, mid, True, "released")
        self._pending = {
            "request_id": rid,
            "module_id": mid,
            "deadline": now + GRASP_TIMEOUT_S,
            "gdone_base": gdone_count,
        }
        self._held = None
        return "grasp", None

    def poll(self, state: dict | None, gdone_count: int, gdone_ok: bool,
             now: float) -> dict | None:
        """Tick -> a grasp_result to publish, or None."""
        p = self._pending
        if p is not None:
            if gdone_count > p["gdone_base"]:
                self._pending = None
                if gdone_ok:
                    self._held = p
                    self._held_seen_grasped = False
                    return _result(p["request_id"], p["module_id"], True, "grasped")
                return _result(p["request_id"], p["module_id"], False, "no object")
            if now >= p["deadline"]:
                # Verdict lost (GSTOP kill / bridge reconnect): the freshest
                # is_grasped sample is the best remaining truth.
                self._pending = None
                ok = bool(state and state.get("is_grasped"))
                if ok:
                    self._held = p
                    self._held_seen_grasped = False
                return _result(
                    p["request_id"], p["module_id"], ok,
                    "grasped (is_grasped fallback)" if ok
                    else "no result from hand (timeout)",
                )
            return None
        h = self._held
        if h is not None and state is not None:
            if state.get("is_grasped"):
                self._held_seen_grasped = True
            elif self._held_seen_grasped:
                # Drop event: a second, failed result under the close
                # request_id — the orchestrator freezes on it (LOST semantics).
                self._held = None
                return _result(h["request_id"], h["module_id"], False, "object lost")
        return None


__all__ = ["HandGraspFsm", "GRASP_TIMEOUT_S"]
