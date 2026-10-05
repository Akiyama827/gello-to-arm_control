"""The Hand side of a dual runtime, shared by every dual backend (direct RT links and supervisors alike).

The Hands are the project's own gripper bridges (``BridgeClient``), never part of an arm's RT link or supervisor. A runtime
that mixes this in keeps ``hands``, ``fingers``, ``finger_limits``, ``_measured_fingers`` and ``_hand_results`` (dicts keyed
by arm id).
"""

import time

import numpy as np


class HandControl:
    def finger_positions(self):
        for a, hand in self.hands.items():
            state = hand.snapshot()
            if state.get("measured") is True:
                self._measured_fingers[a] = float(state["width"]) / 2
        # A stale display retains its last measurement; fresh=False disables
        # motion. Reconnection must never display a fabricated open Hand.
        return dict(self._measured_fingers)

    def set_fingers(self, values):
        for a, width in values.items():
            if not np.isfinite(width) or not self.finger_limits[a][0] <= width <= self.finger_limits[a][1]:
                raise ValueError("Hand target outside configured limits")
            hand = self.hands[a]
            if not hand.send_line(
                f"MOVE {width * 2:.9g} 0.05",
                {"request_id": f"console-{time.monotonic_ns()}", "_position_move": True},
            ):
                raise ValueError(f"{a}: Hand unavailable, busy or missing active-stop support")
            self.fingers[a] = width
            self._hand_results.pop(a, None)

    @property
    def hand_busy(self):
        return any(h.snapshot().get("busy") for h in self.hands.values())

    def cancel_hands(self):
        for hand in self.hands.values():
            hand.cancel_move()
