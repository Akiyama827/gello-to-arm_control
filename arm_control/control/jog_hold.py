"""An operator jog: a hold whose anchor moves, and which dies of old age.

Split out of ``arm_controller`` on 2026-09-10. The console re-sends a setpoint
while a button is held, so a closed tab, a wedged page, a dropped network and a
crashed browser all look identical from here -- setpoints stop arriving. NOT
SENDING IS THE STOP: nothing has to notice and send one.

This owns the setpoint and its freshness only. Re-anchoring the arm when a jog
expires is the controller's job, because the right anchor is the MEASURED pose
and only the controller can build one. The expiry is handed over as a one-shot
latch so that re-anchoring happens exactly once per expiry, not every tick.
"""
from __future__ import annotations

import math

import numpy as np


class JogHold:
    """The current jog setpoint, valid only while it keeps being renewed."""

    def __init__(self, *, timeout_s: float, n_joints: int) -> None:
        self.timeout_s = float(timeout_s)
        # FINITE, not merely positive: inf is a syntactically valid timeout that
        # silently disables the deadman this class exists to be.
        if not (math.isfinite(self.timeout_s) and self.timeout_s > 0.0):
            raise ValueError("jog_timeout_s must be finite and positive")
        self.n_joints = int(n_joints)
        self._q: np.ndarray | None = None
        self._at = 0.0
        self._expired = False

    def set(self, q, now: float) -> None:
        """Renew the setpoint. A wrong-width q is a bug, not a stale jog."""
        q = np.asarray(q, dtype=float).ravel()
        if q.shape != (self.n_joints,):
            raise ValueError(f"jog q must have {self.n_joints} values, got {q.size}")
        self._q = q
        self._at = float(now)

    def clear(self) -> None:
        """Drop the jog without raising the expiry latch (disarm, stop, fault)."""
        self._q = None
        self._expired = False

    @property
    def active(self) -> bool:
        """A jog is held, fresh or not. A PURE read: callers that only need to
        know "is a jog in the way" must not trip the expiry latch, which is
        owed to whoever re-anchors the hold.
        """
        return self._q is not None

    def target(self, now: float) -> np.ndarray | None:
        """The held setpoint, or None once it has gone stale.

        Crossing the timeout raises the expiry latch exactly once.
        """
        if self._q is None:
            return None
        if float(now) - self._at > self.timeout_s:
            self._q = None
            self._expired = True
            return None
        return self._q

    def take_expired(self) -> bool:
        """True once per expiry -- the controller's cue to re-anchor the hold."""
        expired, self._expired = self._expired, False
        return expired


def _self_check() -> None:
    """Renewal keeps a jog alive; silence kills it, exactly once."""
    j = JogHold(timeout_s=0.2, n_joints=7)
    assert j.target(0.0) is None and not j.take_expired(), "no jog is not an expiry"

    j.set(np.ones(7), 1.0)
    assert np.array_equal(j.target(1.1), np.ones(7))
    assert not j.take_expired(), "a live jog must not look expired"

    # Renewal is what keeps it alive -- the held button IS the deadman.
    for t in (1.15, 1.3, 1.45):
        j.set(np.full(7, t), t)
        assert j.target(t + .1) is not None, t

    # Silence past the timeout expires it, and the latch fires ONCE: re-anchoring
    # every tick would ratchet the hold forward on the arm's own gravity sag.
    assert j.target(1.8) is None
    assert j.take_expired() and not j.take_expired()
    assert j.target(1.9) is None and not j.take_expired()

    # clear() is the disarm path: no jog, and no re-anchor owed.
    # `active` must not consume the expiry the hold re-anchor is owed.
    j.set(np.ones(7), 2.0)
    assert j.active
    assert j.active and not j.take_expired(), "a pure read raised the latch"
    assert j.target(2.5) is None and j.active is False
    assert j.take_expired(), "target() past the timeout owes an expiry"

    j.set(np.ones(7), 2.0)
    j.clear()
    assert j.target(2.1) is None and not j.take_expired()

    for bad in (np.ones(6), np.ones((2, 7)), np.ones(0)):
        try:
            j.set(bad, 3.0)
        except ValueError:
            continue
        raise AssertionError(f"accepted q of shape {np.shape(bad)}")

    for bad in (0.0, -1.0, float("inf"), float("nan")):
        try:
            JogHold(timeout_s=bad, n_joints=7)
        except ValueError:
            continue
        raise AssertionError(f"accepted timeout {bad}")
    print("jog_hold: renewal keeps it alive, silence expires it once")


if __name__ == "__main__":
    _self_check()
