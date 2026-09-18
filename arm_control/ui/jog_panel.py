"""Held-button jog state for an operator console.

Split out of ``arm_console`` on 2026-09-10. A jog is a hold whose anchor moves
and which dies of old age, and that is the whole of this file: which direction
is held, at what latched speed, and whether the assertion is still fresh.
Nothing here knows about sliders, plans, hands or HTTP.

THREADING: the caller holds the console lock; this object owns none. See
``hand_panel`` for why.

The MODE GATE (``control_mode``) and the joint count are passed in for the same
reason the arm gate is passed to ``HandPanel``: a jog pad does not own them.
"""
from __future__ import annotations

import time

JOG_AXES = ("x", "y", "z")

#: How long a held-jog assertion stays good at the console. The page re-asserts
#: every 100 ms, so this tolerates a few missed polls and no more -- it is the
#: browser->console half of the jog's two independent deadmen. The controller's
#: own expiry is the other; this one cannot save an arm from a wedged console.
JOG_STALE_S = 0.4


class JogPanel:
    """One held direction, its latched speed, and its freshness."""

    def __init__(self, *, speed_m_s: float, joint_speed_rad_s: float, n_joints: int, log=None):
        self._max = (float(speed_m_s), float(joint_speed_rad_s))
        if not all(v == v and v not in (float("inf"), float("-inf")) for v in self._max) \
                or min(self._max) <= 0:
            raise ValueError("jog speeds must be finite and positive")
        self._speed_m_s, self._joint_speed_rad_s = self._max
        self._n_joints = int(n_joints)
        self._log = log or (lambda _message: None)
        self._held: tuple[str, int] | None = None
        self._at = 0.0
        self._note = ""
        self._latched = self._max
        # Bumped on every fresh press so the consumer can tell a NEW hold from a
        # continued one -- a re-press must re-latch, not resume mid-stroke.
        self._epoch = 0

    @property
    def max_speeds(self) -> tuple[float, float]:
        return self._max

    def set_speeds(self, payload) -> None:
        if not isinstance(payload, dict) or set(payload) != {"speed_m_s", "joint_speed_rad_s"}:
            raise ValueError("provide Cartesian and joint jog speeds only")
        try:
            values = [float(payload["speed_m_s"]), float(payload["joint_speed_rad_s"])]
        except TypeError as error:  # None from a blank field is a refusal, not a crash
            raise ValueError("jog speeds must be numbers") from error
        if not all(v == v and abs(v) != float("inf") for v in values):
            raise ValueError("jog speeds must be finite")
        if any(v <= 0 or v > hi for v, hi in zip(values, self._max)):
            raise ValueError("jog speeds must be positive and no higher than configured limits")
        self._speed_m_s, self._joint_speed_rad_s = values

    def set(self, axis, direction, held, *, control_mode: str) -> None:
        """Hold or release one direction. Unknown axes are refused loudly."""
        if not held:
            self._held = None
            return
        if control_mode == "soft":
            self._note = "jog refused — select Track before jogging"
            return
        axis = str(axis or "")
        if axis not in JOG_AXES and not (
            axis.startswith("j") and axis[1:].isdigit() and int(axis[1:]) < self._n_joints
        ):
            raise ValueError(f"unknown jog axis {axis!r}")
        selected = (axis, 1 if float(direction or 0) >= 0 else -1)
        if self._held != selected or time.monotonic() - self._at > JOG_STALE_S:
            self._latched = (self._speed_m_s, self._joint_speed_rad_s)
            self._epoch += 1
        self._held = selected
        self._at = time.monotonic()

    def release(self) -> None:
        self._held = None

    def command(self):
        """One atomic direction/speed/stroke snapshot, or None if stale."""
        if self._held is None:
            return None
        if time.monotonic() - self._at > JOG_STALE_S:
            self._held = None
            return None
        return (*self._held, *self._latched, self._epoch)

    def held(self) -> tuple[str, int] | None:
        command = self.command()
        return None if command is None else command[:2]

    def set_note(self, note: str) -> None:
        if note != self._note:
            self._note = note
            if note:
                self._log(note)

    def snapshot(self) -> dict:
        return {
            "axes": list(JOG_AXES),
            # The page builds the joint pad from this count; without it the pad hides.
            "joints": self._n_joints,
            "held": None if self._held is None else list(self._held),
            "note": self._note,
            "speed_m_s": self._speed_m_s,
            "joint_speed_rad_s": self._joint_speed_rad_s,
            "max_speed_m_s": self._max[0],
            "max_joint_speed_rad_s": self._max[1],
        }


def _self_check() -> None:
    """Freshness is the deadman; a re-press re-latches; soft mode refuses."""
    def pad():
        return JogPanel(speed_m_s=.01, joint_speed_rad_s=.15, n_joints=7)

    p = pad()
    assert p.command() is None, "nothing held yet"
    p.set("x", 1, True, control_mode="joint")
    axis, sign, lin, ang, epoch = p.command()
    assert (axis, sign, lin, ang) == ("x", 1, .01, .15)

    # Stale = released. The held button IS the deadman: a console that stops
    # re-asserting must stop the arm without anyone sending a stop.
    p._at = time.monotonic() - JOG_STALE_S - .1
    assert p.command() is None and p.held() is None

    # A re-press after staleness is a NEW stroke, not a resumed one.
    p = pad()
    p.set("x", 1, True, control_mode="joint")
    first = p.command()[-1]
    p.set("x", 1, True, control_mode="joint")
    assert p.command()[-1] == first, "a continued hold must not re-latch"
    p._at = time.monotonic() - JOG_STALE_S - .1
    p.set("x", 1, True, control_mode="joint")
    assert p.command()[-1] == first + 1, "a fresh press must re-latch"

    # Soft mode refuses rather than moving, and says why.
    p = pad()
    p.set("x", 1, True, control_mode="soft")
    assert p.command() is None and "Track" in p.snapshot()["note"]

    p = pad()
    for bad in ("w", "j9", "elbow", ""):
        try:
            p.set(bad, 1, True, control_mode="joint")
        except ValueError:
            continue
        raise AssertionError(f"accepted axis {bad!r}")
    p.set("j6", -1, True, control_mode="joint")
    assert p.held() == ("j6", -1)
    assert p.snapshot()["joints"] == 7, "the page hides the joint pad without it"

    # Speeds are bounded by the configured maxima, in both directions.
    p = pad()
    for bad in ({"speed_m_s": .02, "joint_speed_rad_s": .1},
                {"speed_m_s": 0, "joint_speed_rad_s": .1},
                {"speed_m_s": -1, "joint_speed_rad_s": .1},
                {"speed_m_s": float("nan"), "joint_speed_rad_s": .1},
                {"speed_m_s": .01}):
        try:
            p.set_speeds(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted speeds {bad}")
    p.set_speeds({"speed_m_s": .005, "joint_speed_rad_s": .1})
    p.set("y", 1, True, control_mode="joint")
    assert p.command()[2:4] == (.005, .1)
    print("jog_panel: freshness deadman, re-press re-latch, soft refusal, bounds")


if __name__ == "__main__":
    _self_check()
