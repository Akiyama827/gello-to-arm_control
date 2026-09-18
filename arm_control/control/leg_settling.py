"""Bounded settling after a trajectory's reference ends.

Split out of ``arm_controller`` on 2026-09-10. A leg is not finished when its
reference runs out -- the arm is still moving. It is finished when it has
genuinely STOPPED and stayed stopped for a dwell, and it has FAILED when that
never happens inside a timeout. Those are two clocks and one freshness rule,
and they are all this file is.

It DECIDES and does not ACT. The controller freezes the arm, anchors the hold
and reports the leg; this only says which of the three states the leg is in.
Keeping the side effects out is what makes the timing testable without a
plant, an executor or a Dora node.

FRESHNESS is the subtle part: the dwell may only advance on a NEW measurement.
Polling faster than the plant reports would otherwise satisfy a 200 ms dwell
from a single stale sample, and the leg would complete on one lucky reading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SETTLING = "settling"
SETTLED = "settled"
TIMED_OUT = "timed_out"

#: How often a still-settling leg reports progress upward. Faster than this is
#: noise on the operator's page; slower and a long settle looks like a hang.
REPORT_PERIOD_S = 0.25


@dataclass(frozen=True)
class SettleVerdict:
    """Where the leg stands, and how long it has been standing there."""

    state: str
    elapsed_s: float
    remaining_s: float

    @property
    def finished(self) -> bool:
        return self.state != SETTLING


class LegSettling:
    """One leg's settle clocks: dwell to accept, timeout to give up."""

    def __init__(self, *, timeout_s: float, dwell_s: float) -> None:
        self.timeout_s = float(timeout_s)
        self.dwell_s = float(dwell_s)
        if not (math.isfinite(self.timeout_s) and math.isfinite(self.dwell_s)
                and 0 <= self.dwell_s < self.timeout_s):
            raise ValueError("settling requires 0 <= dwell < finite positive timeout")
        self._leg_end = 0.0
        self._settled_since: float | None = None
        self._report_at: float | None = None

    def begin(self, leg_end: float) -> None:
        """Arm the clocks for a leg whose reference ends at ``leg_end``."""
        self._leg_end = float(leg_end)
        self._settled_since = self._report_at = None

    @property
    def started(self) -> bool:
        """True once a settle has been reported on -- the first-message latch."""
        return self._report_at is not None

    def poll(self, *, now: float, fresh: bool, done) -> SettleVerdict | None:
        """None before the reference ends; otherwise the leg's state.

        ``done`` is a CALLABLE and is consulted only on a fresh sample, because
        asking an executor whether it is finished is not always free and,
        more importantly, its answer about a stale sample means nothing.
        """
        if now < self._leg_end:
            return None
        if fresh:
            if done():
                if self._settled_since is None:
                    self._settled_since = now
            else:
                self._settled_since = None
        elapsed = max(0.0, now - self._leg_end)
        remaining = max(0.0, self.timeout_s - elapsed)
        if (fresh and self._settled_since is not None
                and now - self._settled_since >= self.dwell_s
                and elapsed <= self.timeout_s):
            return SettleVerdict(SETTLED, elapsed, remaining)
        if elapsed >= self.timeout_s:
            return SettleVerdict(TIMED_OUT, elapsed, remaining)
        return SettleVerdict(SETTLING, elapsed, remaining)

    def due_to_report(self, now: float) -> bool:
        """Throttle for progress messages; True at most every REPORT_PERIOD_S."""
        if self._report_at is not None and now - self._report_at < REPORT_PERIOD_S:
            return False
        self._report_at = now
        return True


def _self_check() -> None:
    """The dwell needs fresh samples; the timeout does not."""
    def settling(**kw):
        s = LegSettling(**{"timeout_s": 5.0, "dwell_s": 0.2, **kw})
        s.begin(10.0)
        return s

    s = settling()
    assert s.poll(now=9.9, fresh=True, done=lambda: True) is None, "not past the reference"

    # Dwell: stopped is not finished until it has been stopped long enough.
    s = settling()
    assert s.poll(now=10.0, fresh=True, done=lambda: True).state == SETTLING
    assert s.poll(now=10.1, fresh=True, done=lambda: True).state == SETTLING
    # 10.25, not 10.2: `10.2 - 10.0` is 0.19999999999999929 in binary floating
    # point, so an exact-dwell sample is a coin toss. The controller has always
    # behaved this way; the check must not pretend otherwise.
    assert s.poll(now=10.25, fresh=True, done=lambda: True).state == SETTLED

    # THE freshness rule: polling faster than the plant reports must not
    # satisfy the dwell off one stale sample.
    s = settling()
    s.poll(now=10.0, fresh=True, done=lambda: True)
    for t in (10.05, 10.1, 10.2, 10.5, 11.0):
        assert s.poll(now=t, fresh=False, done=lambda: True).state == SETTLING, t
    assert s.poll(now=11.0, fresh=True, done=lambda: True).state == SETTLED

    # Moving again restarts the dwell from zero.
    s = settling()
    s.poll(now=10.0, fresh=True, done=lambda: True)
    assert s.poll(now=10.15, fresh=True, done=lambda: False).state == SETTLING
    assert s.poll(now=10.3, fresh=True, done=lambda: True).state == SETTLING
    assert s.poll(now=10.55, fresh=True, done=lambda: True).state == SETTLED

    # Timeout wins over a settle that arrives too late, and does NOT need a
    # fresh sample -- an arm that stopped reporting must still fail the leg.
    s = settling()
    assert s.poll(now=15.1, fresh=False, done=lambda: True).state == TIMED_OUT
    s = settling()
    s.poll(now=14.95, fresh=True, done=lambda: True)
    assert s.poll(now=15.2, fresh=True, done=lambda: True).state == TIMED_OUT

    # done() is consulted only when the sample is fresh.
    calls = []
    s = settling()
    s.poll(now=10.1, fresh=False, done=lambda: calls.append(1) or True)
    assert not calls, "done() asked about a stale sample"

    # begin() re-arms: a second leg does not inherit the first one's dwell.
    s = settling()
    s.poll(now=10.0, fresh=True, done=lambda: True)
    s.begin(20.0)
    assert s.poll(now=20.0, fresh=True, done=lambda: True).state == SETTLING
    assert not s.started

    for bad in ({"dwell_s": 5.0}, {"dwell_s": -1.0}, {"timeout_s": float("inf")},
                {"timeout_s": float("nan")}, {"dwell_s": float("nan")}):
        try:
            LegSettling(**{"timeout_s": 5.0, "dwell_s": 0.2, **bad})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

    s = settling()
    assert s.due_to_report(10.0) and not s.due_to_report(10.1)
    assert s.due_to_report(10.3) and s.started
    print("leg_settling: dwell needs fresh samples, timeout does not")


if __name__ == "__main__":
    _self_check()
