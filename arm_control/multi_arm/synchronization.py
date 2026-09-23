"""Observable coordination timing for a released pair.

V1 synchronization is a BARRIER START: stage both members, wait for BOTH
staging acknowledgements, then release both. This module owns the timing
record for that release. It measures what the coordinator can actually
observe -- when it accepted the ACKs and when it sent each release -- and
claims nothing more:

- ``stage_skew_sec`` is the arrival spread of the two staging ACKs. It says
  the arms did not finish loading together; it does NOT say their motion
  started apart by that much.
- ``release_skew_sec`` is the send spread of the two execute releases from
  the coordinator's single clock. A barrier start is coordinated, not
  atomic, and no ordinary Python test may claim real-time simultaneity --
  the record is evidence of ordering, not of a timing guarantee.

In-process hardware pairs additionally use SharedReferenceClock for a common
trajectory epoch and per-tick sample time. This removes PC reference phase skew;
the actual release/send timestamps above remain unfrozen and observable.

SCHEDULED START (future, designed for but not implemented here): the record
carries ``requested_start_ns``. A future controller contract accepts a
common monotonic deadline (``execute_at_mono_ns``), each controller waits
locally and begins at T, and per-arm actual start times then join this
record so requested-vs-actual start skew becomes measurable. Nothing in the
V1 coordinator, handle, or wire contract assumes this slot stays ``None``
or that it is filled, which is what keeps the extension compatible.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time


class SharedReferenceClock:
    """One reference time for an in-process release or command batch.

    Batching removes software phase skew between local controllers. It does not
    synchronize transport delivery or robot cycles. Safety/freshness checks must
    keep using the unfrozen monotonic clock. Nested batches retain the outer
    sample; other threads keep their own context.
    """

    def __init__(self, clock=None):
        self._clock = clock or time.monotonic
        self._sample = ContextVar("pair_reference_time", default=None)

    def __call__(self):
        sampled = self._sample.get()
        return self._clock() if sampled is None else sampled

    @contextmanager
    def batch(self):
        token = self._sample.set(self())
        try:
            yield self()
        finally:
            self._sample.reset(token)


def _spread(times: Mapping[str, float]) -> float | None:
    """Max-minus-min over at least two readings; None when under-sampled."""
    if len(times) < 2:
        return None
    values = [float(v) for v in times.values()]
    return max(values) - min(values)


def stage_skew_sec(stage_ack_times: Mapping[str, float]) -> float | None:
    """Arrival spread of the members' staging ACKs (one clock)."""
    return _spread(stage_ack_times)


def release_skew_sec(release_times: Mapping[str, float]) -> float | None:
    """Send spread of the coordinated release (one clock)."""
    return _spread(release_times)


@dataclass(frozen=True)
class StartRecord:
    """One pair's observed coordination timeline."""

    pair_id: str
    stage_ack_times: dict[str, float] = field(default_factory=dict)
    ready_at: float | None = None
    release_times: dict[str, float] = field(default_factory=dict)
    requested_start_ns: int | None = None

    @property
    def stage_skew_sec(self) -> float | None:
        return stage_skew_sec(self.stage_ack_times)

    @property
    def release_skew_sec(self) -> float | None:
        return release_skew_sec(self.release_times)

    def to_payload(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "stage_ack_times": dict(self.stage_ack_times),
            "ready_at": self.ready_at,
            "release_times": dict(self.release_times),
            "stage_skew_sec": self.stage_skew_sec,
            "release_skew_sec": self.release_skew_sec,
            "requested_start_ns": self.requested_start_ns,
        }
