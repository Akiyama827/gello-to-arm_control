"""CTRL-034 B1: envelope age instrumentation for the native v7 stream.

Three distinct ages are reported:

- **decision age** — what the PC stamps into `age_us`: `now - the decision's own
  RT state sample time`. It grows while the fixed-rate sender repeats the same
  decision, and the sender stops sending once it passes the ceiling.
- **transport age** — the extra age the RT adds (network + its own loop), read
  as the RT-reported `envelope_age_us` (= ingress age + the PC's `age_us`) minus
  the decision age of the envelope in flight. The pairing is by nearest send, so
  it is accurate to about one send period.
- **state latency** — PC receipt-to-consumption age of the most recent state
  in this bench. It is not one-way network latency: PC and RT monotonic clocks
  have no common epoch.

The RT-reported total is sampled while a reference is selected; wrapping
`apply_reference_envelope` records the decision age of every envelope handed to
the transport, which is also the "transport age = 0" reference point.

Sampling a shared `RtBackend` is read-only: it calls `latest_state()`, which
only takes the state lock.
"""
from __future__ import annotations

import threading
import time

CEILING_US = 100_000


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class EnvelopeAgeMonitor:
    """Context manager: sample the RT-reported age, decision age and transport age."""

    def __init__(self, rt, *, period=0.001, include_all_states=False):
        self.rt, self.period = rt, period
        self.include_all_states = include_all_states
        self.rt_ages: list[int] = []
        self.decision_ages: list[int] = []
        self.transport_ages: list[int] = []
        self.state_ages: list[int] = []
        self._last_decision = None
        self._last_send_t = None
        self._stop = threading.Event()
        self._orig = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='envelope-age')
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            now = time.monotonic()
            state, received_at = self.rt.latest_state()
            ref = None if state is None else state.reference
            if (state is not None and received_at > 0
                    and (self.include_all_states or (ref and ref.get('reference_id')))):
                self.state_ages.append(max(0, int((now - received_at) * 1e6)))
            if ref is not None:
                age = ref.get('envelope_age_us')
                if age is not None and age != 0xFFFFFFFF:
                    self.rt_ages.append(int(age))
                    # Only pair with a decision while one is actually in flight;
                    # a deliberate send pause (pc_stall) is not transport age.
                    if (self._last_decision is not None
                            and self._last_send_t is not None
                            and now - self._last_send_t < 0.02):
                        self.transport_ages.append(max(0, int(age) - self._last_decision))
            self._stop.wait(self.period)

    def __enter__(self):
        self._orig = self.rt.apply_reference_envelope

        def wrapped(command, **kw):
            age = int((time.monotonic() - kw['sample_received_at']) * 1e6)
            self.decision_ages.append(age)
            self._last_decision = age
            self._last_send_t = time.monotonic()
            return self._orig(command, **kw)

        self.rt.apply_reference_envelope = wrapped
        return self

    def __exit__(self, *_exc):
        if self._orig is not None:
            self.rt.apply_reference_envelope = self._orig
            self._orig = None
        self.stop()

    def stop(self):
        self._stop.set()
        self._thread.join(2)

    def reset(self):
        """Start a new window (e.g. one execution)."""
        self.rt_ages.clear()
        self.decision_ages.clear()
        self.transport_ages.clear()
        self.state_ages.clear()

    def window(self, label, ceiling_us=CEILING_US):
        """Report the window; return how many RT samples exceeded the ceiling."""
        # The decision age is the PC's honest stamp; transport is the RT's own
        # added age. The RT total is what trips the ceiling, so it is the number
        # that must stay under it.
        over = sum(1 for age in self.rt_ages if age > ceiling_us)
        if not self.rt_ages and not self.decision_ages:
            print(f'  age {label}: v7_envelope=n/a | '
                  f'state_rx_to_read n={len(self.state_ages)} '
                  f'p50={percentile(self.state_ages, .5):.0f}us '
                  f'p99.9={percentile(self.state_ages, .999):.0f}us '
                  f'max={max(self.state_ages, default=0)}us')
            return 0
        print(f'  age {label}: decision n={len(self.decision_ages)} '
              f'p99.9={percentile(self.decision_ages, 0.999):.0f}us '
              f'max={max(self.decision_ages, default=0)}us | '
              f'transport p99.9={percentile(self.transport_ages, 0.999):.0f}us '
              f'max={max(self.transport_ages, default=0)}us | '
              f'rt_total p99.9={percentile(self.rt_ages, 0.999):.0f}us '
              f'max={max(self.rt_ages, default=0)}us over_{ceiling_us // 1000}ms={over} | '
              f'state_rx_to_read p50={percentile(self.state_ages, .5):.0f}us '
              f'p99.9={percentile(self.state_ages, .999):.0f}us '
              f'max={max(self.state_ages, default=0)}us')
        return over

    def report(self, label):
        return self.window(label)
