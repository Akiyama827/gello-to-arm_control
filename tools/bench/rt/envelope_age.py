"""CTRL-034 B1: envelope-age instrumentation for the native v7 stream.

The RT reports `envelope_age_us` = ingress age + the PC's declared age_us for
the envelope that currently has authority. A dedicated sampler reads it while a
reference is selected, and wrapping `apply_reference_envelope` records the
PC-side age of every envelope the sender hands to the transport. p99.9 and the
maximum are what the CTRL-034 review asks a run to report; the acceptance run
checks that the stream never spends the 50 ms freshness ceiling in a normal
execution.

Sampling a shared `RtBackend` is read-only: it calls `latest_state()`, which
only takes the state lock.
"""
from __future__ import annotations

import threading
import time

CEILING_US = 50_000


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class EnvelopeAgeMonitor:
    """Context manager: sample the RT-reported age and the PC-side send age."""

    def __init__(self, rt, *, period=0.001):
        self.rt, self.period = rt, period
        self.rt_ages: list[int] = []
        self.pc_ages: list[int] = []
        self._stop = threading.Event()
        self._orig = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='envelope-age')
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            state = self.rt.latest_state()[0]
            ref = None if state is None else state.reference
            if ref is not None:
                age = ref.get('envelope_age_us')
                if age is not None and age != 0xFFFFFFFF:
                    self.rt_ages.append(int(age))
            self._stop.wait(self.period)

    def __enter__(self):
        self._orig = self.rt.apply_reference_envelope

        def wrapped(command, **kw):
            self.pc_ages.append(int((time.monotonic() - kw['sample_received_at']) * 1e6))
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
        self.pc_ages.clear()

    def window(self, label, ceiling_us=CEILING_US):
        """Report the current window and return how many frames exceeded the ceiling."""
        over = sum(1 for age in self.rt_ages if age > ceiling_us)
        print(f'  age {label}: rt n={len(self.rt_ages)} '
              f'p99.9={percentile(self.rt_ages, 0.999):.0f}us '
              f'max={max(self.rt_ages, default=0)}us over_{ceiling_us // 1000}ms={over}'
              + (f' | pc p99.9={percentile(self.pc_ages, 0.999):.0f}us '
                 f'max={max(self.pc_ages, default=0)}us' if self.pc_ages else ''))
        return over

    def report(self, label):
        return self.window(label)
