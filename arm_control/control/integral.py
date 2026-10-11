"""One explicit owner of the clamped joint-position integral torque.

The caller schedules updates after semantic target selection, before tracking
admission. Command previews and fallback construction only read ``bias``.
"""
from __future__ import annotations

import math
import numpy as np

from arm_control.control.gains import validate_integral_vector


class IntegralState:
    I_WINDOW_RAD = 0.02

    def __init__(self, n: int, *, ki=None, i_limit=None, kp_ref=None):
        self.ki = validate_integral_vector(ki, n, where="ki")
        self.i_limit = validate_integral_vector(i_limit, n, where="i_limit")
        # ki is stated at the profile kp (kp_ref). A leg or hold at a lower kp integrates at ki*kp/kp_ref, so the integral
        # keeps the profile's ki/kp time constant instead of overpowering a soft spring (operator 2026-10-10: at the
        # Cartesian floor kp 20 the profile ki 1200 pinned the bias at its clamp). Never above ki. None: unscaled.
        if kp_ref is not None:
            kp_ref = np.asarray(kp_ref, dtype=float)
            if kp_ref.shape != (n,) or not np.isfinite(kp_ref).all() or np.any(kp_ref < 0):
                raise ValueError("integral kp_ref must be n finite nonnegative values (the profile kp)")
            kp_ref = kp_ref.copy()
            kp_ref.flags.writeable = False
        self.kp_ref = kp_ref
        self._bias = np.zeros(n)
        self._time: float | None = None
        self._cycle: int | None = None

    @property
    def bias(self) -> np.ndarray:
        return self._bias.copy()

    def reset(self) -> None:
        """DISARM and existing law changes discard bias and elapsed-time history."""
        self._bias.fill(0.)
        self._time = self._cycle = None

    def update(self, cycle_id: int, cycle_time: float, q_des, q_measured, kp) -> np.ndarray:
        """Advance once per caller cycle; later cycles may reuse a fresh sample.

        Identical timestamps give zero dt. The historical backwards-clock zero
        dt and 50 ms long-gap cap are retained; invalid input never advances time.
        """
        if not isinstance(cycle_id, int) or cycle_id < 0:
            raise ValueError("integral cycle id must be a nonnegative integer")
        now = float(cycle_time)
        values = [np.asarray(v, dtype=float) for v in (q_des, q_measured, kp)]
        if not math.isfinite(now) or any(
            v.shape != self._bias.shape or not np.isfinite(v).all() for v in values
        ):
            raise ValueError("integral update needs finite time and matching finite joint vectors")
        if self._cycle is not None:
            if cycle_id < self._cycle:
                raise ValueError("integral cycle id moved backwards")
            if cycle_id == self._cycle:
                return self.bias
        q_des, q, kp = values
        dt = 0.0 if self._time is None else min(max(now - self._time, 0.0), 0.05)
        err = q_des - q
        live = (kp > 0) & (np.abs(err) < self.I_WINDOW_RAD)
        ki = self.ki
        if self.kp_ref is not None:
            ki = ki * np.clip(np.divide(kp, self.kp_ref, out=np.ones_like(kp), where=self.kp_ref > 0), 0.0, 1.0)
        bias = np.where(live, self._bias + ki * err * dt, self._bias)
        bias = np.clip(bias, -self.i_limit, self.i_limit)
        bias[kp <= 0] = 0.0
        self._bias, self._time, self._cycle = bias, now, cycle_id
        return self.bias


def _self_check():
    state = IntegralState(2, ki=[200., 200.], i_limit=[1., 1.])
    q, target, kp = np.zeros(2), np.full(2, .005), np.array([100., 0.])
    assert np.array_equal(state.update(0, 0., target, q, kp), [0., 0.])
    first = state.update(1, .01, target, q, kp)
    assert np.array_equal(first, [.01, 0.])
    assert np.array_equal(state.update(1, .02, target, q, kp), first)  # duplicate cycle
    assert np.array_equal(state.update(2, .01, target, q, kp), first)  # equal time
    for i in range(3, 203):
        result = state.update(i, .02 + .01*(i-3), target, q, kp)
    assert np.array_equal(result, [1., 0.])
    assert np.array_equal(state.update(203, 2.1, [.02, 0.], q, kp), result)
    state.reset()
    state.update(0, 0., target, q, kp)
    assert np.array_equal(state.update(1, 1., target, q, kp), [.05, 0.])  # 50 ms cap
    saved = state.bias
    for bad in (dict(cycle_time=np.nan), dict(q_des=[np.inf, 0.]),
                dict(q_measured=[np.nan, 0.]), dict(kp=[1.]), dict(cycle_id=-1)):
        args = dict(cycle_id=2, cycle_time=1.01, q_des=target, q_measured=q, kp=kp)
        args.update(bad)
        try:
            state.update(**args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted {bad}')
        assert np.array_equal(state.bias, saved) and state._time == 1.
    assert np.allclose(state.update(2, 1.01, target, q, kp), [.06, 0.])
    assert np.array_equal(state.update(3, 1.02, target, q, np.zeros(2)), q)
    default = IntegralState(2)
    default.update(0, 0., target, q, kp)
    assert np.array_equal(default.update(1, 1., target, q, kp), q)
    for name in ('ki', 'i_limit'):
        for bad in (0., [1.], [1., 2., 3.], [[1., 2.]], [-1., 0.],
                    [np.nan, 0.], [np.inf, 0.], [-np.inf, 0.]):
            try:
                IntegralState(2, **{name: bad})
            except ValueError:
                pass
            else:
                raise AssertionError(f'accepted {name}={bad!r}')
        supplied = np.array([2., 3.])
        admitted = IntegralState(2, **{name: supplied})
        supplied[:] = 99.
        stored = getattr(admitted, name)
        assert np.array_equal(stored, [2., 3.]) and not stored.flags.writeable
    scaled = IntegralState(2, ki=[1200., 1200.], i_limit=[3., 3.], kp_ref=[600., 600.])
    scaled.update(0, 0., target, q, np.array([20., 600.]))
    assert np.allclose(scaled.update(1, .01, target, q, np.array([20., 900.])), [40. * .005 * .01, 1200. * .005 * .01])
    for bad in ([600.], [np.nan, 1.], [-1., 1.]):
        try:
            IntegralState(2, ki=[1., 1.], kp_ref=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted kp_ref={bad!r}')
    print('integral: kp-scaled ki (never above ki), formula/window/clamp/reset, cycle idempotence, time cap and atomic inputs OK')


if __name__ == '__main__':
    _self_check()
