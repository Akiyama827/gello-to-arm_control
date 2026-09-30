"""Console ``control`` updates -> SDK calls, so the SDK owner is the only path to mode/payload.

The console (`arm_console`) emits one Dora ``control`` topic: arm, execute, cancel,
hold, stop, and also gains, Soft (`pose_hold`) and payload. Gain/Soft/payload change
the controller's *mode and load*, which ``controller/1`` on the supervisor owns
(named presets resolved once by :mod:`arm_control.control.presets`; Soft validation,
cancel-before-change and payload identity in the runtime). This adapter splits a
decoded update: those three fields become SDK calls, everything else is returned for
the caller to forward unchanged, in the same message order.

Behaviour is not changed: a gain/Soft request must be exactly a named preset
(the console only ever sends its own presets), and every SDK refusal is reported as
a refusal string rather than swallowed. No new authority: it holds one ``ArmClient``.
"""
from dataclasses import dataclass, field

import numpy as np

from arm_control.execution.controller import PayloadValue
from arm_control.sdk import Refused

MODE_FIELDS = ('gains', 'pose_hold', 'payload')


@dataclass(frozen=True)
class Applied:
    forward: dict                       # fields for the ordinary controller path
    refusals: tuple[str, ...] = ()      # one line per refused mode/payload request
    applied: tuple[str, ...] = field(default_factory=tuple)  # preset ids / 'payload' done


def _canon(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
        return float(value)
    return value


class ConsoleControlAdapter:
    def __init__(self, arm, presets):
        """``arm``: a claimed ArmClient; ``presets``: control.presets.resolve_presets(...)."""
        self.arm, self.presets = arm, presets

    def preset_for_gains(self, gains) -> str | None:
        kp, kd = np.asarray(gains.get('kp'), dtype=float), np.asarray(gains.get('kd'), dtype=float)
        for name, preset in self.presets.items():
            if preset.pose_hold is None and np.array_equal(preset.kp, kp) \
                    and np.array_equal(preset.kd, kd):
                return name
        return None

    def preset_for_soft(self, spec) -> str | None:
        want = _canon({k: v for k, v in dict(spec).items() if k != 'id'})
        for name, preset in self.presets.items():
            if preset.pose_hold is not None and _canon(preset.pose_hold) == want:
                return name
        return None

    def apply(self, update: dict) -> Applied:
        forward = {k: v for k, v in update.items() if k not in MODE_FIELDS}
        refusals, applied = [], []

        def select(name, what):
            if name is None:
                refusals.append(f'{what} is not a named controller preset')
                return
            try:
                self.arm.select_control_preset(name)
                applied.append(name)
            except Refused as exc:
                refusals.append(f'preset {name}: {exc}')

        if 'gains' in update:
            select(self.preset_for_gains(update['gains']), 'gains')
        if 'pose_hold' in update:
            select(self.preset_for_soft(update['pose_hold']), 'Soft request')
        if 'payload' in update:
            body = update['payload'] or {}
            try:
                self.arm.set_payload(PayloadValue(float(body.get('mass_kg', 0.0)), body.get('com_ee')))
                applied.append('payload')
            except (Refused, ValueError) as exc:
                refusals.append(f'payload: {exc}')
        return Applied(forward, tuple(refusals), tuple(applied))
