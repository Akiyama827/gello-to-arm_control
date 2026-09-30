"""Console ``control`` updates -> SDK calls, so the SDK owner is the only path to mode/payload.

The console (`arm_console`) emits one Dora ``control`` topic: arm, execute, cancel,
hold, stop, and also gains, Soft (`pose_hold`) and payload. Gain/Soft/payload change
the controller's *mode and load*, which ``controller/1`` on the supervisor owns
(named presets resolved once by :mod:`arm_control.control.presets`; Soft validation,
cancel-before-change and payload identity in the runtime). Gain/Soft/payload become SDK
calls, applied independently (no all-or-nothing) in ``on_control`` relative order
(payload, gains, pose_hold); every other field is returned in ``Applied.forward`` as an
ordered list of small updates (arm, execute, cancel, hold, stop). The SDK calls take
effect DURING ``apply()`` and the forwarded groups only when the caller sends them
afterwards, so arm/execute in the same message land AFTER payload/gains/Soft (Dora
does them first). The console never sends mixed messages; a real ``send`` interleave
is a CTRL-053 item.

Intended differences from the Dora path (not "no change"; see the history note):
  * gains/Soft must be exactly a named preset (the console only sends its own);
    partial or non-preset gains and ``gains={}`` are refused/ignored differently;
  * payload during a live execution is refused (the SDK rule);
  * a Soft request while already in that Soft preset is refused: the SDK cannot
    re-capture the measured pose, the Dora path does.
A same-preset gains request cancels the live executions given by ``live()`` (the Dora
path cancels a running leg on any gains message). Malformed fields come back as
refusal strings, as is an SDK timeout; ``live()`` must not raise. A same-preset gains
request blocks up to the SDK cancel wait (2 s) per live execution. It holds one ``ArmClient``: no new authority.
"""
from dataclasses import dataclass, field

import numpy as np

from arm_control.execution.controller import PayloadValue
from arm_control.sdk import Refused

MODE_FIELDS = ('gains', 'pose_hold', 'payload')
# on_control order; a reason travels with the group that uses it.
FORWARD_ORDER = (('arm',), ('execute',), ('cancel', 'cancel_plan_id'), ('hold',), ('stop',))
REASON_WITH = ('cancel', 'hold', 'stop')


@dataclass(frozen=True)
class Applied:
    forward: tuple[dict, ...]           # ordered updates for the ordinary controller path
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
    def __init__(self, arm, presets, live=lambda: ()):
        """``arm``: a claimed ArmClient; ``presets``: control.presets.resolve_presets(...);
        ``live``: execution ids currently running, cancelled by a same-preset gains request."""
        self.arm, self.presets, self.live = arm, presets, live

    def preset_for_gains(self, gains) -> str | None:
        try:
            kp, kd = np.asarray(gains.get('kp'), dtype=float), np.asarray(gains.get('kd'), dtype=float)
        except (AttributeError, TypeError, ValueError):
            return None
        for name, preset in self.presets.items():
            if preset.pose_hold is None and np.array_equal(preset.kp, kp) \
                    and np.array_equal(preset.kd, kd):
                return name
        return None

    def preset_for_soft(self, spec) -> str | None:
        try:
            want = _canon({k: v for k, v in dict(spec).items() if k != 'id'})
        except (TypeError, ValueError):
            return None
        for name, preset in self.presets.items():
            if preset.pose_hold is not None and _canon(preset.pose_hold) == want:
                return name
        return None

    def apply(self, update: dict) -> Applied:
        refusals, applied, forward = [], [], []

        def forwarded(*keys):
            group = {k: update[k] for k in keys if k in update}
            if group and any(k in group for k in keys[:1]):
                if group.keys() & set(REASON_WITH) and 'reason' in update:
                    group['reason'] = update['reason']
                forward.append(group)

        def select(name, what, soft):
            if name is None:
                refusals.append(f'{what} is not a named controller preset')
                return
            try:
                if self.arm.controller_state().preset_id == name:
                    if soft:
                        refusals.append(f'preset {name}: already selected; the SDK cannot '
                                        're-capture the measured pose')
                        return
                    for execution_id in self.live():
                        self.arm.cancel(execution_id, 'gains changed mid-leg')
                self.arm.select_control_preset(name)
                applied.append(name)
            except (Refused, TimeoutError) as exc:
                refusals.append(f'preset {name}: {exc!r}' if isinstance(exc, TimeoutError) else f'preset {name}: {exc}')

        forwarded('arm')
        if 'payload' in update:
            body = update['payload'] or {}
            try:
                self.arm.set_payload(PayloadValue(float(body.get('mass_kg', 0.0)), body.get('com_ee')))
                applied.append('payload')
            except (Refused, ValueError, TypeError, AttributeError) as exc:
                refusals.append(f'payload: {exc}')
        forwarded('execute')
        gains = update.get('gains') or {}
        if any(gains.get(k) is not None for k in ('kp', 'kd')) if isinstance(gains, dict) else gains:
            select(self.preset_for_gains(gains), 'gains', soft=False)
        if 'pose_hold' in update:
            select(self.preset_for_soft(update['pose_hold']), 'Soft request', soft=True)
        for keys in FORWARD_ORDER[2:]:
            forwarded(*keys)
        known = {k for keys in FORWARD_ORDER for k in keys} | set(MODE_FIELDS) | {'reason'}
        extra = {k: v for k, v in update.items() if k not in known}
        if extra:
            forward.append(extra)
        return Applied(tuple(forward), tuple(refusals), tuple(applied))
