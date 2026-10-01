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
does them first) unless the caller passes ``send`` (CTRL-053 gate c), which sends each
group at its on_control position. The console never sends mixed messages.

Intended differences from the Dora path (not "no change"; see the history note):
  * gains/Soft must be exactly a named preset (the console only sends its own);
    partial or non-preset gains and ``gains={}`` are refused/ignored differently;
  * payload during a live execution is refused (the SDK rule);
  * a Soft request while already in that Soft preset is refused: the SDK cannot
    re-capture the measured pose, the Dora path does.
A same-preset gains request cancels the live executions given by ``live()`` (the Dora
path cancels a running leg on any gains message). Malformed fields come back as
refusal strings, as is an SDK timeout or a lost connection; ``live()`` must not raise (for SDK errors). A same-preset
gains request waits for its cancels up to ``cancel_budget_s`` in total (approximate: one channel call can run past it).
It holds one ``ArmClient``: no new authority.
"""
from dataclasses import dataclass, field
import threading
import time

import numpy as np

from arm_control.execution.controller import PayloadValue
from arm_control.sdk import Refused, SupervisorError

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


class LiveExecutions:
    """The executions THIS client started and that are not terminal: the ``live()`` supplier (CTRL-053 gate b).

    The supervisor has no list-executions operation, and the client that executes (planner, console, stream owner) is
    the one that knows its ids: register them here (``track``) and ``__call__`` returns those still running.
    An execution another client started is not seen: the SDK owner is one claimed session, so there is none.
    """

    def __init__(self, arm):
        self.arm, self._ids, self._lock = arm, [], threading.Lock()

    def track(self, execution_id: str) -> str:
        with self._lock:
            self._ids.append(execution_id)
        return execution_id

    def __call__(self) -> list:
        # ponytail: the lock is held across the status calls, so a concurrent track() waits for the sweep (worst case
        # N x the channel timeout); single-threaded use is the norm.
        with self._lock:
            alive = []
            for execution_id in self._ids:
                try:
                    if not self.arm.status(execution_id).terminal:
                        alive.append(execution_id)
                except Refused as exc:
                    # the supervisor forgot it (it retains a bounded number): it is over, not running
                    if 'unknown execution' not in str(exc):
                        alive.append(execution_id)
                except (SupervisorError, TimeoutError, OSError, ValueError, KeyError):
                    alive.append(execution_id)  # cannot show it ended: treat as running (the safe side)
            self._ids = alive
            return list(alive)


class ConsoleControlAdapter:
    def __init__(self, arm, presets, live=lambda: (), cancel_budget_s: float = 2.0):
        """``arm``: a claimed ArmClient; ``presets``: control.presets.resolve_presets(...);
        ``live``: execution ids currently running, cancelled by a same-preset gains request (use
        :class:`LiveExecutions`); ``cancel_budget_s``: the total time one ``apply`` may spend waiting for the
        same-preset gains cancels, counted from the first cancel (gate d). If it runs out the request is refused and
        the legs not yet cancelled keep running (earlier ones stay cancelled); the changed-preset path is not bounded
        by it (the supervisor cancels without waiting and switches at once)."""
        self.arm, self.presets, self.live = arm, presets, live
        self.cancel_budget_s = cancel_budget_s

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

    def apply(self, update: dict, send=None) -> Applied:
        """Apply gain/Soft/payload as SDK calls; every other field goes back in ``Applied.forward``.

        ``send(group)`` (gate c): when given, each forwarded group is sent at its ``on_control`` position
        (arm, payload, execute, gains, Soft, cancel, hold, stop), so mixed messages keep the Dora order, and
        ``forward`` comes back empty. Without it the groups are returned for the caller to send afterwards. A
        ``send`` that raises does not stop the remaining steps; its error is reported as a refusal."""
        refusals, applied, forward = [], [], []
        deadline = []  # set at the first cancel

        def forwarded(*keys):
            group = {k: update[k] for k in keys if k in update}
            if group and any(k in group for k in keys[:1]):
                if group.keys() & set(REASON_WITH) and 'reason' in update:
                    group['reason'] = update['reason']
                if send is None:
                    forward.append(group)
                    return
                try:
                    send(group)
                except Exception as exc:  # noqa: BLE001 - reported, the other steps still run
                    refusals.append(f'send {sorted(group)}: {exc!r}')

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
                    cancelled = 0
                    for execution_id in self.live():
                        if not deadline:
                            deadline.append(time.monotonic() + self.cancel_budget_s)
                        left = deadline[0] - time.monotonic()
                        if left <= 0:
                            refusals.append(f'preset {name}: cancel budget {self.cancel_budget_s} s exhausted after '
                                            f'{cancelled} cancel(s); the remaining legs were not cancelled')
                            return
                        try:
                            self.arm.cancel(execution_id, 'gains changed mid-leg', timeout_s=left)
                        except TimeoutError as exc:
                            # the cancel WAS sent; its leg is not yet terminal. Report what was done.
                            refusals.append(f'preset {name}: cancel of {execution_id} not confirmed in time '
                                            f'({exc}); {cancelled} earlier cancel(s) done, its own cancel pending, '
                                            'the remaining legs were not cancelled')
                            return
                        cancelled += 1
                self.arm.select_control_preset(name)
                applied.append(name)
            except (SupervisorError, TimeoutError, OSError, ValueError, KeyError) as exc:  # garbled reply: ValueError/KeyError
                refusals.append(f'preset {name}: {exc!r}' if not isinstance(exc, Refused) else f'preset {name}: {exc}')

        forwarded('arm')
        if 'payload' in update:
            body = update['payload'] or {}
            try:
                self.arm.set_payload(PayloadValue(float(body.get('mass_kg', 0.0)), body.get('com_ee')))
                applied.append('payload')
            except (SupervisorError, OSError, TimeoutError, ValueError, TypeError, AttributeError, KeyError) as exc:
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
            if send is None:
                forward.append(extra)
            else:
                try:
                    send(extra)
                except Exception as exc:  # noqa: BLE001
                    refusals.append(f'send {sorted(extra)}: {exc!r}')
        return Applied(tuple(forward), tuple(refusals), tuple(applied))
