"""The backend-neutral public multi-arm surface.

External consumers depend on THIS module -- not on Dora node internals, not on
MuJoCoBackend, not on FrankaHardwareBackend, not on private ArmController
fields, not on topic naming. The same interface sits above two arm
supervisors, simulated or on the RT link (``SdkArmHandle``), because it speaks
only the per-arm control contracts (plan / control in, controller_event out)
aggregated under arm ids.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from arm_control.multi_arm.coordinator import BimanualCoordinator
from arm_control.multi_arm.types import (
    ArmControlState,
    ArmPlan,
    BimanualPlan,
    BimanualState,
    PairState,
)


@runtime_checkable
class ArmHandle(Protocol):
    """One arm's endpoint as coordination sees it.

    One handle wraps ONE controller endpoint and owns no other arm. The
    per-arm bounded execution, safety state and motor-command authority stay
    with the underlying controller; a handle only forwards and reports.
    """

    @property
    def arm_id(self) -> str: ...

    def state(self) -> ArmControlState: ...

    def stage(self, plan: ArmPlan) -> None: ...

    def execute(self, plan_id: str) -> None: ...

    def cancel(self, reason: str = "", *, plan_id: str | None = None) -> None: ...

    def set_armed(self, armed: bool) -> None: ...

    def set_mode(self, law: str, spec: Mapping | None = None) -> None: ...

    def set_gains(self, kp=None, kd=None) -> None: ...

    def command_gripper(self, finger_m: float) -> None: ...


class BimanualInterface:
    """The two-arm public API over a :class:`BimanualCoordinator`.

    ``left`` and ``right`` are ordered :class:`ArmHandle` slots, not required
    arm-id strings; every pair-level
    operation maps onto the members through explicit ids and lifecycle ACKs.
    The interface adds no policy: what the arms should DO is the caller's.
    """

    def __init__(self, coordinator: BimanualCoordinator) -> None:
        self._ids = tuple(coordinator.arms)
        self._coordinator = coordinator

    @classmethod
    def from_handles(
        cls, left: ArmHandle, right: ArmHandle, **kwargs
    ) -> BimanualInterface:
        return cls(
            BimanualCoordinator({left.arm_id: left, right.arm_id: right}, **kwargs)
        )

    @property
    def coordinator(self) -> BimanualCoordinator:
        return self._coordinator

    @property
    def left(self) -> ArmHandle:
        return self._coordinator.arms[self._ids[0]]

    @property
    def right(self) -> ArmHandle:
        return self._coordinator.arms[self._ids[1]]

    def state(self) -> BimanualState:
        return self._coordinator.state()

    def stage_pair(self, plan: BimanualPlan) -> None:
        self._coordinator.stage_pair(plan)

    def execute_pair(self, pair_id: str) -> None:
        self._coordinator.execute_pair(pair_id)

    def cancel_pair(self, reason: str = "") -> None:
        self._coordinator.cancel_pair(reason)

    def pump(self) -> None:
        """Poll every handle for new controller events and advance the pair."""
        self._coordinator.pump()

    @property
    def active_pair_id(self) -> str | None:
        return self._coordinator.active_pair_id

    def pair_state(self, pair_id: str | None = None) -> PairState:
        return self._coordinator.pair_state(pair_id)

    def wait_pair(
        self,
        pair_id: str,
        *,
        timeout_s: float = 10.0,
        pump: Callable[[], None] | None = None,
    ) -> PairState:
        """Pump until ``pair_id`` reaches a terminal lifecycle (local use).

        Real-time behaviour is the underlying graph's; this loop makes no
        timing promises of its own and never spins without pumping.
        """
        deadline = time.monotonic() + float(timeout_s)
        while True:
            (pump or self.pump)()
            pair = self.pair_state(pair_id)
            if pair.lifecycle.terminal and not pair.cancellation_pending:
                return pair
            if time.monotonic() > deadline:
                self.cancel_pair("wait timeout")
                raise TimeoutError(
                    f"pair {pair_id!r} still {pair.lifecycle.value} after {timeout_s}s"
                )
            if pump is None:
                time.sleep(0.001)
