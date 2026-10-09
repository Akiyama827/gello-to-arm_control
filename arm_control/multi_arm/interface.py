"""The backend-neutral public multi-arm surface.

External consumers depend on THIS module -- not on Dora node internals, not on
MuJoCoBackend, not on FrankaHardwareBackend, not on private ArmController
fields, not on topic naming. The same interface sits above a shared MuJoCo
world today and above two real arm plants tomorrow, because it speaks only
the existing per-arm control contracts (plan / control in,
motor_command / controller_event out) aggregated under arm ids.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from arm_control.control.inprocess import QueueControllerNode
from arm_control.messages import (
    pack_control_update,
    unpack_controller_event,
)
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


class LocalArmHandle:
    """An :class:`ArmHandle` over a live in-process ``ArmController``.

    Events are collected from the controller's node (which must expose
    ``drain_controller_events`` -- :class:`QueueControllerNode` does) and
    dispatched by the coordinator's ``pump``. Nothing here touches a second
    arm's controller.
    """

    def __init__(
        self,
        controller,
        *,
        node: QueueControllerNode | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._controller = controller
        self._node = node if node is not None else controller.node
        if not hasattr(self._node, "drain_controller_events"):
            raise ValueError(
                "LocalArmHandle needs a node exposing drain_controller_events "
                "(use QueueControllerNode)"
            )
        self._clock = clock or time.monotonic

    @property
    def arm_id(self) -> str:
        return self._controller.arm_id

    @property
    def controller(self):
        return self._controller

    def poll_events(self) -> list[dict]:
        """New controller events since the last poll (staging ACKs, results)."""
        return [
            unpack_controller_event(arrow)
            for arrow in self._node.drain_controller_events()
        ]

    def state(self) -> ArmControlState:
        c = self._controller
        # Zeros from a controller that has never measured anything are NOT a
        # measured pose; only report joints once the plant has streamed and
        # the controller went ready at least once.
        measured = c.last_state if c._last_state_at is not None else None
        return ArmControlState(
            arm_id=c.arm_id,
            position=None if measured is None else measured.position,
            velocity=None if measured is None else measured.velocity,
            armed=bool(c.armed),
            ready=bool(c.ready_sent and c.bridge_armed),
            executing=bool(c._running),
            running_plan_id=c.current_plan_id or None,
            mode_law=c.mode_law,
            stopped=bool(c.stopped),
            frozen=bool(c.frozen),
            observed_at=c._last_state_at,
            session_id=c.session_id,
            model_token=c.model_token,
        )

    def stage(self, plan: ArmPlan) -> None:
        self._controller.on_plan(plan.payload)

    def execute(self, plan_id: str) -> None:
        self._controller.on_control(pack_control_update(execute=plan_id))

    def cancel(self, reason: str = "", *, plan_id: str | None = None) -> None:
        self._controller.on_control(
            pack_control_update(cancel=True, reason=reason, cancel_plan_id=plan_id)
        )

    def set_armed(self, armed: bool) -> None:
        # Arming is a LIFECYCLE operation through the controller's own
        # control path (the same message an operator console sends) -- never
        # a flag write, and never implied by a plan arriving.
        self._controller.on_control(pack_control_update(arm=bool(armed)))

    def set_mode(self, law: str, spec: Mapping | None = None) -> None:
        if law == "soft":
            if spec is None:
                raise ValueError("set_mode('soft') requires a pose-hold spec")
            self._controller.on_control(pack_control_update(pose_hold=dict(spec)))
        elif law == "joint":
            # Leaving Soft goes through the controller's own cancel path
            # ("leaving Soft"); a controller already in joint mode is left
            # alone -- cancelling it here would abort a leg for nothing.
            if self._controller.mode_law == "soft":
                self._controller.on_control(
                    pack_control_update(cancel=True, reason="select joint mode")
                )
        else:
            raise ValueError(f"unknown control law {law!r} (joint or soft)")

    def set_gains(self, kp=None, kd=None) -> None:
        fields: dict[str, Any] = {}
        if kp is not None:
            fields["kp"] = kp
        if kd is not None:
            fields["kd"] = kd
        if fields:
            self._controller.on_control(pack_control_update(gains=fields))

    def command_gripper(self, finger_m: float) -> None:
        raise ValueError("an arm controller drives no gripper: the Hand is its own device (hand_bridge / the sim Hand)")


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
