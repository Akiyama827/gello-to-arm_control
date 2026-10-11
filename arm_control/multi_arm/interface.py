"""The backend-neutral public multi-arm surface.

External consumers depend on THIS module -- not on Dora node internals, not on
MuJoCoBackend, not on FrankaHardwareBackend, not on private ArmController
fields, not on topic naming. The same interface sits above two arm
supervisors, simulated or on the RT link (``SdkArmHandle``), because it speaks
only the per-arm control contracts (plan / control in, controller_event out)
aggregated under arm ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from arm_control.multi_arm.types import (
    ArmControlState,
    ArmPlan,
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
