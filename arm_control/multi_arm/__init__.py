"""Coordination of several single-arm controllers.

A bimanual robot is NOT a new 14-DOF ArmController. It is a coordination
group containing two existing single-arm control endpoints, each of which
keeps its own bounded execution, safety state, trajectory execution, holding,
jogging and motor-command authority. This package is the thin layer above
those endpoints:

- ``types``          arm-identity-preserving aggregate state and plan values
- ``interface``      the backend-neutral public surface consumers depend on
- ``coordinator``    pair lifecycle, staging barrier, cancel/fault propagation
- ``synchronization`` observable start-timing records for coordinated release
- ``adapter``        the Dora transport seam
- ``planning``       shared-world joint planning and common retiming
- ``cooperative``    fixed-grasp object kinematics and checked object-path planning
- ``supervised``     an ArmHandle over one arm's supervisor (the SDK)
- ``supervised_hardware`` the dual console's runtime: two supervisors, virtual shared object
- ``world``/``launch`` the simulated pair's world and its supervisors (one hosts, one co-hosts)
- ``config``         composition from existing arm profiles and scene descriptions

The coordinator owns the pair lifecycle. Reusable fixed-grasp math lives beside
it in cooperative.py; grasp selection, pickup, handover and task sequencing
remain consumer policy.
"""

from arm_control.multi_arm.interface import (
    ArmHandle,
)
from arm_control.multi_arm.types import (
    ArmControlState,
    ArmPlan,
    BimanualPlan,
    BimanualState,
    PairLifecycle,
    PairMemberState,
    PairState,
)

__all__ = [
    "ArmControlState",
    "ArmHandle",
    "ArmPlan",
    "BimanualPlan",
    "BimanualState",
    "PairLifecycle",
    "PairMemberState",
    "PairState",
]
