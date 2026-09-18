"""Compatibility imports for motion values and joint-path retiming."""

from arm_control.motion import (
    JointTrajectory as JointTrajectory,
    TrajectoryPoint as TrajectoryPoint,
)
from arm_control.planning.retiming import (
    time_parameterize_blended as time_parameterize_blended,
    _self_check as _self_check,
)

if __name__ == "__main__":
    _self_check()
