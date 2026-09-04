"""Dora node: the BOUNDED half of the arm stack — servo, hold, report.

A thin adapter. Everything real lives in
``arm_control.execution.arm_controller.ArmController``; this file owns the Dora
seam and builds the one thing the controller needs: an executor, via
``build_executor`` from ``arm_control.execution.factory`` — NOT from
``planning.stack``, which would drag in the IK, the OMPL instance, the
collision world, and (through ``preview_rerun``) a hard ``rerun`` import that
this node has no display for. The factory exists so a headless install without
the optional ``[viz]`` extra can still servo.

It is arm-agnostic. ``ARM_CONTROL_CONFIG`` picks the arm; the dataflow picks the
plant. Pair it with ``arm_planner`` (Control-side for a workcell add, but the
contract — plan / control in, motor_command / controller_event out — is generic).
"""

from __future__ import annotations

# ruff: noqa: E402

import os
import sys
from pathlib import Path

from dora import Node

ARM_CONTROL_ROOT = Path(__file__).resolve().parents[1]
if str(ARM_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(ARM_CONTROL_ROOT))

from arm_control.config import load_robot_config
from arm_control.execution.arm_controller import ArmController
from arm_control.execution.factory import build_executor, gripper_command_cfg


def main() -> None:
    cfg = load_robot_config()
    controller = ArmController(
        Node(),
        build_executor(cfg, arm_id=os.environ.get("ARM_ID", "arm")),
        arm_id=os.environ.get("ARM_ID", "arm"),
        gripper=gripper_command_cfg(cfg),
        state_period_s=(
            float(cfg.get("motor_state_period_s"))
            if cfg.get("motor_state_period_s") is not None
            else None
        ),
    )
    print(
        "[arm_controller] idle — DISARMED, streaming nothing until the planner "
        "arms (nothing moves without an operator)",
        flush=True,
    )
    controller.run()


if __name__ == "__main__":
    main()
