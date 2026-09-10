"""Dora node: the BOUNDED half of the arm stack — servo, hold, report.

A thin adapter. Everything real lives in
``arm_control.control.arm_controller.ArmController``; this file owns the Dora
seam and builds the one thing the controller needs: an executor, via
``build_executor`` from ``arm_control.control.factory`` — NOT from
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


from dora import Node


from arm_control.config import arm_joints, load_robot_config
from arm_control.control.arm_controller import ArmController
from arm_control.control.execution_policy import build_execution_policy
from arm_control.control.factory import build_executor, gripper_command_cfg
from arm_control.node_utils import (
    ShutdownFlag,
    _load_mode_config,
    install_signal_handlers,
    resolve_gains,
)


def main() -> None:
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    cfg = load_robot_config()
    mode_cfg = _load_mode_config()
    # Gains span two config styles: an assembly config states `arm.kp`, a
    # motion mode config states `controller.kp` and its robot config states
    # none. resolve_gains prefers the mode config and falls back to the arm
    # table, so one node serves both -- with no mode config (the assembly
    # graphs set none) it resolves exactly what the arm table always gave.
    gains = resolve_gains(
        cfg,
        mode_cfg,
        list(cfg.joint_names or cfg.motor_names),
        len(arm_joints(cfg)),
    )
    executor = build_executor(cfg, arm_id=os.environ.get("ARM_ID", "arm"), gains=gains)
    policy_config = cfg.get("execution_policy")
    if policy_config is not None and gains is None:
        raise ValueError("execution_policy requires resolved per-joint torque limits")
    # The position envelope comes from the robot description, not from YAML --
    # see build_execution_policy.
    policy = build_execution_policy(
        policy_config,
        torque_limits=None if gains is None else gains["torque_limits"][:len(arm_joints(cfg))],
        joint_limits=executor.joint_limits,
    )
    controller = ArmController(
        Node(),
        executor,
        arm_id=os.environ.get("ARM_ID", "arm"),
        gripper=gripper_command_cfg(cfg),
        execution_policy=policy,
        settle_timeout_s=float((cfg.get('arm') or {}).get('settle_timeout_s', 5.0)),
        settle_dwell_s=float((cfg.get('arm') or {}).get('settle_dwell_s', 0.2)),
        # A bare sim plant (mujoco_interface with no sim_bridge) publishes no
        # motor_health, so the controller would wait forever for an armed edge
        # that cannot come. The GRAPH knows whether a bridge is in the path, so
        # the graph says so -- opt-out, never inferred.
        plant_reports_health=os.environ.get("ARM_CONTROL_PLANT_HEALTH", "1")
        not in ("0", "false", "no"),
        # How long a jog setpoint stays live here. Must agree with the console's
        # re-send period with room to spare -- too tight and a normal scheduling
        # hiccup stutters the arm, too loose and it coasts after the button is
        # released. Same mode config both sides read.
        jog_timeout_s=float((mode_cfg.get("jog") or {}).get("timeout_s", 0.2)),
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
    controller.run(shutdown=shutdown)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
