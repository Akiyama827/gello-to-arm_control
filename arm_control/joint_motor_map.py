"""Joint<->motor mapping for the real arm (the 8-vs-7 / gripper seam).

The coordinator/executor work in URDF joint space: 6 arm joints plus a *logical*
gripper expressed as a finger coordinate (metres). The real arm has 7 DM motors:
6 arm joints 1:1 (CAN 0x01-0x06) and ONE gripper motor (0x07) that drives both
prismatic fingers. This module owns the bidirectional map between the two.

The gripper mimic (motor radians <-> finger metres) is a linear interpolation
through the calibrated endpoints in the caller's ``configs/real/assembler.yaml``
``joint_mimics`` (``motor_open``/``motor_closed`` <-> ``lower``/``upper``).
All calibration is read
from that per-joint ``mimic_cfg`` dict; nothing is hardcoded here. The same
calibration was previously applied only for rendering in ``nodes/visualizer.py``;
it now lives here so both directions share one source of truth.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pyarrow as pa

from arm_control.messages import (
    pack_cartesian_block,
    pack_motor_command,
    unpack_motor_state,
)


def _endpoints(mimic_cfg: dict) -> tuple[float, float, float, float]:
    """Read (motor_open, motor_closed, finger_lower, finger_upper) from config.

    Defaults for ``lower``/``upper`` match the visualizer's historical mimic so
    the forward map stays byte-identical after extraction.
    """
    motor_open = float(mimic_cfg["motor_open"])
    motor_closed = float(mimic_cfg["motor_closed"])
    lower = float(mimic_cfg.get("lower", 0.0))
    upper = float(mimic_cfg.get("upper", 1.0))
    return motor_open, motor_closed, lower, upper


def gripper_motor_to_finger(motor_rad: float, mimic_cfg: dict) -> float:
    """Gripper motor angle (rad) -> finger displacement (m), linear through endpoints.

    Forward mimic (the direction the visualizer renders). Not clamped: the
    caller clamps to the physical finger range if it needs to (the visualizer
    does). The expression matches the extracted inline formula operand-for-operand
    so rendered positions are unchanged.
    """
    motor_open, motor_closed, lower, upper = _endpoints(mimic_cfg)
    if motor_closed == motor_open:
        raise ValueError("gripper mimic has identical motor endpoints")
    return lower + (motor_rad - motor_open) / (motor_closed - motor_open) * (upper - lower)


def gripper_finger_to_motor(finger_m: float, mimic_cfg: dict) -> float:
    """Finger displacement (m) -> gripper motor angle (rad); exact inverse of the above."""
    motor_open, motor_closed, lower, upper = _endpoints(mimic_cfg)
    if upper == lower:
        raise ValueError("gripper mimic has identical finger endpoints")
    return motor_open + (finger_m - lower) / (upper - lower) * (motor_closed - motor_open)


def pack_arm_gripper_command(
    arm_q: Sequence[float],
    arm_qd: Sequence[float],
    arm_tau: Sequence[float],
    arm_kp: Sequence[float],
    arm_kd: Sequence[float],
    gripper_finger_m: float,
    gripper_gains: Sequence[float],
    mimic_cfg: dict | None,
    cartesian: dict | None = None,
) -> pa.Array:
    """Pack the 7-motor command: 6 arm joints 1:1 + 1 gripper motor.

    The logical gripper target (finger metres) is mapped to gripper motor radians
    via ``gripper_finger_to_motor``. The gripper slot carries zero velocity/torque
    feedforward and its own ``(kp, kd)`` gains. Wire layout is delegated to
    ``pack_motor_command`` (5 floats/motor) — this helper only arranges the arrays.

    ``mimic_cfg=None`` means this arm's gripper is NOT a motor on the same bus
    (the FR3's Franka Hand is its own device, driven by grasp requests): the
    command is then arm joints only, with no gripper slot appended.

    ``cartesian`` is the optional EE-level impedance block (target pose +
    task-frame K_c/D_c dict, see ``messages.pack_cartesian_block``). It has no
    per-motor structure, so the gripper mapping above does not touch it.
    """
    tail = (
        None
        if cartesian is None
        else pack_cartesian_block(
            cartesian["pose"], cartesian["task_R"], cartesian["kc"], cartesian["dc"]
        )
    )
    if mimic_cfg is None:
        return pack_motor_command(
            np.asarray(arm_q, dtype=float),
            np.asarray(arm_qd, dtype=float),
            np.asarray(arm_tau, dtype=float),
            np.asarray(arm_kp, dtype=float),
            np.asarray(arm_kd, dtype=float),
            tail,
        )
    motor_rad = gripper_finger_to_motor(float(gripper_finger_m), mimic_cfg)
    g_kp, g_kd = float(gripper_gains[0]), float(gripper_gains[1])
    pos = np.concatenate([np.asarray(arm_q, dtype=float), [motor_rad]])
    vel = np.concatenate([np.asarray(arm_qd, dtype=float), [0.0]])
    tor = np.concatenate([np.asarray(arm_tau, dtype=float), [0.0]])
    kp = np.concatenate([np.asarray(arm_kp, dtype=float), [g_kp]])
    kd = np.concatenate([np.asarray(arm_kd, dtype=float), [g_kd]])
    return pack_motor_command(pos, vel, tor, kp, kd, tail)


def unpack_motor_state_to_joint(
    motor_state: pa.Array,
    n_motors: int,
    mimic_cfg: dict | None,
    n_arm: int = 6,
):
    """7-motor ``motor_state`` -> (arm ``JointState`` in URDF space, gripper finger m).

    Arm motors map 1:1 to the first ``n_arm`` URDF joints; the gripper motor (the
    slot after the arm) maps to finger metres via ``gripper_motor_to_finger``.
    ``JointState`` is a neutral motion value, imported lazily for callers that
    only need the mimic (e.g. the visualizer).

    ``mimic_cfg=None`` (gripper is not a motor on this bus — e.g. the FR3's
    Franka Hand) returns ``finger_m=None``; there is no gripper slot to read.
    """
    from arm_control.motion import JointState

    state = unpack_motor_state(motor_state, n_motors)
    arm = JointState(
        position=state["position"][:n_arm].copy(),
        velocity=state["velocity"][:n_arm].copy(),
    )
    if mimic_cfg is None:
        return arm, None
    finger_m = gripper_motor_to_finger(float(state["position"][n_arm]), mimic_cfg)
    return arm, finger_m


__all__ = [
    "gripper_motor_to_finger",
    "gripper_finger_to_motor",
    "pack_arm_gripper_command",
    "unpack_motor_state_to_joint",
]
