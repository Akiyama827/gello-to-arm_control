"""Torque-threshold gripper grasp gate (produces ``grasp_result`` on hardware).

A pure, config-driven state machine over the ONE DM gripper motor (CAN 0x07).
It watches the motor's estimated torque and position each tick and decides:

  ``closing`` — still ramping the motor toward the closed endpoint.
  ``grasped`` — |torque| crossed the grip threshold *before* the fingers reached
                the empty-closed position (something is between the fingers).
                Latched: it stays grasped and holds the closing kp/kd/torque so
                the module is not dropped during the lift.
  ``missed``  — the fingers reached the empty-closed position with torque still
                below threshold (closed on air).
  ``lost``    — a latched grasp whose fingers later passed the empty-closed
                position: the object slipped out (drop mid-lift/transport).
                Unlatches and relaxes toward open; the controller reports a
                failed grasp_result so the orchestrator can freeze.

With a bench-calibrated ``grip_force_per_torque`` (N of grip per N·m of motor
torque, rung 11), thresholds can be configured in Newtons
(``grip_force_thresh_n``) and ``grip_force()`` reports the live squeeze force.

Everything is in motor radians: the gate senses ``step(motor_pos, motor_torque)``
and its ``command`` targets a motor position, so the caller reads/writes the
gripper motor slot directly.  The commanded target is clamped to the valid motor
range, so it never relies on the silent downstream clip (carry-forward from the
Task-2 gripper map).

Endpoints and the clamp range come from the calibrated gripper mimic
(``configs/real/assembler.yaml`` ``joint_mimics``) through the Task-2
``gripper_finger_to_motor`` map — see ``from_config``.  Nothing is hardcoded and
every threshold/gain is bench-tunable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from arm_control.joint_motor_map import gripper_finger_to_motor


class GraspStatus(Enum):
    CLOSING = "closing"
    GRASPED = "grasped"
    MISSED = "missed"
    LOST = "lost"


@dataclass(frozen=True)
class GripperCommand:
    """One gripper-motor MIT setpoint (motor space)."""

    position: float
    velocity: float
    torque: float
    kp: float
    kd: float


class GraspGate:
    def __init__(
        self,
        grip_torque_thresh: float,
        empty_closed_pos: float,
        close_target: float,
        open_target: float = 0.0,
        *,
        close_kp: float,
        close_kd: float,
        open_kp: float | None = None,
        open_kd: float | None = None,
        hold_torque: float = 0.0,
        close_step: float = 0.05,
        motor_min: float | None = None,
        motor_max: float | None = None,
        grip_force_per_torque: float | None = None,
    ) -> None:
        if close_target == open_target:
            raise ValueError("close_target and open_target must differ")
        self.grip_torque_thresh = abs(float(grip_torque_thresh))
        self.empty_closed_pos = float(empty_closed_pos)
        self.close_target = float(close_target)
        self.open_target = float(open_target)
        self.close_kp = float(close_kp)
        self.close_kd = float(close_kd)
        self.open_kp = self.close_kp if open_kp is None else float(open_kp)
        self.open_kd = self.close_kd if open_kd is None else float(open_kd)
        self.hold_torque = float(hold_torque)
        self.close_step = abs(float(close_step))
        lo, hi = sorted((self.open_target, self.close_target))
        self.motor_min = lo if motor_min is None else float(motor_min)
        self.motor_max = hi if motor_max is None else float(motor_max)
        self.grip_force_per_torque = (
            None if grip_force_per_torque is None else abs(float(grip_force_per_torque))
        )
        # +1 if closing increases motor position, -1 if it decreases it.
        self._dir = math.copysign(1.0, self.close_target - self.open_target)
        self._status = GraspStatus.CLOSING
        self._latched = False
        self._opening = False
        self._cmd_target: float | None = None

    @classmethod
    def from_config(cls, grasp_cfg: dict, mimic_cfg: dict) -> "GraspGate":
        """Build from the scenario ``grasp`` block + the gripper ``mimic_cfg``.

        Open/close endpoints and the clamp range come from the mimic finger
        endpoints through ``gripper_finger_to_motor``, so the gate, the joint
        map, and the visualizer all share ONE calibration.
        """
        lower = float(mimic_cfg.get("lower", 0.0))
        upper = float(mimic_cfg.get("upper", 1.0))
        open_target = gripper_finger_to_motor(lower, mimic_cfg)  # == motor_open
        close_target = gripper_finger_to_motor(upper, mimic_cfg)  # == motor_closed
        motor_min, motor_max = sorted((open_target, close_target))
        if "empty_closed_motor" not in grasp_cfg:
            # REQUIRED: it must be bench-calibrated (rung 7). Defaulting it to
            # close_target (full mechanical close) would make _reached_empty_closed
            # fire only at the hard stop, maximizing the false-grasp band — every
            # bottoming-out torque spike short of full close would latch as a grasp.
            raise ValueError(
                "grasp config missing required 'empty_closed_motor' (motor rad the "
                "fingers reach when closing on air); bench-calibrate it (rung 7) — "
                "no safe default exists"
            )
        force_per_torque = grasp_cfg.get("grip_force_per_torque")
        torque_thresh = float(grasp_cfg.get("grip_torque_thresh", 1.5))
        if "grip_force_thresh_n" in grasp_cfg:
            # Newton-denominated threshold needs the rung-11 calibrated map.
            if force_per_torque is None:
                raise ValueError(
                    "grasp config sets 'grip_force_thresh_n' without "
                    "'grip_force_per_torque' (N per N.m, bench-calibrated rung 11)"
                )
            torque_thresh = float(grasp_cfg["grip_force_thresh_n"]) / abs(
                float(force_per_torque)
            )
        return cls(
            grip_torque_thresh=torque_thresh,
            empty_closed_pos=float(grasp_cfg["empty_closed_motor"]),
            close_target=close_target,
            open_target=open_target,
            close_kp=float(grasp_cfg.get("close_kp", 40.0)),
            close_kd=float(grasp_cfg.get("close_kd", 2.0)),
            open_kp=grasp_cfg.get("open_kp"),
            open_kd=grasp_cfg.get("open_kd"),
            hold_torque=float(grasp_cfg.get("hold_torque", 0.0)),
            close_step=float(grasp_cfg.get("close_step_rad", 0.05)),
            motor_min=motor_min,
            motor_max=motor_max,
            grip_force_per_torque=force_per_torque,
        )

    # -- attempt control ------------------------------------------------------
    def close(self) -> None:
        """Begin (or restart) a close attempt."""
        self._status = GraspStatus.CLOSING
        self._latched = False
        self._opening = False
        self._cmd_target = None

    def release(self) -> None:
        """Open the gripper and clear the grasp latch."""
        self._latched = False
        self._opening = True
        self._cmd_target = self.open_target

    @property
    def is_grasped(self) -> bool:
        return self._latched

    @property
    def status(self) -> GraspStatus:
        return self._status

    def grip_force(self, motor_torque_est: float) -> float | None:
        """Estimated grip force in N (None without a calibrated force map)."""
        if self.grip_force_per_torque is None:
            return None
        return abs(float(motor_torque_est)) * self.grip_force_per_torque

    # -- per-tick step --------------------------------------------------------
    def step(self, motor_pos: float, motor_torque_est: float) -> GraspStatus:
        if self._latched:
            if self._reached_empty_closed(motor_pos):
                # Fingers passed the empty-closed point while "holding": the
                # object slipped out. Unlatch and relax; caller reports the drop.
                self._latched = False
                self._status = GraspStatus.LOST
                return self._status
            return GraspStatus.GRASPED  # latched: hold through the lift
        if self._opening:
            return self._status
        base = motor_pos if self._cmd_target is None else self._cmd_target
        self._cmd_target = self._advance(base)
        if self._reached_empty_closed(motor_pos):
            # ponytail: reaching empty-closed means the fingers met with nothing
            # between them; torque here is bottoming-out, never a grasp. Missed
            # regardless of torque (grasped requires NOT having reached it).
            self._status = GraspStatus.MISSED
        elif abs(motor_torque_est) > self.grip_torque_thresh:
            self._status = GraspStatus.GRASPED
            self._latched = True
        else:
            self._status = GraspStatus.CLOSING
        return self._status

    def command(self) -> GripperCommand:
        if self._opening:
            return GripperCommand(
                self._clamp(self.open_target), 0.0, 0.0, self.open_kp, self.open_kd
            )
        if self._latched:
            # HOLD: drive fully closed with closing gains + bias torque to keep grip.
            # The squeeze force here is bounded by close_kp (position error * kp) plus
            # the DM firmware's own torque clamp on the resulting current — NOT by
            # safety.torque_limits[gripper], which caps only the feed-forward tau_ff
            # (hold_torque), not the kp-driven component. Retune close_kp, not the
            # safety limit, to change hold force.
            return GripperCommand(
                self._clamp(self.close_target),
                0.0,
                self.hold_torque,
                self.close_kp,
                self.close_kd,
            )
        if self._status in (GraspStatus.MISSED, GraspStatus.LOST):
            # relax the empty grip rather than strain against the hard stop.
            return GripperCommand(
                self._clamp(self.open_target), 0.0, 0.0, self.open_kp, self.open_kd
            )
        target = self.close_target if self._cmd_target is None else self._cmd_target
        return GripperCommand(
            self._clamp(target), 0.0, 0.0, self.close_kp, self.close_kd
        )

    # -- helpers --------------------------------------------------------------
    def _reached_empty_closed(self, motor_pos: float) -> bool:
        return self._dir * (motor_pos - self.empty_closed_pos) >= 0.0

    def _advance(self, target: float) -> float:
        nxt = target + self._dir * self.close_step
        nxt = min(nxt, self.close_target) if self._dir > 0 else max(nxt, self.close_target)
        return self._clamp(nxt)

    def _clamp(self, pos: float) -> float:
        return min(max(pos, self.motor_min), self.motor_max)


__all__ = ["GraspGate", "GraspStatus", "GripperCommand"]
