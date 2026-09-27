"""Resolve deployment presets once, below applications and the supervisor.

Replaces console-side gain and Soft specification resolution. Numerical law
validation and transition semantics remain owned by ArmController.
"""
from dataclasses import dataclass

import numpy as np

from arm_control.contracts.impedance import pose_hold_values
from arm_control.control.trajectory_executor import gain_error
from arm_control.execution.controller import ControllerPreset
from arm_control.node_utils import expand_named_values


@dataclass(frozen=True)
class ResolvedPreset:
    choice: ControllerPreset
    kp: np.ndarray
    kd: np.ndarray
    pose_hold: dict | None


def resolve_presets(mode, joints, kp, kd):
    result = {}
    for name, raw in (mode.get('gain_presets') or {}).items():
        spec = dict(raw or {})
        gains = tuple(np.array(default, copy=True) if spec.get(key) is None else
                      expand_named_values(spec[key], names=joints, default=0.)
                      for key, default in (('kp', kp), ('kd', kd)))
        error = gain_error(*gains)
        if error or any(x.shape != (len(joints),) for x in gains):
            raise ValueError(f'preset {name}: {error or "wrong gain shape"}')
        soft = spec.get('pose_hold')
        if soft is not None:
            pose_hold_values(dict(soft, id=1))
            soft = dict(soft)
        result[str(name)] = ResolvedPreset(
            ControllerPreset(str(name), 'soft' if soft is not None else 'joint'),
            *gains, soft)
    return result
