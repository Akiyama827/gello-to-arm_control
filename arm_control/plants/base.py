"""Shared observable plant surface; command/lifecycle APIs remain backend-specific.

DM, Franka, remote RT and MuJoCo all expose motor state and resource cleanup.
This contract does not pretend enabling a physical arm is equivalent to loading
a simulator, or replace any backend's command validation or safe-stop behavior.
"""
from typing import Protocol

import numpy as np


class Plant(Protocol):
    @property
    def num_motors(self) -> int: ...

    def motor_state(self) -> dict[str, np.ndarray]: ...

    def close(self) -> None: ...


def _self_check() -> None:
    import inspect

    from arm_control.plants.dm.backend import DmHardwareBackend
    from arm_control.plants.franka.backend import FrankaHardwareBackend
    from arm_control.plants.remote_rt.client import RtBackend
    from arm_control.plants.mujoco.backend import MuJoCoBackend

    for backend in (DmHardwareBackend, FrankaHardwareBackend, RtBackend, MuJoCoBackend):
        for name in ('num_motors', 'motor_state', 'close'):
            assert inspect.getattr_static(backend, name) is not None, (backend, name)
    print('plants.base: OK (structural surface; no hardware instantiated)')


if __name__ == '__main__':
    _self_check()
