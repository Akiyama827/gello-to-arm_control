"""controller/1 values: named deployment choices and complete transient loads.

No device, planner, or controller imports. Payload reference frame is fixed by
controller construction; a declaration cannot redefine it.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControllerPreset:
    id: str
    law: str

    def __post_init__(self):
        if not self.id or self.law not in ('joint', 'soft'):
            raise ValueError('preset requires an id and joint/soft law')


@dataclass(frozen=True)
class PayloadValue:
    mass_kg: float
    com_ee: tuple[float, float, float] | None = None

    def __post_init__(self):
        mass = float(self.mass_kg)
        if not math.isfinite(mass) or mass < 0:
            raise ValueError('payload mass must be finite and nonnegative')
        com = None
        if mass > 0 and self.com_ee is not None:
            com = tuple(float(x) for x in self.com_ee)
            if len(com) != 3 or not all(math.isfinite(x) for x in com):
                raise ValueError('payload CoM must contain three finite values')
        object.__setattr__(self, 'mass_kg', mass)
        object.__setattr__(self, 'com_ee', com)


@dataclass(frozen=True)
class PayloadState:
    revision: int
    value: PayloadValue


@dataclass(frozen=True)
class ControllerState:
    law: str
    preset_id: str | None
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    payload: PayloadState

    def __post_init__(self):
        object.__setattr__(self, 'kp', tuple(self.kp))
        object.__setattr__(self, 'kd', tuple(self.kd))
