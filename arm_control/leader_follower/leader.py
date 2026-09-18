"""小机械臂（leader / 输入设备）抽象与实现。

按仓库里既有的约定，leader 就是一个实现了 `Robot` 语义的设备：
`num_dofs()` / `get_joint_state()` / `get_observations()`。关节角单位**弧度**，
顺序按基座到末端；若配了夹爪，夹爪是最后一个元素。

三个实现
--------------------------------------------------------------------------
    S288LeaderArm      宇树 S288 关节链 + 标定（本项目的小臂）
    GelloLeaderAdapter 直接复用 gello 的 DynamixelRobot.get_joint_state()
    FakeLeaderArm      无硬件的脚本化动作，用于打通整条链路

标定与平滑沿用 gello `DynamixelRobot` 的语义，并修正夹爪归一化方向为
**1 = 张开，0 = 闭合**（与目标夹爪宽度"张开 > 闭合"一致）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# 协议
# --------------------------------------------------------------------------- #
class LeaderArm(Protocol):
    """所有 leader（小臂/输入设备）必须满足的接口。"""

    def num_dofs(self) -> int:
        ...

    def get_joint_state(self) -> np.ndarray:
        """返回关节状态（rad），长度 = num_dofs()；夹爪在最后一位。"""
        ...

    def get_observations(self) -> Dict[str, np.ndarray]:
        ...

    def set_torque_mode(self, enable: bool) -> None:
        ...

    def close(self) -> None:
        ...


# --------------------------------------------------------------------------- #
# S288 leader
# --------------------------------------------------------------------------- #
@dataclass
class S288LeaderArm:
    """一挂 S288 关节构成的小臂，带标定 / 夹爪归一化 / 指数平滑。

    参数
    ----
    chain: S288JointChain          已连好的电机链
    joint_offsets / joint_signs   每关节零位(rad)与方向(±1)
    gripper_index                 夹爪在状态向量中的下标；None 表示无夹爪
    gripper_open_rad              夹爪"张开"时的原始关节角（标定后，rad）
    gripper_close_rad             夹爪"闭合"时的原始关节角（标定后，rad）
    alpha                         指数平滑系数，1.0 = 不平滑
    start_joints                  可选：启动时把各关节零位就近对齐到该位形
    """

    chain: "object"  # S288JointChain；用 object 避免循环导入
    joint_offsets: Sequence[float] = ()
    joint_signs: Sequence[float] = ()
    gripper_index: Optional[int] = None
    gripper_open_rad: float = 0.0
    gripper_close_rad: float = 0.0
    alpha: float = 0.99
    start_joints: Optional[Sequence[float]] = None

    _last_pos: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _joint_offsets: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _joint_signs: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = int(self.chain.n)
        offsets = np.asarray(self.joint_offsets, dtype=float)
        signs = np.asarray(self.joint_signs, dtype=float)
        if offsets.size == 0:
            offsets = np.zeros(n)
        if signs.size == 0:
            signs = np.ones(n)
        if offsets.shape != (n,) or signs.shape != (n,):
            raise ValueError(
                f"joint_offsets/joint_signs 长度应为 {n}，得到 {offsets.shape}/{signs.shape}"
            )
        if not np.all(np.abs(signs) == 1):
            raise ValueError(f"joint_signs 必须为 ±1：{signs}")
        self._joint_offsets = offsets
        self._joint_signs = signs

        if self.start_joints is not None:
            self._align_offsets_to_start()

    # -- 标定 ---------------------------------------------------------------
    def _align_offsets_to_start(self) -> None:
        """把每个零位就近 ±2π 对齐到 start_joints，消除多圈歧义。"""
        start = np.asarray(self.start_joints, dtype=float)
        current = self._calibrated_positions(self.chain.read_positions())
        assert start.shape == current.shape
        arm = slice(None) if self.gripper_index is None else slice(0, -1)
        aligned = self._joint_offsets.copy()
        aligned[arm] = (
            2
            * np.pi
            * np.round((-start[arm] + current[arm]) / (2 * np.pi))
            * self._joint_signs[arm]
            + self._joint_offsets[arm]
        )
        self._joint_offsets = aligned

    def _calibrated_positions(self, raw: np.ndarray) -> np.ndarray:
        return (np.asarray(raw, dtype=float) - self._joint_offsets) * self._joint_signs

    # -- 读取 ---------------------------------------------------------------
    def num_dofs(self) -> int:
        return int(self.chain.n)

    def get_joint_state(self) -> np.ndarray:
        raw = self.chain.read_positions()
        pos = self._calibrated_positions(raw)
        if self.gripper_index is not None:
            if not (self.gripper_open_rad != self.gripper_close_rad):
                raise ValueError("夹爪开/合标定角相同，无法归一化")
            g = (pos[self.gripper_index] - self.gripper_close_rad) / (
                self.gripper_open_rad - self.gripper_close_rad
            )
            pos[self.gripper_index] = float(np.clip(g, 0.0, 1.0))  # 1=张开 0=闭合
        if self._last_pos is None or self.alpha >= 1.0:
            self._last_pos = pos
        else:
            pos = self._last_pos * (1.0 - self.alpha) + pos * self.alpha
            self._last_pos = pos
        return pos

    def get_positions_and_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        raw_pos, raw_vel = self.chain.read_positions_and_velocities()
        pos = self._calibrated_positions(raw_pos)
        vel = np.asarray(raw_vel, dtype=float) * self._joint_signs
        return pos, vel

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_state": self.get_joint_state()}

    def set_torque_mode(self, enable: bool) -> None:
        # S288 用 MIT 帧，没有独立"使能位"；撤销使能 = 发零刚度零阻尼停止帧。
        if not enable:
            self.chain.release()

    def close(self) -> None:
        self.chain.close()


# --------------------------------------------------------------------------- #
# gello 适配器
# --------------------------------------------------------------------------- #
class GelloLeaderAdapter:
    """把 gello 的 `DynamixelRobot`（或任何有 `get_joint_state()` 的对象）接进来。

    这样即便小臂仍跑 gello 原有驱动，也能直接被本遥操作回路消费。leader 状态先
    经过可选的 offsets/signs 再做相同的指数平滑。
    """

    def __init__(
        self,
        robot: object,
        joint_offsets: Optional[Sequence[float]] = None,
        joint_signs: Optional[Sequence[float]] = None,
        alpha: float = 0.99,
    ) -> None:
        self._robot = robot
        n = int(robot.num_dofs())  # type: ignore[attr-defined]
        self._offsets = (
            np.zeros(n)
            if joint_offsets is None
            else np.asarray(joint_offsets, dtype=float)
        )
        self._signs = (
            np.ones(n)
            if joint_signs is None
            else np.asarray(joint_signs, dtype=float)
        )
        if self._offsets.shape != (n,) or self._signs.shape != (n,):
            raise ValueError(f"offsets/signs 长度应为 {n}")
        self._alpha = float(alpha)
        self._last: Optional[np.ndarray] = None

    def num_dofs(self) -> int:
        return int(self._robot.num_dofs())  # type: ignore[attr-defined]

    def get_joint_state(self) -> np.ndarray:
        pos = (
            np.asarray(self._robot.get_joint_state(), dtype=float)  # type: ignore[attr-defined]
            - self._offsets
        ) * self._signs
        if self._last is None or self._alpha >= 1.0:
            self._last = pos
        else:
            pos = self._last * (1.0 - self._alpha) + pos * self._alpha
            self._last = pos
        return pos

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_state": self.get_joint_state()}

    def set_torque_mode(self, enable: bool) -> None:
        fn = getattr(self._robot, "set_torque_mode", None)
        if callable(fn):
            fn(enable)

    def close(self) -> None:
        fn = getattr(self._robot, "close", None)
        if callable(fn):
            fn()


# --------------------------------------------------------------------------- #
# 假 leader
# --------------------------------------------------------------------------- #
class FakeLeaderArm:
    """脚本化小臂：每个关节按不同频率做正弦运动，夹爪在 [0,1] 来回。

    用于在没有 S288 硬件时验证"采集 -> 换算 -> 下发"整条链路。
    """

    def __init__(
        self,
        n_arm_joints: int = 6,
        with_gripper: bool = True,
        amplitude: float = 0.4,
        period_s: float = 6.0,
        speed: float = 1.0,
    ) -> None:
        self._n_arm = int(n_arm_joints)
        self._gripper = bool(with_gripper)
        self._amp = float(amplitude)
        self._period = float(period_s)
        self._speed = float(speed)
        self._t0 = None

    def num_dofs(self) -> int:
        return self._n_arm + (1 if self._gripper else 0)

    def get_joint_state(self) -> np.ndarray:
        import time

        if self._t0 is None:
            self._t0 = time.monotonic()
        t = (time.monotonic() - self._t0) * self._speed
        vals = [
            self._amp * np.sin(2 * np.pi * t / self._period + i * 0.7)
            for i in range(self._n_arm)
        ]
        if self._gripper:
            vals.append(0.5 + 0.5 * np.sin(2 * np.pi * t / self._period))
        return np.asarray(vals, dtype=float)

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_state": self.get_joint_state()}

    def set_torque_mode(self, enable: bool) -> None:
        pass

    def close(self) -> None:
        pass
