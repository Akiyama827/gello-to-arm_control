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
    use_ex_pos                    可选：用**绝对单圈编码器 ExPos**（0..2π）作为角度源，
                                  不依赖多圈计数，上电即为绝对角。代价是只能分辨一圈，
                                  关节行程需落在 (-π, π] 且零点不正对 0/2π 边界。
                                  默认 False（用多圈 q_out）。
    """

    chain: "object"  # S288JointChain；用 object 避免循环导入
    joint_offsets: Sequence[float] = ()
    joint_signs: Sequence[float] = ()
    gripper_index: Optional[int] = None
    gripper_open_rad: float = 0.0
    gripper_close_rad: float = 0.0
    alpha: float = 0.99
    start_joints: Optional[Sequence[float]] = None
    use_ex_pos: bool = False
    # 仅夹爪这一个关节改用绝对单圈 ExPos 作角度源（该电机的转子多圈 q_out 故障时）。
    # q_out 坏但输出端绝对编码器仍可用时，用它把夹爪救回来；臂关节照旧用 q_out。
    gripper_use_ex_pos: bool = False

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
    def _ex_pos_mask(self) -> np.ndarray:
        """哪些关节用**绝对单圈 ExPos**作角度源（其余用多圈 q_out）。"""
        mask = np.zeros(int(self.chain.n), dtype=bool)
        if self.use_ex_pos:
            mask[:] = True
        if self.gripper_use_ex_pos and self.gripper_index is not None:
            mask[self.gripper_index] = True
        return mask

    def ex_pos_mask(self) -> np.ndarray:
        """公开版：供标定 CLI 复用同一套源选择逻辑。"""
        return self._ex_pos_mask()

    def _use_ex_angle(self, raw: np.ndarray, ex: np.ndarray) -> np.ndarray:
        """把角度源换成**绝对单圈编码器 ExPos**（0..2π；nan/未选中的关节保留 q_out）。

        ExPos 是输出端单圈绝对值，跨上电仍是绝对角，因此用它做角度源就不依赖多圈
        计数。代价：可分辨范围只有一圈，配合 ``_calibrated_positions`` 折到 (-π, π]，
        因此该关节机械行程需落在 (-π, π] 且零点不正对编码器 0/2π 边界。默认只对
        ``use_ex_pos`` 的关节生效；``gripper_use_ex_pos`` 时仅夹爪这一个关节生效。
        """
        out = np.asarray(raw, dtype=float).copy()
        ex = np.asarray(ex, dtype=float)
        use = self._ex_pos_mask() & np.isfinite(ex)
        out[use] = ex[use]
        return out

    def read_source_positions(self) -> np.ndarray:
        """当前角度源（按 use_ex_pos / gripper_use_ex_pos 选 q_out 或 ExPos），未标定。"""
        return self._read_raw_positions()

    def _read_raw_positions(self) -> np.ndarray:
        if not (self.use_ex_pos or self.gripper_use_ex_pos):
            return np.asarray(self.chain.read_positions(), dtype=float)
        raw, ex = self.chain.read_positions_and_ex_positions()
        return self._use_ex_angle(raw, ex)

    def _align_offsets_to_start(self) -> None:
        """把每个零位就近 ±2π 对齐到 start_joints，消除多圈歧义。"""
        start = np.asarray(self.start_joints, dtype=float)
        current = self._calibrated_positions(self._read_raw_positions())
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
        pos = (np.asarray(raw, dtype=float) - self._joint_offsets) * self._joint_signs
        mask = self._ex_pos_mask()
        if mask.any():
            # ExPos 只有一圈：折到 (-π, π]，让跨 0/2π 边界时连续
            pos = pos.copy()
            pos[mask] = (pos[mask] + np.pi) % (2 * np.pi) - np.pi
        return pos

    # -- 读取 ---------------------------------------------------------------
    def num_dofs(self) -> int:
        return int(self.chain.n)

    def get_joint_state(self) -> np.ndarray:
        raw = self._read_raw_positions()
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
        raw_pos, raw_vel, ex = self.chain.read_arrays()
        if self.use_ex_pos or self.gripper_use_ex_pos:
            raw_pos = self._use_ex_angle(raw_pos, ex)
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
        center: Optional[Sequence[float]] = None,
    ) -> None:
        self._n_arm = int(n_arm_joints)
        self._gripper = bool(with_gripper)
        self._amp = float(amplitude)
        self._period = float(period_s)
        self._speed = float(speed)
        self._t0 = None
        center_arr = None if center is None else np.asarray(center, dtype=float)
        if center_arr is not None and center_arr.shape != (self._n_arm,):
            raise ValueError(
                f"center 长度应为 {self._n_arm}，得到 {center_arr.shape}"
            )
        self._center = center_arr
        self._phase = np.arange(self._n_arm, dtype=float) * 0.7

    def num_dofs(self) -> int:
        return self._n_arm + (1 if self._gripper else 0)

    def get_joint_state(self) -> np.ndarray:
        import time

        if self._t0 is None:
            self._t0 = time.monotonic()
        t = (time.monotonic() - self._t0) * self._speed
        omega = 2 * np.pi * t / self._period
        if self._center is None:
            vals = list(self._amp * np.sin(omega + self._phase))
        else:
            # 减去 t=0 的相位使初始姿态正好落在 center；系数 0.5 把偏差限制在 ±amp
            vals = list(
                self._center
                + 0.5 * self._amp * (np.sin(omega + self._phase) - np.sin(self._phase))
            )
        if self._gripper:
            vals.append(0.5 + 0.5 * np.sin(omega))
        return np.asarray(vals, dtype=float)

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_state": self.get_joint_state()}

    def set_torque_mode(self, enable: bool) -> None:
        pass

    def close(self) -> None:
        pass
