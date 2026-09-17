"""小臂 -> 大臂的换算（retarget）。

为什么需要换算
--------------------------------------------------------------------------
两臂"体积和电机参数不同"，同一动作在小臂上表现为关节角 θ_small，到大臂上
应当是成比例、可偏置、方向可能相反、并受大臂行程限制的 θ_big。关节空间换算
是这里最稳的做法（也和大臂/小臂现有 factor 映射一致），逐关节：

    θ_big_i = offset_i + scale_i * sign_i * θ_small[src_index_i]

再依次做：夹爪归一化 -> 指数平滑 -> 变化率限幅 -> 关节限位裁剪 ->（可选）软启动。

约定
--------------------------------------------------------------------------
- leader 状态向量：前若干位是臂关节（rad），最后一位（若有）是夹爪，
  归一化到 [0,1]，**1 = 张开，0 = 闭合**（见 leader.S288LeaderArm）。
- 输出：`(arm_target_rad, gripper_width_m)`；夹爪宽度用米，张开 > 闭合。
- 所有限位/变化率都在这里强制，绝不依赖上层。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class JointMapping:
    """单个大臂关节的映射规则。"""

    src_index: int          # 对应 leader 状态向量里的下标
    sign: float = 1.0       # 方向 ±1
    offset: float = 0.0     # 零位偏置（rad）
    scale: float = 1.0      # 行程缩放（小臂 -> 大臂）
    lower: float = -np.inf  # 大臂关节下限（rad）
    upper: float = np.inf   # 大臂关节上限（rad）
    max_rate: float = np.inf  # 最大角速度（rad/s），inf = 不限

    def apply(self, leader_state: np.ndarray, extra_offset: float = 0.0) -> float:
        raw = float(leader_state[self.src_index])
        return self.offset + extra_offset + self.scale * self.sign * raw

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.lower, self.upper))


@dataclass(frozen=True)
class GripperMapping:
    """夹爪：leader 的 [0,1] -> 大臂手指宽度（米）。"""

    src_index: int = -1
    open_width_m: float = 0.075     # leader 张开(1) 对应的大臂宽度
    closed_width_m: float = 0.0     # leader 闭合(0) 对应的大臂宽度
    max_rate_m_s: float = np.inf    # 宽度变化率上限

    def apply(self, leader_state: np.ndarray) -> float:
        g = float(np.clip(leader_state[self.src_index], 0.0, 1.0))  # 1=张开
        w = self.closed_width_m + g * (self.open_width_m - self.closed_width_m)
        return float(np.clip(w, min(self.open_width_m, self.closed_width_m),
                             max(self.open_width_m, self.closed_width_m)))


@dataclass
class Retargeter:
    """把 leader 关节状态换算成大臂关节目标 + 夹爪宽度。"""

    joints: Sequence[JointMapping]
    gripper: Optional[GripperMapping] = None
    alpha: float = 0.9              # 输出指数平滑；1.0 = 不平滑
    ramp_s: float = 0.0             # 软启动时长；auto_align 后通常无需
    hold_last_on_stale: bool = True

    n_out: int = field(init=False)
    _align: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prev_arm: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prev_grip: Optional[float] = field(default=None, init=False, repr=False)
    _start_arm: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _start_grip: Optional[float] = field(default=None, init=False, repr=False)
    _ramp_elapsed: float = field(default=0.0, init=False, repr=False)
    _last_t: Optional[float] = field(default=None, init=False, repr=False)
    _last_ok: Optional[tuple[np.ndarray, float]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.joints = list(self.joints)
        self.n_out = len(self.joints)
        if self.n_out == 0:
            raise ValueError("Retargeter 至少需要一个关节映射")

    # -- 启动对齐：让"当前 leader 姿态"映射到"当前大臂姿态" ----------------
    # 与 gello teleop.mapping.auto_align 等价：上电瞬间目标 == 大臂当前位形，
    # 因此不会产生跳变，操作者从小臂当前姿态开始"接管"。
    def auto_align(
        self, leader_state: np.ndarray, follower_arm_state: Sequence[float]
    ) -> None:
        leader_state = np.asarray(leader_state, dtype=float)
        follower = np.asarray(follower_arm_state, dtype=float)
        if follower.shape != (self.n_out,):
            raise ValueError(f"大臂状态长度应为 {self.n_out}，得到 {follower.shape}")
        self._align = np.array(
            [
                follower[i] - j.apply(leader_state)
                for i, j in enumerate(self.joints)
            ],
            dtype=float,
        )

    # -- 初始化：用大臂当前位形做软启动起点 --------------------------------
    def reset(self, follower_arm_state: Sequence[float], follower_gripper_width: float) -> None:
        self._start_arm = np.asarray(follower_arm_state, dtype=float).copy()
        self._start_grip = float(follower_gripper_width)
        self._prev_arm = self._start_arm.copy()
        self._prev_grip = self._start_grip
        self._ramp_elapsed = 0.0
        self._last_t = None

    # -- 主换算 -------------------------------------------------------------
    def map(
        self, leader_state: np.ndarray, now: float
    ) -> Optional[tuple[np.ndarray, float]]:
        """换算一帧。leader_state 非法或陈旧时按策略返回 None。"""
        leader_state = np.asarray(leader_state, dtype=float)
        if not np.all(np.isfinite(leader_state)):
            return self._last_ok if self.hold_last_on_stale else None

        align = self._align if self._align is not None else np.zeros(self.n_out)
        desired = np.array(
            [j.apply(leader_state, align[i]) for i, j in enumerate(self.joints)],
            dtype=float,
        )
        grip = self.gripper.apply(leader_state) if self.gripper is not None else 0.0

        dt = 0.0 if self._last_t is None else max(now - self._last_t, 0.0)
        self._last_t = now

        # 1) 指数平滑（对大臂目标做，避免抖动传入电机）
        if self._prev_arm is not None and self.alpha < 1.0:
            desired = self._prev_arm * (1.0 - self.alpha) + desired * self.alpha
            if self.gripper is not None:
                grip = self._prev_grip * (1.0 - self.alpha) + grip * self.alpha  # type: ignore[operator]

        # 2) 软启动：从大臂初始位形平滑过渡到映射位形
        if self.ramp_s > 0 and self._start_arm is not None:
            self._ramp_elapsed += dt
            blend = min(self._ramp_elapsed / self.ramp_s, 1.0)
            desired = self._start_arm + blend * (desired - self._start_arm)
            if self.gripper is not None:
                grip = self._start_grip + blend * (grip - self._start_grip)  # type: ignore[operator]

        # 3) 变化率限幅（逐关节）
        if self._prev_arm is not None and dt > 1e-9:
            for i, j in enumerate(self.joints):
                if np.isfinite(j.max_rate):
                    step = j.max_rate * dt
                    desired[i] = np.clip(
                        desired[i], self._prev_arm[i] - step, self._prev_arm[i] + step
                    )
            if self.gripper is not None and np.isfinite(self.gripper.max_rate_m_s):
                step = self.gripper.max_rate_m_s * dt
                grip = float(np.clip(grip, self._prev_grip - step, self._prev_grip + step))  # type: ignore[operator]

        # 4) 关节限位裁剪（硬约束，最后一道）
        desired = np.array(
            [j.clamp(v) for j, v in zip(self.joints, desired)], dtype=float
        )

        self._prev_arm = desired
        self._prev_grip = grip
        self._last_ok = (desired, grip)
        return desired, grip
