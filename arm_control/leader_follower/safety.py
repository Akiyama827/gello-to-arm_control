"""主从遥操作的安全层：碰撞预警/停机 + 大臂"没按预期运行"的停机。

安全目标（用户明确要求）
--------------------------------------------------------------------------
1. **小臂与大臂将要碰撞** -> 先限速，逼近到阈值内立即停机。
2. **大臂没按预期运行** -> 跟踪误差、反馈陈旧/丢失、控制器故障位、
   实测速度异常、目标跳变、关节越限、leader 数据非法/断流，任一触发即停机。

分层
--------------------------------------------------------------------------
- `CollisionGuard`：碰撞/最近距离的可插拔接口。提供
    * `NoCollisionGuard`      显式关闭（需要注释说明为什么）
    * `CallableCollisionGuard` 包装部署侧回调（可接 arm_control 的
      `MuJoCoCollisionWorld`：自碰撞 + 环境 + 把小臂作为场景 actor = 两臂碰撞）
    * `SphereCollisionGuard`  纯 numpy 的两臂球体近似，开箱可用
- `SafetyMonitor`：便宜的即时门（关节限位、跳变、跟踪误差、反馈健康），
  每 tick 都跑；昂贵的碰撞查询按 `collision_every_n` 降频。
- `SafetyStop`：触发时抛出，由 TeleopLoop 捕获并调用 follower.safe_stop()。

设计原则：**宁停勿撞**。任何"拿不准 / 反馈缺失 / 计算异常"都按停机处理，
不停机的情况必须显式配置。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class SafetyStop(RuntimeError):
    """安全门触发，必须停机。reason 会写进日志/操作台。"""

    def __init__(self, reason: str, *, source: str = "safety") -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.source = str(source)


# --------------------------------------------------------------------------- #
# 限值与判定
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SafetyLimits:
    """一组可由操作负责人逐条辩护的数字（沿用 jog 五道闸的风格）。"""

    # -- 关节/目标 --
    joint_margin_rad: float = 0.05        # 距硬限位留的余量
    max_step_rad: float = 0.30            # 单 tick 目标跳变上限（超出=异常）
    max_target_vel_rad_s: float = 2.0     # 目标角速度上限
    gripper_max_step_m: float = 0.02      # 单 tick 夹爪宽度跳变上限

    # -- 跟踪 / 反馈（"大臂没按预期运行"） --
    feedback_required: bool = True        # 没有回读就停机（遥操作必须回读）
    feedback_timeout_s: float = 0.30      # 反馈陈旧上限
    track_err_rad: float = 0.15           # 命令-实测位置误差阈值
    track_err_hold_s: float = 0.30        # 误差持续多久才停（滤瞬态）
    track_vel_rad_s: float = 3.0          # 实测关节速度异常上限
    leader_timeout_s: float = 0.30        # leader 断流上限

    # -- 碰撞 --
    collision_stop_m: float = 0.010       # 最近距离 <= 此值 -> 停机
    collision_warn_m: float = 0.050       # <= 此值 -> 限速
    collision_every_n: int = 1            # 每 n 个 tick 查一次碰撞（降频）
    collision_slow_scale: float = 0.25    # 预警区速度比例

    def __post_init__(self) -> None:
        for name in ("joint_margin_rad", "collision_stop_m", "collision_warn_m"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"safety.{name} 不能为负")
        for name in ("max_step_rad", "max_target_vel_rad_s", "feedback_timeout_s",
                     "track_err_rad", "track_err_hold_s", "leader_timeout_s"):
            if not (float(getattr(self, name)) > 0.0):
                raise ValueError(f"safety.{name} 必须为正")
        if self.collision_warn_m < self.collision_stop_m:
            raise ValueError("collision_warn_m 必须 >= collision_stop_m")
        if int(self.collision_every_n) < 1:
            raise ValueError("collision_every_n 必须 >= 1")


@dataclass(frozen=True)
class SafetyVerdict:
    ok: bool
    reason: str = ""
    speed_scale: float = 1.0

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class FeedbackSample:
    """大臂的一次回读。只有 `arm_q` 必填，其余尽力而为。"""

    arm_q: np.ndarray
    arm_dq: Optional[np.ndarray] = None
    gripper_width_m: Optional[float] = None
    timestamp: float = 0.0
    fault: bool = False
    fault_reason: str = ""
    armed: Optional[bool] = None


# --------------------------------------------------------------------------- #
# 碰撞接口
# --------------------------------------------------------------------------- #
class CollisionGuard(Protocol):
    def set_other_state(self, other_state: Sequence[float]) -> None:
        """更新"另一臂/动态障碍"的状态（例如 leader 关节角）。"""
        ...

    def min_distance(self, arm_q: Sequence[float]) -> float:
        """大臂处于 arm_q 时，与另一臂/环境的最近有符号距离（米）。

        负值表示已穿透；正值为间隙；`inf` 表示该守卫无法判断。
        """
        ...

    def close(self) -> None:
        ...


class NoCollisionGuard:
    """显式关闭碰撞检查。只在明确接受"无碰撞保护"时使用。"""

    def set_other_state(self, other_state: Sequence[float]) -> None:
        pass

    def min_distance(self, arm_q: Sequence[float]) -> float:
        return float("inf")

    def close(self) -> None:
        pass


class CallableCollisionGuard:
    """包装部署侧回调，接 arm_control 的 MuJoCo 碰撞世界最省事。

    `fn(arm_q)` 可返回：
      * float —— 最近有符号距离（米）
      * bool  —— True 表示会碰撞（内部折算成 -1.0 米）
    `set_other_state` 会把 leader 状态转交给可选的 `on_other` 回调，部署侧用它
    把小臂摆到当前位形（例如写进 `MuJoCoCollisionWorld` 的场景 actor_q）。
    """

    def __init__(
        self,
        fn: Callable[[np.ndarray], float | bool],
        *,
        on_other: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._fn = fn
        self._on_other = on_other

    def set_other_state(self, other_state: Sequence[float]) -> None:
        if self._on_other is not None:
            self._on_other(np.asarray(other_state, dtype=float))

    def min_distance(self, arm_q: Sequence[float]) -> float:
        result = self._fn(np.asarray(arm_q, dtype=float))
        if isinstance(result, (bool, np.bool_)):
            return -1.0 if result else float("inf")
        return float(result)

    def close(self) -> None:
        pass


def _as_rt(pose) -> tuple[np.ndarray, np.ndarray]:
    """把 FK 结果统一成 (R(3x3), t(3))。支持 4x4 或 (R,t) 或点(3,)。"""
    p = np.asarray(pose, dtype=float)
    if p.shape == (4, 4):
        return p[:3, :3], p[:3, 3]
    if p.shape == (3, 3):
        return p, np.zeros(3)
    if p.shape == (3,):
        return np.eye(3), p
    raise ValueError(f"无法解析位姿 shape={p.shape}")


@dataclass(frozen=True)
class Sphere:
    """附着在某 FK 坐标系上的碰撞球。"""

    frame: str
    offset: Sequence[float]
    radius: float

    def center(self, poses: dict) -> np.ndarray:
        if self.frame not in poses:
            raise KeyError(f"FK 结果缺少坐标系 {self.frame!r}")
        _, t = _as_rt(poses[self.frame])
        return t + np.asarray(self.offset, dtype=float)


@dataclass
class SphereCollisionGuard:
    """两臂球形近似的最近距离，纯 numpy、开箱可用。

    需要两部 FK：
      * `big_fk(arm_q) -> {frame: 4x4|(R,t)}`      大臂正运动学
      * `small_fk(leader_state) -> {frame: ...}`    小臂正运动学（含基座位姿）

    球形近似是保守的（球一定包住连杆），因此宁可早停，不会漏撞。真正高保真的
    做法是接 `CallableCollisionGuard` + arm_control 的 MuJoCo 世界。
    """

    big_spheres: Sequence[Sphere]
    small_spheres: Sequence[Sphere]
    big_fk: Callable[[Sequence[float]], dict]
    small_fk: Callable[[Sequence[float]], dict]

    _small_poses: Optional[dict] = field(default=None, init=False, repr=False)

    def set_other_state(self, other_state: Sequence[float]) -> None:
        self._small_poses = self.small_fk(np.asarray(other_state, dtype=float))

    def min_distance(self, arm_q: Sequence[float]) -> float:
        if self._small_poses is None:
            return float("inf")  # 还没收到小臂状态，无法判断
        big_poses = self.big_fk(np.asarray(arm_q, dtype=float))
        best = float("inf")
        for b in self.big_spheres:
            cb = b.center(big_poses)
            for s in self.small_spheres:
                cs = s.center(self._small_poses)
                gap = float(np.linalg.norm(cb - cs) - b.radius - s.radius)
                best = min(best, gap)
        return best

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# 监视器
# --------------------------------------------------------------------------- #
@dataclass
class SafetyMonitor:
    """每 tick 的即时安全门。任何一项不通过都应停机。"""

    limits: SafetyLimits
    joint_lower: Sequence[float]
    joint_upper: Sequence[float]
    collision: CollisionGuard = field(default_factory=NoCollisionGuard)

    _n: int = field(init=False)
    _prev_target: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prev_leader_t: float = field(default=0.0, init=False, repr=False)
    _track_since: Optional[float] = field(default=None, init=False, repr=False)
    _last_feedback_t: float = field(default=0.0, init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)
    _last_distance: float = field(default=float("inf"), init=False, repr=False)
    _seen_feedback: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        lo = np.asarray(self.joint_lower, dtype=float).ravel()
        hi = np.asarray(self.joint_upper, dtype=float).ravel()
        if lo.shape != hi.shape or lo.size == 0:
            raise ValueError("joint_lower/upper 形状不一致或为空")
        if not np.all(lo <= hi):
            raise ValueError("joint_lower 必须 <= joint_upper")
        self.joint_lower = lo
        self.joint_upper = hi
        self._n = int(lo.size)

    # -- leader -------------------------------------------------------------
    def check_leader(self, leader_state: np.ndarray, now: float) -> SafetyVerdict:
        arr = np.asarray(leader_state, dtype=float).ravel()
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return SafetyVerdict(False, "leader 数据非法（NaN/inf/空）")
        if self._prev_leader_t > 0.0:
            gap = now - self._prev_leader_t
            if gap > self.limits.leader_timeout_s:
                return SafetyVerdict(
                    False, f"leader 断流 {gap:.2f}s（阈值 {self.limits.leader_timeout_s:.2f}s）"
                )
        self._prev_leader_t = now
        return SafetyVerdict(True)

    # -- 目标（下发前） -----------------------------------------------------
    def check_and_clamp_target(
        self, target: np.ndarray, now: float
    ) -> tuple[SafetyVerdict, np.ndarray]:
        """关节限位 + 跳变 + 碰撞；返回（判定, 已限速/裁剪的目标）。"""
        tgt = np.asarray(target, dtype=float).ravel()
        if tgt.size != self._n or not np.all(np.isfinite(tgt)):
            return SafetyVerdict(False, f"目标非法：shape={tgt.shape}/非有限"), tgt

        # 1) 关节限位（含余量）
        lo = self.joint_lower + self.limits.joint_margin_rad
        hi = self.joint_upper - self.limits.joint_margin_rad
        over = np.where((tgt < lo) | (tgt > hi))[0]
        if over.size:
            i = int(over[0])
            return (
                SafetyVerdict(
                    False,
                    f"关节越限：joint{i}={tgt[i]:.3f} rad 不在 "
                    f"[{lo[i]:.3f}, {hi[i]:.3f}]",
                ),
                tgt,
            )

        # 2) 跳变检测（相对上一次目标）
        if self._prev_target is not None:
            step = np.abs(tgt - self._prev_target)
            bad = np.where(step > self.limits.max_step_rad)[0]
            if bad.size:
                i = int(bad[0])
                return (
                    SafetyVerdict(
                        False,
                        f"目标跳变：joint{i} 单步变化 {step[i]:.3f} rad "
                        f"超过 {self.limits.max_step_rad:.3f} rad",
                    ),
                    tgt,
                )

        # 3) 碰撞（降频；预警区限速）
        speed_scale = 1.0
        self._tick += 1
        if self._tick % self.limits.collision_every_n == 0:
            self._last_distance = self.collision.min_distance(tgt)
        d = self._last_distance
        if d <= self.limits.collision_stop_m:
            return (
                SafetyVerdict(
                    False, f"碰撞危险：两臂最近距离 {d * 1000:.1f} mm "
                    f"<= {self.limits.collision_stop_m * 1000:.1f} mm"
                ),
                tgt,
            )
        if d <= self.limits.collision_warn_m:
            speed_scale = self.limits.collision_slow_scale

        self._prev_target = tgt.copy()
        return SafetyVerdict(True, speed_scale=speed_scale), tgt

    # -- 反馈（下发后） -----------------------------------------------------
    def check_feedback(
        self, sample: Optional[FeedbackSample], now: float
    ) -> SafetyVerdict:
        if sample is None:
            if self.limits.feedback_required:
                if not self._seen_feedback:
                    return SafetyVerdict(False, "未收到大臂反馈（feedback_required=true）")
                gap = now - self._last_feedback_t
                if gap > self.limits.feedback_timeout_s:
                    return SafetyVerdict(
                        False, f"大臂反馈陈旧 {gap:.2f}s"
                    )
            return SafetyVerdict(True)
        self._seen_feedback = True
        # 用样本自带的到达时刻判新鲜度，而不是"这一 tick 调用了就算新鲜"：
        # 否则调用方每 tick 递回同一个缓存样本，反馈陈旧门永远不会触发。
        t = float(sample.timestamp) if sample.timestamp else now
        self._last_feedback_t = t
        if now - t > self.limits.feedback_timeout_s:
            return SafetyVerdict(
                False,
                f"大臂反馈陈旧 {now - t:.2f}s（阈值 {self.limits.feedback_timeout_s:.2f}s）",
            )

        q = np.asarray(sample.arm_q, dtype=float).ravel()
        if q.size != self._n or not np.all(np.isfinite(q)):
            return SafetyVerdict(False, "大臂反馈非法（NaN/inf/长度不符）")

        if sample.fault:
            return SafetyVerdict(False, f"控制器故障：{sample.fault_reason or 'unknown'}")
        if sample.armed is False:
            return SafetyVerdict(False, "大臂已失能（plant 报告 disarmed）")

        if sample.arm_dq is not None:
            dq = np.abs(np.asarray(sample.arm_dq, dtype=float).ravel())
            if dq.size == self._n and np.all(np.isfinite(dq)):
                if float(dq.max()) > self.limits.track_vel_rad_s:
                    i = int(np.argmax(dq))
                    return SafetyVerdict(
                        False,
                        f"大臂速度异常：joint{i} |dq|={dq[i]:.2f} rad/s "
                        f"> {self.limits.track_vel_rad_s:.2f}",
                    )

        # 跟踪误差：命令(上一 tick 目标) vs 实测，需持续超阈才停。
        if self._prev_target is not None:
            err = float(np.max(np.abs(self._prev_target - q)))
            if err > self.limits.track_err_rad:
                if self._track_since is None:
                    self._track_since = now
                elif now - self._track_since > self.limits.track_err_hold_s:
                    i = int(np.argmax(np.abs(self._prev_target - q)))
                    return SafetyVerdict(
                        False,
                        f"跟踪误差过大：joint{i} 偏差 {err:.3f} rad 持续 "
                        f"{now - self._track_since:.2f}s（阈值 "
                        f"{self.limits.track_err_rad:.3f} rad / "
                        f"{self.limits.track_err_hold_s:.2f}s）",
                    )
            else:
                self._track_since = None
        return SafetyVerdict(True)

    @property
    def last_collision_distance(self) -> float:
        return self._last_distance

    def close(self) -> None:
        self.collision.close()
