"""主从遥操作的实时主循环与运行入口。

一次 tick 的顺序（每一步都可能触发安全停机）：

    1. 读小臂           leader.get_joint_state()
    2. leader 安全门     合法性 / 断流
    3. 换算             retargeter.map()  -> (大臂关节目标, 夹爪宽度)
    4. 碰撞守卫更新小臂    collision.set_other_state(leader_state)
    5. 目标安全门        关节限位 / 跳变 / 两臂碰撞（含预警限速）
    6. 下发大臂           follower.send()
    7. 回读大臂           feedback()
    8. 反馈安全门        跟踪误差 / 反馈陈旧 / 故障位 / 异常速度
    9. 按 hz 睡到下一个 deadline

任一安全门不通过 -> 抛 `SafetyStop` -> 立即 `follower.safe_stop()`。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .leader import LeaderArm
from .mapping import Retargeter
from .follower import FollowerArm
from .safety import FeedbackSample, SafetyMonitor, SafetyStop


@dataclass
class TeleopStats:
    ticks: int = 0
    sends: int = 0
    safety_stops: int = 0
    mean_dt_s: float = 0.0
    max_dt_s: float = 0.0
    last_leader_age_s: float = 0.0
    last_collision_distance_m: float = float("inf")


@dataclass
class TeleopLoop:
    leader: LeaderArm
    retargeter: Retargeter
    follower: FollowerArm
    monitor: SafetyMonitor
    feedback: Optional[Callable[[], Optional[FeedbackSample]]] = None
    hz: float = 100.0
    auto_align: bool = True
    log_period_s: float = 1.0
    verbose: bool = True

    _stop: bool = field(default=False, init=False)
    stats: TeleopStats = field(default_factory=TeleopStats, init=False)
    _last_log_t: float = field(default=0.0, init=False)

    # -- 控制 ---------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop = True

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # -- 反馈 ---------------------------------------------------------------
    def _read_feedback(self, now: float) -> Optional[FeedbackSample]:
        if self.feedback is None:
            return None
        try:
            return self.feedback()
        except Exception as exc:  # 回读异常本身就是"没按预期运行"
            n = getattr(self.follower, "num_arm_joints", 0)
            return FeedbackSample(
                arm_q=np.full(n, np.nan, dtype=float),
                timestamp=now,
                fault=True,
                fault_reason=f"反馈读取异常：{exc}",
            )

    # -- 启动对齐 -----------------------------------------------------------
    def _startup_align(self) -> None:
        leader_state = self.leader.get_joint_state()
        follower_arm, follower_grip = self.follower.read_state()
        if self.auto_align:
            self.retargeter.auto_align(leader_state, follower_arm)
            self._log(
                "[teleop] 已 auto_align：当前大臂位形对齐当前小臂姿态，无跳变接管"
            )
        self.retargeter.reset(follower_arm, follower_grip)

    # -- 单步（可单独测试） -------------------------------------------------
    def once(self, now: float) -> None:
        leader_state = self.leader.get_joint_state()

        verdict = self.monitor.check_leader(leader_state, now)
        if not verdict:
            raise SafetyStop(verdict.reason, source="leader")

        mapped = self.retargeter.map(leader_state, now)
        if mapped is None:
            raise SafetyStop("retarget 未产生目标（leader 数据非法）", source="retarget")
        arm_target, gripper_width = mapped

        self.monitor.collision.set_other_state(leader_state)

        verdict, arm_target = self.monitor.check_and_clamp_target(arm_target, now)
        if not verdict:
            raise SafetyStop(verdict.reason, source="target")

        # 预警区限速：把这一步的变化再按 speed_scale 收缩（在 retargeter 变化率
        # 限幅之外的第二道，专门应对"正在靠近碰撞"）。
        if verdict.speed_scale < 1.0 and self.retargeter._prev_arm is not None:  # type: ignore[attr-defined]
            prev = self.retargeter._prev_arm  # type: ignore[attr-defined]
            arm_target = prev + (arm_target - prev) * verdict.speed_scale
            self.retargeter._prev_arm = arm_target  # type: ignore[attr-defined]

        self.follower.send(arm_target, gripper_width)
        self.stats.sends += 1

        sample = self._read_feedback(now)
        verdict = self.monitor.check_feedback(sample, now)
        if not verdict:
            raise SafetyStop(verdict.reason, source="feedback")
        self.stats.ticks += 1

    # -- 主循环 -------------------------------------------------------------
    def run(self, duration_s: Optional[float] = None) -> TeleopStats:
        period = 1.0 / float(self.hz) if self.hz > 0 else 0.0
        self.follower.open()
        self._startup_align()
        self._log(f"[teleop] 启动：{self.hz:.0f} Hz，Ctrl-C 停止")

        start_t = time.monotonic()
        next_t = start_t
        prev_tick_t = start_t
        try:
            while not self._stop:
                now = time.monotonic()
                if duration_s is not None and now - start_t >= duration_s:
                    self._log(f"[teleop] 到达 duration={duration_s}s，正常结束")
                    break
                try:
                    self.once(now)
                except SafetyStop as stop:
                    self.stats.safety_stops += 1
                    self._log(
                        f"\n[teleop][安全停机] source={stop.source} 原因：{stop.reason}"
                    )
                    self.follower.safe_stop()
                    return self.stats

                # 统计与限频日志
                if self.stats.ticks > 1:
                    dt = now - prev_tick_t
                    self.stats.mean_dt_s += (dt - self.stats.mean_dt_s) / self.stats.ticks
                    self.stats.max_dt_s = max(self.stats.max_dt_s, dt)
                prev_tick_t = now
                self.stats.last_collision_distance_m = self.monitor.last_collision_distance
                if self.verbose and now - self._last_log_t >= self.log_period_s:
                    self._last_log_t = now
                    rate = 1.0 / self.stats.mean_dt_s if self.stats.mean_dt_s > 0 else 0.0
                    d = self.stats.last_collision_distance_m
                    d_txt = f"{d * 1000:.0f}mm" if np.isfinite(d) else "n/a"
                    self._log(
                        f"[teleop] tick={self.stats.ticks} 实际≈{rate:.1f}Hz "
                        f"最大间隔={self.stats.max_dt_s * 1000:.1f}ms 最近距离={d_txt}"
                    )

                if period > 0.0:
                    next_t += period
                    sleep_s = next_t - time.monotonic()
                    if sleep_s < -period:
                        next_t = time.monotonic()  # 落后太多则重新对表
                    elif sleep_s > 0:
                        time.sleep(sleep_s)
        except KeyboardInterrupt:
            self._log("\n[teleop] 收到 Ctrl-C，安全停机")
            self.follower.safe_stop()
        finally:
            self.follower.close()
            self.leader.close()
            self.monitor.close()
        return self.stats


# --------------------------------------------------------------------------- #
# 便捷反馈源
# --------------------------------------------------------------------------- #
class FollowerFeedback:
    """从 follower.read_state() 构造 FeedbackSample（dry_run / fake / rt 可用）。

    对 DoraJogFollower 这类 jog 通道不回读的后端，应改用订阅 motor_state 的
    实现，否则跟踪误差门会误报。
    """

    def __init__(self, follower: FollowerArm, armed: Optional[bool] = None) -> None:
        self._follower = follower
        self._armed = armed

    def __call__(self) -> Optional[FeedbackSample]:
        arm_q, grip = self._follower.read_state()
        return FeedbackSample(
            arm_q=np.asarray(arm_q, dtype=float),
            gripper_width_m=float(grip),
            timestamp=time.monotonic(),
            armed=self._armed,
        )
