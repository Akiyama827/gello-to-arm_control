"""大臂（arm_control）与小臂（GELLO/S288）之间的实时主从遥操作。

    leader（小臂/输入）  --换算-->  follower（大臂/被控）
        S288LeaderArm                 DryRunFollower
        GelloLeaderAdapter            DoraJogFollower
        FakeLeaderArm                 RtFollower / FakeFollower

核心模块：
    s288       宇树 S288 规格、MIT 协议编解码、总线、关节读取
    leader     小臂抽象与 S288 / gello / fake 实现
    mapping    关节空间换算（sign/offset/scale/限位/夹爪归一化/align）
    follower   大臂三种下发后端
    safety     碰撞预警停机 + "没按预期运行"停机
    loop       实时主循环
    dora_feedback  大臂回读（motor_state/health/controller_event）
    node       Dora 节点入口
    config     YAML 配置与装配

注意：`follower` 子模块与包同名，导入时请用 `from arm_control.leader_follower
import follower as follower_mod` 或直接用 `build_follower`。
"""
from __future__ import annotations

from .mapping import GripperMapping, JointMapping, Retargeter
from .loop import FollowerFeedback, TeleopLoop, TeleopStats
from .dora_feedback import DoraFollowerFeedback
from .safety import (
    CallableCollisionGuard,
    CollisionGuard,
    FeedbackSample,
    NoCollisionGuard,
    SafetyLimits,
    SafetyMonitor,
    SafetyStop,
    SafetyVerdict,
    Sphere,
    SphereCollisionGuard,
)

__all__ = [
    "GripperMapping",
    "JointMapping",
    "Retargeter",
    "TeleopLoop",
    "TeleopStats",
    "FollowerFeedback",
    "DoraFollowerFeedback",
    "CollisionGuard",
    "NoCollisionGuard",
    "CallableCollisionGuard",
    "SphereCollisionGuard",
    "Sphere",
    "SafetyLimits",
    "SafetyMonitor",
    "SafetyStop",
    "SafetyVerdict",
    "FeedbackSample",
]
