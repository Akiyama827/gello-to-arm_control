"""leader-follower 遥操作的配置加载与装配。

一份 YAML 描述四件事：小臂怎么读、大臂怎么下发、怎么换算、安全阈值多少。
`build_pipeline()` 把它们装配成可直接交给 `TeleopLoop` 的对象。

碰撞守卫（CollisionGuard）依赖运动学/几何，无法写进 YAML，因此由调用方通过
`collision_guard=` 注入；不注入就是 `NoCollisionGuard`（显式关闭）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import follower as follower_mod
from . import leader as leader_mod
from .mapping import GripperMapping, JointMapping, Retargeter
from .safety import (
    CollisionGuard,
    NoCollisionGuard,
    SafetyLimits,
    SafetyMonitor,
)
from .s288 import (
    FakeS288Bus,
    S288Codec,
    S288JointChain,
    S288Spec,
    SerialS288Bus,
    UnitreeSdkS288Bus,
)


# --------------------------------------------------------------------------- #
# 配置数据类
# --------------------------------------------------------------------------- #
@dataclass
class LeaderConfig:
    kind: str = "fake"                 # s288 | gello | fake
    n_arm_joints: int = 7              # 默认按 8 个 S288 = 7 臂关节 + 1 夹爪
    with_gripper: bool = True
    # s288
    bus: str = "unitree_sdk"           # unitree_sdk | serial_raw | fake
    motor_type: str = "S288"           # 官方 SDK MotorType 枚举名
    port: str = "/dev/ttyUSB0"
    motor_ids: Sequence[int] = ()
    gripper_index: Optional[int] = None
    joint_offsets: Sequence[float] = ()
    joint_signs: Sequence[float] = ()
    gripper_open_rad: float = 0.0
    gripper_close_rad: float = 0.0
    alpha: float = 0.99
    start_joints: Sequence[float] = ()
    use_fake_bus: bool = False         # 兼容旧字段：等价于 bus=fake
    # fake
    fake_amplitude: float = 0.4
    fake_period_s: float = 6.0


@dataclass
class FollowerConfig:
    kind: str = "dry_run"              # dry_run | fake | dora | rt
    n_arm_joints: int = 7              # FR3：7 关节 + Franka Hand
    # 仿真/空跑后端的初始位形（让 auto_align 从一个合法位形起步）
    initial: Sequence[float] = ()
    initial_finger_m: float = 0.0
    # rt
    joint_names: Sequence[str] = ()
    host: str = "127.0.0.1"
    udp_port: int = 47800
    tcp_port: int = 47801
    gripper_slot: bool = False
    kp: Sequence[float] = ()
    kd: Sequence[float] = ()
    # dora：open() 时是否自动发 control(arm=True)。真机上若希望保留操作台
    # 的 ARM/DISARM 门，设为 False，由操作台显式使能。
    auto_arm: bool = True


@dataclass
class MappingConfig:
    auto_align: bool = True
    alpha: float = 0.9
    ramp_s: float = 0.0
    joints: Sequence[JointMapping] = ()
    gripper: Optional[GripperMapping] = None


@dataclass
class LoopConfig:
    hz: float = 100.0
    log_period_s: float = 1.0
    duration_s: Optional[float] = None


@dataclass
class SafetyConfig:
    limits: SafetyLimits = field(default_factory=SafetyLimits)
    joint_lower: Sequence[float] = ()
    joint_upper: Sequence[float] = ()


@dataclass
class LeaderFollowerConfig:
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    follower: FollowerConfig = field(default_factory=FollowerConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
FR3_JOINTS: tuple[str, ...] = (
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
)


def _joint_mapping(raw: dict, default_src: int) -> JointMapping:
    return JointMapping(
        src_index=int(raw.get("src", default_src)),
        sign=float(raw.get("sign", 1.0)),
        offset=float(raw.get("offset", 0.0)),
        scale=float(raw.get("scale", 1.0)),
        lower=float(raw.get("lower", -np.inf)),
        upper=float(raw.get("upper", np.inf)),
        max_rate=float(raw.get("max_rate", np.inf)),
    )


def _default_joint_mappings(n: int) -> list[JointMapping]:
    return [JointMapping(src_index=i) for i in range(n)]


def config_from_dict(raw: dict) -> LeaderFollowerConfig:
    raw = raw or {}
    lead_raw = dict(raw.get("leader") or {})
    foll_raw = dict(raw.get("follower") or {})
    map_raw = dict(raw.get("mapping") or {})
    loop_raw = dict(raw.get("loop") or {})
    safe_raw = dict(raw.get("safety") or {})

    leader = LeaderConfig(**{
        k: v for k, v in lead_raw.items() if k in LeaderConfig.__dataclass_fields__
    })
    follower = FollowerConfig(**{
        k: v for k, v in foll_raw.items() if k in FollowerConfig.__dataclass_fields__
    })

    joints_raw = list(map_raw.pop("joints", []) or [])
    if joints_raw:
        joints = [
            m if isinstance(m, JointMapping) else _joint_mapping(m, i)
            for i, m in enumerate(joints_raw)
        ]
    else:
        joints = _default_joint_mappings(int(map_raw.pop("n_arm_joints", follower.n_arm_joints)))

    grip_raw = map_raw.pop("gripper", None)
    gripper = None
    if grip_raw is not None:
        gripper = GripperMapping(
            src_index=int(grip_raw.get("src", -1)),
            open_finger_m=float(
                grip_raw.get("open_finger_m", grip_raw.get("open_width_m", 0.04))
            ),
            closed_finger_m=float(
                grip_raw.get("closed_finger_m", grip_raw.get("closed_width_m", 0.0))
            ),
            max_rate_m_s=float(grip_raw.get("max_rate_m_s", np.inf)),
        )
    mapping = MappingConfig(
        auto_align=bool(map_raw.get("auto_align", True)),
        alpha=float(map_raw.get("alpha", 0.9)),
        ramp_s=float(map_raw.get("ramp_s", 0.0)),
        joints=joints,
        gripper=gripper,
    )

    limits_raw = dict(safe_raw.get("limits") or {})
    limits_fields = SafetyLimits.__dataclass_fields__
    limits = SafetyLimits(**{k: v for k, v in limits_raw.items() if k in limits_fields})

    safety = SafetyConfig(
        limits=limits,
        joint_lower=list(safe_raw.get("joint_lower", []) or []),
        joint_upper=list(safe_raw.get("joint_upper", []) or []),
    )
    loop = LoopConfig(**{k: v for k, v in loop_raw.items() if k in LoopConfig.__dataclass_fields__})

    cfg = LeaderFollowerConfig(
        leader=leader, follower=follower, mapping=mapping, loop=loop, safety=safety
    )
    _validate(cfg)
    return cfg


def config_from_yaml(path: str | Path) -> LeaderFollowerConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return config_from_dict(yaml.safe_load(handle) or {})


def _validate(cfg: LeaderFollowerConfig) -> None:
    n_arm = int(cfg.follower.n_arm_joints)
    if len(cfg.mapping.joints) != n_arm:
        raise ValueError(
            f"mapping.joints 数量 {len(cfg.mapping.joints)} 与大臂关节数 {n_arm} 不一致"
        )


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #
def build_leader(cfg: LeaderConfig):
    if cfg.kind == "fake":
        return leader_mod.FakeLeaderArm(
            n_arm_joints=cfg.n_arm_joints,
            with_gripper=cfg.with_gripper,
            amplitude=cfg.fake_amplitude,
            period_s=cfg.fake_period_s,
        )
    if cfg.kind == "s288":
        ids = list(cfg.motor_ids)
        if not ids:
            raise ValueError("leader.motor_ids 不能为空（s288）")
        # 8 个 S288 时总 dof 应与配置一致，早报错好过把关节顺序搞错。
        expected = cfg.n_arm_joints + (1 if cfg.with_gripper else 0)
        if len(ids) != expected:
            raise ValueError(
                f"leader.motor_ids 有 {len(ids)} 个，与 n_arm_joints={cfg.n_arm_joints}"
                f" + 夹爪={cfg.with_gripper}（应为 {expected}）不一致"
            )
        spec = S288Spec()
        bus_kind = "fake" if cfg.use_fake_bus else cfg.bus
        if bus_kind == "fake":
            bus = FakeS288Bus(ids, spec=spec)
        elif bus_kind == "serial_raw":
            bus = SerialS288Bus(ids, port=cfg.port, spec=spec, codec=S288Codec())
        elif bus_kind == "unitree_sdk":
            # 官方 unitree_actuator_sdk：串口通信，转子侧 q 在总线内换算。
            bus = UnitreeSdkS288Bus(
                ids, port=cfg.port, spec=spec, motor_type=cfg.motor_type
            )
        else:
            raise ValueError(
                f"未知 leader.bus={bus_kind!r}（unitree_sdk | serial_raw | fake）"
            )
        chain = S288JointChain(motor_ids=ids, bus=bus, spec=spec, codec=S288Codec())
        return leader_mod.S288LeaderArm(
            chain=chain,
            joint_offsets=cfg.joint_offsets,
            joint_signs=cfg.joint_signs,
            gripper_index=cfg.gripper_index,
            gripper_open_rad=cfg.gripper_open_rad,
            gripper_close_rad=cfg.gripper_close_rad,
            alpha=cfg.alpha,
            start_joints=cfg.start_joints or None,
        )
    if cfg.kind == "gello":
        raise ValueError(
            "leader.kind=gello 需要在部署侧注入已构造的 gello Robot，"
            "用 leader.GelloLeaderAdapter(robot) 包一层后传给 TeleopLoop"
        )
    raise ValueError(f"未知 leader.kind: {cfg.kind!r}")


def build_follower(cfg: FollowerConfig):
    initial = list(cfg.initial) or None
    if cfg.kind == "dry_run":
        return follower_mod.DryRunFollower(
            num_arm_joints=cfg.n_arm_joints,
            initial_q=initial,
            initial_finger_m=cfg.initial_finger_m,
        )
    if cfg.kind == "fake":
        return follower_mod.FakeFollower(
            num_arm_joints=cfg.n_arm_joints,
            initial_q=initial,
            initial_finger_m=cfg.initial_finger_m,
        )
    if cfg.kind == "dora":
        return follower_mod.DoraJogFollower(
            num_arm_joints=cfg.n_arm_joints, auto_arm=cfg.auto_arm
        )
    if cfg.kind == "rt":
        names = list(cfg.joint_names) or list(FR3_JOINTS[: cfg.n_arm_joints])
        return follower_mod.RtFollower(
            joint_names=names,
            host=cfg.host,
            udp_port=cfg.udp_port,
            tcp_port=cfg.tcp_port,
            kp=list(cfg.kp) or None,
            kd=list(cfg.kd) or None,
            gripper_slot=cfg.gripper_slot,
        )
    raise ValueError(f"未知 follower.kind: {cfg.kind!r}")


def build_retargeter(cfg: MappingConfig) -> Retargeter:
    return Retargeter(
        joints=cfg.joints,
        gripper=cfg.gripper,
        alpha=cfg.alpha,
        ramp_s=cfg.ramp_s,
    )


def build_monitor(
    cfg: SafetyConfig,
    joints: Sequence[JointMapping],
    collision_guard: Optional[CollisionGuard] = None,
) -> SafetyMonitor:
    n = len(joints)
    lower = list(cfg.joint_lower) or [j.lower for j in joints]
    upper = list(cfg.joint_upper) or [j.upper for j in joints]
    lower = [(-np.inf if v == -np.inf else float(v)) for v in lower]
    upper = [(np.inf if v == np.inf else float(v)) for v in upper]
    return SafetyMonitor(
        limits=cfg.limits,
        joint_lower=lower,
        joint_upper=upper,
        collision=collision_guard or NoCollisionGuard(),
    )


def build_pipeline(cfg: LeaderFollowerConfig, *, collision_guard: Optional[CollisionGuard] = None):
    """返回 (leader, retargeter, follower, monitor)。"""
    return (
        build_leader(cfg.leader),
        build_retargeter(cfg.mapping),
        build_follower(cfg.follower),
        build_monitor(cfg.safety, cfg.mapping.joints, collision_guard),
    )
