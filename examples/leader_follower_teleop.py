#!/usr/bin/env python3
"""主从遥操作可运行示例：小臂(leader) -> 大臂(follower)。

默认读 examples/configs/leader_follower.yaml，用 FakeLeader + FakeFollower
在不接任何硬件的情况下跑通全链路，方便先核对换算与安全阈值。

    python examples/leader_follower_teleop.py --duration 10
    python examples/leader_follower_teleop.py --collision-demo --duration 10

接真机时：
    * 改配置里的 leader.kind=s288 / follower.kind=dora|rt
    * 用 `--collision-guard mujoco` 之类的部署侧代码把 arm_control 的
      MuJoCoCollisionWorld 包进 CallableCollisionGuard（本示例用 --collision-demo
      展示一个玩具球体守卫，证明接线方式）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许从仓库根直接运行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_control.leader_follower import (  # noqa: E402
    FollowerFeedback,
    Sphere,
    SphereCollisionGuard,
    TeleopLoop,
)
from arm_control.leader_follower.config import (  # noqa: E402
    build_pipeline,
    config_from_yaml,
)


def _planar_fk(q, base, link_len: float, n_links: int) -> dict:
    """玩具平面 FK：每关节绕 z 转、连杆沿 x。仅用于演示碰撞守卫接线。"""
    import numpy as np

    poses = {}
    t = np.asarray(base, dtype=float)
    angle = 0.0
    for i in range(n_links):
        angle += float(q[i]) if i < len(q) else 0.0
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        poses[f"link{i}"] = M
        t = t + R @ np.array([link_len, 0.0, 0.0])
    return poses


def make_demo_collision_guard(n_arm: int, link_len: float = 0.30):
    """两臂各 6 球、沿 y 分开 0.6m 的玩具场景，演示两臂碰撞检测接线。"""
    big_spheres = [
        Sphere(frame=f"link{i}", offset=[link_len / 2, 0.0, 0.0], radius=0.05)
        for i in range(n_arm)
    ]
    small_spheres = list(big_spheres)
    return SphereCollisionGuard(
        big_spheres=big_spheres,
        small_spheres=small_spheres,
        big_fk=lambda q: _planar_fk(q, [0.0, 0.0, 0.0], link_len, n_arm),
        small_fk=lambda q: _planar_fk(q, [0.0, 0.6, 0.0], link_len, n_arm),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="小臂->大臂实时遥操作")
    parser.add_argument(
        "--config",
        default=str(ROOT / "examples" / "configs" / "leader_follower.yaml"),
        help="YAML 配置路径",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="运行秒数")
    parser.add_argument("--hz", type=float, default=None, help="覆盖控制频率")
    parser.add_argument(
        "--collision-demo",
        action="store_true",
        help="接一个玩具球体碰撞守卫（演示两臂碰撞停机接线）",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="不接大臂回读（会关闭跟踪误差门，仅用于排查）",
    )
    args = parser.parse_args()

    cfg = config_from_yaml(args.config)
    if args.hz:
        cfg.loop.hz = args.hz

    guard = make_demo_collision_guard(cfg.follower.n_arm_joints) if args.collision_demo else None
    leader, retargeter, follower, monitor = build_pipeline(cfg, collision_guard=guard)

    # dry_run / fake / rt 的后端都能 read_state；dora 的 jog 通道不回读，
    # 需要另接订阅 motor_state 的反馈实现。
    feedback = None
    if not args.no_feedback and cfg.follower.kind in ("dry_run", "fake", "rt"):
        feedback = FollowerFeedback(follower, armed=True)

    loop = TeleopLoop(
        leader=leader,
        retargeter=retargeter,
        follower=follower,
        monitor=monitor,
        feedback=feedback,
        hz=cfg.loop.hz,
        auto_align=cfg.mapping.auto_align,
        log_period_s=cfg.loop.log_period_s,
    )
    stats = loop.run(duration_s=args.duration)
    print(
        f"\n[示例结束] ticks={stats.ticks} sends={stats.sends} "
        f"安全停机={stats.safety_stops} 平均周期={stats.mean_dt_s * 1000:.2f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
