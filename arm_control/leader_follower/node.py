"""Dora 节点入口：在小臂串口与大臂 Dora 图之间跑实时遥操作。

进程内只有一个 Dora `Node`，它同时：

* **发** `jog` / `control` / `gripper`（`DoraJogFollower`）
* **收** `motor_state` / `motor_health` / `controller_event` / `gripper_state`
  （`DoraFollowerFeedback`）

启动前先 `prime()` 等到首帧实测位形，再用它做 `auto_align`——不能拿零点当
FR3 的对齐基准（`fr3_joint4` 的零位本身就不在限位内）。

用法（由 dataflow 拉起，一般不手动跑）::

    LEADER_FOLLOWER_CONFIG=examples/configs/leader_follower.yaml \\
        python nodes/leader_teleop.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from .config import config_from_yaml, build_pipeline
from .dora_feedback import DoraFollowerFeedback
from .loop import TeleopLoop
from .safety import CollisionGuard, NoCollisionGuard

DEFAULT_CONFIG = "examples/configs/leader_follower.yaml"


def build_dora_node(
    cfg,
    *,
    node,
    collision_guard: Optional[CollisionGuard] = None,
) -> tuple[TeleopLoop, DoraFollowerFeedback]:
    """把已解析的配置 + 一个 Dora `Node` 装配成 (loop, feedback)。"""
    leader, retargeter, follower, monitor = build_pipeline(
        cfg, collision_guard=collision_guard
    )

    feedback = DoraFollowerFeedback(node, cfg.follower.n_arm_joints)
    # 回读接入 follower：启动对齐用实测位形，`jog` 通道仍只管下发。
    follower.node = node
    follower.state_provider = feedback.latest_arm_state

    loop = TeleopLoop(
        leader=leader,
        retargeter=retargeter,
        follower=follower,
        monitor=monitor,
        feedback=feedback.sample,
        hz=cfg.loop.hz,
        auto_align=cfg.mapping.auto_align,
        log_period_s=cfg.loop.log_period_s,
    )
    return loop, feedback


def _make_node():
    from dora import Node

    return Node()


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="leader → FR3 实时遥操作 Dora 节点")
    parser.add_argument(
        "--config",
        default=os.environ.get("LEADER_FOLLOWER_CONFIG", DEFAULT_CONFIG),
        help="leader-follower YAML 路径",
    )
    parser.add_argument("--duration", type=float, default=None, help="运行时长（秒）")
    parser.add_argument(
        "--prime-timeout",
        type=float,
        default=float(os.environ.get("LEADER_FOLLOWER_PRIME_S", "10")),
        help="等待首帧 motor_state 的超时（秒）",
    )
    parser.add_argument(
        "--skip-prime",
        action="store_true",
        help="跳过启动回读等待（仅用于确认接线，真机不要用）",
    )
    args = parser.parse_args(argv)

    cfg = config_from_yaml(args.config)
    if getattr(cfg.follower, "kind", "") != "dora":
        print(
            f"[leader_teleop] 拒绝启动：follower.kind={cfg.follower.kind!r} 不是 'dora'。\n"
            "  本节点只通过 Dora 图下发 jog/control/gripper；请把部署配置里的\n"
            "  follower.kind 设为 dora，并用 LEADER_FOLLOWER_CONFIG 指向它。",
            file=sys.stderr,
        )
        return 2

    node = _make_node()
    loop, feedback = build_dora_node(cfg, node=node, collision_guard=NoCollisionGuard())
    if not args.skip_prime:
        feedback.prime(args.prime_timeout)
    loop.run(duration_s=args.duration if args.duration else cfg.loop.duration_s)
    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
