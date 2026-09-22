#!/usr/bin/env python3
"""直接驱动真实 FR3 模型：**不经过小臂、不做映射、不走遥操作回路**。

用途：单独验证"程序到底能不能命令 FR3（尤其是夹爪）动"。

本脚本把 FR3 的 7 个臂关节和 Franka Hand 的双指按脚本信号（正弦 / 开合）
直接写进 MuJoCo 的 ``qpos``，实时显示。全程：

* 不读小臂、不读串口；
* 不做 ``Retargeter`` 换算、没有 auto_align；
* 没有 ``TeleopLoop`` / 安全门。

因此这是一个**纯下发/显示链路**的自证：

* 窗口里 FR3 臂 + 手指按预期动 → 命令与显示这一段没问题，之前的"夹爪不动"
  只可能出在**输入（小臂读数）**或**映射被关掉**上；
* 某个量始终不动 → 问题就在这一段，和传感器无关。

真机呢？
------
本脚本只碰 **MuJoCo 里的 FR3 模型**（安全、无需硬件）。要在**真 FR3** 上做同样的
"不经小臂"直控，把遥操作配置里的 ``follower.kind`` 设为 ``dora``（或 ``rt``），
再跑 ``examples/leader_follower_teleop.py``；或者用下面的 `--gripper-cmd` 生成
单指米制指令喂给 ``franka_gripper`` 节点。

运行（fish）：
    source .venv/bin/activate.fish
    PYTHONPATH=. python -B examples/franka_direct_demo.py            # 臂 + 夹爪都动
    PYTHONPATH=. python -B examples/franka_direct_demo.py --gripper-only
    PYTHONPATH=. python -B examples/franka_direct_demo.py --headless --duration 3
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_control.simulation.mujoco_model import build_mujoco_model  # noqa: E402

FR3_URDF = ROOT / "franka" / "urdf" / "fr3.urdf"
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
FINGER_MAX_M = 0.04
# 一个合法的待机位形（与示例配置一致）。
FR3_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


def _adr(model, joint_name: str) -> int:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"模型里没有关节 {joint_name!r}")
    return int(model.jnt_qposadr[jid])


def _arm_target(home: np.ndarray, t: float, amp: float, period: float) -> np.ndarray:
    """每个关节一个固定相位的慢正弦，幅度小、不同轴错开，便于肉眼确认。"""
    n = home.size
    phase = np.arange(n, dtype=float) * 0.6
    return home + amp * np.sin(2 * np.pi * t / period + phase)


def _finger_target(t: float, period: float) -> float:
    """0 <-> FINGER_MAX_M 的平滑开合（单指位移，米）。"""
    return 0.5 * FINGER_MAX_M * (1.0 - np.cos(2 * np.pi * t / period))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="直接驱动真实 FR3 模型（不经小臂 / 映射 / 遥操作）"
    )
    parser.add_argument("--gripper-only", action="store_true",
                        help="臂保持 home，只让夹爪开合（聚焦夹爪）")
    parser.add_argument("--amp", type=float, default=0.25, help="臂关节正弦幅度（rad）")
    parser.add_argument("--period", type=float, default=6.0, help="臂正弦周期（秒）")
    parser.add_argument("--gripper-period", type=float, default=4.0, help="夹爪开合周期（秒）")
    parser.add_argument("--duration", type=float, default=None, help="运行时长（秒）")
    parser.add_argument("--headless", action="store_true", help="不开窗口，仅自检并打印范围")
    args = parser.parse_args(argv)

    import mujoco

    cache = Path(tempfile.gettempdir()) / "arm_control_fr3_mjcache"
    staged = build_mujoco_model(FR3_URDF, cache_dir=cache, keep_visual=True)
    model = mujoco.MjModel.from_xml_path(str(staged))
    data = mujoco.MjData(model)

    arm_adr = [_adr(model, n) for n in ARM_JOINTS]
    finger_adr = [_adr(model, n) for n in FINGER_JOINTS]

    home = np.asarray(FR3_HOME, dtype=float)
    for adr, value in zip(arm_adr, home):
        data.qpos[adr] = float(value)
    for adr in finger_adr:
        data.qpos[adr] = 0.0
    mujoco.mj_forward(model, data)

    print(
        f"[direct] 已加载 FR3：{len(arm_adr)} 臂关节 + {len(finger_adr)} 指；"
        f"模式={'仅夹爪' if args.gripper_only else '臂+夹爪'}",
        flush=True,
    )

    t0 = time.monotonic()
    finger_lo, finger_hi = np.inf, -np.inf
    arm_lo = np.full(home.size, np.inf)
    arm_hi = np.full(home.size, -np.inf)

    def step(t: float) -> tuple[np.ndarray, float]:
        arm = home.copy() if args.gripper_only else _arm_target(home, t, args.amp, args.period)
        finger = _finger_target(t, args.gripper_period)
        return arm, float(finger)

    if args.headless:
        duration = args.duration if args.duration else 3.0
        while time.monotonic() - t0 < duration:
            t = time.monotonic() - t0
            arm, finger = step(t)
            for adr, value in zip(arm_adr, arm):
                data.qpos[adr] = float(value)
            for adr in finger_adr:
                data.qpos[adr] = float(finger)
            mujoco.mj_forward(model, data)
            finger_lo, finger_hi = min(finger_lo, finger), max(finger_hi, finger)
            arm_lo = np.minimum(arm_lo, arm)
            arm_hi = np.maximum(arm_hi, arm)
            time.sleep(1.0 / 100.0)
        print(f"[direct] 单指位移范围 [{finger_lo * 1000:.1f}, {finger_hi * 1000:.1f}] mm"
              f"（模型假手行程 0–40mm）")
        print("[direct] 各臂关节运动范围(rad)：")
        for i in range(home.size):
            print(f"    J{i + 1}: [{arm_lo[i]:+.3f}, {arm_hi[i]:+.3f}]")
        return 0

    try:
        import mujoco.viewer  # noqa: F401
    except Exception as exc:
        print(f"[direct] 无法加载 MuJoCo viewer：{exc}", file=sys.stderr)
        print("[direct] 可加 --headless 先做无窗口自检。", file=sys.stderr)
        return 2

    try:
        with mujoco.viewer.launch_passive(model, data) as handle:
            handle.cam.lookat[:] = [0.0, 0.0, 0.45]
            handle.cam.distance = 1.4
            handle.cam.azimuth = 130.0
            handle.cam.elevation = -20.0
            while handle.is_running():
                t = time.monotonic() - t0
                if args.duration is not None and t >= args.duration:
                    print("[direct] 到达 --duration，退出", flush=True)
                    break
                arm, finger = step(t)
                with handle.lock():
                    for adr, value in zip(arm_adr, arm):
                        data.qpos[adr] = float(value)
                    for adr in finger_adr:
                        data.qpos[adr] = float(finger)
                    mujoco.mj_forward(model, data)
                try:
                    handle.set_texts([
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            "DIRECT FR3（不经小臂/映射）",
                            f"夹爪单指={finger * 1000:.0f}mm  t={t:.1f}s",
                        )
                    ])
                except Exception:
                    pass
                handle.sync()
                time.sleep(1.0 / 60.0)
    finally:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
