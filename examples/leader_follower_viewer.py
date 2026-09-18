#!/usr/bin/env python3
"""主从遥操作 3D 可视化：MuJoCo 原生窗口，两条 7-DOF 臂实时跟随。

左边小臂（leader，蓝），右边大臂（follower，橙）。不吃外部 URDF/mesh，
用程序化的 capsule 链搭骨架上场，所以无需 staging 任何资产即可运行。

驱动方式：跑真正的 `TeleopLoop`（默认配置是 FakeLeader + FakeFollower），
在**后台线程**里 tick；主线程用 MuJoCo 被动查看器把记录的关节角写进
qpos 并 sync。核心代码零改动——用轻量代理包住 leader/follower 记录状态。

    python -B examples/leader_follower_viewer.py                  # 边跑边看
    python -B examples/leader_follower_viewer.py --duration 20    # 20s 后停
    python -B examples/leader_follower_viewer.py --collision-demo # 演示碰撞停机
    python -B examples/leader_follower_viewer.py --headless --duration 3  # 无窗口自检

窗口操作：鼠标左键旋转、右键平移、滚轮缩放；左上角显示状态。
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_control.leader_follower import (  # noqa: E402
    FollowerFeedback,
    SafetyStop,
    TeleopLoop,
)
from arm_control.leader_follower.config import (  # noqa: E402
    build_pipeline,
    config_from_yaml,
)


# --------------------------------------------------------------------------- #
# 程序化 7-DOF 臂（capsule 链）
# --------------------------------------------------------------------------- #
# 每节：关节轴 + 连杆长 + 半径。轴序 = 基座 yaw + 平面 pitch + 前臂/腕 roll。
_LINKS = [
    ("0 0 1", 0.10, 0.050),
    ("0 1 0", 0.30, 0.048),
    ("0 1 0", 0.27, 0.042),
    ("1 0 0", 0.11, 0.034),
    ("0 1 0", 0.10, 0.030),
    ("1 0 0", 0.08, 0.027),
    ("0 1 0", 0.07, 0.024),
]


def _arm_xml(prefix: str, scale: float, pos: str, quat: str, rgba: str) -> str:
    """一条 7 关节 capsule 臂。关节名 ``{prefix}_j1..j7``。"""
    out = [f'<body name="{prefix}_base" pos="{pos}" quat="{quat}">']
    out.append(
        f'  <geom type="cylinder" size="{0.07 * scale:.4f} {0.03 * scale:.4f}"'
        f' pos="0 0 {-0.02 * scale:.4f}" rgba="{rgba}"/>'
    )
    indent = "  "
    prev = 0.0
    for i, (axis, length, radius) in enumerate(_LINKS):
        ln, r = length * scale, radius * scale
        out.append(f'{indent}<body name="{prefix}_l{i + 1}" pos="{prev:.4f} 0 0">')
        out.append(
            f'{indent}  <joint name="{prefix}_j{i + 1}" type="hinge"'
            f' axis="{axis}" range="-3.0 3.0" damping="0.05"/>'
        )
        out.append(
            f'{indent}  <geom type="capsule" fromto="0 0 0 {ln:.4f} 0 0"'
            f' size="{r:.4f}" rgba="{rgba}"/>'
        )
        indent += "  "
        prev = ln
    out.append("</body>" * (len(_LINKS) + 1))
    return "\n".join(out)


def build_model_xml() -> str:
    return f"""<mujoco model="leader_follower_teleop">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.22 0.24 0.28 1"/>
    <geom name="midline" type="box" size="0.004 0.6 0.001" pos="0 0 0.001"
          rgba="0.5 0.5 0.55 0.6"/>
    {_arm_xml("lead", 0.55, "-0.62 0 0.30", "1 0 0 0", "0.20 0.62 1.0 1.0")}
    {_arm_xml("foll", 1.00, "0.62 0 0.30", "0 0 0 1", "1.00 0.55 0.10 1.0")}
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------- #
# 状态记录代理（核心零改动）
# --------------------------------------------------------------------------- #
class _Record:
    def __init__(self, n: int) -> None:
        self.leader = np.zeros(n, dtype=float)
        self.target = np.zeros(n, dtype=float)
        self.measured = np.zeros(n, dtype=float)
        self.gripper = 0.0
        self.tick = 0
        self.has_measured = False
        self.stopped = False
        self.stop_reason = ""


def _instrument(leader, follower, loop: TeleopLoop, rec: _Record) -> None:
    """包住 leader.get_joint_state / follower.send|read_state 以及 loop.once。

    只把**真正的** ``SafetyStop`` 记为停机；``follower.close()`` 内部也会调
    ``safe_stop()``，那是正常收尾，不能算安全事件。
    """
    orig_get = leader.get_joint_state

    def get_joint_state():
        q = np.asarray(orig_get(), dtype=float)
        rec.leader = q[: rec.leader.size]
        return q

    leader.get_joint_state = get_joint_state  # type: ignore[method-assign]

    orig_send = follower.send

    def send(arm_q, gripper_finger_m):
        rec.target = np.asarray(arm_q, dtype=float).copy()
        rec.gripper = float(gripper_finger_m)
        rec.tick += 1
        return orig_send(arm_q, gripper_finger_m)

    follower.send = send  # type: ignore[method-assign]

    orig_read = follower.read_state

    def read_state():
        arm, grip = orig_read()
        rec.measured = np.asarray(arm, dtype=float).copy()
        rec.has_measured = True
        return arm, grip

    follower.read_state = read_state  # type: ignore[method-assign]

    orig_once = loop.once

    def once(now: float):
        try:
            return orig_once(now)
        except SafetyStop as stop:
            src = getattr(stop, "source", "?")
            rec.stopped = True
            rec.stop_reason = f"source={src} {stop.reason}"
            raise

    loop.once = once  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# 无窗口自检
# --------------------------------------------------------------------------- #
def run_headless(loop: TeleopLoop, rec: _Record, duration_s: float) -> None:
    loop.verbose = False
    stats = loop.run(duration_s=duration_s)
    state = f"安全停机（{rec.stop_reason}）" if rec.stopped else "正常结束"
    print(f"[viewer] headless 完成：ticks={stats.ticks} sends={stats.sends} {state}")
    print(f"[viewer] 安全事件次数={stats.safety_stops}")
    print(f"[viewer] leader  q 范围 [{rec.leader.min():+.3f}, {rec.leader.max():+.3f}]")
    print(f"[viewer] target  q 范围 [{rec.target.min():+.3f}, {rec.target.max():+.3f}]")


# --------------------------------------------------------------------------- #
# MuJoCo 窗口
# --------------------------------------------------------------------------- #
def run_viewer(loop: TeleopLoop, rec: _Record, monitor, duration_s, hold_s: float = 0.0) -> int:
    try:
        import mujoco
        import mujoco.viewer
    except Exception as exc:  # 依赖/显示不可用
        print(f"[viewer] 无法加载 MuJoCo viewer：{exc}", file=sys.stderr)
        print("[viewer] 可改用 --headless 自检，或安装 mujoco/修复显示环境。", file=sys.stderr)
        return 2

    model = mujoco.MjModel.from_xml_string(build_model_xml())
    data = mujoco.MjData(model)

    joint_adr: dict[str, int] = {}
    for prefix in ("lead", "foll"):
        for i in range(7):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_j{i + 1}")
            joint_adr[f"{prefix}_j{i + 1}"] = int(model.jnt_qposadr[jid])

    thread = threading.Thread(
        target=loop.run, kwargs={"duration_s": duration_s}, daemon=True
    )
    thread.start()

    def text(msg: str, sub: str = "") -> list:
        return [(mujoco.mjtFontScale.mjFONTSCALE_150,
                 mujoco.mjtGridPos.mjGRID_TOPLEFT, msg, sub)]

    try:
        with mujoco.viewer.launch_passive(model, data) as handle:
            loop_done_t: float | None = None
            while handle.is_running():
                if thread.is_alive():
                    loop_done_t = None
                elif loop_done_t is None:
                    loop_done_t = time.monotonic()
                    print("[viewer] 遥操作已结束，窗口保留中（关闭窗口退出）", flush=True)
                elif hold_s > 0.0 and time.monotonic() - loop_done_t >= hold_s:
                    break
                with handle.lock():
                    foll_q = rec.measured if rec.has_measured else rec.target
                    for i in range(7):
                        data.qpos[joint_adr[f"lead_j{i + 1}"]] = float(rec.leader[i])
                        data.qpos[joint_adr[f"foll_j{i + 1}"]] = float(foll_q[i])
                    mujoco.mj_forward(model, data)
                try:
                    if rec.stopped:
                        line1 = f"tick={rec.tick}  STOPPED"
                        line2 = rec.stop_reason or "安全停机"
                    else:
                        err = (
                            float(np.max(np.abs(rec.target - rec.measured)))
                            if rec.has_measured else 0.0
                        )
                        d = monitor.last_collision_distance
                        d_txt = f"{d * 1000:.0f}mm" if np.isfinite(d) else "n/a"
                        line1 = f"tick={rec.tick}  RUNNING   蓝=小臂(leader) 橙=大臂(follower实测)"
                        line2 = (
                            f"gripper={rec.gripper:.3f}m  跟踪误差max={err:.3f}rad  "
                            f"两臂最近={d_txt}"
                        )
                    handle.set_texts(text(line1, line2))
                except Exception:
                    pass
                handle.sync()
                time.sleep(1.0 / 60.0)
    finally:
        loop.request_stop()
        thread.join(timeout=2.0)
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="主从遥操作 MuJoCo 3D 可视化")
    parser.add_argument("--config", default=str(ROOT / "examples/configs/leader_follower.yaml"))
    parser.add_argument("--duration", type=float, default=None, help="遥操作运行时长（秒）")
    parser.add_argument("--headless", action="store_true", help="不开窗口，仅自检并打印")
    parser.add_argument("--hold", type=float, default=0.0,
                        help="遥操作结束后窗口再保留几秒（0=一直保留到手动关闭）")
    parser.add_argument("--collision-demo", action="store_true", help="用玩具球体守卫演示碰撞停机")
    parser.add_argument("--hz", type=float, default=None, help="覆盖 loop.hz")
    args = parser.parse_args(argv)

    cfg = config_from_yaml(args.config)
    if args.hz:
        cfg.loop.hz = args.hz

    guard = None
    if args.collision_demo:
        # 复用示例里的玩具平面 FK 球体守卫（示例有 __main__ 守卫，导入不会跑 main）
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "leader_follower_teleop", str(ROOT / "examples" / "leader_follower_teleop.py")
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        guard = mod.make_demo_collision_guard(cfg.follower.n_arm_joints)

    leader, retargeter, follower, monitor = build_pipeline(cfg, collision_guard=guard)
    loop = TeleopLoop(
        leader=leader,
        retargeter=retargeter,
        follower=follower,
        monitor=monitor,
        feedback=FollowerFeedback(follower, armed=True),
        hz=cfg.loop.hz,
        auto_align=cfg.mapping.auto_align,
        log_period_s=cfg.loop.log_period_s,
    )

    rec = _Record(cfg.follower.n_arm_joints)
    _instrument(leader, follower, loop, rec)

    if args.headless:
        run_headless(loop, rec, args.duration if args.duration else 3.0)
        return 0

    print("[viewer] 打开 MuJoCo 窗口；关闭窗口或 Ctrl-C 结束。", flush=True)
    return run_viewer(loop, rec, monitor, args.duration, hold_s=args.hold)


if __name__ == "__main__":
    raise SystemExit(main())
