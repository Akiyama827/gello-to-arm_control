#!/usr/bin/env python3
"""真实 FR3 外形 + 时间序列曲线，单窗口（Rerun）。

与 ``examples/leader_follower_viewer.py``（MuJoCo 窗口、胶囊示意臂）互补：
这里用 **staged 的真实 FR3 网格**渲染大臂，并在同一 Rerun 录制里画曲线：

* 3D：右侧 = 真实 FR3（视觉网格，跟随 ``follower.measured``；手指跟夹爪）；
       左侧 = 小臂(leader)骨架线框（S288 无公开网格，用 7 段线示意）。
* 曲线：每个关节的 leader / follower 指令 / follower 实测 / 跟踪误差，
        以及夹爪、两臂最近距离。

前置：先 staging 真实 FR3 描述（脚本会自动 clone franka_description + 用
xacro 生成 URDF + 把网格转成 .stl）：

    pip install xacro trimesh pycollada
    python tools/assets/fetch_fr3_description.py

运行（fish）：

    source .venv/bin/activate.fish
    PYTHONPATH=. python -B examples/leader_follower_rerun.py                 # 开 Rerun 窗口
    PYTHONPATH=. python -B examples/leader_follower_rerun.py --duration 20   # 跑 20s
    PYTHONPATH=. python -B examples/leader_follower_rerun.py --collision-demo
    PYTHONPATH=. python -B examples/leader_follower_rerun.py --save /tmp/opencode/teleop.rrd  # 存盘，无窗口

核心遥操作代码零改动：用轻量代理采集 leader 状态、follower 指令/实测，在
``loop.once`` 之外做 FK 与 Rerun 记录。
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import rerun as rr  # noqa: E402

from arm_control.leader_follower import (  # noqa: E402
    FollowerFeedback,
    SafetyStop,
    TeleopLoop,
)
from arm_control.leader_follower.config import (  # noqa: E402
    build_pipeline,
    config_from_yaml,
)
from arm_control.simulation.mujoco_model import build_mujoco_model  # noqa: E402

FR3_URDF = ROOT / "franka" / "urdf" / "fr3.urdf"
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
FINGER_MAX_M = 0.04
APP_ID = "arm_control_leader_follower"

# 小臂骨架：与 MuJoCo 版同一套连杆约定（基座 yaw + 平面 pitch + 前臂/腕 roll）。
_LINKS = [
    ("0 0 1", 0.10),
    ("0 1 0", 0.30),
    ("0 1 0", 0.27),
    ("1 0 0", 0.11),
    ("0 1 0", 0.10),
    ("1 0 0", 0.08),
    ("0 1 0", 0.07),
]
_LEADER_SCALE = 0.6
_LEADER_BASE = (-0.65, 0.0, 0.25)
_LEADER_COLOR = [80, 160, 255]


# --------------------------------------------------------------------------- #
# 小臂正运动学（骨架线框）
# --------------------------------------------------------------------------- #
def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def leader_fk(q: np.ndarray, scale: float, base) -> np.ndarray:
    """返回 7 段骨的 8 个端点（世界坐标）。"""
    T = np.eye(4)
    T[:3, 3] = np.asarray(base, dtype=float)
    pts = [T[:3, 3].copy()]
    for i, (axis, length) in enumerate(_LINKS):
        T = T @ _axis_angle(np.array([float(a) for a in axis.split()]), float(q[i]))
        tip = T @ np.array([length * scale, 0.0, 0.0, 1.0])
        pts.append(tip[:3].copy())
        T = T.copy()
        T[:3, 3] = tip[:3]
    return np.asarray(pts)


# --------------------------------------------------------------------------- #
# 只渲染视觉网格（group==1）的镜像器：资产上传一次，每帧只发 Transform3D
# --------------------------------------------------------------------------- #
class _VisualMirror:
    def __init__(self, model, data, prefix: str) -> None:
        self._m = model
        self._d = data
        self._prefix = prefix.rstrip("/")
        self._geoms: list[tuple[int, str]] = []
        for i in range(model.ngeom):
            if model.geom_group[i] != 1:  # 0=碰撞 1=视觉 3=视觉的碰撞孪生
                continue
            kind = int(model.geom_type[i])
            if kind not in (
                int(mujoco.mjtGeom.mjGEOM_MESH),
                int(mujoco.mjtGeom.mjGEOM_BOX),
            ):
                continue
            name = model.geom(i).name or f"geom{i}"
            body = model.body(model.geom_bodyid[i]).name or "world"
            entity = f"{self._prefix}/{body}/{name}"
            self._log_asset(i, entity)
            self._geoms.append((i, entity))

    def _color(self, i: int) -> list[int]:
        rgba = np.clip(self._m.geom_rgba[i], 0.0, 1.0)
        if rgba[3] <= 0.0:
            rgba = np.array([0.9, 0.9, 0.92, 1.0])
        return [int(round(c * 255)) for c in rgba]

    def _log_asset(self, i: int, entity: str) -> None:
        color = self._color(i)
        if int(self._m.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            did = self._m.geom_dataid[i]
            v0, nv = self._m.mesh_vertadr[did], self._m.mesh_vertnum[did]
            f0, nf = self._m.mesh_faceadr[did], self._m.mesh_facenum[did]
            rr.log(
                entity,
                rr.Mesh3D(
                    vertex_positions=self._m.mesh_vert[v0 : v0 + nv],
                    triangle_indices=self._m.mesh_face[f0 : f0 + nf],
                    albedo_factor=color,
                ),
                static=True,  # 几何是 timeless：否则会落在 log_time 时间轴，切到 tick 就不显示
            )
        else:
            rr.log(
                entity,
                rr.Boxes3D(
                    half_sizes=[self._m.geom_size[i]], colors=[color], fill_mode="solid"
                ),
                static=True,
            )

    def update(self) -> None:
        for i, entity in self._geoms:
            rr.log(
                entity,
                rr.Transform3D(
                    translation=self._d.geom_xpos[i],
                    mat3x3=self._d.geom_xmat[i].reshape(3, 3),
                ),
            )


# --------------------------------------------------------------------------- #
# 状态采集（核心零改动）
# --------------------------------------------------------------------------- #
class _State:
    def __init__(self, n: int) -> None:
        self.leader = np.zeros(n)
        self.target = np.zeros(n)
        self.measured = np.zeros(n)
        self.gripper = 0.0
        self.tick = 0
        self.has_measured = False
        self.stopped = False
        self.reason = ""


def _instrument(leader, follower, loop, st: _State) -> None:
    orig_get = leader.get_joint_state

    def get_joint_state():
        q = np.asarray(orig_get(), dtype=float)
        st.leader = q[: st.leader.size]
        return q

    leader.get_joint_state = get_joint_state  # type: ignore[method-assign]

    orig_send = follower.send

    def send(arm_q, gripper_finger_m):
        st.target = np.asarray(arm_q, dtype=float).copy()
        st.gripper = float(gripper_finger_m)
        return orig_send(arm_q, gripper_finger_m)

    follower.send = send  # type: ignore[method-assign]

    orig_read = follower.read_state

    def read_state():
        arm, grip = orig_read()
        st.measured = np.asarray(arm, dtype=float).copy()
        st.has_measured = True
        return arm, grip

    follower.read_state = read_state  # type: ignore[method-assign]

    orig_once = loop.once

    def once(now: float):
        try:
            result = orig_once(now)
        except SafetyStop as stop:
            st.stopped = True
            st.reason = f"source={getattr(stop, 'source', '?')} {stop.reason}"
            rr.log("status/stop", rr.TextLog(st.reason))
            raise
        st.tick += 1
        return result

    loop.once = once  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="真实 FR3 + 时间序列（Rerun）")
    parser.add_argument("--config", default=str(ROOT / "examples/configs/leader_follower.yaml"))
    parser.add_argument("--duration", type=float, default=None, help="遥操作运行时长（秒）")
    parser.add_argument("--collision-demo", action="store_true", help="用玩具球体守卫演示碰撞停机")
    parser.add_argument("--save", default=None, help="把录制存成 .rrd（不弹窗，适合无显示/存档）")
    parser.add_argument("--no-spawn", action="store_true", help="初始化 Rerun 但不自动拉起查看器")
    parser.add_argument("--hz", type=float, default=None, help="覆盖 loop.hz")
    args = parser.parse_args(argv)

    if not FR3_URDF.exists():
        print(f"[rerun] 找不到 FR3 URDF：{FR3_URDF}", file=sys.stderr)
        print(
            "[rerun] 先 staging 真实 FR3 描述：\n"
            "  pip install xacro trimesh pycollada\n"
            "  python tools/assets/fetch_fr3_description.py",
            file=sys.stderr,
        )
        return 2

    cfg = config_from_yaml(args.config)
    if args.hz:
        cfg.loop.hz = args.hz

    # --- Rerun 会话 ---
    rr.init(APP_ID)
    if args.save:
        rr.save(args.save)
        print(f"[rerun] 录制写入 {args.save}（不弹窗）", flush=True)
    elif not args.no_spawn:
        rr.spawn()
        print("[rerun] 已拉起 Rerun 查看器（3D + Time series）", flush=True)
    else:
        print("[rerun] 已初始化，但未自动拉起查看器（--no-spawn）", flush=True)

    # --- 真实 FR3 模型 + 视觉镜像 ---
    cache = Path(tempfile.gettempdir()) / "arm_control_fr3_mjcache"
    staged = build_mujoco_model(FR3_URDF, cache_dir=cache, keep_visual=True)
    model = mujoco.MjModel.from_xml_path(str(staged))
    data = mujoco.MjData(model)
    qadr = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in (*ARM_JOINTS, *FINGER_JOINTS)
    }
    mirror = _VisualMirror(model, data, prefix="follower")

    # 坐标约定放在根实体上（两只臂都在根下），静态、任何时间轴都生效
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- 遥操作装配 ---
    guard = None
    if args.collision_demo:
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

    st = _State(cfg.follower.n_arm_joints)
    _instrument(leader, follower, loop, st)

    def log_tick() -> None:
        rr.set_time("tick", sequence=st.tick)
        foll_q = st.measured if st.has_measured else st.target
        for i, name in enumerate(ARM_JOINTS):
            data.qpos[qadr[name]] = float(foll_q[i])
        grip = float(np.clip(st.gripper, 0.0, FINGER_MAX_M))
        for name in FINGER_JOINTS:
            data.qpos[qadr[name]] = grip
        mujoco.mj_forward(model, data)
        mirror.update()

        pts = leader_fk(st.leader, _LEADER_SCALE, _LEADER_BASE)
        rr.log("leader/skeleton", rr.LineStrips3D([pts], colors=[_LEADER_COLOR], radii=0.022))
        rr.log("leader/joints", rr.Points3D(pts, radii=0.032, colors=[_LEADER_COLOR]))

        for i in range(st.leader.size):
            rr.log(f"plots/leader/q{i + 1}", rr.Scalars(float(st.leader[i])))
            rr.log(f"plots/follower_cmd/q{i + 1}", rr.Scalars(float(st.target[i])))
            rr.log(f"plots/follower_meas/q{i + 1}", rr.Scalars(float(st.measured[i])))
            rr.log(f"plots/track_err/q{i + 1}", rr.Scalars(float(st.target[i] - st.measured[i])))
        rr.log("plots/gripper_m", rr.Scalars(grip))
        d = monitor.last_collision_distance
        if np.isfinite(d):
            rr.log("plots/collision_distance_m", rr.Scalars(float(d)))

    # 把 log_tick 挂到 loop.once 之后（每次成功 tick 记一帧）
    orig_once = loop.once

    def once_with_log(now: float):
        result = orig_once(now)
        if not st.stopped:
            log_tick()
        return result

    loop.once = once_with_log  # type: ignore[method-assign]

    print(
        f"[rerun] 启动：{cfg.loop.hz:.0f}Hz  duration={args.duration or '∞'}  "
        f"（Ctrl-C 结束；关查看器不影响循环）",
        flush=True,
    )
    try:
        stats = loop.run(duration_s=args.duration)
    except KeyboardInterrupt:
        stats = loop.stats
    print(
        f"[rerun] 结束：ticks={stats.ticks} sends={stats.sends} "
        f"安全停机={stats.safety_stops}"
        + (f"（{st.reason}）" if st.stopped else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
