#!/usr/bin/env python3
"""真实 FR3 外形 + 小臂模型 + 时间序列曲线，单窗口（Rerun）。

与另外两个查看器互补：

* ``examples/leader_follower_viewer.py``       MuJoCo 窗口、两条胶囊示意臂；
* ``examples/leader_follower_interactive.py``  MuJoCo 窗口、可**鼠标拖拽**小臂；
* 本文件：Rerun 单窗口，用 **staged 的真实 FR3 网格** + **等比缩小的 FR3 孪生**
  一起画，并在同一录制里画曲线。

3D（两条臂同构：``leader_fr3_joint_i`` 与 ``fr3_joint_i`` 一一对应，同号连杆同色，
关节位置标 ``J1..J7``）：
* 右侧 = 真实 FR3（视觉网格，跟随 ``follower.measured``；手指跟夹爪）；
* 左侧 = 小臂（缩小的 FR3，跟随 ``leader``）。
曲线：每个关节的 leader / follower 指令 / follower 实测 / 跟踪误差，
以及夹爪、两臂最近距离。

前置：先 staging 真实 FR3 描述（脚本会自动 clone franka_description + 用
xacro 生成 URDF + 把网格转成 .stl）：

    pip install xacro trimesh pycollada
    python tools/assets/fetch_fr3_description.py

运行（fish）：

    source .venv/bin/activate.fish
    PYTHONPATH=. python -B examples/leader_follower_rerun.py                 # 开 Rerun 窗口
    PYTHONPATH=. python -B examples/leader_follower_rerun.py --duration 20   # 跑 20s
    PYTHONPATH=. python -B examples/leader_follower_rerun.py --separation 1.4
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
from arm_control.leader_follower.leader import FakeLeaderArm  # noqa: E402
from arm_control.simulation.leader_arm_model import (  # noqa: E402
    GELLO_LEADER_SCALE,
    GRIP_TRAVEL_M,
    JOINT_COLORS,
    build_combined_spec,
    force_identity_arm_mapping,
    joint_qpos_addresses,
)
from arm_control.simulation.mujoco_model import build_mujoco_model  # noqa: E402

FR3_URDF = ROOT / "franka" / "urdf" / "fr3.urdf"
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
FINGER_MAX_M = 0.04
APP_ID = "arm_control_leader_follower"
# follower 的合法 home；仿真里小臂也绕它摆动，使两臂同形
FR3_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


# --------------------------------------------------------------------------- #
# 只渲染视觉几何（group==1）的镜像器：资产上传一次，每帧只发 Transform3D
# --------------------------------------------------------------------------- #
class _VisualMirror:
    def __init__(self, model, data, prefix: str, geom_prefix: str) -> None:
        self._m = model
        self._d = data
        self._prefix = prefix.rstrip("/")
        self._geoms: list[tuple[int, str]] = []
        for i in range(model.ngeom):
            if model.geom_group[i] != 1:  # 0=碰撞 1=视觉
                continue
            name = model.geom(i).name or f"geom{i}"
            if not name.startswith(geom_prefix):
                continue
            kind = int(model.geom_type[i])
            if kind not in (
                int(mujoco.mjtGeom.mjGEOM_MESH),
                int(mujoco.mjtGeom.mjGEOM_BOX),
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            ):
                continue
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
        m = self._m
        color = self._color(i)
        kind = int(m.geom_type[i])
        size = m.geom_size[i]
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            did = m.geom_dataid[i]
            v0, nv = m.mesh_vertadr[did], m.mesh_vertnum[did]
            f0, nf = m.mesh_faceadr[did], m.mesh_facenum[did]
            rr.log(
                entity,
                rr.Mesh3D(
                    vertex_positions=m.mesh_vert[v0 : v0 + nv],
                    triangle_indices=m.mesh_face[f0 : f0 + nf],
                    albedo_factor=color,
                ),
                static=True,  # 几何是 timeless：否则会落在 log_time 时间轴，切到 tick 就不显示
            )
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            rr.log(
                entity,
                rr.Boxes3D(half_sizes=[size], colors=[color], fill_mode="solid"),
                static=True,
            )
        elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            # Rerun 胶囊从 (0,0,0) 沿 +z 到 (0,0,length)（端帽球心）；MuJoCo 的
            # capsule 以 geom 中心为原点、半长 size[1]。故 length=2*size[1]，
            # 并沿 -z 平移 size[1] 把它对回中心。
            rr.log(
                entity,
                rr.Capsules3D(
                    lengths=[2.0 * float(size[1])],
                    radii=[float(size[0])],
                    translations=[[0.0, 0.0, -float(size[1])]],
                    colors=[color],
                    fill_mode="solid",
                ),
                static=True,
            )
        else:  # CYLINDER：Rerun 以中心为原点，length 即全长
            rr.log(
                entity,
                rr.Cylinders3D(
                    lengths=[2.0 * float(size[1])],
                    radii=[float(size[0])],
                    colors=[color],
                    fill_mode="solid",
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
        self.leader_grip = 1.0  # 1=张开
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
        if q.size > st.leader.size:
            st.leader_grip = float(np.clip(q[st.leader.size], 0.0, 1.0))
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
    parser = argparse.ArgumentParser(description="真实 FR3 + 小臂模型 + 时间序列（Rerun）")
    parser.add_argument("--config", default=str(ROOT / "examples/configs/leader_follower.yaml"))
    parser.add_argument("--duration", type=float, default=None, help="遥操作运行时长（秒）")
    parser.add_argument("--separation", type=float, default=1.15, help="两臂基座间距（米）")
    parser.add_argument("--leader-scale", type=float, default=None,
                        help=f"小臂模型缩放（默认：gello 外观 {GELLO_LEADER_SCALE}，孪生 0.75）")
    parser.add_argument("--leader-appearance", choices=("gello", "twin"), default="gello",
                        help="小臂外观：gello=Franka 官方 GELLO 真实零件；twin=缩小 FR3 孪生")
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

    # --- 合并模型：真实 FR3 网格 + 缩小的小臂（默认用 GELLO 真实零件外观） ---
    cache = Path(tempfile.gettempdir()) / "arm_control_fr3_mjcache"
    staged = build_mujoco_model(FR3_URDF, cache_dir=cache, keep_visual=True)
    use_gello = args.leader_appearance == "gello"
    leader_scale = args.leader_scale
    if leader_scale is None:
        leader_scale = GELLO_LEADER_SCALE if use_gello else 0.75
    spec, refs = build_combined_spec(
        str(staged),
        leader_position=(-args.separation, 0.0, 0.0),
        leader_scale=leader_scale,
        leader_gello_parts=use_gello,
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    qadr = joint_qpos_addresses(model, ARM_JOINTS + FINGER_JOINTS)
    leader_adr = joint_qpos_addresses(model, refs.joint_names)
    leader_arm_adr = [leader_adr[n] for n in refs.arm_joint_names]
    leader_finger_adr = [
        int(model.jnt_qposadr[jid])
        for jid in range(model.njnt)
        if (model.joint(jid).name or "").startswith(refs.prefix + "_")
        and "finger" in (model.joint(jid).name or "")
    ]

    follower_mirror = _VisualMirror(model, data, prefix="follower", geom_prefix="fr3")
    leader_mirror = _VisualMirror(model, data, prefix="leader", geom_prefix=refs.prefix)

    # 坐标约定放在根实体上（两只臂都在根下），静态、任何时间轴都生效
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- 遥操作装配 ---
    guard = None
    if args.collision_demo:
        import importlib.util

        spec_mod = importlib.util.spec_from_file_location(
            "leader_follower_teleop", str(ROOT / "examples" / "leader_follower_teleop.py")
        )
        mod = importlib.util.module_from_spec(spec_mod)
        assert spec_mod.loader is not None
        spec_mod.loader.exec_module(mod)
        guard = mod.make_demo_collision_guard(cfg.follower.n_arm_joints)

    # 小臂是 FR3 孪生：关节映射改为直连，大臂才会和小臂同形跟动
    force_identity_arm_mapping(cfg)
    leader, retargeter, follower, monitor = build_pipeline(cfg, collision_guard=guard)
    # 让假小臂绕 FR3 home 摆动（初值正好落在 home），两条臂保持同形
    initial = np.asarray(cfg.follower.initial, dtype=float)
    if initial.shape != (cfg.follower.n_arm_joints,):
        initial = np.asarray(FR3_HOME, dtype=float)[: cfg.follower.n_arm_joints]
    leader = FakeLeaderArm(
        n_arm_joints=cfg.leader.n_arm_joints,
        with_gripper=cfg.leader.with_gripper,
        amplitude=cfg.leader.fake_amplitude,
        period_s=cfg.leader.fake_period_s,
        center=initial,
    )
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
        for i, adr in enumerate(leader_arm_adr):
            data.qpos[adr] = float(st.leader[i])
        leader_grip_m = float(np.clip(st.leader_grip, 0.0, 1.0)) * GRIP_TRAVEL_M
        for adr in leader_finger_adr:
            data.qpos[adr] = leader_grip_m
        mujoco.mj_forward(model, data)
        follower_mirror.update()
        leader_mirror.update()

        # 关节位置标 J1..J7（两臂同号同色）
        for arm_joints, view in ((ARM_JOINTS, "follower"), (refs.arm_joint_names, "leader")):
            positions = np.empty((len(arm_joints), 3))
            colors = []
            labels = []
            for i, name in enumerate(arm_joints):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                positions[i] = data.xanchor[jid]
                colors.append(JOINT_COLORS[i][:3])
                labels.append(f"J{i + 1}")
            rr.log(
                f"labels/{view}",
                rr.Points3D(positions, colors=colors, labels=labels, radii=0.008),
            )

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
        f"间距={args.separation:.2f}m（Ctrl-C 结束；关查看器不影响循环）",
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
    # 主动 flush / 断开：MuJoCo 在解释器退出时可能段错误，析构会被跳过，
    # 否则 .rrd 可能只落盘了一部分（例如小臂网格丢失）。
    try:
        rr.disconnect()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
