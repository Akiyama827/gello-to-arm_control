#!/usr/bin/env python3
"""交互式主从遥操作：**鼠标拖拽小臂，真实 FR3 实时跟随**（MuJoCo 被动窗口）。

与另外两个查看器的区别：这个窗口是**可输入**的。

* 左：小臂模型（**等比缩小的 FR3 孪生**，与右侧同构，关节一一对应、同号同色），
  可用鼠标拖拽；
* 右：staged 的**真实 FR3 网格**，按 ``Retargeter`` 实时跟动。

拖拽走 MuJoCo 自带的扰动（perturbation）：按住被拖的连杆拖动会给它一个弹簧力，
小臂关节在阻尼下运动；主线程每帧读出小臂 7 关节 + 夹爪，交给真正的
``TeleopLoop`` 换算（含 ``SafetyMonitor``、``auto_align``、真实几何碰撞守卫），
再写回 FR3 的 qpos。**安全门与真机一致**：越限 / 跳变 / 跟踪异常 / 两臂将碰都会
安全停机并冻结大臂。

窗口操作（MuJoCo 原生）：
* 默认已经选中小臂末端；按住 **Ctrl + 鼠标右键拖动** = 平移施力（推荐，最能体现
  "拖小臂"），**Ctrl + 鼠标左键拖动** = 绕选中点旋转施力；
* 想拖别的连杆：先 **鼠标左键双击** 选中那一节，再按上面的方式拖动；
* 鼠标左键拖动 = 旋转视角；右键 = 平移视角；滚轮 = 缩放；
* 界面内空格 = 暂停/继续物理；Esc/q = 退出（也可直接关窗口）。

前置：先 staging 真实 FR3 描述：``python tools/assets/fetch_fr3_description.py``。

运行（fish）：
    source .venv/bin/activate.fish
    PYTHONPATH=. python -B examples/leader_follower_interactive.py
    PYTHONPATH=. python -B examples/leader_follower_interactive.py --separation 1.4
    PYTHONPATH=. python -B examples/leader_follower_interactive.py --duration 60
    PYTHONPATH=. python -B examples/leader_follower_interactive.py --no-collision-guard
    PYTHONPATH=. python -B examples/leader_follower_interactive.py --rerun   # 边拖边看曲线
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from arm_control.leader_follower import (  # noqa: E402
    FollowerFeedback,
    NoCollisionGuard,
    TeleopLoop,
)
from arm_control.leader_follower.config import (  # noqa: E402
    build_monitor,
    build_retargeter,
    config_from_yaml,
)
from arm_control.simulation.leader_arm_model import (  # noqa: E402
    GELLO_LEADER_SCALE,
    GRIP_TRAVEL_M,
    build_combined_spec,
    force_identity_arm_mapping,
    leader_state_from_qpos,
)
from arm_control.simulation.mj_collision_guard import (  # noqa: E402
    MjGeomDistanceGuard,
    select_geoms_by_name,
)
from arm_control.simulation.mujoco_model import build_mujoco_model  # noqa: E402

FR3_URDF = ROOT / "franka" / "urdf" / "fr3.urdf"
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]
FINGER_MAX_M = 0.04
# FR3 的一个合法 home（与 examples/configs/leader_follower.yaml 一致）；配置里没给
# follower.initial 时兜底，否则 auto_align 会从全零起步、第一 tick 就因越限停机。
FR3_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
APP_ID = "arm_control_leader_follower_interactive"


def _adr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"模型里没有关节 {joint_name!r}")
    return int(model.jnt_qposadr[jid])


def _dof_adr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_dofadr[jid])


def _leader_gripper_joints(model: mujoco.MjModel, refs) -> list[str]:
    """小臂上所有夹爪指关节名（Franka Hand 是双指）。"""
    out: list[str] = []
    for jid in range(model.njnt):
        name = model.joint(jid).name or ""
        if name.startswith(refs.prefix + "_") and "finger" in name:
            out.append(name)
    return out


def _leader_tip_body_id(model: mujoco.MjModel, refs) -> int:
    """默认拖拽目标：小臂末节连杆；找不到就退化为最深的小臂 body。

    MuJoCo 的扰动机制要求先"选中"一个 body（``perturb.select > 0``）才会施力，
    默认替用户选中末端，于是不必先双击即可直接 Ctrl+拖动。
    """
    for candidate in (
        f"{refs.prefix}_fr3_link7",
        f"{refs.prefix}_l7",
        f"{refs.prefix}_finger",
    ):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid > 0:
            return int(bid)
    best, best_depth = -1, -1
    for bid in range(model.nbody):
        name = model.body(bid).name or ""
        if not name.startswith(refs.prefix + "_"):
            continue
        depth, cur = 0, bid
        while model.body_parentid[cur] > 0:
            cur = int(model.body_parentid[cur])
            depth += 1
        if depth > best_depth:
            best, best_depth = bid, depth
    return int(best)


# --------------------------------------------------------------------------- #
# 把"被拖拽的小臂"接成 leader，把"被写 qpos 的 FR3"接成 follower
# --------------------------------------------------------------------------- #
class DragLeader:
    """实现 ``LeaderArm``：状态由 MuJoCo 查看器线程每帧写入。"""

    def __init__(self, n_arm: int, with_gripper: bool, grip_open: float = 1.0) -> None:
        self._n = int(n_arm) + (1 if with_gripper else 0)
        self._state = np.zeros(self._n, dtype=float)
        if with_gripper:
            self._state[-1] = float(grip_open)
        self._lock = threading.Lock()

    def num_dofs(self) -> int:
        return self._n

    def get_joint_state(self) -> np.ndarray:
        with self._lock:
            return self._state.copy()

    def set_state(self, state: np.ndarray) -> None:
        with self._lock:
            self._state = np.asarray(state, dtype=float).ravel()[: self._n].copy()

    def get_observations(self) -> dict:
        return {"joint_positions": self.get_joint_state()}

    def set_torque_mode(self, enable: bool) -> None:
        pass

    def close(self) -> None:
        pass


class KinematicFollower:
    """实现 ``FollowerArm``：只记录目标，viewer 线程把它写进 FR3 的 qpos。

    ``read_state`` 返回**最近一次下发**的目标（等价于理想伺服），因此跟踪误差为 0，
    安全门仍会检查限位 / 跳变 / 反馈新鲜度。``safe_stop`` 后冻结、不再接受新目标。
    """

    def __init__(self, n_arm: int, initial_q, initial_finger_m: float) -> None:
        self.num_arm_joints = int(n_arm)
        self._q = np.asarray(initial_q, dtype=float).copy()
        self._grip = float(initial_finger_m)
        self._lock = threading.Lock()
        self._stopped = False

    def open(self) -> None:
        with self._lock:
            self._stopped = False

    def current(self) -> tuple[np.ndarray, float]:
        with self._lock:
            return self._q.copy(), self._grip

    def read_state(self) -> tuple[np.ndarray, float]:
        with self._lock:
            return self._q.copy(), self._grip

    def send(self, arm_q, gripper_finger_m: float) -> None:
        with self._lock:
            if self._stopped:
                return
            self._q = np.asarray(arm_q, dtype=float).copy()
            self._grip = float(gripper_finger_m)

    def safe_stop(self) -> None:
        with self._lock:
            self._stopped = True

    def close(self) -> None:
        with self._lock:
            self._stopped = True


class _RerunLogger:
    """可选的 Rerun 曲线记录；未启用时全部 no-op。"""

    def __init__(self, enabled: bool) -> None:
        self._rr = None
        self._n_leader = 0
        if not enabled:
            return
        import rerun as rr

        self._rr = rr

    def frame(self, tick: int, leader_state, follower_q, finger_m, distance) -> None:
        rr = self._rr
        if rr is None:
            return
        rr.set_time("tick", sequence=int(tick))
        n_arm = len(leader_state) - 1 if leader_state.size > 0 else 0
        for i in range(n_arm):
            rr.log(f"plots/leader/q{i + 1}", rr.Scalars(float(leader_state[i])))
        for i, v in enumerate(follower_q):
            rr.log(f"plots/follower/q{i + 1}", rr.Scalars(float(v)))
        rr.log("plots/gripper_m", rr.Scalars(float(finger_m)))
        if np.isfinite(distance):
            rr.log("plots/collision_distance_m", rr.Scalars(float(distance)))


# --------------------------------------------------------------------------- #
def _make_guard(args, model, refs, cfg):
    if args.no_collision_guard:
        print("[interactive] 碰撞守卫：已关闭（--no-collision-guard）", flush=True)
        return NoCollisionGuard()
    return MjGeomDistanceGuard(
        model,
        follower_arm_qpos_adr=[_adr(model, n) for n in ARM_JOINTS],
        leader_arm_qpos_adr=[_adr(model, n) for n in refs.arm_joint_names],
        leader_grip_qpos_adr=(_adr(model, refs.gripper_name) if refs.gripper_name else None),
        leader_geom_ids=select_geoms_by_name(
            model,
            refs.prefix,
            # gello 外观下小臂几何在 group 1（FR3 网格已被替换）；孪生外观在 group 0。
            group=(1 if args.leader_appearance == "gello" else 0),
        ),
        follower_geom_ids=select_geoms_by_name(model, "fr3", group=0),
        distmax_m=max(0.3, float(cfg.safety.limits.collision_warn_m) * 5.0),
    )


def _write_follower(qpos, fr3_arm_adr, fr3_finger_adr, arm_q, finger_m) -> None:
    for adr, value in zip(fr3_arm_adr, arm_q):
        qpos[adr] = float(value)
    fm = float(np.clip(finger_m, 0.0, FINGER_MAX_M))
    for adr in fr3_finger_adr:
        qpos[adr] = fm


def run_interactive(loop, kin, drag_leader, refs, model, data, args, monitor, logger) -> int:
    try:
        import mujoco.viewer  # noqa: F401  # 子模块不会随 import mujoco 自动加载
    except Exception as exc:  # 依赖/显示不可用
        print(f"[interactive] 无法加载 MuJoCo viewer：{exc}", file=sys.stderr)
        print("[interactive] 可改用其它查看器，或修复显示环境。", file=sys.stderr)
        return 2

    leader_arm_adr = [_adr(model, n) for n in refs.arm_joint_names]
    grip_adr = _adr(model, refs.gripper_name) if refs.gripper_name else None
    fr3_arm_adr = [_adr(model, n) for n in ARM_JOINTS]
    fr3_finger_adr = [_adr(model, n) for n in FINGER_JOINTS]
    fr3_dof_adr = [_dof_adr(model, n) for n in (*ARM_JOINTS, *FINGER_JOINTS)]

    dt = float(model.opt.timestep)
    substeps = max(1, int(round((1.0 / 60.0) / dt)))

    thread = threading.Thread(
        target=loop.run, kwargs={"duration_s": args.duration}, daemon=True
    )
    thread.start()

    def text(msg: str, sub: str = "") -> list:
        return [(mujoco.mjtFontScale.mjFONTSCALE_150,
                 mujoco.mjtGridPos.mjGRID_TOPLEFT, msg, sub)]

    try:
        with mujoco.viewer.launch_passive(model, data) as handle:
            handle.cam.lookat[:] = [-args.separation * 0.5, 0.0, 0.35]
            handle.cam.distance = args.separation * 0.9 + 1.4
            handle.cam.azimuth = 90.0
            handle.cam.elevation = -18.0

            # 默认选中小臂末端：MuJoCo 要先选中 body 才施力，省去"先双击"这一步
            tip_body = _leader_tip_body_id(model, refs)
            if tip_body > 0:
                handle.perturb.select = int(tip_body)
                handle.perturb.localpos[:] = 0.0

            while handle.is_running():
                with handle.lock():
                    if tip_body > 0 and handle.perturb.select <= 0:
                        # 双击空白会清掉选择，这里兜底重新选上
                        handle.perturb.select = int(tip_body)
                    # 1) 大臂先对齐到 loop 最近一次下发
                    fq, fg = kin.current()
                    _write_follower(data.qpos, fr3_arm_adr, fr3_finger_adr, fq, fg)
                    for adr in fr3_dof_adr:
                        data.qvel[adr] = 0.0
                    # 2) 施加鼠标拖拽力，让小臂动力学走几步
                    for _ in range(substeps):
                        mujoco.mjv_applyPerturbForce(model, data, handle.perturb)
                        mujoco.mj_step(model, data)
                    # 3) 读回被拖出来的小臂状态，交给 TeleopLoop
                    ls = leader_state_from_qpos(data.qpos, leader_arm_adr, grip_adr)
                    drag_leader.set_state(ls)
                    # 4) 再写一次大臂，覆盖 step 期间的任何漂移
                    fq, fg = kin.current()
                    _write_follower(data.qpos, fr3_arm_adr, fr3_finger_adr, fq, fg)
                    for adr in fr3_dof_adr:
                        data.qvel[adr] = 0.0
                    mujoco.mj_forward(model, data)

                distance = monitor.last_collision_distance
                logger.frame(loop.stats.ticks, ls, fq, fg, distance)

                try:
                    if loop.stats.safety_stops > 0:
                        line1 = f"[安全停机] tick={loop.stats.ticks}  大臂已冻结"
                        line2 = "小臂仍可拖动；重新运行本脚本可恢复"
                    else:
                        d_txt = f"{distance * 1000:.0f}mm" if np.isfinite(distance) else "n/a"
                        line1 = "RUNNING   左=小臂(可拖)  右=真实 FR3  同号同色=对应关节"
                        line2 = (
                            f"tick={loop.stats.ticks}  两臂最近={d_txt}  "
                            f"拖拽=Ctrl+右键(平移)/Ctrl+左键(旋转)，已选中末端  "
                            f"夹爪={fg * 1000:.0f}mm"
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
def _start_rerun(args) -> None:
    import rerun as rr

    rr.init(APP_ID)
    if args.rerun_save:
        rr.save(args.rerun_save)
        print(f"[interactive] Rerun 录制写入 {args.rerun_save}（不弹窗）", flush=True)
    else:
        rr.spawn()
        print("[interactive] 已拉起 Rerun 查看器（曲线随拖拽更新）", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="交互式拖拽小臂 -> 真实 FR3 跟随（MuJoCo）")
    parser.add_argument("--config", default=str(ROOT / "examples/configs/leader_follower.yaml"))
    parser.add_argument("--separation", type=float, default=1.15, help="两臂基座间距（米）")
    parser.add_argument("--leader-scale", type=float, default=None,
                        help=f"小臂模型缩放（默认：gello 外观 {GELLO_LEADER_SCALE}，孪生 0.8）")
    parser.add_argument("--leader-appearance", choices=("gello", "twin"), default="twin",
                        help="小臂外观：twin=缩小 FR3 孪生（默认，连贯）；"
                             "gello=Franka 官方 GELLO 零件（装配位姿为反求近似，待官方 CAD 精确对齐）")
    parser.add_argument("--duration", type=float, default=None, help="遥操作运行时长（秒）")
    parser.add_argument("--no-collision-guard", action="store_true", help="关闭碰撞守卫")
    parser.add_argument("--hz", type=float, default=None, help="覆盖 loop.hz")
    parser.add_argument("--rerun", action="store_true", help="同时把曲线记到 Rerun（实时）")
    parser.add_argument("--rerun-save", default=None, help="把 Rerun 录制存成 .rrd（不弹窗）")
    args = parser.parse_args(argv)

    if not FR3_URDF.exists():
        print(f"[interactive] 找不到 FR3 URDF：{FR3_URDF}", file=sys.stderr)
        print(
            "[interactive] 先 staging 真实 FR3 描述：\n"
            "  pip install xacro trimesh pycollada\n"
            "  python tools/assets/fetch_fr3_description.py",
            file=sys.stderr,
        )
        return 2

    cfg = config_from_yaml(args.config)
    if args.hz:
        cfg.loop.hz = args.hz

    # --- 合并模型：真实 FR3 + 缩小的小臂（默认孪生外观；--leader-appearance gello 切零件外观） ---
    cache = Path(tempfile.gettempdir()) / "arm_control_fr3_mjcache"
    staged = build_mujoco_model(FR3_URDF, cache_dir=cache, keep_visual=True)
    use_gello = args.leader_appearance == "gello"
    leader_scale = args.leader_scale
    if leader_scale is None:
        leader_scale = GELLO_LEADER_SCALE if use_gello else 0.8
    spec, refs = build_combined_spec(
        str(staged),
        leader_position=(-args.separation, 0.0, 0.0),
        leader_scale=leader_scale,
        leader_gello_parts=use_gello,
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    initial = np.asarray(cfg.follower.initial, dtype=float)
    if initial.shape != (cfg.follower.n_arm_joints,):
        initial = np.asarray(FR3_HOME, dtype=float)[: cfg.follower.n_arm_joints]
    # follower 与 leader（FR3 孪生）从同一个合法 home 起步：auto_align 之后映射
    # 退化为直连，两条臂保持"同形"（只差一个缩放），关节对应关系最直观。
    for adr, value in zip((_adr(model, n) for n in ARM_JOINTS), initial):
        data.qpos[adr] = float(value)
    for adr in (_adr(model, n) for n in FINGER_JOINTS):
        data.qpos[adr] = float(cfg.follower.initial_finger_m)
    for adr, value in zip((_adr(model, n) for n in refs.arm_joint_names), initial):
        data.qpos[adr] = float(value)
    for name in _leader_gripper_joints(model, refs):
        data.qpos[_adr(model, name)] = GRIP_TRAVEL_M  # 初始张开
    mujoco.mj_forward(model, data)

    # --- 装配 TeleopLoop（与真机同一条安全链路）---
    drag_leader = DragLeader(cfg.leader.n_arm_joints, cfg.leader.with_gripper, grip_open=1.0)
    # 小臂初始状态 = FR3 home + 夹爪张开，保证 auto_align 时 offset ≈ 0
    drag_leader.set_state(np.concatenate([initial, [1.0]]))
    kin = KinematicFollower(cfg.follower.n_arm_joints, initial, cfg.follower.initial_finger_m)
    guard = _make_guard(args, model, refs, cfg)
    # 小臂是 FR3 孪生：关节映射改为直连，大臂才会和小臂同形跟动
    force_identity_arm_mapping(cfg)
    retargeter = build_retargeter(cfg.mapping)
    monitor = build_monitor(cfg.safety, cfg.mapping.joints, collision_guard=guard)
    loop = TeleopLoop(
        leader=drag_leader,
        retargeter=retargeter,
        follower=kin,
        monitor=monitor,
        feedback=FollowerFeedback(kin, armed=True),
        hz=cfg.loop.hz,
        auto_align=cfg.mapping.auto_align,
        log_period_s=cfg.loop.log_period_s,
    )

    logger = _RerunLogger(bool(args.rerun or args.rerun_save))
    if args.rerun or args.rerun_save:
        _start_rerun(args)

    print(
        f"[interactive] 窗口打开：左=小臂(FR3 孪生，Ctrl+右键拖动)，右=真实 FR3；"
        f"间距={args.separation:.2f}m  "
        f"碰撞守卫={'关' if args.no_collision_guard else '开'}",
        flush=True,
    )
    return run_interactive(loop, kin, drag_leader, refs, model, data, args, monitor, logger)


if __name__ == "__main__":
    raise SystemExit(main())
