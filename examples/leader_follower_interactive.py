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

真机驱动模式（``--leader-source config``）：
    用配置里的**真机 S288 小臂**驱动左臂模型，鼠标只用于转视角/缩放；
    适合边掰真臂边在原生窗口里看孪生，**用来标 ``joint_signs``**：

        python -B examples/leader_follower_interactive.py \\
            --config examples/configs/leader_follower_s288.local.yaml \\
            --leader-source config

    此时渲染线程不读串口，真机读数由 100 Hz 的 ``TeleopLoop`` 线程采样后转存。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

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
    build_leader,
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


def _hide_follower_geoms(model: mujoco.MjModel) -> int:
    """把大臂（FR3 follower）的视觉几何设为不可见。

    合并模型里 follower 的 body 名以 ``fr3`` 开头，leader 的是 ``leader_*``。
    只改 alpha（视觉），不动几何本身——安全守卫仍可查询。
    """
    hidden = 0
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        name = model.body(bid).name or ""
        if name.startswith("fr3"):
            model.geom_rgba[gid, 3] = 0.0
            hidden += 1
    return hidden


def _hide_leader_geoms(model: mujoco.MjModel) -> int:
    """把小臂（leader 孪生）的视觉几何设为不可见；只改 alpha，不动几何本身。"""
    hidden = 0
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        name = model.body(bid).name or ""
        if name.startswith("leader_"):
            model.geom_rgba[gid, 3] = 0.0
            hidden += 1
    return hidden


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


class _LeaderTap:
    """缓存真机 leader 最近一次 ``get_joint_state`` 的结果。

    TeleopLoop 在 100 Hz 线程里读真机；渲染线程不能再去读同一个串口，故用这个
    轻量缓存把"最近一次读数"转发给渲染线程。
    """

    def __init__(self, n_arm: int, with_gripper: bool) -> None:
        n = int(n_arm) + (1 if with_gripper else 0)
        self._state = np.zeros(n, dtype=float)
        if with_gripper:
            self._state[-1] = 1.0  # 1=张开
        self._lock = threading.Lock()

    def record(self, state: np.ndarray) -> None:
        s = np.asarray(state, dtype=float).ravel()
        with self._lock:
            self._state[: min(s.size, self._state.size)] = s[: self._state.size]

    def state(self) -> np.ndarray:
        with self._lock:
            return self._state.copy()


def _instrument_leader(leader, tap: "_LeaderTap") -> None:
    """包一层 ``get_joint_state``，把每次真机读数转存到 ``tap``。"""
    orig_get = leader.get_joint_state

    def get_joint_state():
        q = np.asarray(orig_get(), dtype=float)
        tap.record(q)
        return q

    leader.get_joint_state = get_joint_state  # type: ignore[method-assign]


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
class _SliderPanel:
    """独立 tkinter 小面板：每个小臂关节一个「滑块 + 跟随」复选框。

    * 勾选「跟随」= 该关节继续由鼠标拖拽控制（默认）；
    * 取消勾选    = 该关节**由滑条接管**（滑块优先），滑块值会被注入到小臂
      ``ls`` 的对应关节，再交给 ``TeleopLoop``。

    面板跑在独立 daemon 线程（``tkinter.mainloop``），通过线程安全的
    共享状态与 MuJoCo 渲染线程交换，互不阻塞。纯仿真、不碰真机。
    """

    def __init__(
        self,
        joint_names: tuple[str, ...],
        labels: tuple[str, ...],
        lower: Sequence[float],
        upper: Sequence[float],
        variants: int = 200,
        initial: Optional[Sequence[float]] = None,
    ) -> None:
        self._names = tuple(joint_names)
        self._labels = tuple(labels)
        n = len(self._names)
        if len(lower) != n or len(upper) != n:
            raise ValueError("lower/upper 长度必须与关节数一致")
        self._lower = [float(x) for x in lower]
        self._upper = [float(x) for x in upper]
        self._init = (
            [float(x) for x in initial]
            if initial is not None and len(initial) == n
            else None
        )
        # 线程安全共享状态（MuJoCo 循环读）：
        #   override[i] True -> 该关节用 slider_val[i]，否则跟随拖拽
        self._lock = threading.Lock()
        self._override = [False] * n
        self._value = [float(x) for x in lower]
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self._root = None
        self._vars: list = []
        self._chk: list = []
        self._variants = int(variants)
        # tkinter 延迟导入：无显示环境时用户可 --no-slider 跳过
        try:
            import tkinter as tk
        except Exception as exc:  # pragma: no cover - 取决于显示环境
            raise RuntimeError(
                f"无法导入 tkinter（{exc}）。无图形环境请用 --no-slider 关闭滑块面板。"
            ) from exc
        self._tk = tk

    # -- 状态读取（供 MuJoCo 渲染线程调用，非阻塞）--------------------------------
    def overrides(self) -> list[tuple[int, float]]:
        """返回 (关节下标, 滑块值) 列表：所有被接管（未勾选跟随）的关节。"""
        with self._lock:
            return [(i, self._value[i]) for i in range(len(self._value))
                    if self._override[i]]

    def launch(self) -> None:
        """在独立 daemon 线程里启动 tkinter 主循环。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="slider-panel"
        )
        self._thread.start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    # -- tkinter 侧 -----------------------------------------------------------------
    def _run_loop(self) -> None:  # pragma: no cover - GUI 线程
        import tkinter as tk

        root = tk.Tk()
        self._root = root
        root.title("小臂关节滑条（并排面板）")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        # 自动贴合到 MuJoCo 窗口右侧：放在屏幕靠右，便于和画面并排观察
        try:
            w = root.winfo_screenwidth()
            h = root.winfo_screenheight()
            pw, ph = 300, min(320, int(h * 0.9))
            x = max(0, w - pw - 20)
            y = max(0, (h - ph) // 2)
            root.wm_geometry(f"{pw}x{ph}+{x}+{y}")
        except Exception:
            pass
        n = len(self._names)
        self._last_vals: list[float] = []

        def on_var(_idx: int) -> "tk.BooleanVar":
            from tkinter import IntVar

            var = IntVar(value=1)  # 默认跟随
            self._chk.append(var)
            return var

        self._scales: list = []
        self._labels_widgets: list = []
        for i, name in enumerate(self._names):
            row = tk.Frame(root)
            row.pack(fill="x", padx=4, pady=1)
            chk = on_var(i)

            def _toggle(idx: int = i):
                with self._lock:
                    # 1 = 跟随(释放)；0 = 接管
                    self._override[idx] = self._chk[idx].get() == 0
                    if self._override[idx]:
                        self._value[idx] = float(self._vars[idx].get())

            from tkinter import DoubleVar, Scale, Checkbutton

            low, high = self._lower[i], self._upper[i]
            sv = DoubleVar(
                value=self._init[i] if self._init else (low + high) / 2.0
            )
            self._vars.append(sv)
            self._last_vals.append(float(sv.get()))
            # 关节名 + 实时角度值
            val_lbl = tk.Label(row, text=f"{self._labels[i]}  {sv.get():+.2f} rad",
                               width=12, anchor="w")
            val_lbl.pack(side="left")
            self._labels_widgets.append(val_lbl)
            Checkbutton(
                row, text="跟随", variable=chk, command=_toggle
            ).pack(side="left")

            # command 为辅助触发（真实拖动有效时锦上添花）；主机制靠下面的
            # 定时轮询 _sync_from_widgets（100% 可靠，不依赖任何回调）。
            def _on_command(_val=None, idx: int = i, lbl: "tk.Label" = val_lbl):
                self._sync_one(idx)
                try:
                    lbl.configure(
                        text=f"{self._labels[idx]}  {self._vars[idx].get():+.2f} rad"
                    )
                except Exception:
                    pass

            sc = Scale(
                row,
                from_=low,
                to=high,
                resolution=(high - low) / self._variants,
                orient="horizontal",
                variable=sv,
                command=_on_command,
                length=210,
                showvalue=False,
            )
            sc.pack(side="left", fill="x", expand=True)
            self._scales.append(sc)

        foot = tk.Label(
            root,
            text="拖滑条=接管该关节并实时到位；勾选「跟随」=释放回拖拽。",
            fg="#555",
        )
        foot.pack(pady=2)
        # 定时轮询：拖动滑块必然被检测到（比 trace/command 可靠）
        try:
            root.after(40, self._sync_loop)
        except Exception:
            pass
        root.mainloop()

    def _sync_one(self, idx: int) -> None:  # pragma: no cover - GUI 线程
        try:
            if idx >= len(self._vars):
                return
            val = float(self._vars[idx].get())
            with self._lock:
                self._value[idx] = val
                self._override[idx] = True  # 拖动即接管
            try:
                self._chk[idx].set(0)
            except Exception:
                pass
        except Exception:
            pass

    def _sync_loop(self) -> None:  # pragma: no cover - GUI 线程
        """定时轮询每个滑块的当前值；值变了就把该关节标记为接管并同步。"""
        try:
            for i in range(len(self._vars)):
                cur = float(self._vars[i].get())
                if abs(cur - self._last_vals[i]) > 1e-9:
                    self._last_vals[i] = cur
                    self._sync_one(i)
                    try:
                        self._labels_widgets[i].configure(
                            text=f"{self._labels[i]}  {cur:+.2f} rad"
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self._root.after(40, self._sync_loop)
        except Exception:
            pass

    def _on_close(self) -> None:  # pragma: no cover - GUI 线程
        with self._lock:
            self._closed = True
            # 关闭时释放所有接管，恢复到纯拖拽
            self._override[:] = [False] * len(self._override)
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass


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


def run_interactive(
    loop,
    kin,
    drag_leader,
    refs,
    model,
    data,
    args,
    monitor,
    logger,
    *,
    tap=None,
    use_config_leader: bool = False,
    slider: Optional["_SliderPanel"] = None,
    real_feedback: Optional[object] = None,
    teleop: Optional[dict] = None,
    spawn_teleop: Optional[Callable[[], None]] = None,
    send_arm: Optional[Callable[[bool], None]] = None,
) -> int:
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
    render_fps = 120.0
    substeps = max(1, int(round((1.0 / render_fps) / dt)))

    def _text(lines) -> list:
        """lines: 可迭代的 (grid_pos, msg, sub)。"""
        return [
            (mujoco.mjtFontScale.mjFONTSCALE_130, pos, msg, sub)
            for pos, msg, sub in lines
        ]

    def _arm_line(tag: str, vec, tail: str = "") -> str:
        """一行关节角（rad，2 位小数），便于逐关节比对。"""
        n = len(leader_arm_adr)
        parts = [f"J{i + 1}={vec[i]:+.2f}" for i in range(n)]
        line = f"{tag} " + " ".join(parts)
        return f"{line}  {tail}" if tail else line

    # 人工在环 ARM/STOP 键盘：仅 dora 真机模式启用（send_arm 非 None）。
    # 键值用 glfw 常量；此回调由 MuJoCo passive viewer 在按键时调用。
    def _key_callback(key_code: int) -> None:
        if send_arm is None:
            return
        try:
            import glfw
        except Exception:
            return
        if key_code == glfw.KEY_A:  # ARM
            send_arm(True)
        elif key_code == glfw.KEY_S:  # STOP / DISARM
            send_arm(False)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as handle:
            handle.cam.lookat[:] = [-args.separation * 0.5, 0.0, 0.35]
            handle.cam.distance = args.separation * 0.9 + 1.4
            handle.cam.azimuth = 90.0
            handle.cam.elevation = -18.0

            # 只有鼠标拖拽模式才需要"选中末端"以施力；真机模式鼠标只用于转视角
            tip_body = 0 if use_config_leader else _leader_tip_body_id(model, refs)
            if tip_body > 0:
                # 默认选中小臂末端：MuJoCo 要先选中 body 才施力，省去"先双击"这一步
                handle.perturb.select = int(tip_body)
                handle.perturb.localpos[:] = 0.0

            n_leader_arm = len(leader_arm_adr)
            ls = np.zeros(n_leader_arm + (1 if grip_adr is not None else 0))
            fq, fg = kin.current()

            def _display_follower():
                """返回用于场景大臂显示的 (arm_q, finger_m)。

                dora 真机模式优先用真机实测位形回填（真实位置），否则退回
                ``kin`` 的仿真目标（保持原有行为）。
                """
                if real_feedback is not None:
                    real_q = None
                    try:
                        real_q = real_feedback.cached_arm_state()
                    except Exception:
                        real_q = None
                    if real_q is not None and real_q.shape == (len(fr3_arm_adr),):
                        return real_q, fg
                return kin.current()

            while handle.is_running():
                # 兼容调用方未传 teleop/spawn（老签名/其它入口）
                _teleop = teleop or {}
                _spawn = spawn_teleop or (lambda: None)
                # dora 模式 + 真机 ARM 后自动恢复：loop 因未使能安全停机后，
                # 一旦真机被人工 ARM（armed=True 回读到位），自动重建续跑。
                # teleop 线程停死后由渲染线程接管 drain，保持 armed/位形最新。
                if real_feedback is not None:
                    cur_thread = _teleop.get("thread")
                    if cur_thread is not None and not cur_thread.is_alive():
                        try:
                            real_feedback.drain()  # 只在 teleop 停时 drain，避免争抢
                            armed = real_feedback.cached_armed()
                        except Exception:
                            armed = None
                        if armed:
                            print(
                                "[interactive] 检测到真机已 ARM，自动恢复跟随...",
                                flush=True,
                            )
                            _spawn()
                            loop = _teleop.get("loop", loop)
                with handle.lock():
                    if use_config_leader and tap is not None:
                        # 1) 小臂 = 真机最近一次读数（由 loop 线程 100Hz 采样）
                        ls = tap.state()
                        for i, adr in enumerate(leader_arm_adr):
                            data.qpos[adr] = float(ls[i])
                        if grip_adr is not None and ls.size > n_leader_arm:
                            data.qpos[grip_adr] = (
                                float(np.clip(ls[n_leader_arm], 0.0, 1.0)) * GRIP_TRAVEL_M
                            )
                        for adr in leader_arm_adr:
                            data.qvel[adr] = 0.0
                        # 2) 大臂对齐到 loop 最近一次下发
                        fq, fg = _display_follower()
                        _write_follower(data.qpos, fr3_arm_adr, fr3_finger_adr, fq, fg)
                        for adr in fr3_dof_adr:
                            data.qvel[adr] = 0.0
                        mujoco.mj_forward(model, data)
                    else:
                        if tip_body > 0 and handle.perturb.select <= 0:
                            # 双击空白会清掉选择，这里兜底重新选上
                            handle.perturb.select = int(tip_body)
                        # 1) 大臂先对齐到 loop 最近一次下发
                        fq, fg = _display_follower()
                        _write_follower(data.qpos, fr3_arm_adr, fr3_finger_adr, fq, fg)
                        for adr in fr3_dof_adr:
                            data.qvel[adr] = 0.0
                        # 2) 施加鼠标拖拽力，让小臂动力学走几步
                        for _ in range(substeps):
                            mujoco.mjv_applyPerturbForce(model, data, handle.perturb)
                            mujoco.mj_step(model, data)
                        # 3) 读回被拖出来的小臂状态，交给 TeleopLoop
                        ls = leader_state_from_qpos(data.qpos, leader_arm_adr, grip_adr)
                        if slider is not None:
                            # 滑块优先：被接管的关节用滑块值覆写小臂状态
                            for i, val in slider.overrides():
                                if 0 <= i < len(ls):
                                    ls[i] = float(val)
                                    # 视觉同步：窗口里小臂该关节也转到滑块角度
                                    if i < len(leader_arm_adr):
                                        data.qpos[leader_arm_adr[i]] = float(val)
                                        data.qvel[leader_arm_adr[i]] = 0.0
                        drag_leader.set_state(ls)
                        # 4) 再写一次大臂，覆盖 step 期间的任何漂移
                        fq, fg = _display_follower()
                        _write_follower(data.qpos, fr3_arm_adr, fr3_finger_adr, fq, fg)
                        for adr in fr3_dof_adr:
                            data.qvel[adr] = 0.0
                        mujoco.mj_forward(model, data)

                monitor = _teleop.get("monitor", monitor)
                distance = monitor.last_collision_distance
                # ARM 后自动重启 teleop 时更新本地引用，显示/统计跟最新一次 loop
                loop = _teleop.get("loop", loop)
                logger.frame(loop.stats.ticks, ls, fq, fg, distance)

                try:
                    if loop.stats.safety_stops > 0:
                        line1 = f"[安全停机] tick={loop.stats.ticks}  大臂已冻结"
                        line2 = (
                            "真机小臂仍在跟随；重新运行本脚本可恢复"
                            if use_config_leader
                            else "小臂仍可拖动；重新运行本脚本可恢复"
                        )
                    else:
                        d_txt = f"{distance * 1000:.0f}mm" if np.isfinite(distance) else "n/a"
                        right_txt = "" if args.hide_follower else "  右=真实 FR3"
                        # 人工在环提示：仅 dora 真机模式显示 ARM/STOP 热键
                        arm_hint = "  A=ARM  S=STOP" if send_arm is not None else ""
                        if use_config_leader:
                            left_txt = "" if args.hide_leader else "  左=真机小臂(实时)"
                            line1 = (
                                f"RUNNING{left_txt}{right_txt}  "
                                f"同号同色=对应关节"
                            )
                            line2 = (
                                f"tick={loop.stats.ticks}  两臂最近={d_txt}  "
                                f"鼠标=左键旋转/右键平移  "
                                f"夹爪指令={fg * 1000:.0f}mm{arm_hint}"
                            )
                        else:
                            left_txt = "" if args.hide_leader else "  左=小臂(可拖)"
                            line1 = (
                                f"RUNNING{left_txt}{right_txt}  "
                                f"同号同色=对应关节"
                            )
                            line2 = (
                                f"tick={loop.stats.ticks}  两臂最近={d_txt}  "
                                f"拖拽=Ctrl+右键(平移)/Ctrl+左键(旋转)，已选中末端  "
                                f"夹爪指令={fg * 1000:.0f}mm{arm_hint}"
                            )
                    lead_tail = (
                        f"grip={ls[len(leader_arm_adr)]:+.2f}"
                        if ls.size > len(leader_arm_adr)
                        else ""
                    )
                    handle.set_texts(_text([
                        (mujoco.mjtGridPos.mjGRID_TOPLEFT, line1,
                         _arm_line("小臂L(rad)", ls, lead_tail)),
                        (mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, line2,
                         _arm_line("大臂F(rad)", fq, f"grip={fg * 1000:.0f}mm")),
                    ]))
                except Exception:
                    pass
                handle.sync()
                time.sleep(1.0 / render_fps)
    finally:
        _teleop = teleop or {}
        cur_loop = _teleop.get("loop", loop)
        cur_thread = _teleop.get("thread", thread)
        cur_loop.request_stop()
        cur_thread.join(timeout=2.0)
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
    parser.add_argument("--config",
                        default=os.environ.get("LEADER_FOLLOWER_CONFIG")
                        or str(ROOT / "examples/configs/leader_follower.yaml"))
    parser.add_argument("--separation", type=float, default=1.15, help="两臂基座间距（米）")
    parser.add_argument("--leader-scale", type=float, default=None,
                        help=f"小臂模型缩放（默认：gello 外观 {GELLO_LEADER_SCALE}，孪生 0.8）")
    parser.add_argument("--leader-appearance", choices=("gello", "twin"), default="twin",
                        help="小臂外观：twin=缩小 FR3 孪生（默认，连贯）；"
                             "gello=Franka 官方 GELLO 零件（装配位姿为反求近似，待官方 CAD 精确对齐）")
    parser.add_argument("--duration", type=float, default=None, help="遥操作运行时长（秒）")
    parser.add_argument("--no-collision-guard", action="store_true", help="关闭碰撞守卫")
    parser.add_argument("--leader-source", choices=("drag", "config"), default="drag",
                        help="小臂来源：drag=鼠标拖拽虚拟小臂（默认）；"
                             "config=读配置里的真机小臂（S288），此时鼠标只用于转视角")
    parser.add_argument("--hide-follower", action="store_true",
                        help="隐藏大臂（FR3 follower），只看小臂")
    parser.add_argument("--hide-leader", action="store_true",
                        help="隐藏小臂（leader 孪生），只看大臂")
    parser.add_argument("--hz", type=float, default=None, help="覆盖 loop.hz")
    parser.add_argument("--rerun", action="store_true", help="同时把曲线记到 Rerun（实时）")
    parser.add_argument("--rerun-save", default=None, help="把 Rerun 录制存成 .rrd（不弹窗）")
    parser.add_argument("--no-slider", action="store_true",
                        help="不弹 tkinter 关节滑块面板（无显示环境时用）")
    parser.add_argument("--follower", choices=("kin", "dora"), default=None,
                        help="大臂下发后端：kin=仿真/随从（默认）；"
                             "dora=经 Dora 图发 jog 到 supervised_arm->真机 FR3"
                             "（需在 dora 图里作为节点运行，配 follower.kind=dora）。"
                             "也可用环境变量 LEADER_FOLLOWER_DORA=1 触发（dora 图内用）")
    args = parser.parse_args(argv)
    if args.follower is None:
        # dora 图节点用 LEADER_FOLLOWER_DORA=1 触发真机模式，不必写死命令行
        args.follower = "dora" if os.environ.get("LEADER_FOLLOWER_DORA", "") == "1" else "kin"

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
    if args.hide_follower:
        n_hidden = _hide_follower_geoms(model)
        print(f"[interactive] 已隐藏大臂（{n_hidden} 个几何）", flush=True)
    if args.hide_leader:
        n_hidden = _hide_leader_geoms(model)
        print(f"[interactive] 已隐藏小臂（{n_hidden} 个几何）", flush=True)
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
    use_config_leader = args.leader_source == "config"
    drag_leader = DragLeader(cfg.leader.n_arm_joints, cfg.leader.with_gripper, grip_open=1.0)
    # 小臂初始状态 = FR3 home + 夹爪张开，保证 auto_align 时 offset ≈ 0
    drag_leader.set_state(np.concatenate([initial, [1.0]]))
    tap = None
    if use_config_leader:
        # 真机小臂：复用与其它示例同一条 build_leader（串口在 build_leader 里打开）
        leader = build_leader(cfg.leader)
        tap = _LeaderTap(cfg.leader.n_arm_joints, cfg.leader.with_gripper)
        _instrument_leader(leader, tap)
        print(
            f"[interactive] 小臂来源：真机（kind={cfg.leader.kind}, bus={cfg.leader.bus}）",
            flush=True,
        )
    else:
        leader = drag_leader
    kin = KinematicFollower(cfg.follower.n_arm_joints, initial, cfg.follower.initial_finger_m)
    guard = _make_guard(args, model, refs, cfg)
    if not use_config_leader:
        # 小臂是 FR3 孪生：关节映射改为直连，大臂才会和小臂同形跟动
        force_identity_arm_mapping(cfg)

    # --- follower / 反馈：仿真（kin）或真机（dora）---
    real_feedback = None  # dora 模式：真机实测位形回读（场景显示 + 安全反馈）
    if args.follower == "dora":
        from arm_control.leader_follower.dora_feedback import DoraFollowerFeedback
        from arm_control.leader_follower.follower import DoraJogFollower
        if cfg.follower.kind != "dora":
            raise SystemExit(
                "--follower dora 需要配置里 follower.kind=dora（LEADER_FOLLOWER_CONFIG）。"
            )
        from dora import Node

        dora_node = Node()
        # 安全：不让节点自动使能（auto_arm=False）。ARM 由操作台人工执行。
        kin_follower = DoraJogFollower(
            cfg.follower.n_arm_joints,
            auto_arm=False,
            node=dora_node,
        )
        kin_follower.state_provider = None  # auto_align 基准由 prime 后回读注入
        real_feedback = DoraFollowerFeedback(dora_node, cfg.follower.n_arm_joints)
        kin_follower.state_provider = real_feedback.latest_arm_state
        # 首帧真机位形回来后再 auto_align，避免拿零点对齐（fr3_joint4 零位越限）
        real_feedback.prime()
        print("[interactive] 大臂后端：Dora -> supervised_arm -> 真机 FR3（未 ARM，人工使能）",
              flush=True)
        follower = kin_follower
        loop_feedback = real_feedback.sample

        # 人工在环 ARM/STOP：往 supervised_arm 发 control(arm=...) 使能/失能真机。
        # 这只是"按键 -> 一条 control 消息"；真机是否真的 ARM/DISARM 由
        # supervisor 裁决（未 ARM 时 STOP 仍安全；ARM 前请确认周围无人+急停在手）。
        def _send_arm(armed: bool) -> None:
            from arm_control.contracts.motion import pack_control_update

            dora_node.send_output("control", pack_control_update(arm=bool(armed)))
            print(
                f"[interactive] 人工{'ARM' if armed else 'STOP/DISARM'}："
                f"已发 control(arm={bool(armed)})",
                flush=True,
            )

    else:
        follower = kin
        loop_feedback = FollowerFeedback(kin, armed=True)
        _send_arm = None

    # dora 模式支持"人工 ARM 后自动恢复跟随"：未使能时 loop 会因
    # `大臂已失能` 安全停机（安全门：未 ARM 不允许下发 jog）；一旦操作台
    # ARM 使能，渲染线程检测到 armed 就重建 teleop loop 自动续跑，无需重启
    # 脚本。重建只换成新鲜的 retargeter/monitor（避免停机态残留），而
    # leader / follower / 反馈复用同一份（share 同一个 dora 节点，不相撞）。
    teleop = {}  # 存放当前 (loop, thread)；渲染线程按需更新/读取

    def _spawn_teleop():
        """（重新）装配 TeleopLoop 并启动其线程，更新 ``teleop`` 容器。"""
        old_thread = teleop.get("thread")
        if old_thread is not None and old_thread.is_alive():
            return  # 仍在跑，不重复启动
        if old_thread is not None:
            old_thread.join(timeout=2.0)
        fresh_retargeter = build_retargeter(cfg.mapping)
        fresh_monitor = build_monitor(
            cfg.safety, cfg.mapping.joints, collision_guard=guard
        )
        fresh_loop = TeleopLoop(
            leader=leader,
            retargeter=fresh_retargeter,
            follower=follower,
            monitor=fresh_monitor,
            feedback=loop_feedback,
            hz=cfg.loop.hz,
            auto_align=cfg.mapping.auto_align,
            log_period_s=cfg.loop.log_period_s,
        )
        fresh_thread = threading.Thread(
            target=fresh_loop.run,
            kwargs={"duration_s": args.duration},
            daemon=True,
        )
        teleop["loop"], teleop["thread"] = fresh_loop, fresh_thread
        teleop["monitor"] = fresh_monitor
        fresh_thread.start()

    _spawn_teleop()
    loop = teleop["loop"]
    thread = teleop["thread"]
    monitor = teleop["monitor"]

    logger = _RerunLogger(bool(args.rerun or args.rerun_save))
    if args.rerun or args.rerun_save:
        _start_rerun(args)

    if use_config_leader:
        print(
            f"[interactive] 窗口打开：左=真机小臂(实时)，右=真实 FR3；"
            f"间距={args.separation:.2f}m  "
            f"碰撞守卫={'关' if args.no_collision_guard else '开'}",
            flush=True,
        )
    else:
        print(
            f"[interactive] 窗口打开：左=小臂(FR3 孪生，Ctrl+右键拖动)，右=真实 FR3；"
            f"间距={args.separation:.2f}m  "
            f"碰撞守卫={'关' if args.no_collision_guard else '开'}",
            flush=True,
        )
    slider = None
    if not use_config_leader and not args.no_slider:
        try:
            lo = list(cfg.safety.joint_lower)
            hi = list(cfg.safety.joint_upper)
            if len(lo) < cfg.leader.n_arm_joints or len(hi) < cfg.leader.n_arm_joints:
                lo = [x for x in lo] + [0.0] * (cfg.leader.n_arm_joints - len(lo))
                hi = [x for x in hi] + [0.0] * (cfg.leader.n_arm_joints - len(hi))
            slider = _SliderPanel(
                tuple(refs.arm_joint_names),
                tuple(f"J{i + 1}" for i in range(cfg.leader.n_arm_joints)),
                lo[: cfg.leader.n_arm_joints],
                hi[: cfg.leader.n_arm_joints],
                initial=tuple(initial)[: cfg.leader.n_arm_joints],
            )
            slider.launch()
            print("[interactive] 已开启关节滑块面板（Slide 控制；勾选跟随=拖拽）", flush=True)
        except Exception as exc:
            slider = None
            print(f"[interactive] 滑块面板不可用（已跳过）：{exc}", file=sys.stderr)
    try:
        return run_interactive(
            loop,
            kin,
            drag_leader,
            refs,
            model,
            data,
            args,
            monitor,
            logger,
            tap=tap,
            use_config_leader=use_config_leader,
            slider=slider,
            real_feedback=real_feedback,
            teleop=teleop,
            spawn_teleop=_spawn_teleop,
            send_arm=_send_arm,
        )
    finally:
        if slider is not None:
            slider.close()
        if use_config_leader:
            leader.close()


if __name__ == "__main__":
    # mujoco 3.x 的 ``MjSpec.attach`` 在解释器退出、GC 清理 spec 时会段错误
    # （仅在"真正退出/被回收"时触发；独立交互运行因循环不退出而不必暴露，
    #  但在 dora 节点进程里节点返回主线程后必然触发）。
    # 规避：flush 后用 ``os._exit`` 终止进程，跳过解释器的对象析构阶段。
    # 正常返回与异常路径都走这里，避免异常回溯清理时代理 spec 崩溃。
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)
