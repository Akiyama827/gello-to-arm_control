"""程序化的宇树 S288 小臂（leader）模型（**近似外形**，非官方 CAD）。

S288 没有公开的网格/URDF，这里用一组 capsule/cylinder/box 拼出"基座 + 7 段
连杆 + 夹爪"的外形，供三处共用，保证"小臂长什么样"只有一处定义：

* ``examples/leader_follower_rerun.py``       在真实 FR3 旁画出小臂模型；
* ``examples/leader_follower_interactive.py`` 把小臂放进可拖拽的 MuJoCo 场景；
* 其它需要小臂外形的可视化。

约定（与仓库原有骨架一致）
--------------------------------------------------------------------------
* 关节轴序：``z, y, y, x, y, x, y``（基座 yaw + 平面 pitch + 前臂/腕 roll）；
* 每节连杆沿局部 ``+x`` 伸出，下一关节位于上一节末端；
* 夹爪是最后一个 **prismatic** 关节，行程 ``[0, GRIP_TRAVEL_M]``，位置越大
  越张开（与 ``GripperMapping`` 的 ``[0,1]``、``1=张开`` 对齐）。

这些都写在小臂模型里；真实零位/转向的标定仍走 ``S288LeaderArm`` 的
``joint_offsets / joint_signs``，与渲染无关。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import mujoco
import numpy as np

# (关节轴, 连杆长 m, 半径 m)
LEADER_LINKS: tuple[tuple[str, float, float], ...] = (
    ("0 0 1", 0.10, 0.050),
    ("0 1 0", 0.30, 0.048),
    ("0 1 0", 0.27, 0.042),
    ("1 0 0", 0.11, 0.034),
    ("0 1 0", 0.10, 0.030),
    ("1 0 0", 0.08, 0.027),
    ("0 1 0", 0.07, 0.024),
)
N_ARM = len(LEADER_LINKS)

# 夹爪单指行程（米）。与 FR3 Hand 同量级，纯粹为了观感与"1=张开"的归一化。
GRIP_TRAVEL_M = 0.04


def leader_joint_names(prefix: str = "leader", with_gripper: bool = True) -> list[str]:
    """小臂模型里的关节名：``{prefix}_j1..j7``（+ ``{prefix}_grip``）。"""
    names = [f"{prefix}_j{i + 1}" for i in range(N_ARM)]
    if with_gripper:
        names.append(f"{prefix}_grip")
    return names


@dataclass
class LeaderArmRefs:
    """挂到场景里之后的关键名字，方便调用方取 qpos / 找关节。"""

    prefix: str
    joint_names: list[str]
    gripper_name: str | None = None

    @property
    def arm_joint_names(self) -> list[str]:
        return [n for n in self.joint_names if n != self.gripper_name]


def leader_state_from_qpos(
    qpos: np.ndarray,
    arm_qpos_adr: Sequence[int],
    grip_qpos_adr: int | None,
    *,
    grip_travel_m: float = GRIP_TRAVEL_M,
) -> np.ndarray:
    """从 MuJoCo 的 ``qpos`` 取小臂状态：7 臂关节(rad) + 夹爪归一化 ``[0,1]``。

    夹爪 **1 = 张开**，与 ``S288LeaderArm`` / ``GripperMapping`` 的约定一致。
    """
    arm = np.array([float(qpos[a]) for a in arm_qpos_adr], dtype=float)
    if grip_qpos_adr is None:
        return arm
    g = float(np.clip(qpos[grip_qpos_adr] / max(grip_travel_m, 1e-9), 0.0, 1.0))
    return np.concatenate([arm, [g]])


def add_leader_arm(
    parent,
    *,
    position: Sequence[float] = (-1.0, 0.0, 0.30),
    quat: Sequence[float] | None = None,
    scale: float = 1.0,
    prefix: str = "leader",
    with_gripper: bool = True,
    rgba: Sequence[float] = (0.20, 0.62, 1.0, 1.0),
    joint_limit_rad: float = 2.8,
    link_damping: float = 0.5,
    visual_group: int = 1,
    disable_contacts: bool = True,
) -> LeaderArmRefs:
    """把一条 7-DOF 小臂（+夹爪）挂到 ``parent``（通常是 ``spec.worldbody``）。

    返回 :class:`LeaderArmRefs`。几何默认放 visual group、关闭接触，这样它
    只是"看得见 + 可被鼠标拖拽"，不会和场景里的 FR3 发生接触力。
    """
    kw = {
        "group": int(visual_group),
        "rgba": list(rgba),
    }
    if disable_contacts:
        kw["contype"] = 0
        kw["conaffinity"] = 0

    base = parent.add_body(
        name=f"{prefix}_base",
        pos=list(position),
        quat=list(quat) if quat is not None else [1.0, 0.0, 0.0, 0.0],
    )
    # 基座用 box（而非 cylinder）：Rerun 的 Boxes3D/Capsules3D 定位语义明确，
    # 省得在渲染侧再纠结 cylinder 是"以中心还是以端点为原点"。
    base.add_geom(
        name=f"{prefix}_pedestal",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.07 * scale, 0.07 * scale, 0.03 * scale],
        pos=[0.0, 0.0, -0.03 * scale],
        **kw,
    )

    body = base
    prev = 0.0
    for i, (axis, length, radius) in enumerate(LEADER_LINKS):
        ln = float(length) * scale
        r = float(radius) * scale
        b = body.add_body(name=f"{prefix}_l{i + 1}", pos=[prev, 0.0, 0.0])
        b.add_joint(
            name=f"{prefix}_j{i + 1}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[float(a) for a in str(axis).split()],
            range=[-float(joint_limit_rad), float(joint_limit_rad)],
            damping=float(link_damping),
        )
        b.add_geom(
            name=f"{prefix}_link{i + 1}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, ln, 0.0, 0.0],
            size=[r, 0.0, 0.0],
            **kw,
        )
        body = b
        prev = ln

    refs = LeaderArmRefs(
        prefix=prefix,
        joint_names=leader_joint_names(prefix, with_gripper),
        gripper_name=f"{prefix}_grip" if with_gripper else None,
    )

    if with_gripper:
        hand = body.add_body(name=f"{prefix}_hand", pos=[prev, 0.0, 0.0])
        hand.add_geom(
            name=f"{prefix}_palm",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02 * scale, 0.05 * scale, 0.03 * scale],
            **kw,
        )
        # 固定指
        hand.add_geom(
            name=f"{prefix}_finger_fixed",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.015 * scale, 0.006 * scale, 0.025 * scale],
            pos=[0.055 * scale, -0.030 * scale, 0.0],
            **kw,
        )
        # 动指：沿 +y 平移，位置越大越张开
        moving = hand.add_body(
            name=f"{prefix}_finger", pos=[0.055 * scale, -0.005 * scale, 0.0]
        )
        moving.add_joint(
            name=refs.gripper_name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 1.0, 0.0],
            range=[0.0, float(GRIP_TRAVEL_M)],
            damping=0.05,
        )
        moving.add_geom(
            name=f"{prefix}_finger_moving",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.015 * scale, 0.006 * scale, 0.025 * scale],
            **kw,
        )

    return refs


def joint_qpos_addresses(
    model: mujoco.MjModel, names: Sequence[str]
) -> dict[str, int]:
    """名字 -> ``qpos`` 下标；不存在的关节直接跳过。"""
    out: dict[str, int] = {}
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            out[name] = int(model.jnt_qposadr[jid])
    return out


def build_combined_spec(
    fr3_model_path: str,
    *,
    leader_position: Sequence[float] = (-1.0, 0.0, 0.30),
    leader_quat: Sequence[float] | None = None,
    leader_scale: float = 1.0,
    leader_prefix: str = "leader",
    leader_rgba: Sequence[float] = (0.20, 0.62, 1.0, 1.0),
    with_gripper: bool = True,
) -> tuple[mujoco.MjSpec, LeaderArmRefs]:
    """把真实 FR3 与小臂模型合进同一个 :class:`mujoco.MjSpec`。

    FR3 走 staged 的 URDF/MJCF（``build_mujoco_model`` 的产物），小臂由
    :func:`add_leader_arm` 程序化生成。返回 ``(spec, refs)``，其中 ``spec`` 需
    再 ``spec.compile()``。
    """
    spec = mujoco.MjSpec.from_file(str(fr3_model_path))
    spec.option.gravity = [0.0, 0.0, 0.0]
    refs = add_leader_arm(
        spec.worldbody,
        position=leader_position,
        quat=leader_quat,
        scale=leader_scale,
        prefix=leader_prefix,
        rgba=leader_rgba,
        with_gripper=with_gripper,
    )
    return spec, refs
