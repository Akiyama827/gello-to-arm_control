"""小臂（leader）在仿真/可视化里的替身：**等比缩小的 FR3 视觉孪生**。

宇树 S288 没有公开的网格/URDF。早先这里用 capsule/box 程序化拼了一个"近似
外形"，但和真实 FR3 网格摆在一起完全不像，也看不出关节对应关系。现在改成直接
复用真实 FR3 的描述（同一套网格、同一套连杆坐标系），整体等比缩小后挂到 leader
一侧，于是：

    小臂 leader_fr3_joint_i  <->  大臂 fr3_joint_i     （i = 1..7，一一对应）
    小臂 leader_fr3_finger_* <->  大臂 fr3_finger_*    （Franka Hand 两指）

两条臂同构，任一转关节时"哪一节在动"一眼可对；再配合 :func:`color_arm_links`
把每对对应连杆涂成同一颜色，对应关系更直观（J1..J7 七彩，夹爪灰）。

注意
----
* 小臂只是**视觉/仿真替身**，不是真实 S288 的外观；真实 S288 的零位/转向标定
  仍走 ``S288LeaderArm`` 的 ``joint_offsets / joint_signs``，与渲染无关。
* 因为小臂与大臂同构，仿真里的关节映射应改为**直连**（``sign=+1``、``scale=1``），
  见 :func:`force_identity_arm_mapping`；否则大臂会和小臂"镜像"而不是"同形"。
* 夹爪沿用 Franka Hand 双指，读取时取一指并按行程 ``GRIP_TRAVEL_M`` 归一化，
  约定 **1 = 张开**（与 ``GripperMapping`` 一致）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import mujoco
import numpy as np

ARM_JOINT_COUNT = 7
# 夹爪单指行程（米），与 Franka Hand 的 fr3_finger_joint 行程同量级。
GRIP_TRAVEL_M = 0.04

# 每对对应连杆的颜色（J1..J7）；基座与夹爪另给中性色。两臂同号连杆同色。
JOINT_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.90, 0.22, 0.22, 1.0),  # J1 红
    (0.95, 0.55, 0.15, 1.0),  # J2 橙
    (0.92, 0.85, 0.20, 1.0),  # J3 黄
    (0.30, 0.75, 0.35, 1.0),  # J4 绿
    (0.20, 0.72, 0.80, 1.0),  # J5 青
    (0.30, 0.45, 0.92, 1.0),  # J6 蓝
    (0.62, 0.35, 0.86, 1.0),  # J7 紫
)
BASE_COLOR = (0.55, 0.55, 0.58, 1.0)
GRIPPER_COLOR = (0.82, 0.82, 0.84, 1.0)


@dataclass
class LeaderArmRefs:
    """挂到场景里之后的关键名字，方便调用方取 qpos / 找关节。"""

    prefix: str
    joint_names: list[str]
    gripper_name: Optional[str] = None

    @property
    def arm_joint_names(self) -> list[str]:
        return [n for n in self.joint_names if n != self.gripper_name]


def joint_color_for_link(name: str) -> tuple[float, float, float, float]:
    """按连杆名（``...linkN...`` / ``...hand...`` / ``...finger...``）取颜色。

    几何名形如 ``fr3_link3_visual`` / ``leader_fr3_link3_visual``，两臂同号连杆
    会落到同一个颜色上。
    """
    name = name or ""
    if "finger" in name or "hand" in name:
        return GRIPPER_COLOR
    match = re.search(r"link(\d+)", name)
    if match:
        index = int(match.group(1))
        if 1 <= index <= ARM_JOINT_COUNT:
            return JOINT_COLORS[index - 1]
    return BASE_COLOR


def color_arm_links(spec: mujoco.MjSpec) -> None:
    """给 `spec` 里所有**视觉**几何按连杆编号着色（两臂同号连杆同色）。

    只改 ``group == 1`` 的视觉几何；碰撞几何保持原样。follower 与 leader 一起
    调用，于是 ``fr3_link3_visual`` 与 ``leader_fr3_link3_visual`` 同色。
    """
    for geom in spec.geoms:
        if int(geom.group) != 1:
            continue
        name = geom.name or ""
        if not any(token in name for token in ("link", "hand", "finger")):
            continue
        geom.rgba = list(joint_color_for_link(name))


def _scale_spec(spec: mujoco.MjSpec, scale: float) -> None:
    """把一棵 spec 的几何/位形/惯量整体按 ``scale`` 缩放（绕它的根）。"""
    s = float(scale)

    def walk(body: mujoco.MjsBody) -> None:
        body.pos = (np.asarray(body.pos, dtype=float) * s).tolist()
        if bool(body.explicitinertial):
            # 质量 ~ s^3，惯量 ~ s^5，质心 ~ s
            body.mass = float(body.mass) * s ** 3
            body.ipos = (np.asarray(body.ipos, dtype=float) * s).tolist()
            full = np.asarray(body.fullinertia, dtype=float)
            if full.size == 6:
                body.fullinertia = (full * s ** 5).tolist()
            diag = np.asarray(body.inertia, dtype=float)
            if diag.size == 3 and np.any(diag):
                body.inertia = (diag * s ** 5).tolist()
        for geom in body.geoms:
            geom.pos = (np.asarray(geom.pos, dtype=float) * s).tolist()
            geom.size = (np.asarray(geom.size, dtype=float) * s).tolist()
            fromto = np.asarray(geom.fromto, dtype=float)
            # mesh 几何的 fromto 是 NaN，只有 capsule/cylinder 等才有意义
            if fromto.size == 6 and np.isfinite(fromto).all() and np.any(fromto):
                geom.fromto = (fromto * s).tolist()
        for joint in body.joints:
            joint.pos = (np.asarray(joint.pos, dtype=float) * s).tolist()
        for site in body.sites:
            site.pos = (np.asarray(site.pos, dtype=float) * s).tolist()
        for child in body.bodies:
            walk(child)

    walk(spec.worldbody)


def leader_state_from_qpos(
    qpos: np.ndarray,
    arm_qpos_adr: Sequence[int],
    grip_qpos_adr: Optional[int],
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
    spec: mujoco.MjSpec,
    fr3_spec: mujoco.MjSpec,
    *,
    position: Sequence[float] = (-1.0, 0.0, 0.30),
    quat: Optional[Sequence[float]] = None,
    scale: float = 1.0,
    prefix: str = "leader",
    with_gripper: bool = True,
    joint_limit_rad: Optional[float] = None,
    link_damping: float = 0.8,
    armature: float = 0.1,
) -> LeaderArmRefs:
    """把一棵 FR3 spec 等比缩小后作为小臂挂到 ``spec`` 的 ``position`` 处。

    ``fr3_spec`` 会被 :meth:`mujoco.MjSpec.attach` 消费（网格/几何复制到 ``spec``
    里、名字加 ``{prefix}_``）。关节默认**不设限位**（``joint_limit_rad=None``），
    这样可自由拖拽；仿真里真正的限位由安全层按 FR3 限位把关。
    """
    _scale_spec(fr3_spec, scale)
    mount = spec.worldbody.add_frame(
        name=f"{prefix}_mount",
        pos=list(position),
        quat=list(quat) if quat is not None else [1.0, 0.0, 0.0, 0.0],
    )
    spec.attach(fr3_spec, prefix=f"{prefix}_", frame=mount)

    arm_joints = [f"{prefix}_fr3_joint{i}" for i in range(1, ARM_JOINT_COUNT + 1)]
    finger_joints = [f"{prefix}_fr3_finger_joint{i}" for i in (1, 2)]

    for body in spec.bodies:
        if not (body.name or "").startswith(f"{prefix}_"):
            continue
        for geom in body.geoms:
            # 小臂只"看得见 + 可拖"，不与场景里的 FR3 产生接触力
            geom.contype = 0
            geom.conaffinity = 0
        for joint in body.joints:
            kind = int(joint.type)
            if kind == int(mujoco.mjtJoint.mjJNT_HINGE):
                joint.damping = [float(link_damping), 0.0, 0.0]
                joint.armature = float(armature)
                if joint_limit_rad is None:
                    joint.limited = False
                else:
                    joint.limited = True
                    joint.range = [-float(joint_limit_rad), float(joint_limit_rad)]
            elif kind == int(mujoco.mjtJoint.mjJNT_SLIDE):
                joint.damping = [0.05, 0.0, 0.0]

    return LeaderArmRefs(
        prefix=prefix,
        joint_names=arm_joints + (finger_joints[:1] if with_gripper else []),
        gripper_name=finger_joints[0] if with_gripper else None,
    )


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


def force_identity_arm_mapping(cfg) -> None:
    """把小臂当 FR3 孪生时，关节映射要直连，否则大臂会和小臂"镜像"。

    只动 7 个臂关节的 ``sign/offset/scale``；限位、``auto_align``、夹爪映射都
    保持配置原样。真实 S288 的标定仍以 ``leader_follower.yaml`` 为准。
    """
    cfg.mapping.joints = [
        replace(joint, sign=1.0, offset=0.0, scale=1.0)
        for joint in cfg.mapping.joints
    ]


def build_combined_spec(
    fr3_model_path: str,
    *,
    leader_position: Sequence[float] = (-1.0, 0.0, 0.30),
    leader_quat: Optional[Sequence[float]] = None,
    leader_scale: float = 1.0,
    leader_prefix: str = "leader",
    with_gripper: bool = True,
) -> tuple[mujoco.MjSpec, LeaderArmRefs]:
    """真实 FR3（follower）+ 等比缩小的 FR3 孪生（leader）合进一个 :class:`mujoco.MjSpec`。

    返回 ``(spec, refs)``，其中 ``spec`` 需再 ``spec.compile()``。两条臂都会按
    连杆编号着色（同号同色）。
    """
    spec = mujoco.MjSpec.from_file(str(fr3_model_path))
    spec.option.gravity = [0.0, 0.0, 0.0]
    leader_src = mujoco.MjSpec.from_file(str(fr3_model_path))
    refs = add_leader_arm(
        spec,
        leader_src,
        position=leader_position,
        quat=leader_quat,
        scale=leader_scale,
        prefix=leader_prefix,
        with_gripper=with_gripper,
    )
    color_arm_links(spec)
    return spec, refs
