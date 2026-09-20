"""小臂（leader）在仿真/可视化里的替身。

宇树 S288 没有公开的网格/URDF。这里用两段拼出小臂：

**骨架** —— 复用真实 FR3 的描述（同一套连杆坐标系）整体等比缩小，于是：

    小臂 leader_fr3_joint_i  <->  大臂 fr3_joint_i     （i = 1..7，一一对应）
    小臂 leader_fr3_finger_* <->  大臂 fr3_finger_*    （Franka Hand 两指）

**外观** —— 默认直接用骨架自己的 FR3 视觉网格（"缩小 FR3 孪生"，连贯保真）。
另提供 **Franka 官方 GELLO 的真实 3D 打印件**外观：`--leader-appearance gello` /
``gello_parts=True`` 时把各连杆视觉网格换成 ``gello_leader/franka_fr3/*.STL``
（见 :func:`apply_gello_appearance` 与 :data:`GELLO_LINK_PARTS`）。

.. warning::
   GELLO 零件外观是**反求近似装配**：上游只发布零件级 STL，没有装配体/CAD，
   :data:`GELLO_LINK_PARTS` 的位姿是自动估的，法兰 roll 与具体摆放不保证正确，
   待拿到官方 CAD 后替换为精确位姿。

两条臂同构，任一转关节时"哪一节在动"一眼可对；再配合 :func:`color_arm_links`
把每对对应连杆涂成同一颜色，对应关系更直观（J1..J7 七彩，夹爪灰）。

注意
----
* 小臂外观只是**视觉/仿真替身**；真实 S288 的零位/转向标定仍走 ``S288LeaderArm``
  的 ``joint_offsets / joint_signs``，与渲染无关。
* GELLO 零件外观默认**不启用**（用孪生）；启用时其相对位姿是"按零件孔轴 + 包围盒自动
  摆出来的**近似**装配"——公共渠道只有零件级 STL，没有装配体；**绕各关节轴的法兰 roll
  需人工微调或替换为官方 CAD**（改 :data:`GELLO_LINK_PARTS`）。
* 因为小臂骨架与大臂同构，仿真里的关节映射应改为**直连**（``sign=+1``、``scale=1``），
  见 :func:`force_identity_arm_mapping`；否则大臂会和小臂"镜像"而不是"同形"。
* 夹爪沿用 Franka Hand 双指，读取时取一指并按行程 ``GRIP_TRAVEL_M`` 归一化，
  约定 **1 = 张开**（与 ``GripperMapping`` 一致）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
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


# ---------------------------------------------------------------------------
# GELLO 真实零件外观（复用 franka_fr3 的 3D 打印件）
# ---------------------------------------------------------------------------
#
# 背景：Franka 官方的 GELLO 硬件本身就是"按 FR3 等比缩小的运动学等价件"，
# 但公共渠道只放出了**零件级 STL**（各自在自身局部坐标里、单位 mm），没有装配体
# /URDF/STEP。所以这里保留缩小的 FR3 连杆链作为骨架，把每一节的 FR3 视觉网格
# **换成对应的 GELLO 零件网格**，关节映射仍是 `leader_fr3_joint_i <-> fr3_joint_i`。
#
# `GELLO_LINK_PARTS` 是按零件孔轴 + 包围盒自动摆出来的**近似**装配；绕各关节轴的
# 法兰 roll 无法从零件反求，需要人工微调。改这张表即可，不必动其它代码；用
# `examples/gello_leader_preview.py` 可以边看边调。

#: franka_fr3（Franka 官方 GELLO 小臂）零件 STL 目录，随仓库分发。
GELLO_PART_DIR = Path(__file__).resolve().parents[2] / "gello_leader" / "franka_fr3"

#: 与 GELLO 零件**实际尺寸**匹配的 FR3 缩放。零件是实尺（mm），所以用零件外观时
#: leader 的缩放应取这个值，否则零件与连杆骨架对不上。
GELLO_LEADER_SCALE = 0.4

#: 每个 leader 连杆坐标系里挂的 GELLO 零件。
#: 键 = leader 的 FR3 连杆号（1..7）；值 = [(STL 名(不含 .STL), 位置(m), rpy(rad)), ...]，
#: 位置/姿态都在该连杆坐标系里。**法兰对齐（rpy）请按实物微调。**
GELLO_LINK_PARTS: dict[
    int, list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]
] = {
    1: [("03_A12_CONNECTOR",   (-0.0278,  0.0140, -0.0265), (0.0000,  0.0000, -1.5708))],
    2: [("04_A23_MOTOR_FLANGE",(-0.0121, -0.0539, -0.0349), (0.0000, -0.8773, -1.5708))],
    3: [("05_A34_CORNER_LINK", (-0.0308,  0.0270, -0.0400), (0.0000, -1.5708,  3.1416))],
    4: [("06_A45_CORNER_LINK", (-0.0321, -0.0077, -0.0390), (-1.1797, -1.5708, -0.1798))],
    5: [("07_A56_MOTOR_FLANGE",(-0.0120,  0.0050, -0.0205), (0.0000, -1.5708,  3.1416))],
    6: [("08_A67_CONNECTOR",   (-0.0228, -0.0112, -0.0220), (0.3218, -1.5708, -1.8927))],
    7: [("20_TRIGGER_BODY",    ( 0.0000,  0.0000, -0.0672), (2.0344, -0.7297,  2.0344))],
}

#: 底座的 GELLO 零件（挂在 leader 的 link0，即 FR3 基座坐标系）。
GELLO_BASE_PARTS: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
    ("01_BASE",         (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("02_BASE_BEARING", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]

#: 可选的桌面安装板；默认不挂，需要时由调用方加进 `extra_base_parts`。
GELLO_TABLE_MOUNT = ("00_TABLE_BASE_MOUNT_SINGLE", (0.0, 0.0, -0.004), (0.0, 0.0, 0.0))

#: STL 是毫米；MuJoCo 网格要乘 0.001 变成米。
_MM_TO_M = [0.001, 0.001, 0.001]


def _rpy_to_R(rpy: Sequence[float]) -> np.ndarray:
    """``Rz(y) @ Ry(p) @ Rx(r)``，与 MuJoCo/URDF 的 ``rpy`` 约定一致。"""
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rpy_to_quat(rpy: Sequence[float]) -> list[float]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, _rpy_to_R(rpy).flatten())
    return quat.tolist()


def apply_gello_appearance(
    spec: mujoco.MjSpec,
    *,
    prefix: str = "leader",
    part_dir: Optional[str | Path] = None,
    link_parts: Optional[dict] = None,
    base_parts: Optional[Sequence] = None,
) -> None:
    """把 leader 那一侧的 FR3 视觉/碰撞网格换成 ``franka_fr3`` 的 GELLO 零件网格。

    只影响名字以 ``{prefix}_fr3_`` 开头的 body：删掉它们的 ``*_visual`` / ``*_collision``
    网格（连杆 body 的惯量是显式给的，删网格不影响运动链），再按
    :data:`GELLO_LINK_PARTS` / :data:`GELLO_BASE_PARTS` 挂上 GELLO 零件。

    零件几何放在 ``group == 1``，名字里带 ``linkN``，于是 :func:`color_arm_links`
    会把它们和 follower 的 ``fr3_linkN`` 涂成同一颜色，关节对应关系照旧可见。
    """
    part_dir = Path(part_dir) if part_dir is not None else GELLO_PART_DIR
    link_parts = link_parts if link_parts is not None else GELLO_LINK_PARTS
    base_parts = base_parts if base_parts is not None else GELLO_BASE_PARTS

    leader_body_prefix = f"{prefix}_fr3_"

    # 1) 去掉 leader 自带的 FR3 网格（视觉 + 碰撞），只留骨架运动链。
    for body in spec.bodies:
        if not (body.name or "").startswith(leader_body_prefix):
            continue
        for geom in list(body.geoms):
            gname = geom.name or ""
            if gname.endswith("_visual") or gname.endswith("_collision"):
                spec.delete(geom)

    # 2) 按表挂 GELLO 零件。
    def _attach(body_name: str, stem: str, pos, rpy, tag: str) -> None:
        body = spec.body(body_name)
        if body is None:
            return
        mesh_name = f"{prefix}_gello_{tag}"
        spec.add_mesh(
            name=mesh_name,
            file=str(part_dir / f"{stem}.STL"),
            scale=list(_MM_TO_M),
        )
        body.add_geom(
            name=f"{prefix}_gello_{tag}_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            pos=[float(v) for v in pos],
            quat=_rpy_to_quat(rpy),
            contype=0,
            conaffinity=1,  # contype=conaffinity=0 的几何会被 MuJoCo 丢弃
            group=1,
            rgba=list(BASE_COLOR),
        )

    for link_index, pieces in link_parts.items():
        body_name = f"{leader_body_prefix}link{link_index}"
        for j, (stem, pos, rpy) in enumerate(pieces):
            # 名字里带 linkN，便于 color_arm_links 同号同色
            _attach(body_name, stem, pos, rpy, f"link{link_index}_{j}")

    for j, (stem, pos, rpy) in enumerate(base_parts):
        _attach(f"{leader_body_prefix}link0", stem, pos, rpy, f"base_{j}")


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
    position: Sequence[float] = (-1.0, 0.0, 0.0),
    quat: Optional[Sequence[float]] = None,
    scale: float = 1.0,
    prefix: str = "leader",
    with_gripper: bool = True,
    joint_limit_rad: Optional[float] = None,
    link_damping: float = 0.8,
    armature: float = 0.1,
    gello_parts: bool = False,
    gello_part_dir: Optional[str | Path] = None,
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

    if gello_parts:
        apply_gello_appearance(spec, prefix=prefix, part_dir=gello_part_dir)

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
    leader_position: Sequence[float] = (-1.0, 0.0, 0.0),
    leader_quat: Optional[Sequence[float]] = None,
    leader_scale: float = 1.0,
    leader_prefix: str = "leader",
    with_gripper: bool = True,
    leader_gello_parts: bool = False,
    leader_gello_part_dir: Optional[str | Path] = None,
) -> tuple[mujoco.MjSpec, LeaderArmRefs]:
    """真实 FR3（follower）+ 等比缩小的 FR3 孪生（leader）合进一个 :class:`mujoco.MjSpec`。

    ``leader_gello_parts=True`` 时，leader 一侧的视觉网格会换成 ``franka_fr3`` 的
    GELLO 真实零件（见 :func:`apply_gello_appearance`）；此时 leader 缩放建议取
    :data:`GELLO_LEADER_SCALE`，否则零件与骨架对不上。

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
        gello_parts=leader_gello_parts,
        gello_part_dir=leader_gello_part_dir,
    )
    color_arm_links(spec)
    return spec, refs
