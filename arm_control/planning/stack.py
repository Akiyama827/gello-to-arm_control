"""One arm's planning + execution stack, wired from its config.

Pure interface: IK, collision world, OMPL, retiming planner, dynamics,
executor, Rerun preview. Nothing here knows about phases, grasps, or
scenarios — the pick-and-dock coordinator wraps this in its own policy layer
(``assembly.build``), and the teleop/executor nodes use it directly. One
recipe instead of three hand-rolled copies.

Every identity comes from the config's ``arm:`` block (``arm.joints``,
``arm.ee_frame``, ``arm.urdf``, ``arm.kp``…): a new robot is a new YAML,
never a code edit, and a MISSING key is a loud error — inheriting another
arm's gains is how a silently limp (or hot) arm happens.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arm_control import frames
from arm_control.config import CONTROL_ROOT, _arm_block, arm_joints, ee_frame, gripper_joints
from arm_control.dynamics import PinocchioDynamics
from arm_control.execution.trajectory_executor import JointTrajectoryExecutor
from arm_control.planning.high_level import ArmPlanner, build_collision_stack
# Scene-derived obstacles; preview_rerun imports rerun/numpy only at module
# level, so this stays importable on the executor hosts.
from arm_control.planning.preview_rerun import scene_obstacle_geoms
from arm_control.planning.ik import PinocchioIK

__all__ = [
    "PlanningStack",
    "arm_urdf",
    "build_planning_stack",
    "gain_vector",
    "gripper_command_cfg",
    "arm_joints",
    "gripper_joints",
    "ee_frame",
]


def arm_urdf(cfg) -> str:
    raw = _arm_block(cfg).get("urdf") or cfg.get("urdf_path")
    if not raw:
        raise ValueError("config missing an arm URDF (arm.urdf / urdf_path)")
    path = Path(str(raw))
    return str(path if path.is_absolute() else CONTROL_ROOT / path)


def gain_vector(cfg, key: str, n: int) -> np.ndarray:
    """Per-joint vector from ``arm.<key>``; a scalar broadcasts to ``n``."""
    value = _arm_block(cfg).get(key)
    if value is None:
        raise ValueError(
            f"config missing arm.{key}: every arm declares its own per-joint "
            f"values (there is no robot-shaped default to inherit)"
        )
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    if arr.shape != (n,):
        raise ValueError(f"arm.{key} must have {n} values for this arm, got {arr.size}")
    return arr


def gripper_command_cfg(cfg) -> dict:
    """Gripper packing parameters for the arm+gripper motor command.

    Returns ``None`` for ``mimic`` on arms whose gripper is not a motor slot on
    the same bus (the FR3's Franka Hand is its own device, driven by grasp
    requests rather than a packed command word).
    """
    mimics = cfg.get("joint_mimics") or {}
    mimic = next((m for m in mimics.values() if isinstance(m, dict)), None)
    grasp = cfg.get("grasp") or {}
    return {
        "mimic": mimic,
        # Held open through the arm-motion legs; the bridge's grasp gate owns the
        # gripper slot from close_gripper onward, so this never fights the grasp.
        "open_finger_m": float(
            cfg.get("gripper_open_finger_m", (mimic or {}).get("lower", 0.0))
        ),
        "gains": (float(grasp.get("close_kp", 40.0)), float(grasp.get("close_kd", 2.0))),
        "n_motors": int(cfg.get("num_motors", 7)),
    }


def _build_preview(cfg, world, ik, urdf: str, joints: list[str], grip: list[str]):
    """(PreviewScene, MeasuredGhost) for Rerun, or (None, None) when unconfigured."""
    preview_cfg = dict(cfg.get("planner") or {}).get("preview")
    if not preview_cfg or world is None:
        return None, None
    from arm_control.planning.preview_rerun import (
        MeasuredGhost,
        PreviewScene,
        init_preview_stream,
    )

    init_preview_stream(preview_cfg if isinstance(preview_cfg, dict) else {})
    # Preview/ghost entities are authored in the ARM-BASE frame; pin their roots
    # to the arm's world mount so they land in the same coordinates as the sim
    # mirror's world-frame ground truth.
    world_T_arm = frames.world_T_arm(cfg)
    preview = PreviewScene(
        ik.fk,
        urdf_path=urdf,
        joint_names=joints,
        ee_link=ee_frame(cfg),
        world_T_arm=world_T_arm,
        gripper_joints=grip,
    )
    ghost = MeasuredGhost(urdf, joints + grip, world_T_arm=world_T_arm)
    print(
        "[planning.stack] plan preview streaming to Rerun "
        "(STL measured arm, green planned motion, orange plan target)",
        flush=True,
    )
    return preview, ghost


@dataclass
class PlanningStack:
    """One arm's wired interface stack — no task policy attached."""

    arm_id: str
    ik: PinocchioIK
    planner: ArmPlanner
    executor: JointTrajectoryExecutor
    joints: list[str]
    gripper_joints: list[str]
    urdf: str
    world: Any = None
    preview: Any = None
    ghost: Any = None

    @property
    def n_arm(self) -> int:
        return len(self.joints)


def build_planning_stack(
    cfg, *, arm_id: str = "arm", collision_world=None
) -> PlanningStack:
    """IK + collision world + OMPL + planner + executor (+ preview) from config."""
    urdf = arm_urdf(cfg)
    joints = arm_joints(cfg)
    grip_joints = gripper_joints(cfg)
    n = len(joints)

    ik = PinocchioIK(urdf, ee_frame=ee_frame(cfg), joint_names=joints)

    # Collision-checked planning when the config carries a planner block; absent
    # block keeps the historical straight-line behaviour (sim graphs unchanged).
    planner_cfg = dict(cfg.get("planner") or {})
    world = ompl = None
    if collision_world is not None:
        world = collision_world
        if planner_cfg.get("use_ompl", bool(planner_cfg)):
            from arm_control.planning.ompl_planner import OMPLPlanner

            ompl = OMPLPlanner(
                list(zip(world.lower, world.upper)), world.in_collision,
                solve_time_sec=float(planner_cfg.get("solve_time_sec", 2.0)),
                simplify_time_sec=float(planner_cfg.get("simplify_time_sec", 0.5)),
                resolution_frac=float(planner_cfg.get("resolution_frac", 0.005)),
            )
    elif planner_cfg.get("use_ompl", bool(planner_cfg)):
        world, ompl = build_collision_stack(
            urdf,
            joints,
            planner_cfg,
            cache_dir=CONTROL_ROOT / ".cache" / "planning",
            # Same rule as the teleop node: whatever the viewers draw as scene
            # bodies is also an obstacle to plan around.
            environment=list(cfg.get("environment") or [])
            + scene_obstacle_geoms(cfg),
        )

    preview, ghost = _build_preview(cfg, world, ik, urdf, joints, grip_joints)

    planner = ArmPlanner(
        arm_id=arm_id,
        ik=ik,
        ompl=ompl,
        max_vel=gain_vector(cfg, "max_vel", n),
        max_acc=gain_vector(cfg, "max_acc", n),
        world=world,
    )
    # done() tolerances are per-arm feedback facts (DM quantization, payload
    # sag); absent keys keep the executor's sim-tuned defaults.
    arm_blk = _arm_block(cfg)
    tolerances = {
        k: float(arm_blk[k]) for k in ("done_pos_tol", "done_vel_tol") if k in arm_blk
    }
    executor = JointTrajectoryExecutor(
        arm_id=arm_id,
        joint_names=joints,
        dynamics=PinocchioDynamics(urdf, joints),
        kp_default=gain_vector(cfg, "kp", n),
        kd_default=gain_vector(cfg, "kd", n),
        max_torque=gain_vector(cfg, "max_tau", n),
        # The plant compensates gravity (the FR3 control box on the bench, and
        # now the twin too), so ship RNEA MINUS gravity or the arm gets it
        # twice. nodes/trajectory_executor.py already honoured this flag; this
        # path silently ignored it.
        gravity_comp=bool(arm_blk.get("plant_gravity_comp", False)),
        **tolerances,
    )
    # Payload feedforward frame (mass toggles on grasp/release results): RNEA
    # knows only the bare arm; a held module otherwise sags on the soft contact-
    # phase gains (centimeters at the EE — measured in the twin).
    executor.set_payload(0.0, ee_frame(cfg))

    return PlanningStack(
        arm_id=arm_id,
        ik=ik,
        planner=planner,
        executor=executor,
        joints=joints,
        gripper_joints=grip_joints,
        urdf=urdf,
        world=world,
        preview=preview,
        ghost=ghost,
    )
