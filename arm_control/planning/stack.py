"""One arm's planning stack, wired from its config.

Pure interface: IK, collision world, OMPL, retiming planner, Rerun preview.
Nothing here knows about phases, grasps, or
scenarios — the pick-and-dock coordinator wraps this in its own policy layer
(``assembly.runtime.build``). Servo construction lives in ``control.factory``.

Every identity comes from the config's ``arm:`` block (``arm.joints``,
``arm.ee_frame``, ``arm.urdf``, ``arm.kp``…): a new robot is a new YAML,
never a code edit, and a MISSING key is a loud error — inheriting another
arm's gains is how a silently limp (or hot) arm happens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from arm_control import frames
from arm_control.config import CONTROL_ROOT, arm_joints, ee_frame, gripper_joints
from arm_control.control.factory import (
    arm_urdf,
    gain_vector,
    gripper_command_cfg,
)
from arm_control.planning.high_level import ArmPlanner, build_collision_stack
# Scene-derived obstacles. NOTE: preview_rerun imports `rerun` AT MODULE
# LEVEL, so importing this module requires the optional [viz] extra. That
# is why the executor factory lives in arm_control.control.factory and
# not here -- a controller host must not need a visualiser to servo.
from arm_control.planning.preview_rerun import scene_obstacle_geoms
from arm_control.planning.ik import PinocchioIK

__all__ = [
    "PlannerStack",
    "arm_urdf",
    "build_planner",
    "build_preview",
    "gain_vector",
    "gripper_command_cfg",
    "arm_joints",
    "gripper_joints",
    "ee_frame",
]


def build_preview(cfg, world, ik, urdf: str, joints: list[str], grip: list[str]):
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
    # Both go behind DeferredPreview: every Rerun call is queued and drained on
    # a daemon thread, so a viewer nobody opened costs dropped frames and never
    # a blocked caller. See its docstring for why that is not optional.
    from arm_control.planning.preview_rerun import DeferredPreview

    preview, ghost = DeferredPreview(preview), DeferredPreview(ghost)
    print(
        "[planning.stack] plan preview streaming to Rerun "
        "(STL measured arm, green planned motion, orange plan target)",
        flush=True,
    )
    return preview, ghost


@dataclass
class PlannerStack:
    """One arm's wired interface stack — no task policy attached."""

    arm_id: str
    ik: PinocchioIK
    planner: ArmPlanner
    joints: list[str]
    gripper_joints: list[str]
    urdf: str
    world: Any = None
    preview: Any = None
    ghost: Any = None

    @property
    def n_arm(self) -> int:
        return len(self.joints)


def build_planner(
    cfg, *, arm_id: str = "arm", collision_world=None, refinement_world=None
) -> PlannerStack:
    """Build a planner; refinement_world returns the latest raw MuJoCo world.

    Scene adapters that recompile must supply that callable, not capture an
    initial model. It is called synchronously in the planner process only.
    """
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

    refinement = planner_cfg.get('refinement')
    if refinement not in (None, 'trajopt'):
        raise ValueError(f'unknown planner refinement: {refinement!r}')
    refiner = None
    if refinement == 'trajopt':
        if world is None or ompl is None:
            raise ValueError('TrajOpt refinement requires a collision world and OMPL seed')
        from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
        from arm_control.planning.trajopt import refine_trajectory

        if refinement_world is None:
            if not isinstance(world, MuJoCoCollisionWorld):
                raise ValueError('scene adapters must supply the latest refinement_world callable')
            def refinement_world():
                return world
        if not callable(refinement_world):
            raise ValueError('refinement_world must be callable')

        def refiner(seed, vmax, amax):
            return refine_trajectory(
                refinement_world(), seed, vmax, amax,
                soft_clearance_exempt_pairs=planner_cfg.get('soft_clearance_exempt_pairs'),
                arm_id=arm_id,
            )

    preview, ghost = build_preview(cfg, world, ik, urdf, joints, grip_joints)

    planner = ArmPlanner(
        arm_id=arm_id,
        ik=ik,
        ompl=ompl,
        max_vel=gain_vector(cfg, "max_vel", n),
        max_acc=gain_vector(cfg, "max_acc", n),
        world=world,
        trajectory_refiner=refiner,
        ik_candidate_attempts=planner_cfg.get('ik_candidate_attempts', 0),
        ik_limit_margin_fraction=planner_cfg.get('ik_limit_margin_fraction', 0.05),
    )

    return PlannerStack(
        arm_id=arm_id,
        ik=ik,
        planner=planner,
        joints=joints,
        gripper_joints=grip_joints,
        urdf=urdf,
        world=world,
        preview=preview,
        ghost=ghost,
    )
