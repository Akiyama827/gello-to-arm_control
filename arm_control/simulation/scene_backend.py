"""Composed-scene MuJoCo backend construction from a config ``scene`` block.

Shared by the plant node (``mujoco_interface``) and the simulated perception
node (``sim_perception``): both build the SAME composed twin scene from the
scenario ``scene`` config, so the sampled cloud tracks the physics the plant
steps.

Scene slots are ROLES — ``arm`` / ``base`` / ``module``; which robot fills a
slot is config data (``model_path`` + ``prefix``), never a name in this code.
"""
from __future__ import annotations

from pathlib import Path

from arm_control.simulation.mujoco_backend import (
    MuJoCoBackend,
    MuJoCoSceneSpec,
    SceneModelSpec,
)

from arm_control import CONTROL_ROOT  # deployment root (env-overridable seam)


def _resolve(p: str) -> str:
    pp = Path(p)
    return str(pp) if pp.is_absolute() else str(CONTROL_ROOT / pp)


def _model_spec(scene_cfg: dict, key: str) -> SceneModelSpec:
    entry = scene_cfg[key]
    return SceneModelSpec(
        model_path=_resolve(str(entry["model_path"])),
        name=key,
        prefix=str(entry["prefix"]),
        world_pos=tuple(entry.get("world_pos", (0.0, 0.0, 0.0))),
        world_rpy=tuple(entry.get("world_rpy", (0.0, 0.0, 0.0))),
    )


def build_scene_backend(
    scene_cfg: dict,
    control_period: float,
    launch_viewer: bool = False,
    enable_self_collision: bool = False,
) -> tuple[MuJoCoBackend, dict, list[str], int]:
    spec = MuJoCoSceneSpec(
        arm=_model_spec(scene_cfg, "arm"),
        base=_model_spec(scene_cfg, "base"),
        module=_model_spec(scene_cfg, "module"),
        timestep=float(scene_cfg.get("timestep", 0.001)),
    )
    # Actuated joints are PREFIXED in the composed model (mjSpec really renames,
    # unlike the SDF include which only hinted).
    joint_names: list[str] = []
    for slot in ("arm", "base"):
        prefix = str(scene_cfg.get(slot, {}).get("prefix", ""))
        joint_names.extend(
            f"{prefix}{name}" for name in scene_cfg.get(slot, {}).get("joint_names", [])
        )
    arm_slices = {
        str(name): {"start": int(info["start"]), "n": int(info["n"])}
        for name, info in (scene_cfg.get("arm_slices") or {}).items()
    }
    n_motors = int(scene_cfg.get("num_motors", len(joint_names)))
    ground_z = scene_cfg.get("ground_z")
    weld_cfg = dict(scene_cfg.get("welds") or {})
    missing = [k for k in ("ee_body", "module_body", "dock_site", "module_site") if not weld_cfg.get(k)]
    if missing:
        raise ValueError(
            f"scene.welds missing {missing}: the composed scene's mate bodies/"
            "sites are model facts and must be stated (there is no default robot)"
        )
    backend = MuJoCoBackend(
        scene=spec,
        joint_names=joint_names,
        control_period=control_period,
        ground_z=None if ground_z is None else float(ground_z),
        static_boxes=list(scene_cfg.get("static_boxes") or []),
        launch_viewer=launch_viewer,
        enable_self_collision=enable_self_collision,
        default_joint_positions={
            f"{scene_cfg.get('arm', {}).get('prefix', '')}{k}": float(v)
            for k, v in (scene_cfg.get("default_joint_positions") or {}).items()
        },
        ee_body=str(weld_cfg["ee_body"]),
        module_body=str(weld_cfg["module_body"]),
        dock_site=str(weld_cfg["dock_site"]),
        module_site=str(weld_cfg["module_site"]),
        dock_capture_m=float(weld_cfg.get("dock_capture_m", 0.05)),
        dock_capture_deg=float(weld_cfg.get("dock_capture_deg", 20.0)),
        gripper_joints=tuple(str(j) for j in (scene_cfg.get("gripper_joints") or [])),
        finger_body_match=tuple(
            str(m) for m in (scene_cfg.get("finger_body_match") or [])
        ),
    )
    return backend, arm_slices, joint_names, n_motors
