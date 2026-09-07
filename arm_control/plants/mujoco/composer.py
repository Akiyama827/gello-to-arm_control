"""Generic scene construction and explicit deployment-builder injection.

The generic path accepts SceneSpec, not a project's loaded-workcell wrapper.
Deployment roles and legacy scene dictionaries belong in the supplied builder.
"""
from pathlib import Path
from typing import Callable

from arm_control import CONTROL_ROOT
from arm_control.node_utils import entry_point
from arm_control.plants.mujoco.backend import MuJoCoBackend
from arm_control.scene import load_scene


def _resolve(p: str) -> str:
    pp = Path(p)
    return str(pp) if pp.is_absolute() else str(CONTROL_ROOT / pp)


def build_scene_backend(
    scene_cfg: dict,
    control_period: float,
    launch_viewer: bool = False,
    enable_self_collision: bool = False,
    *,
    chain_factory: Callable | None = None,
    mate_policy: object = None,
) -> tuple[MuJoCoBackend, dict, list[str], int]:
    """Adapt a deployment-specific dictionary using its explicit builder."""
    builder = entry_point("SCENE_BACKEND_FACTORY")
    if builder is None:
        raise ValueError("SCENE_BACKEND_FACTORY must name the deployment scene builder")
    return builder(
        scene_cfg, control_period, launch_viewer, enable_self_collision,
        chain_factory=chain_factory, mate_policy=mate_policy,
    )


def build_workcell_backend(
    scene_path: str | Path,
    control_period: float,
    *,
    launch_viewer: bool = False,
    enable_self_collision: bool = False,
    loader: Callable | None = None,
    chain_factory: Callable | None = None,
) -> tuple[MuJoCoBackend, dict[str, dict[str, int]], list[str], int]:
    """Build a SceneSpec directly, or use the graph's deployment adapter."""
    builder = entry_point("WORKCELL_BACKEND_FACTORY")
    if builder is not None:
        return builder(
            scene_path, control_period, launch_viewer=launch_viewer,
            enable_self_collision=enable_self_collision, loader=loader,
            chain_factory=chain_factory,
        )
    scene = loader(scene_path) if loader is not None else load_scene(scene_path)
    backend = MuJoCoBackend.from_workcell_scene(
        scene, chain_factory=chain_factory, control_period=control_period,
        launch_viewer=launch_viewer, enable_self_collision=enable_self_collision,
    )
    start = 0
    slices = {}
    for actor in scene.actors:
        width = len(actor.joints)
        slices[actor.name] = {"start": start, "n": width}
        start += width
    return backend, slices, list(backend.joint_names), len(backend.joint_names)
