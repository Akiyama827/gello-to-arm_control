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
from typing import Callable

from arm_control.simulation.mujoco_backend import (
    ModuleSlot,
    MuJoCoBackend,
    MuJoCoSceneSpec,
    SceneModelSpec,
    _prefixed,
)
from arm_control.scene import load_scene

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


def _module_slots(scene_cfg: dict) -> tuple[ModuleSlot, ...]:
    """Inventory modules from either scene schema.

    ``scene.inventory`` is the plural form: one entry per physical module in
    its nest, in the order the assembly plan will consume them. ``scene.module``
    stays supported unchanged — it is exactly a one-entry inventory, and the
    bench scenarios that use it are owned by the hardware thread.

    Port SITE names default to the connector standard because the dock IS one
    printed part shared by every module (the caller derives the standard from
    its own connector model); ``body`` and ``joints`` have no such standard
    and must be stated, same rule as every other model fact here.
    """
    entries = list(scene_cfg.get("inventory") or [])
    if entries and scene_cfg.get("module"):
        raise ValueError(
            "scene declares BOTH module and inventory: pick one "
            "(inventory is the plural form; module is a 1-entry inventory)"
        )
    if not entries:
        entry = scene_cfg.get("module")
        if not entry:
            return ()
        # Legacy single-module scene: the body/site names live in scene.welds
        # PREFIXED, so strip the prefix back off to recover the model facts.
        welds = dict(scene_cfg.get("welds") or {})
        prefix = str(entry.get("prefix", ""))
        body = str(welds.get("module_body", "")).removeprefix(prefix)
        site = str(welds.get("module_site", "")).removeprefix(prefix)
        entries = [
            {
                **entry,
                # One nest, and nothing ever consumed a slot id before, so any
                # stable id will do; the prefix is already unique.
                "slot": prefix.rstrip("_") or "module",
                "module_id": body.removesuffix("_body"),
                "body": body,
                "passive_site": site,
                # No driven joints: the legacy scene never actuated the module
                # (it docks as scenery), and adding one would silently widen
                # that scenario's motor vector.
                "joints": [],
            }
        ]
    out = []
    for i, entry in enumerate(entries):
        entry = dict(entry)
        for key in ("slot", "module_id", "body", "model_path"):
            if not entry.get(key):
                raise ValueError(f"scene inventory entry {i}: missing {key!r}")
        prefix = str(entry.get("prefix") or f"{entry['slot']}_")
        out.append(
            ModuleSlot(
                slot=str(entry["slot"]),
                module_id=str(entry["module_id"]),
                spec=SceneModelSpec(
                    model_path=_resolve(str(entry["model_path"])),
                    name=str(entry["slot"]),
                    prefix=prefix,
                    world_pos=tuple(entry.get("world_pos", (0.0, 0.0, 0.0))),
                    world_rpy=tuple(entry.get("world_rpy", (0.0, 0.0, 0.0))),
                ),
                body=str(entry["body"]),
                joints=tuple(str(j) for j in (entry.get("joints") or ())),
                passive_site=str(entry.get("passive_site") or "passive_connector"),
                active_site=str(entry.get("active_site") or "active_connector"),
                grasp_site=str(entry.get("grasp_site") or "grasp_frame"),
            )
        )
    slots = [m.slot for m in out]
    if len(set(slots)) != len(slots):
        raise ValueError(f"scene inventory slot ids must be unique, got {slots}")
    return tuple(out)


def build_scene_backend(
    scene_cfg: dict,
    control_period: float,
    launch_viewer: bool = False,
    enable_self_collision: bool = False,
) -> tuple[MuJoCoBackend, dict, list[str], int]:
    modules = _module_slots(scene_cfg)
    spec = MuJoCoSceneSpec(
        arm=_model_spec(scene_cfg, "arm"),
        base=_model_spec(scene_cfg, "base"),
        modules=modules,
        timestep=float(scene_cfg.get("timestep", 0.001)),
    )
    # Actuated joints are PREFIXED in the composed model (mjSpec really renames,
    # unlike the SDF include which only hinted).
    joint_names: list[str] = []
    for role in ("arm", "base"):
        prefix = str(scene_cfg.get(role, {}).get("prefix", ""))
        joint_names.extend(
            f"{prefix}{name}" for name in scene_cfg.get(role, {}).get("joint_names", [])
        )
    # Every inventory module's joints are RESERVED in the command vector from
    # the start, in plan order, even though they are only actuated once that
    # module docks. A fixed-width vector means num_motors, arm_slices and the
    # Dora message shapes never have to be renegotiated mid-run — the robot
    # grows, the contract does not.
    for module in modules:
        joint_names.extend(module.joint_names)
    arm_slices = {
        str(name): {"start": int(info["start"]), "n": int(info["n"])}
        for name, info in (scene_cfg.get("arm_slices") or {}).items()
    }
    n_motors = int(scene_cfg.get("num_motors", len(joint_names)))
    if n_motors != len(joint_names):
        raise ValueError(
            f"scene.num_motors {n_motors} != {len(joint_names)} joints "
            f"({joint_names}) — with reserved module joints these must agree, "
            "or the plant silently truncates the modular arm's slice"
        )
    ground_z = scene_cfg.get("ground_z")
    weld_cfg = dict(scene_cfg.get("welds") or {})
    missing = [k for k in ("ee_body", "dock_site") if not weld_cfg.get(k)]
    if missing:
        raise ValueError(
            f"scene.welds missing {missing}: the composed scene's EE body and "
            "the base's dock port are model facts and must be stated (there is "
            "no default robot)"
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
        dock_site=str(weld_cfg["dock_site"]),
        # No fallback: the mate's lead-in is the caller's hardware fact. A
        # missing key fails by name in MuJoCoSceneSpec.load() rather than
        # silently seating against a number this package invented.
        dock_capture_m=float(weld_cfg["dock_capture_m"]),
        dock_capture_deg=float(weld_cfg["dock_capture_deg"]),
        gripper_joints=tuple(str(j) for j in (scene_cfg.get("gripper_joints") or [])),
        finger_body_match=tuple(
            str(m) for m in (scene_cfg.get("finger_body_match") or [])
        ),
    )
    return backend, arm_slices, joint_names, n_motors


def build_workcell_backend(
    scene_path: str | Path,
    control_period: float,
    *,
    launch_viewer: bool = False,
    enable_self_collision: bool = False,
    loader: Callable | None = None,
    chain_factory: Callable | None = None,
) -> tuple[MuJoCoBackend, dict[str, dict[str, int]], list[str], int]:
    """Load a generic workcell file and derive actor ports from its schema."""
    loaded = loader(scene_path) if loader is not None else load_scene(scene_path)
    scene = getattr(loaded, "scene", loaded)
    state = getattr(loaded, "state", None)
    object_joint_owner = getattr(loaded, "docking_base", None)
    editor = getattr(loaded, "grasp_editor", None)
    assembler = getattr(loaded, "assembler", None)
    backend = MuJoCoBackend.from_workcell_scene(
        scene,
        state,
        object_joint_owner=object_joint_owner,
        chain_factory=chain_factory,
        control_period=control_period,
        launch_viewer=launch_viewer,
        enable_self_collision=enable_self_collision,
        gripper_joints=tuple(
            _prefixed(assembler, name) for name in getattr(editor, "finger_joints", ())
        ),
        # The arm holds itself up; the base and modules do not (see
        # MuJoCoBackend.gravcomp_prefixes).
        gravcomp_prefixes=((f"{assembler}__",) if assembler is not None else ()),
        finger_body_match=tuple(
            _prefixed(assembler, name)
            for name in getattr(editor, "intended_contact_links", ())
        ),
        ee_body=(
            _prefixed(assembler, editor.ee_frame)
            if assembler is not None and editor is not None
            else None
        ),
    )
    start = 0
    slices = {}
    for actor in scene.actors:
        width = len(actor.joints)
        if actor.name == object_joint_owner:
            width += len(backend._joint_object)
        slices[actor.name] = {"start": start, "n": width}
        start += width
    return backend, slices, list(backend.joint_names), len(backend.joint_names)
