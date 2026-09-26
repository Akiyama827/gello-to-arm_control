"""Composition of a generic two-arm example from existing per-arm profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from arm_control.config import RobotConfig, arm_joints, ee_frame, load_config_tree
from arm_control import frames
from arm_control.model_bundle import execution_gravity, model_token
from arm_control.control.factory import arm_urdf
from arm_control.dynamics import PinocchioDynamics
from arm_control.scene import ActorSpec, ObjectSpec, SceneSpec, WeldSpec, load_scene
from arm_control.multi_arm.cooperative import (
    RigidGrasps,
    rigid_transform,
)


@dataclass
class SharedObject:
    name: str
    body: str
    world_T_object: np.ndarray
    grasps: RigidGrasps
    fingers: dict[str, float]
    virtual: bool = False

    @property
    def support(self):
        return f"{self.name}__support"

    @property
    def grasp_welds(self):
        return tuple(f"grasp_{a}" for a in self.fingers)


@dataclass
class PairConfig:
    scene: SceneSpec
    robots: dict[str, RobotConfig]
    model_tokens: dict[str, str]
    ground_z: float | None
    velocity: float
    acceleration: float
    source: Path
    content_digest: str
    shared_object: SharedObject | None = None
    workspace_xy: tuple[tuple[float, float], tuple[float, float]] | None = None
    backend: str = "shared_sim"

    @property
    def scene_token(self):
        return hashlib.sha256(
            repr(
                (self.scene, self.model_tokens, self.ground_z, self.content_digest)
            ).encode()
        ).hexdigest()


def scene_for_mode(config, mode):
    """Select constraints for planning; changing mode never changes joint poses."""
    if mode not in {"independent", "shared"}:
        raise ValueError("mode must be independent or shared")
    obj = config.shared_object
    if obj is None:
        if mode == "shared":
            raise ValueError("this setup has no shared object")
        return config.scene
    removed = {obj.support} if mode == "shared" else set(obj.grasp_welds)
    return replace(
        config.scene,
        welds=tuple(w for w in config.scene.welds if w.name not in removed),
    )


def load_pair_config(path) -> PairConfig:
    source = Path(path).resolve()
    raw = load_config_tree(source)
    if (
        not isinstance(raw, dict)
        or raw.get("version") != 1
        or raw.get("backend") not in {"shared_sim", "remote_rt"}
    ):
        raise ValueError(
            "dual console config requires version: 1 and backend: shared_sim or remote_rt"
        )
    arms = raw.get("arms")
    if not isinstance(arms, dict) or len(arms) != 2:
        raise ValueError("dual console config must contain exactly two arms")
    robots, actors, tokens = {}, [], {}
    scene = load_scene(source.parent / raw["scene"]) if raw.get("scene") else None
    for name, entry in arms.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError("arm id must be nonempty without slashes")
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), str):
            raise ValueError(f"{name}: config must name an existing arm profile")
        cfg_raw = load_config_tree(source.parent / entry["config"])
        cfg_raw["arm"] = dict(cfg_raw["arm"])
        if scene:
            actor = next(a for a in scene.actors if a.name == name)
            cfg_raw["arm"].update(
                urdf=str(actor.path),
                world_origin=list(actor.pos),
                world_rpy=list(actor.rpy),
            )
        else:
            for key in ("world_origin", "world_rpy"):
                if key in entry:
                    cfg_raw["arm"][key] = entry[key]
        cfg = RobotConfig.from_mapping(cfg_raw)
        joints = tuple(arm_joints(cfg))
        urdf = Path(arm_urdf(cfg)).resolve()
        if not urdf.is_file():
            raise ValueError(f"{name}: missing URDF {urdf}; stage example assets first")
        if not scene:
            initial = cfg.get("sim_default_joint_positions", {})
            actor = ActorSpec(
                name,
                urdf,
                joints,
                tuple(cfg_raw["arm"].get("world_origin", [0.0, 0.0, 0.0])),
                tuple(cfg_raw["arm"].get("world_rpy", [0.0, 0.0, 0.0])),
                tuple(
                    entry.get("initial_joints", [initial.get(j, 0.0) for j in joints])
                ),
            )
        if cfg_raw["arm"].get("max_tau") is None:
            # Ratings come from the description; explicit operating caps win.
            cfg_raw["arm"]["max_tau"] = PinocchioDynamics(
                str(urdf), joints
            ).effort_limits.tolist()
            cfg = RobotConfig.from_mapping(cfg_raw)
        if actor.joints != joints:
            raise ValueError(f"{name}: scene and controller joint order differ")
        for label, values, n in (
            ("world_origin", actor.pos, 3),
            ("world_rpy", actor.rpy, 3),
            ("initial joints", actor.q, len(joints)),
        ):
            if len(values) != n or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in values
            ):
                raise ValueError(f"{name}: {label} needs {n} finite numbers")
        actors.append(actor)
        robots[name] = cfg
        digest = hashlib.sha256(
            urdf.read_bytes() + json.dumps((actor.pos, actor.rpy, joints)).encode()
        ).hexdigest()
        tokens[name] = model_token(cfg) or f"{name}:sim:{digest}"
        if raw["backend"] == "remote_rt" and (
            not model_token(cfg) or (cfg.get("rt") or {}).get("instance_id") != name
            or not cfg.get("execution_policy")
        ):
            raise ValueError(f"{name}: hardware needs model binding, RT identity and execution policy")
        if raw["backend"] == "remote_rt":
            execution_gravity(cfg, urdf)
    if scene is None:
        scene = SceneSpec(tuple(actors), (), ())
    if set(scene.actor_names) != set(robots):
        raise ValueError("scene must contain exactly the configured arms")
    if scene.objects:
        raise ValueError(
            "this console example supports static scenery; object dynamics need a consumer adapter"
        )
    ground_z = raw.get("ground_z", 0.0)
    if ground_z is not None and (
        isinstance(ground_z, bool)
        or not isinstance(ground_z, (int, float))
        or not math.isfinite(ground_z)
    ):
        raise ValueError("ground_z must be finite or null")
    for name, default in [("max_velocity", 0.4), ("max_acceleration", 0.8)]:
        value = raw.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    workspace = raw.get("workspace_xy")
    if workspace is not None:
        bounds = np.asarray(workspace, dtype=float)
        if (
            bounds.shape != (2, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1])
        ):
            raise ValueError(
                "workspace_xy requires finite ordered lower/upper XY bounds"
            )
        workspace = tuple(tuple(float(v) for v in row) for row in bounds)
    digest = hashlib.sha256(
        json.dumps(
            {"pair": raw, "arms": {a: c.raw for a, c in robots.items()}},
            sort_keys=True,
            default=str,
        ).encode()
    )
    shared = None
    if raw.get("shared_object") is not None:
        obj = raw["shared_object"]
        if (
            not isinstance(obj, dict)
            or not isinstance(obj.get("grasps"), dict)
            or set(obj["grasps"]) != set(robots)
        ):
            raise ValueError("shared_object.grasps must name both arms")
        virtual = obj.get("virtual", False)
        if not isinstance(virtual, bool):
            raise ValueError("shared_object.virtual must be a boolean")
        if raw["backend"] == "remote_rt" and not virtual:
            raise ValueError("physical hardware shared-object grasps are not qualified")
        if virtual and raw["backend"] != "remote_rt":
            raise ValueError("virtual shared objects require the remote_rt backend")
        name, body = obj.get("name"), obj.get("body")
        if (
            not all(isinstance(v, str) and v and "/" not in v for v in (name, body))
            or name in robots
        ):
            raise ValueError("shared object needs distinct name and root body")
        path = (source.parent / obj["path"]).resolve()
        pose = rigid_transform(obj["pose"])
        grasps = RigidGrasps(
            {a: rigid_transform(g["object_T_tcp"]) for a, g in obj["grasps"].items()}
        )
        fingers = {a: float(g["finger_position"]) for a, g in obj["grasps"].items()}
        if any(not math.isfinite(v) or v < 0 for v in fingers.values()):
            raise ValueError(
                "shared grasp finger positions must be finite and nonnegative"
            )
        shared = SharedObject(name, f"{name}__{body}", pose, grasps, fingers, virtual)
        object_spec = ObjectSpec(
            name,
            path,
            tuple(pose[:3, 3]),
            tuple(Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")),
        )
        welds = tuple(
            WeldSpec(
                f"grasp_{a}",
                f"{a}__{ee_frame(robots[a])}",
                shared.body,
                tuple(frames.T_to_pose_xyzquat(np.linalg.inv(T))),
            )
            for a, T in grasps.object_T_tcp.items()
        )
        exclusions = []
        for a, g in obj["grasps"].items():
            bodies = g.get("contact_bodies", [])
            if not isinstance(bodies, list) or any(
                not isinstance(b, str) or not b for b in bodies
            ):
                raise ValueError("grasp contact_bodies must be body names")
            exclusions.extend((f"{a}__{b}", shared.body) for b in bodies)
        scene = replace(
            scene,
            objects=(object_spec,),
            welds=scene.welds + (() if virtual else welds + (
                WeldSpec(
                    shared.support,
                    "world",
                    shared.body,
                    tuple(frames.T_to_pose_xyzquat(pose)),
                ),
            )),
            contact_exclusions=scene.contact_exclusions + tuple(exclusions),
        )
    for item in (*scene.actors, *scene.objects, *scene.fixtures, *scene.obstacles):
        if item.path:
            digest.update(item.path.read_bytes())
    result = PairConfig(
        scene,
        robots,
        tokens,
        ground_z,
        float(raw.get("max_velocity", 0.4)),
        float(raw.get("max_acceleration", 0.8)),
        source,
        digest.hexdigest(),
        shared,
        workspace,
        raw["backend"],
    )
    return result
