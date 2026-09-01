"""Generic URDF geometry and contact preview for grasp editing."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from arm_control import frames
from arm_control.planning.preview_rerun import _mesh_package_dirs


@dataclass(frozen=True)
class FixedUrdf:
    name: str
    urdf: Path
    world_T_root: np.ndarray
    joints: tuple[tuple[str, float], ...] = ()
    emphasized: bool = False


@dataclass(frozen=True)
class FixedMesh:
    name: str
    mesh: Path
    world_T_mesh: np.ndarray
    color: tuple[float, float, float, float] = (0.55, 0.60, 0.60, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    emphasized: bool = False


@dataclass(frozen=True)
class FixedHalfspace:
    name: str
    normal: tuple[float, float, float]
    offset: float


@dataclass(frozen=True)
class VisualGeometry:
    group: str
    link: str
    mesh: Path
    color: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    T: np.ndarray
    emphasized: bool = False


@dataclass(frozen=True)
class _CollisionGeometry:
    group: str
    link: str
    geometry: object
    T: np.ndarray


class _UrdfGeometry:
    def __init__(self, urdf: Path) -> None:
        import pinocchio as pin

        self.pin = pin
        self.urdf = Path(urdf).resolve(strict=True)
        self.model, self.visual, self.collision = pin.buildModelsFromUrdf(
            str(self.urdf),
            package_dirs=_mesh_package_dirs(self.urdf) or None,
            geometry_types=[pin.GeometryType.VISUAL, pin.GeometryType.COLLISION],
        )
        self.data = self.model.createData()
        self.visual_data = self.visual.createData()
        self.collision_data = self.collision.createData()
        self.q = pin.neutral(self.model)
        self.update()

    def update(self) -> None:
        self.pin.forwardKinematics(self.model, self.data, self.q)
        self.pin.updateFramePlacements(self.model, self.data)
        self.pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.visual,
            self.visual_data,
            self.q,
        )
        self.pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.collision,
            self.collision_data,
            self.q,
        )

    def set_joints(self, joints: tuple[tuple[str, float], ...]) -> None:
        for name, value in joints:
            if not self.model.existJointName(str(name)):
                raise ValueError(f"unknown fixed joint: {name}")
            joint_id = self.model.getJointId(str(name))
            joint = self.model.joints[joint_id]
            value = float(value)
            if joint.nq != 1 or not np.isfinite(value):
                raise ValueError(f"fixed joint {name!r} needs one finite position")
            self.q[joint.idx_q] = value
        self.update()

    def frame_T(self, name: str) -> np.ndarray:
        frame_id = self.model.getFrameId(name)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"unknown URDF frame: {name}")
        return np.asarray(self.data.oMf[frame_id].homogeneous).copy()

    def geometry(
        self,
        group: str,
        selected_links: frozenset[str],
        origin_T_root: np.ndarray,
        emphasized: bool = False,
    ) -> tuple[list[VisualGeometry], list[_CollisionGeometry], dict[str, np.ndarray]]:
        visuals = []
        for geom_id, geom in enumerate(self.visual.geometryObjects):
            link = self.model.frames[geom.parentFrame].name
            mesh = Path(geom.meshPath)
            if link not in selected_links or not mesh.is_file():
                continue
            visuals.append(
                VisualGeometry(
                    group,
                    link,
                    mesh,
                    tuple(float(value) for value in geom.meshColor),
                    tuple(float(value) for value in geom.meshScale),
                    origin_T_root
                    @ np.asarray(self.visual_data.oMg[geom_id].homogeneous),
                    emphasized,
                )
            )
        collisions = []
        for geom_id, geom in enumerate(self.collision.geometryObjects):
            link = self.model.frames[geom.parentFrame].name
            if link in selected_links:
                collisions.append(
                    _CollisionGeometry(
                        group,
                        link,
                        geom.geometry,
                        origin_T_root
                        @ np.asarray(self.collision_data.oMg[geom_id].homogeneous),
                    )
                )
        return (
            visuals,
            collisions,
            {
                link: origin_T_root @ self.frame_T(link)
                for link in selected_links
            },
        )


def _links(urdf: Path) -> frozenset[str]:
    return frozenset(
        link.get("name")
        for link in ET.parse(urdf).getroot().findall("link")
        if link.get("name")
    )


def _subtree_links(urdf: Path, root_link: str) -> tuple[str, ...]:
    root = ET.parse(urdf).getroot()
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            children.setdefault(parent.get("link", ""), []).append(
                child.get("link", "")
            )
    if root_link not in _links(urdf):
        raise ValueError(f"unknown tool root link: {root_link}")
    result = []
    pending = [root_link]
    while pending:
        link = pending.pop()
        result.append(link)
        pending.extend(reversed(children.get(link, [])))
    return tuple(result)


def _finite_transform(transform: np.ndarray, label: str) -> np.ndarray:
    out = np.asarray(transform, dtype=float)
    if out.shape != (4, 4) or not np.isfinite(out).all():
        raise ValueError(f"{label} must be a finite 4x4")
    return out


class GraspVisual:
    """Render fixed context and a movable tool, and classify its contacts."""

    def __init__(
        self,
        *,
        module: FixedUrdf,
        fixture: FixedUrdf,
        tool_urdf: str | Path,
        tool_root_link: str,
        ee_frame: str,
        finger_joints: tuple[str, ...],
        intended_tool_links: frozenset[str],
        halfspaces: tuple[FixedHalfspace, ...] = (),
        context: tuple[FixedUrdf, ...] = (),
        meshes: tuple[FixedMesh, ...] = (),
    ) -> None:
        self.module_urdf = Path(module.urdf).resolve(strict=True)
        self.fixture_urdf = Path(fixture.urdf).resolve(strict=True)
        self.tool_urdf = Path(tool_urdf).resolve(strict=True)
        self.tool_root_link = tool_root_link
        self.ee_frame = ee_frame
        self.finger_joints = tuple(finger_joints)
        self.tool_links = _subtree_links(self.tool_urdf, tool_root_link)
        tool_link_set = frozenset(self.tool_links)
        if ee_frame not in tool_link_set:
            raise ValueError("ee_frame must belong to the selected tool subtree")
        if not intended_tool_links <= tool_link_set:
            raise ValueError("intended contact links must belong to the tool subtree")
        self.intended_tool_links = frozenset(intended_tool_links)
        self._halfspaces = []
        for halfspace in halfspaces:
            normal = np.asarray(halfspace.normal, dtype=float)
            norm = float(np.linalg.norm(normal))
            if (
                not halfspace.name
                or normal.shape != (3,)
                or not np.isfinite(normal).all()
                or not np.isfinite(halfspace.offset)
                or norm == 0.0
            ):
                raise ValueError("halfspace needs a name, finite normal, and offset")
            self._halfspaces.append(
                (normal / norm, float(halfspace.offset) / norm)
            )

        self._module = _UrdfGeometry(self.module_urdf)
        self._fixture = _UrdfGeometry(self.fixture_urdf)
        self._tool = _UrdfGeometry(self.tool_urdf)
        self._frames: dict[str, dict[str, np.ndarray]] = {}
        self._visuals: list[VisualGeometry] = []
        self._fixed_collisions: list[_CollisionGeometry] = []
        self._tool_collisions: list[_CollisionGeometry] = []

        fixed_models = [(module, self._module), (fixture, self._fixture)]
        fixed_models.extend((item, _UrdfGeometry(item.urdf)) for item in context)
        for asset, model in fixed_models:
            world_T_root = _finite_transform(
                asset.world_T_root, f"{asset.name}.world_T_root"
            )
            model.set_joints(asset.joints)
            visual, collision, fixed_frames = model.geometry(
                asset.name,
                _links(model.urdf),
                world_T_root,
                asset.emphasized,
            )
            self._visuals.extend(visual)
            self._fixed_collisions.extend(collision)
            self._frames[asset.name] = fixed_frames
        for mesh in meshes:
            path = Path(mesh.mesh).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{mesh.name}.mesh must be a file")
            color = tuple(float(value) for value in mesh.color)
            scale = tuple(float(value) for value in mesh.scale)
            if len(color) != 4 or not np.isfinite(color).all():
                raise ValueError(f"{mesh.name}.color must have four finite values")
            if len(scale) != 3 or not np.isfinite(scale).all() or min(scale) <= 0.0:
                raise ValueError(
                    f"{mesh.name}.scale must have three finite positive values"
                )
            self._visuals.append(
                VisualGeometry(
                    mesh.name,
                    mesh.name,
                    path,
                    color,
                    scale,
                    _finite_transform(mesh.world_T_mesh, f"{mesh.name}.world_T_mesh"),
                    mesh.emphasized,
                )
            )

        if not self.finger_joints or len(set(self.finger_joints)) != len(
            self.finger_joints
        ):
            raise ValueError("finger_joints must be non-empty and unique")
        lower = []
        upper = []
        for name in self.finger_joints:
            joint_id = self._tool.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"unknown finger joint: {name}")
            joint = self._tool.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"finger joint {name!r} must have one position")
            lower.append(float(self._tool.model.lowerPositionLimit[joint.idx_q]))
            upper.append(float(self._tool.model.upperPositionLimit[joint.idx_q]))
        count = len(self.finger_joints)
        self.finger_width_limits = (max(lower) * count, min(upper) * count)
        self._finger_width_m = 0.0
        self._fixed_visual_count = len(self._visuals)
        self._update_tool()

    @property
    def finger_width_m(self) -> float:
        return self._finger_width_m

    def set_finger_width(self, width_m: float) -> None:
        width = float(width_m)
        lower, upper = self.finger_width_limits
        if not np.isfinite(width) or not lower <= width <= upper:
            raise ValueError("finger width is outside URDF limits")
        value = width / len(self.finger_joints)
        for name in self.finger_joints:
            joint_id = self._tool.model.getJointId(name)
            self._tool.q[self._tool.model.joints[joint_id].idx_q] = value
        self._finger_width_m = width
        self._tool.update()
        self._update_tool()

    def _update_tool(self) -> None:
        tcp_T_root = frames.invert(self._tool.frame_T(self.ee_frame))
        visual, collision, tool_frames = self._tool.geometry(
            "tool",
            frozenset(self.tool_links),
            tcp_T_root,
        )
        del self._visuals[self._fixed_visual_count :]
        self._visuals.extend(visual)
        self._tool_collisions = collision
        self._frames["tool"] = tool_frames

    def frame_T(self, group: str, frame: str) -> np.ndarray:
        try:
            return self._frames[group][frame].copy()
        except KeyError as exc:
            raise ValueError(f"unknown editor frame: {group}/{frame}") from exc

    def visuals(self) -> tuple[VisualGeometry, ...]:
        return tuple(self._visuals)

    def contacts(self, world_T_tcp: np.ndarray) -> dict[str, object]:
        import coal

        world_T_tcp = np.asarray(world_T_tcp, dtype=float)
        if world_T_tcp.shape != (4, 4) or not np.isfinite(world_T_tcp).all():
            raise ValueError("world_T_tcp must be a finite 4x4")
        intended = set()
        forbidden = set()
        request = coal.CollisionRequest()
        for tool in self._tool_collisions:
            world_T_tool = world_T_tcp @ tool.T
            tool_pose = coal.Transform3s(
                world_T_tool[:3, :3], world_T_tool[:3, 3]
            )
            for other in self._fixed_collisions:
                result = coal.CollisionResult()
                other_pose = coal.Transform3s(other.T[:3, :3], other.T[:3, 3])
                coal.collide(
                    tool.geometry,
                    tool_pose,
                    other.geometry,
                    other_pose,
                    request,
                    result,
                )
                if not result.isCollision():
                    continue
                if (
                    other.group == "module"
                    and tool.link in self.intended_tool_links
                ):
                    intended.add(tool.link)
                else:
                    forbidden.add(tool.link)
            for normal, offset in self._halfspaces:
                result = coal.CollisionResult()
                coal.collide(
                    tool.geometry,
                    tool_pose,
                    coal.Halfspace(normal, offset),
                    coal.Transform3s(),
                    request,
                    result,
                )
                if result.isCollision():
                    forbidden.add(tool.link)
        return {
            "intended_tool_links": sorted(intended),
            "forbidden_tool_links": sorted(forbidden),
            "ok": not forbidden,
        }


__all__ = [
    "FixedHalfspace",
    "FixedMesh",
    "FixedUrdf",
    "GraspVisual",
    "VisualFK",
    "VisualGeometry",
]


def _pose_json(pin, M) -> dict:
    quat = pin.Quaternion(M.rotation).coeffs()  # x, y, z, w
    return {
        "p": [round(float(v), 5) for v in M.translation],
        "q": [round(float(v), 6) for v in quat],
    }


class VisualFK:
    """Visual-geom FK feeding the web page: mesh list once, poses per config.

    Thread-safe (own lock): the HTTP handler poses the measured/target robots
    per poll while the node loop builds plan-playback frames."""

    def __init__(
        self,
        urdf_path,
        joint_names: list[str],
        ee_frame: str,
        finger_joints: list[str] | None = None,
        world_T_root: np.ndarray | None = None,
    ) -> None:
        import pinocchio as pin

        from arm_control.planning.preview_rerun import _mesh_package_dirs

        self._pin = pin
        root = np.eye(4) if world_T_root is None else np.asarray(world_T_root, dtype=float)
        if root.shape != (4, 4) or not np.isfinite(root).all():
            raise ValueError("world_T_root must be a finite 4x4")
        self._world_T_root = pin.SE3(root[:3, :3], root[:3, 3])
        self.model, self.visual = pin.buildModelsFromUrdf(
            str(urdf_path),
            package_dirs=_mesh_package_dirs(urdf_path) or None,
            geometry_types=[pin.GeometryType.VISUAL],
        )
        self.data = self.model.createData()
        self.vdata = self.visual.createData()
        self._q_idx = [
            self.model.joints[self.model.getJointId(str(n))].idx_q for n in joint_names
        ]
        # Finger joints ride along so every ghost mirrors the gripper slider.
        # Whichever finger joints THIS arm declares (arm.gripper_joints), kept
        # only if the URDF really has them — no robot names baked in here.
        self.finger_joints = [
            n for n in (finger_joints or []) if self.model.existJointName(n)
        ]
        self._finger_idx = [
            self.model.joints[self.model.getJointId(n)].idx_q
            for n in self.finger_joints
        ]
        self._ee_id = self.model.getFrameId(str(ee_frame))
        self._geom_ids = [
            i
            for i, g in enumerate(self.visual.geometryObjects)
            if Path(g.meshPath).is_file()
        ]
        # Static scene bodies (the modular base / dock) appended AFTER the arm's
        # geoms, so `mesh/<k>` keys stay stable for the robot itself. Same
        # source as the Rerun recordings — the page and the viewer cannot
        # disagree about where the dock stands.
        self._static: list[tuple[Path, dict]] = []
        self._lock = threading.Lock()

    def add_static_scene(self, cfg) -> None:
        """Append the config's non-arm scene bodies as fixed page geometry."""
        from arm_control.planning.preview_rerun import static_scene_geoms

        for _body, _name, mesh_path, T in static_scene_geoms(cfg):
            quat = self._pin.Quaternion(T[:3, :3].copy()).coeffs()  # x,y,z,w
            self._static.append(
                (
                    Path(mesh_path),
                    {
                        "p": [round(float(v), 5) for v in T[:3, 3]],
                        "q": [round(float(v), 6) for v in quat],
                    },
                )
            )
        if self._static:
            print(f"[teleop] page scene: {len(self._static)} static meshes",
                  flush=True)

    def scene_json(self, mesh_prefix: str = "mesh") -> list[dict]:
        """Geometry list for the page: one cache-busted mesh URL per visual geom.

        The ``?v=`` stamp is load-bearing. ``mesh/<k>`` is a stable, OPAQUE key
        whose CONTENT changes whenever the arm changes or its meshes are
        restaged, and the handler serves it with ``max-age=86400`` — so without a
        content-dependent URL the browser happily renders yesterday's robot for a
        day. Measured: after converting the FR3 visuals from .obj to .stl, Chrome
        kept returning the cached OBJ body (5.1 MB, "# https://github.com/mikedh/
        trimesh") with no network request at all, and the page silently showed
        nothing. Stamping mtime+size keeps caching effective and makes staleness
        impossible.
        """
        out = []
        for k, gid in enumerate(self._geom_ids):
            g = self.visual.geometryObjects[gid]
            path = self.mesh_path(k)
            try:
                st = path.stat()
                stamp = f"{int(st.st_mtime)}-{st.st_size}"
            except OSError:
                stamp = "0"
            out.append(
                {
                    "mesh": f"{mesh_prefix}/{k}?v={stamp}",
                    "color": [float(v) for v in g.meshColor],
                    "scale": [float(v) for v in g.meshScale],
                }
            )
        return out

    def static_json(self) -> list[dict]:
        """Fixed scene geometry (the dock/modular base): mesh + colour + POSE.

        Deliberately NOT part of scene_json(): the page builds THREE robots
        (measured, target, plan) from that list, so anything appended there is
        drawn three times in three tints. These carry their own pose and are
        drawn once.
        """
        out = []
        for j, (path, pose) in enumerate(self._static):
            try:
                st = path.stat()
                stamp = f"{int(st.st_mtime)}-{st.st_size}"
            except OSError:
                stamp = "0"
            out.append(
                {
                    # Muted grey: a backdrop for judging geometry, never
                    # mistakable for the live robot's own STL colours.
                    "mesh": f"mesh/{len(self._geom_ids) + j}?v={stamp}",
                    "color": [0.55, 0.57, 0.60],
                    "scale": [1.0, 1.0, 1.0],
                    **pose,
                }
            )
        return out

    def mesh_path(self, k: int) -> Path | None:
        if 0 <= k < len(self._geom_ids):
            return Path(self.visual.geometryObjects[self._geom_ids[k]].meshPath)
        j = k - len(self._geom_ids)
        if 0 <= j < len(self._static):
            return self._static[j][0]
        return None

    def poses(self, q_arm, finger_m: float) -> dict:
        """``{'geoms': [{p,q}...], 'ee': {p,q}}`` at arm config + finger opening."""
        pin = self._pin
        with self._lock:
            q = pin.neutral(self.model)
            for idx, value in zip(self._q_idx, np.asarray(q_arm, dtype=float)):
                q[idx] = value
            for idx in self._finger_idx:
                q[idx] = float(finger_m)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pin.updateGeometryPlacements(
                self.model, self.data, self.visual, self.vdata, q
            )
            geoms = [
                _pose_json(pin, self._world_T_root * self.vdata.oMg[gid])
                for gid in self._geom_ids
            ]
            return {
                "geoms": geoms,
                "ee": _pose_json(pin, self._world_T_root * self.data.oMf[self._ee_id]),
            }
