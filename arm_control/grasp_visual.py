"""Generic URDF geometry and contact preview for grasp editing."""

from __future__ import annotations

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


@dataclass(frozen=True)
class VisualGeometry:
    group: str
    link: str
    mesh: Path
    color: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    T: np.ndarray


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

        self._module = _UrdfGeometry(self.module_urdf)
        self._fixture = _UrdfGeometry(self.fixture_urdf)
        self._tool = _UrdfGeometry(self.tool_urdf)
        self._frames: dict[str, dict[str, np.ndarray]] = {}
        self._visuals: list[VisualGeometry] = []
        self._fixed_collisions: list[_CollisionGeometry] = []
        self._tool_collisions: list[_CollisionGeometry] = []

        for asset, model in ((module, self._module), (fixture, self._fixture)):
            world_T_root = np.asarray(asset.world_T_root, dtype=float)
            if world_T_root.shape != (4, 4) or not np.isfinite(world_T_root).all():
                raise ValueError(f"{asset.name}.world_T_root must be a finite 4x4")
            visual, collision, fixed_frames = model.geometry(
                asset.name,
                _links(model.urdf),
                world_T_root,
            )
            self._visuals.extend(visual)
            self._fixed_collisions.extend(collision)
            self._frames[asset.name] = fixed_frames

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
        return {
            "intended_tool_links": sorted(intended),
            "forbidden_tool_links": sorted(forbidden),
            "ok": not forbidden,
        }


__all__ = ["FixedUrdf", "GraspVisual", "VisualGeometry"]
