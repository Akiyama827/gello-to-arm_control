"""Local calibration console and visualization-only module grasp editor."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import yaml

from arm_control import frames
from arm_control.calibration_console import CalibrationStore, validate_grasp_profile
# One allowlist, shared with the operator panel -- see arm_control/console_assets.py.
from arm_control.console_assets import CONSOLE_ASSETS, console_asset  # noqa: F401
from arm_control.console_server import ConsoleServer, file_asset
from arm_control.grasp_visual import GraspVisual, VisualFK


def _pose_json(transform: np.ndarray) -> dict[str, list[float]]:
    pose = frames.T_to_pose_xyzquat(transform)
    return {
        "p": [float(value) for value in pose[:3]],
        "q": [float(pose[4]), float(pose[5]), float(pose[6]), float(pose[3])],
    }


@dataclass(frozen=True)
class GraspCheck:
    status: str
    detail: str = ""
    q: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "checking",
            "valid",
            "unreachable",
            "collision",
            "error",
        }:
            raise ValueError(f"unknown grasp check status: {self.status}")
        if self.q is not None and (
            self.status not in {"valid", "collision"}
            or not self.q
            or not np.isfinite(np.asarray(self.q, dtype=float)).all()
        ):
            raise ValueError("only valid or colliding checks may carry finite q")


@dataclass(frozen=True)
class GraspEditorWorkspace:
    targets: Mapping[str, Path]
    default_placements: Mapping[str, str]
    placements: tuple[str, ...]
    contexts: tuple[str, ...]
    build_visual: Callable[[str, str, str], GraspVisual]
    evaluate: Callable[[str, str, Mapping[str, object]], Mapping[str, GraspCheck]]
    arm_visual: VisualFK


class _NoArmVisual:
    def scene_json(self, _mesh_prefix: str = "arm-mesh") -> list[dict]:
        return []

    def mesh_path(self, _index: int) -> Path | None:
        return None

    def poses(self, _q, _finger_m: float) -> dict:
        return {"geoms": []}


class GraspEditorPanel:
    def __init__(
        self,
        workspace: GraspEditorWorkspace | Path,
        visual: GraspVisual | None = None,
        *,
        bind: str,
        port: int,
        target: str | None = None,
        storage: str | None = None,
        placement: str | None = None,
        context: str | None = None,
    ) -> None:
        if storage is not None:
            if placement is not None and placement != storage:
                raise ValueError("storage and placement disagree")
            placement = storage
        if isinstance(workspace, GraspEditorWorkspace):
            self.workspace = workspace
        else:
            if visual is None:
                raise ValueError("visual is required for a single grasp profile")
            profile_path = Path(workspace).resolve(strict=True)
            name = profile_path.stem
            self.workspace = GraspEditorWorkspace(
                targets={name: profile_path},
                default_placements={name: "default"},
                placements=("default",),
                contexts=("default",),
                build_visual=lambda _target, _placement, _context: visual,
                evaluate=lambda _target, _placement, _raw: {
                    "default": GraspCheck("checking")
                },
                arm_visual=_NoArmVisual(),
            )
            target = target or name
            placement = placement or "default"
            context = context or "default"
        self._validate_workspace()
        self.store = CalibrationStore(
            tuple(path.parent for path in self.workspace.targets.values())
        )
        self.log = ["Visualization only — editing cannot command hardware"]
        self._lock = threading.Lock()
        self._check_lock = threading.Lock()
        self.dirty = False
        self.edit_revision = 0
        self.target = target or next(iter(self.workspace.targets))
        self.placement = placement or self.workspace.default_placements[self.target]
        self.context = context or self.workspace.contexts[0]
        self.profile_path: Path
        self.visual: GraspVisual
        self.revision: str
        self.raw: Mapping[str, object]
        self.grasp: object
        self._activate(self.target, self.placement, self.context)

        self._server = ConsoleServer(
            name="grasp editor", bind=bind, port=port,
            get=self._get, post=self._post, index="editor.html",
        )
        self.server = self._server.server
        self.port = self._server.port

    # -- routes ---------------------------------------------------------------
    def _get(self, route: str):
        """JSON-able, an :class:`Asset`, or None for 404."""
        if route == "state":
            return self.state()
        if route == "scene":
            return self.scene()
        if route == "plan":
            return {"version": 0, "times": [], "frames": []}
        # Meshes are content-addressed by index; file_asset turns a missing one
        # straight into a 404, and ConsoleServer answers 304 to a client that
        # already holds the bytes (this used to re-send every STL per load).
        for prefix, lookup in (
            ("mesh/", self.mesh_path),
            ("arm-mesh/", self.workspace.arm_visual.mesh_path),
        ):
            if route.startswith(prefix):
                try:
                    index = int(route[len(prefix):])
                except ValueError:
                    return None
                return file_asset(lookup(index))
        return None

    def _post(self, route: str, payload: dict):
        # Exact routes, not the endswith() this used to do -- that also matched
        # anything ending in the name, e.g. /anything/selection.
        if route == "grasp/check":
            return self.check(payload)
        if route == "grasp":
            editor = self.update(payload)
        elif route == "grasp/save":
            self.save(payload)
            editor = self.state()["module"]
        elif route == "selection":
            editor = self.select(payload)
        else:
            return None
        return {"ok": True, "revision": self.revision, "editor": editor}

    def _validate_workspace(self) -> None:
        if not self.workspace.targets:
            raise ValueError("workspace targets must be non-empty")
        if (
            not self.workspace.placements
            or len(set(self.workspace.placements)) != len(self.workspace.placements)
            or not self.workspace.contexts
            or len(set(self.workspace.contexts)) != len(self.workspace.contexts)
        ):
            raise ValueError("workspace choices must be non-empty and unique")
        for name, path in self.workspace.targets.items():
            if not name:
                raise ValueError("workspace target names must be non-empty")
            if not Path(path).resolve(strict=True).is_file():
                raise ValueError(f"profile path must be a file: {name}")
            placement = self.workspace.default_placements.get(name)
            if placement not in self.workspace.placements:
                raise ValueError(f"missing default storage for module: {name}")

    def _activate(self, target: str, placement: str, context: str) -> None:
        if target not in self.workspace.targets:
            raise ValueError(f"unknown module: {target}")
        if placement not in self.workspace.placements:
            raise ValueError(f"unknown storage: {placement}")
        if context not in self.workspace.contexts:
            raise ValueError(f"unknown context: {context}")
        load = not hasattr(self, "profile_path") or target != self.target
        self.target = target
        self.placement = placement
        self.context = context
        self.profile_path = Path(self.workspace.targets[target]).resolve(strict=True)
        self.visual = self.workspace.build_visual(target, placement, context)
        if load:
            self.revision = self.store.revision(self.profile_path)
            self.raw = yaml.safe_load(self.profile_path.read_text())
            self.grasp = self._validate(self.raw)
            self.dirty = False
        self.visual.set_finger_width(self.grasp.finger_width_m)

    def _validate(self, raw) -> object:
        return validate_grasp_profile(
            raw,
            module_urdf=self.visual.module_urdf,
            tool_urdf=self.visual.tool_urdf,
            finger_joints=self.visual.finger_joints,
        )

    @staticmethod
    def _offset(vector: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, 3] = vector
        return transform

    def _editor_state(self) -> dict[str, object]:
        reference_T = self.visual.frame_T("module", self.grasp.reference_link)
        poses = {
            "pregrasp": reference_T
            @ self._offset(self.grasp.approach_offset_m)
            @ self.grasp.link_T_ee,
            "grasp": reference_T @ self.grasp.link_T_ee,
            "retreat": reference_T
            @ self._offset(self.grasp.retreat_offset_m)
            @ self.grasp.link_T_ee,
        }
        return {
            "type": self.grasp.module_type,
            "status": self.grasp.calibration_status,
            "reference_link": self.grasp.reference_link,
            "ee_frame": self.grasp.ee_frame,
            "reference_pose": frames.T_to_pose_xyzquat(reference_T),
            "ee_pose": frames.T_to_pose_xyzquat(poses["grasp"]),
            "pregrasp_pose": frames.T_to_pose_xyzquat(poses["pregrasp"]),
            "retreat_pose": frames.T_to_pose_xyzquat(poses["retreat"]),
            "link_T_ee": frames.T_to_pose_xyzquat(self.grasp.link_T_ee),
            "approach_offset_m": self.grasp.approach_offset_m.tolist(),
            "retreat_offset_m": self.grasp.retreat_offset_m.tolist(),
            "finger_opening": {
                "name": "Finger opening",
                "min": self.visual.finger_width_limits[0],
                "max": self.visual.finger_width_limits[1],
                "step": 0.001,
                "value": self.grasp.finger_width_m,
            },
            "contacts": {
                name: self.visual.contacts(transform)
                for name, transform in poses.items()
            },
            "revision": self.revision,
            "edit_revision": self.edit_revision,
            "choices": {
                "modules": list(self.workspace.targets),
                "storages": list(self.workspace.placements),
                "contexts": list(self.workspace.contexts),
            },
            "selection": {
                "module": self.target,
                "storage": self.placement,
                "context": self.context,
            },
        }

    def state(self) -> dict[str, object]:
        with self._lock:
            return {
                "mode": "module_editor",
                "ident": f"{self.grasp.module_type} / draft grasp",
                "log": list(self.log[-8:]),
                "module": self._editor_state(),
            }

    def update(self, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            raw = yaml.safe_load(yaml.safe_dump(self.raw, sort_keys=False))
            raw["grasp"].update(
                {
                    "link_T_ee": {
                        "pos": payload["pos"],
                        "quat": payload["quat"],
                    },
                    "finger_width_m": payload.get(
                        "finger_width_m", self.grasp.finger_width_m
                    ),
                    "approach_offset_m": payload.get(
                        "approach_offset_m", self.grasp.approach_offset_m.tolist()
                    ),
                    "retreat_offset_m": payload.get(
                        "retreat_offset_m", self.grasp.retreat_offset_m.tolist()
                    ),
                }
            )
            grasp = self._validate(raw)
            changed = (
                not np.allclose(grasp.link_T_ee, self.grasp.link_T_ee)
                or not np.allclose(
                    grasp.approach_offset_m, self.grasp.approach_offset_m
                )
                or not np.allclose(grasp.retreat_offset_m, self.grasp.retreat_offset_m)
                or not np.isclose(grasp.finger_width_m, self.grasp.finger_width_m)
            )
            if not changed:
                return self._editor_state()
            normalized = frames.T_to_pose_xyzquat(grasp.link_T_ee)
            raw["grasp"]["link_T_ee"] = {
                "pos": normalized[:3],
                "quat": normalized[3:],
            }
            self.raw, self.grasp = raw, grasp
            self.visual.set_finger_width(grasp.finger_width_m)
            self.raw["calibration_status"] = "draft"
            self.grasp = self._validate(self.raw)
            self.dirty = True
            self.edit_revision += 1
            return self._editor_state()

    def save(self, payload: dict[str, object]) -> None:
        if payload.get("confirm") is not True:
            raise ValueError("explicit save confirmation is required")
        expected = payload.get("expected_revision")
        if not isinstance(expected, str):
            raise ValueError("expected_revision is required")
        with self._lock:
            if not self.dirty:
                if expected != self.revision:
                    raise ValueError("stale revision")
                return
            self.revision = self.store.save_yaml(
                self.profile_path,
                self.raw,
                expected_revision=expected,
            )
            self.dirty = False
            self.log.append("Draft grasp saved with dated backup")

    def select(self, payload: Mapping[str, object]) -> dict[str, object]:
        with self._lock:
            target = str(payload.get("module", self.target))
            placement = str(payload.get("storage", self.placement))
            context = str(payload.get("context", self.context))
            if target not in self.workspace.targets:
                raise ValueError(f"unknown module: {target}")
            if placement not in self.workspace.placements:
                raise ValueError(f"unknown storage: {placement}")
            if context not in self.workspace.contexts:
                raise ValueError(f"unknown context: {context}")
            if (
                target != self.target
                and self.dirty
                and payload.get("discard") is not True
            ):
                raise ValueError(
                    "unsaved grasp; save or discard before switching module"
                )
            self._activate(target, placement, context)
            self.edit_revision += 1
            return self._editor_state()

    def check(self, payload: Mapping[str, object]) -> dict[str, object]:
        requested = int(payload.get("edit_revision", -1))
        with self._lock:
            if requested != self.edit_revision:
                return {"stale": True, "edit_revision": self.edit_revision}
            target = self.target
            placement = self.placement
            grasp_raw = dict(self.raw["grasp"])
            finger_width = self.grasp.finger_width_m
        with self._check_lock:
            checks = self.workspace.evaluate(target, placement, grasp_raw)
        with self._lock:
            if requested != self.edit_revision:
                return {"stale": True, "edit_revision": self.edit_revision}
        result = {}
        for name in self.workspace.contexts:
            check = checks[name]
            arm = None
            if check.q is not None:
                finger = finger_width / len(self.visual.finger_joints)
                arm = self.workspace.arm_visual.poses(check.q, finger)["geoms"]
            result[name] = {
                "status": check.status,
                "detail": check.detail,
                "arm_geoms": arm,
            }
        return {"stale": False, "edit_revision": requested, "checks": result}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def mesh_path(self, index: int) -> Path | None:
        with self._lock:
            visuals = self.visual.visuals()
        if not 0 <= index < len(visuals):
            return None
        return visuals[index].mesh

    def scene(self) -> dict[str, object]:
        with self._lock:
            visual = self.visual
            grasp = self.grasp
            arm = self.workspace.arm_visual
        result: dict[str, object] = {
            "fixture": [],
            "module": [],
            "tool": [],
            "fixed": [],
            "arm": arm.scene_json("arm-mesh"),
            "frames": {
                grasp.reference_link: _pose_json(
                    visual.frame_T("module", grasp.reference_link)
                ),
                grasp.ee_frame: _pose_json(
                    visual.frame_T("tool", grasp.ee_frame)
                ),
            },
        }
        for index, item in enumerate(visual.visuals()):
            stat = item.mesh.stat()
            entry = {
                "mesh": f"mesh/{index}?v={int(stat.st_mtime)}-{stat.st_size}",
                "link": item.link,
                "color": list(item.color),
                "scale": list(item.scale),
                "emphasized": bool(getattr(item, "emphasized", False)),
                **_pose_json(item.T),
            }
            result.setdefault(item.group, []).append(entry)
            if item.group != "tool":
                result["fixed"].append(entry)
        return result


def main() -> None:
    try:
        from nodes.arm_console import main as motion_main
    except ModuleNotFoundError:
        from arm_console import main as motion_main
    motion_main()


if __name__ == "__main__":
    main()
