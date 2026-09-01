"""Local calibration console and visualization-only module grasp editor."""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable, Mapping

import numpy as np
import yaml

from arm_control import frames
from arm_control.calibration_console import CalibrationStore, validate_grasp_profile
from arm_control.grasp_visual import GraspVisual
try:
    from nodes.motion_teleop import VisualFK, console_asset, main as motion_main
except ModuleNotFoundError:
    from motion_teleop import VisualFK, console_asset, main as motion_main


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
        placement: str | None = None,
        context: str | None = None,
    ) -> None:
        try:
            loopback = ipaddress.ip_address(bind).is_loopback
        except ValueError as exc:
            raise ValueError("grasp editor bind must be a loopback address") from exc
        if not loopback:
            raise ValueError("grasp editor bind must be a loopback address")
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
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def json(self, value: object, code: int = 200) -> None:
                body = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                route = self.path.split("?", 1)[0].strip("/")
                if route in ("", "index.html") or route.startswith("static/"):
                    try:
                        content_type, body, cache = console_asset(route)
                    except KeyError:
                        self.json({"error": "not found"}, 404)
                        return
                    etag = hashlib.sha256(body).hexdigest()
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", cache)
                    self.send_header("ETag", etag)
                    self.end_headers()
                    self.wfile.write(body)
                elif route == "state":
                    self.json(panel.state())
                elif route == "scene":
                    self.json(panel.scene())
                elif route == "plan":
                    self.json({"version": 0, "times": [], "frames": []})
                elif route.startswith("mesh/"):
                    try:
                        path = panel.mesh_path(int(route.split("/", 1)[1]))
                    except ValueError:
                        path = None
                    if path is None:
                        self.json({"error": "not found"}, 404)
                        return
                    body = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif route.startswith("arm-mesh/"):
                    try:
                        path = panel.workspace.arm_visual.mesh_path(
                            int(route.split("/", 1)[1])
                        )
                    except ValueError:
                        path = None
                    if path is None:
                        self.json({"error": "not found"}, 404)
                        return
                    body = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.json({"error": "not found"}, 404)

            def do_POST(self) -> None:
                origin = self.headers.get("Origin")
                host = self.headers.get("Host", "")
                if origin is not None and origin not in (f"http://{host}", f"https://{host}"):
                    self.json({"error": "cross-origin refused"}, 403)
                    return
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length > 64 * 1024:
                    self.json({"error": "body too large"}, 413)
                    return
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if self.path.endswith("/grasp"):
                        editor = panel.update(payload)
                    elif self.path.endswith("/grasp/save"):
                        panel.save(payload)
                        editor = panel.state()["module"]
                    elif self.path.endswith("/selection"):
                        editor = panel.select(payload)
                    elif self.path.endswith("/grasp/check"):
                        self.json(panel.check(payload))
                        return
                    else:
                        self.json({"error": "not found"}, 404)
                        return
                except (KeyError, TypeError, ValueError) as exc:
                    self.json({"error": str(exc)}, 400)
                    return
                self.json(
                    {"ok": True, "revision": panel.revision, "editor": editor}
                )

        self.server = ThreadingHTTPServer((bind, int(port)), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

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
            Path(path).resolve(strict=True)
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
        visuals = self.visual.visuals()
        if not 0 <= index < len(visuals):
            return None
        return visuals[index].mesh

    def scene(self) -> dict[str, object]:
        result: dict[str, object] = {
            "fixture": [],
            "module": [],
            "tool": [],
            "fixed": [],
            "arm": self.workspace.arm_visual.scene_json("arm-mesh"),
            "frames": {
                self.grasp.reference_link: _pose_json(
                    self.visual.frame_T("module", self.grasp.reference_link)
                ),
                self.grasp.ee_frame: _pose_json(
                    self.visual.frame_T("tool", self.grasp.ee_frame)
                ),
            },
        }
        for index, item in enumerate(self.visual.visuals()):
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
    motion_main()


if __name__ == "__main__":
    main()
