"""Local calibration console and visualization-only module grasp editor."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import numpy as np
import yaml

from arm_control import frames
from arm_control.calibration_console import CalibrationStore, validate_grasp_profile
from motion_teleop import console_asset, main as motion_main


def _pose_json(pin, transform) -> dict[str, list[float]]:
    quat = pin.Quaternion(transform.rotation).coeffs()
    return {
        "p": [float(value) for value in transform.translation],
        "q": [float(value) for value in quat],
    }


class ModuleVisual:
    def __init__(self, urdf: Path) -> None:
        import pinocchio as pin

        from arm_control.planning.preview_rerun import _mesh_package_dirs

        self.pin = pin
        self.model, self.visual = pin.buildModelsFromUrdf(
            str(urdf),
            package_dirs=_mesh_package_dirs(urdf) or None,
            geometry_types=[pin.GeometryType.VISUAL],
        )
        self.data = self.model.createData()
        self.vdata = self.visual.createData()
        q = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(self.model, self.data, self.visual, self.vdata, q)
        self.geom_ids = [
            index
            for index, geom in enumerate(self.visual.geometryObjects)
            if Path(geom.meshPath).is_file()
        ]

    def mesh_path(self, index: int) -> Path | None:
        if not 0 <= index < len(self.geom_ids):
            return None
        return Path(self.visual.geometryObjects[self.geom_ids[index]].meshPath)

    def scene(self) -> dict[str, object]:
        geoms = []
        for index, geom_id in enumerate(self.geom_ids):
            geom = self.visual.geometryObjects[geom_id]
            path = self.mesh_path(index)
            stat = path.stat()
            geoms.append({
                "mesh": f"mesh/{index}?v={int(stat.st_mtime)}-{stat.st_size}",
                "color": [float(value) for value in geom.meshColor],
                "scale": [float(value) for value in geom.meshScale],
                **_pose_json(self.pin, self.vdata.oMg[geom_id]),
            })
        frames_json = {}
        for name in ("Passive_Side", "Active_Side"):
            frame_id = self.model.getFrameId(name)
            if frame_id < len(self.model.frames):
                frames_json[name] = _pose_json(self.pin, self.data.oMf[frame_id])
        return {"module": geoms, "frames": frames_json}

    def frame_T(self, name: str) -> np.ndarray:
        frame_id = self.model.getFrameId(name)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"unknown module link: {name}")
        return np.asarray(self.data.oMf[frame_id].homogeneous).copy()


class GraspEditorPanel:
    def __init__(
        self,
        profile_path: Path,
        module_urdf: Path,
        *,
        bind: str,
        port: int,
    ) -> None:
        self.profile_path = profile_path.resolve(strict=True)
        self.module_urdf = module_urdf.resolve(strict=True)
        self.store = CalibrationStore((self.profile_path.parent,))
        self.revision = self.store.revision(self.profile_path)
        self.raw = yaml.safe_load(self.profile_path.read_text())
        self.grasp = validate_grasp_profile(self.raw, module_urdf=self.module_urdf)
        self.visual = ModuleVisual(self.module_urdf)
        self.log = ["Visualization only — editing cannot command hardware"]
        self._lock = threading.Lock()
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
                    self.json(panel.visual.scene())
                elif route == "plan":
                    self.json({"version": 0, "times": [], "frames": []})
                elif route.startswith("mesh/"):
                    try:
                        path = panel.visual.mesh_path(int(route.split("/", 1)[1]))
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
                        panel.update(payload)
                    elif self.path.endswith("/grasp/save"):
                        panel.save(payload)
                    else:
                        self.json({"error": "not found"}, 404)
                        return
                except (KeyError, TypeError, ValueError) as exc:
                    self.json({"error": str(exc)}, 400)
                    return
                self.json({"ok": True, "revision": panel.revision})

        self.server = ThreadingHTTPServer((bind, int(port)), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def state(self) -> dict[str, object]:
        with self._lock:
            reference_T = self.visual.frame_T(self.grasp.reference_link)
            ee_T = reference_T @ self.grasp.link_T_ee
            return {
                "mode": "module_editor",
                "ident": f"{self.grasp.module_type} / draft grasp",
                "log": list(self.log[-8:]),
                "module": {
                    "type": self.grasp.module_type,
                    "status": self.grasp.calibration_status,
                    "reference_link": self.grasp.reference_link,
                    "reference_pose": frames.T_to_pose_xyzquat(reference_T),
                    "ee_pose": frames.T_to_pose_xyzquat(ee_T),
                    "link_T_ee": frames.T_to_pose_xyzquat(self.grasp.link_T_ee),
                    "approach_offset_m": self.grasp.approach_offset_m.tolist(),
                    "retreat_offset_m": self.grasp.retreat_offset_m.tolist(),
                    "revision": self.revision,
                },
            }

    def update(self, payload: dict[str, object]) -> None:
        with self._lock:
            raw = yaml.safe_load(yaml.safe_dump(self.raw, sort_keys=False))
            raw["grasp"] = {
                "reference_link": self.grasp.reference_link,
                "link_T_ee": {"pos": payload["pos"], "quat": payload["quat"]},
                "approach_offset_m": payload.get(
                    "approach_offset_m", self.grasp.approach_offset_m.tolist()
                ),
                "retreat_offset_m": payload.get(
                    "retreat_offset_m", self.grasp.retreat_offset_m.tolist()
                ),
            }
            grasp = validate_grasp_profile(raw, module_urdf=self.module_urdf)
            normalized = frames.T_to_pose_xyzquat(grasp.link_T_ee)
            raw["grasp"]["link_T_ee"] = {
                "pos": normalized[:3],
                "quat": normalized[3:],
            }
            self.raw, self.grasp = raw, grasp

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
            self.log.append("Draft grasp saved with dated backup")

    def close(self) -> None:
        self.server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-profile")
    parser.add_argument("--module-urdf")
    parser.add_argument("--visualization-only", action="store_true")
    parser.add_argument("--http-bind", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=7500)
    args, unknown = parser.parse_known_args()
    if args.module_profile is None and args.module_urdf is None:
        if unknown:
            raise SystemExit(f"unknown arguments: {' '.join(unknown)}")
        motion_main()
        return
    if not args.visualization_only:
        raise SystemExit("module grasp editing requires --visualization-only")
    if not args.module_profile or not args.module_urdf:
        raise SystemExit("--module-profile and --module-urdf are both required")
    panel = GraspEditorPanel(
        Path(args.module_profile),
        Path(args.module_urdf),
        bind=args.http_bind,
        port=args.http_port,
    )
    print(f"http://{args.http_bind}:{panel.port}", flush=True)
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        panel.close()


if __name__ == "__main__":
    main()
