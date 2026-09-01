"""Teleop + planner node: drag joint sliders, plan, preview, execute.

The control panel (open ``http://<arm-pc>:<teleop.http_port>`` from the desk
browser) is an interactive 3D page: the robot renders from its real STLs
(three.js via CDN import; the node itself stays stdlib-HTTP and serves page +
meshes + FK poses). Color protocol matches Rerun: REAL STL colors = the live
measured arm, ORANGE = the target you set, GREEN = planned-motion playback.
A gizmo on the end effector drags ONE axis at a time (arrows translate,
rings rotate) — each drag runs damped-LS IK seeded from the current target
(no restarts, so the arm never branch-jumps under your hand) and the orange
robot follows. One gripper slider (finger metres, published on the
``gripper`` output in 2-finger ``motor_command_gripper`` format — wired
straight to the sim plant, or into the executor's gripper hold slot on the
real arm). Joint sliders survive behind a debug toggle. Design + calibration
end-goal: ``docs/cartesian-teleop.md``. Rerun previews still stream
(plan_t timeline etc.). Buttons:

    Sync target to robot   copy the measured pose into the sliders
    Plan + preview         plan measured -> sliders, publish the preview
    Execute                send the previewed trajectory to the executor
    Stop (hold)            executor drops to a compliant hold at measured pose

Planning = OMPL RRT-Connect over joint space with the MuJoCo self-collision
oracle, then blended time-parameterization. Requires mujoco + ompl (this
node only — the executor stays engine-free).
"""
from __future__ import annotations

# ruff: noqa: E402

import json
import hashlib
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from dora import Node

from arm_control import CONTROL_ROOT

from arm_control.calibration_console import ConsoleAuthority
from arm_control.config import arm_joints, ee_frame, gripper_joints, load_robot_config
from arm_control.joint_motor_map import gripper_motor_to_finger
from arm_control.messages import (
    pack_json_message,
    pack_motor_command,
    pack_trajectory,
    unpack_json_message,
    unpack_motor_state,
)
from arm_control.planning.high_level import build_collision_stack
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.planning.ompl_planner import OMPLPlanner
from arm_control.planning.trajectory import (
    JointTrajectory,
    time_parameterize_blended,
)
from arm_control.node_utils import _load_mode_config, expand_named_values

_BUTTONS = ("Sync target to robot", "Plan + preview", "Execute", "Stop (hold)")
# The operator gate, when this node OWNS it (real motion graphs wire our `arm`
# output straight into the bridge — one page, one producer). Rendered as a
# separate styled row with the armed badge, never mixed into the generic
# button strip: DISARM is the authority drop and must be findable instantly.
_GATE_BUTTONS = ("ARM", "DISARM")

_CART_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

_CONSOLE_DIR = Path(__file__).resolve().parent / "console"
_CONSOLE_ASSETS = {
    "": ("text/html; charset=utf-8", "index.html", "no-store"),
    "index.html": ("text/html; charset=utf-8", "index.html", "no-store"),
    "static/style.css": ("text/css; charset=utf-8", "style.css", "no-store"),
    "static/app.js": ("text/javascript", "app.js", "no-store"),
    "static/vendor/three.module.js": (
        "text/javascript",
        "vendor/three.module.js",
        "public, max-age=31536000, immutable",
    ),
    "static/vendor/OrbitControls.js": (
        "text/javascript",
        "vendor/OrbitControls.js",
        "public, max-age=31536000, immutable",
    ),
    "static/vendor/TransformControls.js": (
        "text/javascript",
        "vendor/TransformControls.js",
        "public, max-age=31536000, immutable",
    ),
    "static/vendor/STLLoader.js": (
        "text/javascript",
        "vendor/STLLoader.js",
        "public, max-age=31536000, immutable",
    ),
}


def console_asset(route: str) -> tuple[str, bytes, str]:
    """Return one allowlisted offline console asset."""
    try:
        content_type, relative, cache = _CONSOLE_ASSETS[route.strip("/")]
    except KeyError as exc:
        raise KeyError(f"unknown console asset: {route}") from exc
    return content_type, (_CONSOLE_DIR / relative).read_bytes(), cache


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


class ControlPanel:
    """Sliders + buttons over stdlib HTTP; thread-safe state shared with the
    node loop. The page polls state; slider drags and button clicks POST back."""

    def __init__(
        self,
        names: list[str],
        lower,
        upper,
        port: int,
        vfk: VisualFK,
        grip_range: tuple[float, float] = (0.0, 0.0),
        bind: str = "127.0.0.1",
        ident: str = "",
    ) -> None:
        # grip_range (0,0) = no gripper slider; real travel comes from the
        # arm config (gripper_range_m) or its joint_mimics entry — never a
        # robot-shaped constant here.
        self._lock = threading.Lock()
        self._vfk = vfk
        self._names = list(names)
        self._lower = [float(v) for v in lower]
        self._upper = [float(v) for v in upper]
        self._values = [0.0] * len(names)
        self._measured: np.ndarray | None = None
        # (axis, abs_value|None, rot_delta|None); translation = latest wins,
        # rotation increments accumulate until the node loop consumes them.
        self._cart_pending: tuple[int, float | None, float | None] | None = None
        self._plan: dict = {"version": 0, "times": [], "frames": []}
        self._grip_lo, self._grip_hi = float(grip_range[0]), float(grip_range[1])
        self._grip = self._grip_hi               # start open
        self._grip_dirty = False
        self._clicks = {name: 0 for name in _BUTTONS + _GATE_BUTTONS}
        self._seen = {name: 0 for name in _BUTTONS + _GATE_BUTTONS}
        self._armed: bool | None = None  # None = no health wire (sim: no gate)
        self._fault = ""  # server latched-fault text; badge shows FAULTED
        self._measured_grip: float | None = None  # live finger m (Franka Hand)
        self._log: deque[str] = deque(maxlen=200)  # page shows [-8:]; a stuck
        # gizmo drag logged at loop rate once grew this without bound
        self._ident = ident
        self._authority = ConsoleAuthority(("robot",), deadman_timeout_s=0.3)
        self._authority.select("robot", now=time.monotonic())
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # quiet
                pass

            def _json(self, payload: dict, code: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                route = self.path.split("?")[0].strip("/")
                if route in _CONSOLE_ASSETS:
                    content_type, body, cache = console_asset(route)
                    etag = hashlib.sha256(body).hexdigest()
                    if self.headers.get("If-None-Match") == etag:
                        self.send_response(304)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", cache)
                    self.send_header("ETag", etag)
                    self.end_headers()
                    self.wfile.write(body)
                elif route == "state":
                    self._json(panel._state())
                elif route == "scene":
                    self._json(
                        {
                            "geoms": panel._vfk.scene_json(),
                            "static": panel._vfk.static_json(),
                        }
                    )
                elif route == "plan":
                    self._json(panel.plan_json())
                elif route.startswith("mesh/"):
                    try:
                        mesh = panel._vfk.mesh_path(int(route.split("/", 1)[1]))
                    except ValueError:
                        mesh = None
                    if mesh is None:
                        self._json({"error": "not found"}, 404)
                        return
                    body = mesh.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "max-age=86400")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:
                # This server ARMS a torque-controlled arm. Two cheap gates:
                # a cross-origin browser POST always carries an Origin header
                # that won't match ours (kills the CSRF class — any website
                # the operator visits could otherwise click ARM/Execute), and
                # a declared multi-GB body must not OOM the only node that
                # can DISARM.
                origin = self.headers.get("Origin")
                if origin is not None:
                    host = self.headers.get("Host", "")
                    if origin not in (f"http://{host}", f"https://{host}"):
                        self._json({"error": "cross-origin refused"}, 403)
                        return
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length > 64 * 1024:
                    self._json({"error": "body too large"}, 413)
                    return
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self._json({"error": "bad json"}, 400)
                    return
                if self.path.endswith("sliders"):
                    panel.set_sliders(payload.get("values") or [])
                elif self.path.endswith("cart"):
                    panel.set_cart_pending(
                        payload.get("axis"),
                        value=payload.get("value"),
                        delta=payload.get("delta"),
                    )
                elif self.path.endswith("gripper"):
                    panel.set_gripper(payload.get("value"), dirty=True)
                elif self.path.endswith("deadman"):
                    panel.set_deadman(bool(payload.get("held")))
                elif self.path.endswith("click"):
                    panel._click(str(payload.get("button", "")))
                else:
                    self._json({"error": "not found"}, 404)
                    return
                self._json({"ok": True})

        # Loopback by default: every mutating endpoint on this server can
        # ARM and move the arm, unauthenticated. LAN exposure is an
        # explicit config decision (teleop.http_bind), not a default.
        self._server = ThreadingHTTPServer((str(bind), int(port)), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _state(self) -> dict:
        self.expire_deadman()
        with self._lock:
            values = list(self._values)
            measured = None if self._measured is None else list(self._measured)
            grip = self._grip
            mgrip = self._measured_grip if self._measured_grip is not None else grip
            payload = {
                "sliders": [
                    {"name": n, "min": lo, "max": hi, "value": v}
                    for n, lo, hi, v in zip(
                        self._names, self._lower, self._upper, values
                    )
                ],
                "gripper": {
                    "name": "opening",
                    "min": self._grip_lo,
                    "max": self._grip_hi,
                    "step": 0.001,
                    "value": grip,
                },
                "buttons": list(_BUTTONS),
                "armed": self._armed,
                "fault": self._fault,
                "log": list(self._log)[-8:],  # deque: copy THEN slice
                "plan_version": self._plan["version"],
                "ident": self._ident,
                "deadman": self._authority.may_move(
                    "robot", now=time.monotonic()
                ),
            }
        target_fk = self._vfk.poses(values, grip)   # vfk has its own lock
        payload["target_geoms"] = target_fk["geoms"]
        payload["ee"] = target_fk["ee"]
        payload["measured_geoms"] = (
            None if measured is None else self._vfk.poses(measured, mgrip)["geoms"]
        )
        return payload

    def _click(self, button: str) -> None:
        with self._lock:
            if button == "Execute" and not self._authority.may_move(
                "robot", now=time.monotonic()
            ):
                self._log.append("REFUSED: hold the deadman before Execute")
                return
            if button in self._clicks:
                self._clicks[button] += 1

    def _authority_actions(self, actions: list[tuple[str, str]]) -> None:
        for action, _actor in actions:
            button = "Stop (hold)" if action == "hold" else "DISARM"
            self._clicks[button] += 1

    def set_deadman(self, held: bool) -> None:
        with self._lock:
            actions = self._authority.set_deadman(held, now=time.monotonic())
            self._authority_actions(actions)

    def expire_deadman(self) -> None:
        with self._lock:
            self._authority_actions(
                self._authority.expire(now=time.monotonic())
            )

    def set_sliders(self, values) -> None:
        with self._lock:
            for i, value in enumerate(values[: len(self._values)]):
                self._values[i] = float(
                    np.clip(float(value), self._lower[i], self._upper[i])
                )

    def sliders(self) -> np.ndarray:
        with self._lock:
            return np.array(self._values)

    def set_cart_pending(self, axis, value=None, delta=None) -> None:
        try:
            axis = int(axis)
            value = None if value is None else float(value)
            delta = None if delta is None else float(delta)
        except (TypeError, ValueError):
            return
        if not 0 <= axis < 6 or (value is None) == (delta is None):
            return
        with self._lock:
            prev = self._cart_pending
            if (
                delta is not None
                and prev is not None
                and prev[0] == axis
                and prev[2] is not None
            ):
                delta += prev[2]  # never drop an unconsumed rotation increment
            self._cart_pending = (axis, value, delta)

    def pop_cart(self) -> tuple[int, float | None, float | None] | None:
        with self._lock:
            pending, self._cart_pending = self._cart_pending, None
            return pending

    def set_armed(self, armed: bool | None, fault: str = "") -> None:
        with self._lock:
            self._armed = armed
            self._fault = fault

    def set_measured_grip(self, finger_m: float) -> None:
        with self._lock:
            self._measured_grip = finger_m

    def set_measured(self, q) -> None:
        with self._lock:
            self._measured = np.asarray(q, dtype=float).copy()

    def gripper_value(self) -> float:
        with self._lock:
            return self._grip

    def set_plan(self, times, frames) -> None:
        with self._lock:
            self._plan = {
                "version": self._plan["version"] + 1,
                "times": [float(t) for t in times],
                "frames": frames,
            }

    def clear_plan(self) -> None:
        with self._lock:
            self._plan = {"version": self._plan["version"] + 1, "times": [], "frames": []}

    def plan_json(self) -> dict:
        with self._lock:
            return dict(self._plan)

    def set_gripper(self, value, dirty: bool) -> None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        with self._lock:
            if dirty and not self._authority.may_move(
                "robot", now=time.monotonic()
            ):
                self._log.append("REFUSED: hold the deadman to move the Hand")
                return
            self._grip = float(np.clip(value, self._grip_lo, self._grip_hi))
            self._grip_dirty = self._grip_dirty or dirty

    def pop_gripper(self) -> float | None:
        with self._lock:
            if not self._grip_dirty:
                return None
            self._grip_dirty = False
            return self._grip

    def clicked(self, button: str) -> bool:
        with self._lock:
            fresh = self._clicks[button] > self._seen[button]
            self._seen[button] = self._clicks[button]
            return fresh

    def log(self, message: str) -> None:
        print(f"[motion_teleop] {message}", flush=True)
        with self._lock:
            self._log.append(message)

    def close(self) -> None:
        self._server.shutdown()


def plan_trajectory(
    world: MuJoCoCollisionWorld,
    ompl: OMPLPlanner,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    vmax: np.ndarray,
    amax: np.ndarray,
    soft_speed_frac: float = 0.5,
    soft_acc_floor: float = 0.15,
) -> JointTrajectory:
    """Plan start -> goal or raise ValueError with an operator-readable reason."""
    q_start = np.clip(q_start, world.lower, world.upper)  # measured can sit eps outside
    if world.in_collision(q_goal):
        raise ValueError("target pose is in self-collision")
    if world.in_collision(q_start):
        raise ValueError("current pose is in self-collision (check padding)")
    if np.allclose(q_start, q_goal, atol=5e-3):
        # 5 mrad: below anything worth planning. The old 1e-4 gate let a
        # near-zero path through, and the retimer's soft-launch taper turned
        # it into a 2-sample "plan" with a 5-day duration (seen on the bench).
        raise ValueError("already at the target (within 5 mrad)")
    waypoints = ompl.plan(q_start, q_goal)
    if waypoints is None:
        raise ValueError("no collision-free path found (try a different target)")
    return time_parameterize_blended(
        waypoints,
        vmax,
        amax,
        soft_speed_frac=soft_speed_frac,
        soft_acc_floor=soft_acc_floor,
    )


def main() -> None:
    import pinocchio as pin

    from arm_control.planning.ik import PinocchioIK
    from arm_control.planning.preview_rerun import (
        MeasuredGhost,
        PreviewScene,
        RobotGhost,
        init_preview_stream,
        log_static_scene,
        scene_obstacle_geoms,
    )

    cfg = load_robot_config()
    mode_cfg = _load_mode_config()
    planner_cfg = dict(mode_cfg.get("planner") or {})
    teleop_cfg = dict(mode_cfg.get("teleop") or {})
    names = list(cfg.joint_names or cfg.motor_names)
    n = cfg.num_motors
    # Default to the arm's own joint list rather than "all motors but the last":
    # that guess only holds for arms whose gripper is a trailing motor slot.
    planned = [str(j) for j in (planner_cfg.get("joints") or arm_joints(cfg))]
    # Wire contract: trajectory columns land on the FIRST len(planned) motor
    # slots in order (see pack_trajectory) — the planned joints must be a
    # prefix of the motor list.
    if planned != names[: len(planned)]:
        raise ValueError(f"planner.joints {planned} must be a prefix of {names}")
    vmax = expand_named_values(planner_cfg.get("vel_limits", 0.5), names=planned, default=0.5)
    amax = expand_named_values(planner_cfg.get("acc_limits", 1.0), names=planned, default=1.0)
    soft_speed_frac = float(planner_cfg.get("soft_speed_frac", 0.5))
    soft_acc_floor = float(planner_cfg.get("soft_acc_floor", 0.15))

    world, ompl = build_collision_stack(
        cfg.urdf_path,
        planned,
        planner_cfg,
        cache_dir=CONTROL_ROOT / ".cache" / "planning",
        # Scene bodies are obstacles, not scenery: the dock was drawn in every
        # viewer and invisible to the planner, which would route straight
        # through it.
        environment=list(cfg.get("environment") or []) + scene_obstacle_geoms(cfg),
    )
    init_preview_stream(teleop_cfg.get("preview"))
    ee_link = ee_frame(cfg)
    vfk = VisualFK(cfg.urdf_path, planned, ee_link, gripper_joints(cfg))
    # The dock/base on the teleop page AND in the preview recording — the
    # planner's `environment:` boxes are invisible, so without this the operator
    # judges reach against an empty table.
    vfk.add_static_scene(cfg)
    log_static_scene(cfg)
    # URDF-present finger joints (vfk already filtered them). The Rerun ghosts
    # carry them too: target fingers mirror the SLIDER, measured-ghost fingers
    # mirror the LIVE width — frozen URDF-default fingers on the live ghost
    # read as a rendering bug (bench-reported, same round as PreviewScene's).
    gj = list(vfk.finger_joints)
    target_robot = RobotGhost(cfg.urdf_path, planned + gj, "target")  # solid = target
    ghost = MeasuredGhost(cfg.urdf_path, planned + gj)
    # restarts=1: only the current-target seed, so a Cartesian drag follows the
    # NEAREST IK branch and never jumps the arm to a different fold mid-drag.
    ik = PinocchioIK(cfg.urdf_path, ee_link, planned, restarts=1)
    # Preview draws with the SAME FK the plan was solved against (see
    # PreviewScene: the MuJoCo collision world has no body for a fixed-joint EE
    # frame such as the FR3's fr3_hand_tcp).
    scene = PreviewScene(
        ik.fk, urdf_path=cfg.urdf_path, joint_names=planned, ee_link=ee_link,
        gripper_joints=gj,
    )
    # First mimic entry = the gripper's motor slot, whatever this arm calls it.
    mimic = next(
        (dict(m) for m in (cfg.get("joint_mimics") or {}).values() if isinstance(m, dict)),
        {},
    )
    # Finger travel for the slider. An arm with no `joint_mimics` (its gripper is
    # not a motor on this bus — the FR3's Franka Hand) has no mimic to read, and
    # the old fallback silently handed it the DM arm's 0.0439 m. State it in the
    # arm config instead: `gripper_range_m: [lower, upper]`, per FINGER.
    explicit = cfg.get("gripper_range_m")
    if explicit is not None:
        grip_range = (float(explicit[0]), float(explicit[1]))
    else:
        # A mimic without travel bounds is a config error for a slider; (0,0)
        # renders no slider rather than inventing another robot's numbers.
        grip_range = (float(mimic.get("lower", 0.0)), float(mimic.get("upper", 0.0)))
    # The page must SAY which arm it drives: a browser tab from a sim
    # rehearsal silently reattaches to a real graph on the same port, and the
    # first click lands on hardware.
    ident = os.environ.get("ARM_CONTROL_CONFIG", "") or "unknown config"
    bind = str(teleop_cfg.get("http_bind", "127.0.0.1"))
    panel = ControlPanel(
        planned,
        world.lower,
        world.upper,
        port=int(teleop_cfg.get("http_port", 7500)),
        vfk=vfk,
        grip_range=grip_range,
        bind=bind,
        ident=ident,
    )
    print(
        f"[motion_teleop] control panel at http://{bind}:{panel.port} "
        f"[{ident}] — visuals in Rerun: solid robot = target, green ghost = "
        "live arm, orange = plan",
        flush=True,
    )
    node = Node()
    measured: np.ndarray | None = None
    # None = this graph wires no motor_health (sim: no arm gate) -> allow.
    # False = a health stream says DISARMED -> Execute is refused VISIBLY on
    # the panel (the bridge would swallow the commands and the executor would
    # abort 2 s later with only a terminal line — a silent no-op at the page).
    armed: bool | None = None
    fault = ""
    pending: JointTrajectory | None = None
    shown: np.ndarray | None = None
    # Cartesian drag reference: (position, rotation) latched when a drag streak
    # starts on an axis; the untouched axes are held to it so IK tolerance
    # can't accumulate into drift across a long drag.
    cart_ref: tuple[np.ndarray, np.ndarray] | None = None
    cart_ref_q: np.ndarray | None = None
    cart_axis = -1
    synced_once = False
    grip_synced = False
    last_ui = 0.0
    last_ghost = 0.0
    last_health_t = 0.0
    health_stale_shown = False
    mgrip_f = 0.0     # live measured finger metres (gripper_state topic)
    shown_grip: float | None = None
    # Planning runs on a WORKER thread: the 2.5 s OMPL solve used to run
    # inline on the same loop that services ARM/DISARM/Stop clicks — the
    # authority drop was unresponsive for the whole solve, during live motion.
    plan_thread: threading.Thread | None = None
    plan_box: list = []  # worker appends ("ok", traj, goal_q) | ("err", msg)

    while True:
        event = node.next(timeout=0.05)
        now = time.monotonic()
        panel.expire_deadman()
        if event is not None:
            if event["type"] == "INPUT" and event["id"] == "motor_state":
                pos = unpack_motor_state(event["value"], n)["position"]
                measured = pos[: len(planned)]
                panel.set_measured(measured)
                if not synced_once:
                    # First state: start the target at the real pose, no surprises.
                    panel.set_sliders(measured)
                    if n > len(planned) and mimic:
                        panel.set_gripper(
                            gripper_motor_to_finger(float(pos[len(planned)]), mimic),
                            dirty=False,
                        )
                    synced_once = True
                if now - last_ghost >= 0.1:
                    last_ghost = now
                    ghost.update(np.append(measured, np.full(len(gj), mgrip_f)))
            elif event["type"] == "INPUT" and event["id"] == "gripper_state":
                width = float(unpack_json_message(event["value"]).get("width", 0.0))
                mgrip_f = width / 2.0
                panel.set_measured_grip(mgrip_f)
                scene.set_finger_state(mgrip_f)
                if not grip_synced:
                    # First real width: start the TARGET slider at reality —
                    # the same no-surprises rule as the arm sliders' first-
                    # state sync. It initialized to gripper_range_m's max
                    # ("start open") and lied until the first touch, which
                    # then commanded from that wrong baseline. dirty=False:
                    # a sync must never itself send a MOVE.
                    grip_synced = True
                    panel.set_gripper(mgrip_f, dirty=False)
                if measured is not None and now - last_ghost >= 0.1:
                    # Width changes with the arm parked still animate the ghost.
                    last_ghost = now
                    ghost.update(np.append(measured, np.full(len(gj), mgrip_f)))
            elif event["type"] == "INPUT" and event["id"] == "motor_health":
                health = unpack_json_message(event["value"])
                last_health_t = now
                was, was_fault = armed, fault
                armed = bool(health.get("armed", False))
                # A fault-holding server keeps its ARMED flag on purpose — the
                # latched fault is a SEPARATE dimension and the badge must
                # show it, or Execute looks legitimate while the server is
                # parked and ignoring every command (seen live on rung 3).
                fault = str(health.get("latched_fault") or "")
                panel.set_armed(armed, fault)
                if fault and fault != was_fault:
                    panel.log(f"SERVER FAULT: {fault} — DISARM then ARM to recover")
                    # Stop the executor NOW, same as DISARM does: a latched
                    # server ignores commands while the executor's clock keeps
                    # marching — the quick DISARM->ARM recovery would otherwise
                    # re-arm onto a target up to abort-tol away (a saturated-
                    # torque yank). The plan preview is kept for re-Execute.
                    node.send_output("trajectory", pack_trajectory([], [], []))
                elif was_fault and not fault:
                    panel.log("fault cleared")
                if was is not None and was != armed:
                    panel.log("ARMED" if armed else "DISARMED — Execute is gated")
            elif event["type"] == "STOP":
                break

        if now - last_ui < 0.05:
            continue
        last_ui = now

        # The badge is otherwise a LATCH: if motor_health stops arriving
        # (plant node dead, partial graph teardown) it would show green
        # ARMED forever. Unattended, "armed and healthy" and "everything
        # downstream is dead" must not render identically.
        if armed is not None and last_health_t and now - last_health_t > 1.0:
            if not health_stale_shown:
                health_stale_shown = True
                panel.set_armed(
                    armed, fault or "no health from the plant bridge — state UNKNOWN"
                )
                panel.log("motor_health stale — plant bridge down?")
        elif health_stale_shown:
            health_stale_shown = False
            panel.set_armed(armed, fault)

        if panel.clicked("Sync target to robot") and measured is not None:
            panel.set_sliders(measured)

        target = panel.sliders()
        cart_req = panel.pop_cart()
        if cart_req is not None:
            axis, value, delta = cart_req
            if (
                cart_ref is None
                or axis != cart_axis
                or cart_ref_q is None
                or np.max(np.abs(target - cart_ref_q)) > 1e-6
            ):
                T = ik.fk(target)
                cart_ref = (T[:3, 3].copy(), T[:3, :3].copy())
            cart_axis = axis
            pos, rot = cart_ref[0].copy(), cart_ref[1].copy()
            if axis < 3:
                pos[axis] = value
            else:
                # Gizmo rings send world-axis rotation INCREMENTS — compose
                # them onto the latched rotation (no rpy, no gimbal trouble).
                axvec = np.zeros(3)
                axvec[axis - 3] = 1.0
                rot = pin.AngleAxis(float(delta), axvec).matrix() @ rot
            goal_T = np.eye(4)
            goal_T[:3, :3] = rot
            goal_T[:3, 3] = pos
            sol = ik.solve(goal_T, target)
            if sol is None:
                panel.log(f"IK: {_CART_NAMES[axis]} drag unreachable from here")
            else:
                panel.set_sliders(sol)
                cart_ref = (pos, rot)
                cart_ref_q = panel.sliders()
                target = cart_ref_q

        tgrip = panel.gripper_value()
        if (
            shown is None
            or np.max(np.abs(target - shown)) > 1e-4
            or shown_grip is None
            or abs(tgrip - shown_grip) > 1e-4
        ):
            target_robot.update(np.append(target, np.full(len(gj), tgrip)))
            shown, shown_grip = target, tgrip

        grip = panel.pop_gripper()
        if grip is not None:
            zeros2 = np.zeros(2)
            node.send_output(
                "gripper",
                pack_motor_command([grip, grip], zeros2, zeros2, zeros2, zeros2),
            )

        if panel.clicked("Plan + preview"):
            if measured is None:
                panel.log("no motor state yet — cannot plan")
            elif plan_thread is not None:
                panel.log("still planning — wait for the current plan")
            else:
                t_plan = time.monotonic()
                m_snap, t_snap = measured.copy(), target.copy()

                def _plan_worker(m=m_snap, t=t_snap, t0=t_plan):
                    try:
                        traj = plan_trajectory(
                            world, ompl, m, t, vmax, amax,
                            soft_speed_frac, soft_acc_floor,
                        )
                        plan_box.append(("ok", traj, t, t0))
                    except ValueError as exc:
                        plan_box.append(("err", str(exc), t, t0))

                plan_thread = threading.Thread(target=_plan_worker, daemon=True)
                plan_thread.start()
                panel.log("planning…")

        if plan_thread is not None and plan_box:
            result = plan_box.pop()
            plan_thread = None
            if result[0] == "ok":
                _, pending, goal_q, t_plan = result
                panel.log(
                    f"plan OK: {len(pending.times)} samples, "
                    f"{pending.duration_sec:.2f}s, planned in "
                    f"{time.monotonic() - t_plan:.2f}s — scrub 'plan_t' in "
                    "Rerun, press Execute to run"
                )
                scene.show_target("goal", ik.fk(goal_q))
                scene.show_ee_path("planned_path", pending.positions)
                scene.animate(pending.times, pending.positions)
                # Green playback frames for the web page (downsampled).
                grip = panel.gripper_value()
                idx = np.linspace(
                    0, len(pending.times) - 1, min(len(pending.times), 45)
                ).astype(int)
                panel.set_plan(
                    [pending.times[i] for i in idx],
                    [vfk.poses(pending.positions[i], grip)["geoms"] for i in idx],
                )
                # An Execute click that queued up DURING the solve would fire
                # on this brand-new, never-reviewed plan — consume and drop it
                # (clicked() is consume-and-report). Execute must postdate the
                # preview it executes.
                panel.clicked("Execute")
            else:
                _, msg, _, _ = result
                pending = None
                panel.clear_plan()
                panel.log(f"plan FAILED: {msg}")

        if panel.clicked("Execute"):
            if pending is None:
                panel.log("nothing to execute — plan first")
            elif fault:
                panel.log(
                    f"REFUSED: server fault latched ({fault}) — press DISARM "
                    "then ARM to recover, then Execute"
                )
            elif armed is False:
                # Keep the plan: after arming, Execute again without replanning.
                panel.log("REFUSED: DISARMED — press ARM above, then Execute")
            else:
                node.send_output(
                    "trajectory",
                    pack_trajectory(pending.times, pending.positions, pending.velocities),
                )
                panel.log("trajectory sent")
                pending = None
                panel.clear_plan()

        if panel.clicked("Stop (hold)"):
            node.send_output("trajectory", pack_trajectory([], [], []))
            pending = None
            panel.clear_plan()
            panel.log("STOP sent — executor holds measured pose")

        # The operator gate, owned by this page in the real motion graphs
        # (our `arm` output feeds the bridge directly — single producer).
        # `armed is None` = no motor_health wire = this graph has no gate
        # (sim), and the `arm` output is likely unwired too: don't send.
        if panel.clicked("ARM"):
            if armed is None:
                panel.log("no arm gate in this graph (sim, or health not up yet)")
            else:
                node.send_output("arm", pack_json_message("arm", {"armed": True}))
                panel.log("ARM requested — bridge enables, server holds this pose")
        if panel.clicked("DISARM"):
            if armed is not None:
                # Executor to hold FIRST: a disarm+rearm inside the executor's
                # 2 s runaway window must never resume a stale trajectory.
                node.send_output("trajectory", pack_trajectory([], [], []))
                node.send_output("arm", pack_json_message("arm", {"armed": False}))
                panel.log("DISARM requested — authority drop (bridge verifies)")
            else:
                # Reachable by a programmatic supervisor before the first
                # health message: never a SILENT no-op on the stop path.
                panel.log("DISARM ignored — no health wire yet (sim, or "
                          "plant bridge still starting)")


if __name__ == "__main__":
    main()
