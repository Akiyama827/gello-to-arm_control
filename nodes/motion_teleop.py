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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from dora import Node

from arm_control import CONTROL_ROOT


from arm_control.config import arm_joints, ee_frame, gripper_joints, load_robot_config
from arm_control.joint_motor_map import gripper_motor_to_finger
from arm_control.messages import (
    pack_motor_command,
    pack_trajectory,
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

_CART_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

_PAGE = (Path(__file__).resolve().parent / "teleop_page.html").read_text()


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

    def __init__(self, urdf_path, joint_names: list[str], ee_frame: str,
                 finger_joints: list[str] | None = None) -> None:
        import pinocchio as pin

        from arm_control.planning.preview_rerun import _mesh_package_dirs

        self._pin = pin
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
        self._lock = threading.Lock()

    def scene_json(self) -> list[dict]:
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
                    "mesh": f"mesh/{k}?v={stamp}",
                    "color": [float(v) for v in g.meshColor],
                    "scale": [float(v) for v in g.meshScale],
                }
            )
        return out

    def mesh_path(self, k: int) -> Path | None:
        if 0 <= k < len(self._geom_ids):
            return Path(self.visual.geometryObjects[self._geom_ids[k]].meshPath)
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
            geoms = [_pose_json(pin, self.vdata.oMg[gid]) for gid in self._geom_ids]
            return {"geoms": geoms, "ee": _pose_json(pin, self.data.oMf[self._ee_id])}


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
        self._clicks = {name: 0 for name in _BUTTONS}
        self._seen = {name: 0 for name in _BUTTONS}
        self._log: list[str] = []
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
                if route in ("", "index.html"):
                    body = _PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif route == "state":
                    self._json(panel._state())
                elif route == "scene":
                    self._json({"geoms": panel._vfk.scene_json()})
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
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
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
                elif self.path.endswith("click"):
                    panel._click(str(payload.get("button", "")))
                self._json({"ok": True})

        self._server = ThreadingHTTPServer(("0.0.0.0", int(port)), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _state(self) -> dict:
        with self._lock:
            values = list(self._values)
            measured = None if self._measured is None else list(self._measured)
            grip = self._grip
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
                "log": list(self._log[-8:]),
                "plan_version": self._plan["version"],
            }
        target_fk = self._vfk.poses(values, grip)   # vfk has its own lock
        payload["target_geoms"] = target_fk["geoms"]
        payload["ee"] = target_fk["ee"]
        payload["measured_geoms"] = (
            None if measured is None else self._vfk.poses(measured, grip)["geoms"]
        )
        return payload

    def _click(self, button: str) -> None:
        with self._lock:
            if button in self._clicks:
                self._clicks[button] += 1

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
    if np.allclose(q_start, q_goal, atol=1e-4):
        raise ValueError("already at the target")
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
        environment=cfg.get("environment"),
    )
    init_preview_stream(teleop_cfg.get("preview"))
    ee_link = ee_frame(cfg)
    target_robot = RobotGhost(cfg.urdf_path, planned, "target")  # solid = target
    ghost = MeasuredGhost(cfg.urdf_path, planned)
    # restarts=1: only the current-target seed, so a Cartesian drag follows the
    # NEAREST IK branch and never jumps the arm to a different fold mid-drag.
    ik = PinocchioIK(cfg.urdf_path, ee_link, planned, restarts=1)
    # Preview draws with the SAME FK the plan was solved against (see
    # PreviewScene: the MuJoCo collision world has no body for a fixed-joint EE
    # frame such as the FR3's fr3_hand_tcp).
    scene = PreviewScene(
        ik.fk, urdf_path=cfg.urdf_path, joint_names=planned, ee_link=ee_link
    )
    vfk = VisualFK(cfg.urdf_path, planned, ee_link, gripper_joints(cfg))
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
    panel = ControlPanel(
        planned,
        world.lower,
        world.upper,
        port=int(teleop_cfg.get("http_port", 7500)),
        vfk=vfk,
        grip_range=grip_range,
    )
    print(
        f"[motion_teleop] control panel at http://0.0.0.0:{panel.port} — visuals "
        "in Rerun: solid robot = target, green ghost = live arm, orange = plan",
        flush=True,
    )
    node = Node()
    measured: np.ndarray | None = None
    pending: JointTrajectory | None = None
    shown: np.ndarray | None = None
    # Cartesian drag reference: (position, rotation) latched when a drag streak
    # starts on an axis; the untouched axes are held to it so IK tolerance
    # can't accumulate into drift across a long drag.
    cart_ref: tuple[np.ndarray, np.ndarray] | None = None
    cart_ref_q: np.ndarray | None = None
    cart_axis = -1
    synced_once = False
    last_ui = 0.0
    last_ghost = 0.0

    while True:
        event = node.next(timeout=0.05)
        now = time.monotonic()
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
                    ghost.update(measured)
            elif event["type"] == "STOP":
                break

        if now - last_ui < 0.05:
            continue
        last_ui = now

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

        if shown is None or np.max(np.abs(target - shown)) > 1e-4:
            target_robot.update(target)
            shown = target

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
            else:
                t_plan = time.monotonic()
                try:
                    pending = plan_trajectory(
                        world, ompl, measured, target, vmax, amax,
                        soft_speed_frac, soft_acc_floor,
                    )
                    panel.log(
                        f"plan OK: {len(pending.times)} samples, "
                        f"{pending.duration_sec:.2f}s, planned in "
                        f"{time.monotonic() - t_plan:.2f}s — scrub 'plan_t' in "
                        "Rerun, press Execute to run"
                    )
                    scene.show_target("goal", ik.fk(target))
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
                except ValueError as exc:
                    pending = None
                    panel.clear_plan()
                    panel.log(f"plan FAILED: {exc}")

        if panel.clicked("Execute"):
            if pending is None:
                panel.log("nothing to execute — plan first")
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


if __name__ == "__main__":
    main()
