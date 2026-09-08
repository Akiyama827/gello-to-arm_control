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
end-goal: ``docs/history/cartesian-teleop.md``. Rerun previews still stream
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

import os
import threading
import uuid
import time
from collections import deque
from contextlib import ExitStack

import numpy as np
from dora import Node

from arm_control import CONTROL_ROOT, REPO_ROOT

from arm_control.config import arm_joints, ee_frame, gripper_joints, load_robot_config
from arm_control.contracts.impedance import pose_hold_values
from arm_control.contracts.gripper import pack_grasp_request, unpack_grasp_result
from arm_control.end_effectors.franka_hand import (
    GRASP_TIMEOUT_S, MIN_FORCE_N, MAX_FORCE_N, resolve_grasp_parameters,
)
from arm_control.grasp_visual import VisualFK
from arm_control.joint_motor_map import gripper_motor_to_finger
from arm_control.messages import (
    pack_control_update,
    pack_motor_command,
    pack_jog,
    pack_plan,
    unpack_json_message,
    unpack_motor_state,
)
from arm_control.planning.high_level import build_collision_stack
from arm_control.planning.jog import JogLimits, check_step
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.planning.ompl_planner import OMPLPlanner
from arm_control.motion import JointTrajectory
from arm_control.planning.retiming import time_parameterize_blended
from arm_control.node_utils import (
    ShutdownFlag,
    _load_mode_config,
    expand_named_values,
    install_signal_handlers,
    next_event_gil_friendly,
    resolve_gains,
)
from arm_control.console_server import ConsoleServer, file_asset

_BUTTONS = ("Sync target to robot", "Plan + preview", "Execute", "Stop (hold)")
#: Gain presets are just buttons -- the page renders whatever `state.buttons`
#: lists, so a control law the operator can switch mid-session costs no page
#: code at all. Named at runtime from the mode config's `gain_presets`.
_GAIN_PREFIX = "Gains: "
# The operator requests authority; arm_controller owns the arm output.
# Rendered as a
# separate styled row with the armed badge, never mixed into the generic
# button strip: DISARM is the authority drop and must be findable instantly.
_GATE_BUTTONS = ("ARM", "DISARM")

_CART_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
#: Jog axes the page offers. World frame for translation -- an operator asking
#: for "down" means down in the room, not down along a tool that may be tilted.
_JOG_AXES = ("x", "y", "z")
#: How long a held-jog assertion stays good at the console. The page re-asserts
#: every 100 ms, so this tolerates a few missed polls and no more -- it is the
#: browser->console half of the jog's two independent deadmen.
JOG_STALE_S = 0.4
HAND_STALE_S = 0.6


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
        extra_buttons: tuple[str, ...] = (),
        jog_speed_m_s: float = 0.01,
        jog_joint_speed_rad_s: float = 0.15,
        gripper_cfg: dict | None = None,
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
        # A held jog direction: ("x".."z" | "rx".."rz" | "j0".."jN", +1/-1), or
        # None while nothing is held. The page re-asserts it; releasing clears
        # it. Held state, not a queue -- a jog is "keep going", not "go once".
        self._jog: tuple[str, int] | None = None
        self._jog_at = 0.0
        self._jog_note = ""
        self._jog_speed_m_s = float(jog_speed_m_s)
        self._jog_joint_speed_rad_s = float(jog_joint_speed_rad_s)
        self._plan: dict = {"version": 0, "times": [], "frames": []}
        self._grip_lo, self._grip_hi = float(grip_range[0]), float(grip_range[1])
        self._grip = self._grip_hi               # start open
        self._grip_dirty = False
        self._hand_cfg = dict(gripper_cfg or {})
        self._hand_defaults = resolve_grasp_parameters(self._hand_cfg, {}) if self._hand_cfg else None
        self._hand_state: dict = {}
        self._hand_seq = None
        self._hand_sample_at = float("-inf")
        self._hand_pending: dict | None = None
        self._hand_request_id = ""
        self._hand_waiting = False
        self._hand_sent_at = 0.0
        self._hand_submit_seq = None
        self._hand_status = "Idle"
        self._hand_disarm_requested = False
        # Gain presets (and anything else runtime-named) join the same strip:
        # the page renders `state.buttons`, so this costs no page code.
        self._buttons = tuple(_BUTTONS) + tuple(extra_buttons)
        self._clicks = {name: 0 for name in self._buttons + _GATE_BUTTONS}
        self._seen = {name: 0 for name in self._buttons + _GATE_BUTTONS}
        self._armed: bool | None = None  # Unknown until authority is reported.
        self._fault = ""  # server latched-fault text; badge shows FAULTED
        self._supports_pose_hold = False
        self._control_mode = "joint"
        self._measured_grip: float | None = None  # live finger m (Franka Hand)
        self._log: deque[str] = deque(maxlen=200)  # page shows [-8:]; a stuck
        # gizmo drag logged at loop rate once grew this without bound
        self._ident = ident

        # Loopback by REFUSAL, not merely by default. This inherited
        # `require_loopback=False` from motion_teleop, whose page could only
        # plan -- a reviewed move an operator approves before it runs. This
        # page can JOG: unreviewed motion, unauthenticated, at the press of a
        # button. `http_bind` alone is one config typo away from a robot the
        # whole subnet can drive, so the bind address is now VALIDATED: a
        # non-loopback `http_bind` raises at startup instead of quietly
        # working. (The kernel does the actual enforcing -- a socket bound to
        # 127.0.0.1 never sees a packet off the wire; this just refuses to
        # bind anywhere else.)
        # Remote access is an explicit SSH tunnel:
        #     ssh -L 7500:127.0.0.1:7500 <host>
        self._server = ConsoleServer(
            name="control panel", bind=str(bind), port=int(port),
            get=self._get, post=self._post, require_loopback=True,
            index="console.html",
        )
        self.port = self._server.port

    # -- routes ---------------------------------------------------------------
    def _get(self, route: str):
        if route == "state":
            return self._state()
        if route == "scene":
            return {
                "geoms": self._vfk.scene_json(),
                "static": self._vfk.static_json(),
            }
        if route == "plan":
            return self.plan_json()
        if route.startswith("mesh/"):
            try:
                index = int(route[len("mesh/"):])
            except ValueError:
                return None
            return file_asset(self._vfk.mesh_path(index))
        return None

    def _post(self, route: str, payload: dict):
        # Exact routes, not endswith(): that also matched /anything/click.
        if route == "sliders":
            self.set_sliders(payload.get("values") or [])
        elif route == "cart":
            self.set_cart_pending(
                payload.get("axis"),
                value=payload.get("value"),
                delta=payload.get("delta"),
            )
        elif route == "gripper":
            self.set_gripper(payload.get("value"), dirty=True)
        elif route == "hand":
            self.request_hand(payload)
        elif route == "jog":
            self.set_jog(payload.get("axis"), payload.get("dir"), payload.get("held"))
        elif route == "click":
            self._click(str(payload.get("button", "")))
        else:
            return None
        return {"ok": True}

    def _state(self) -> dict:
        with self._lock:
            values = list(self._values)
            measured = None if self._measured is None else list(self._measured)
            grip = self._grip
            # Unknown feedback must not borrow the commanded finger target.
            mgrip = self._measured_grip if self._measured_grip is not None else 0.0
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
                "hand": self._hand_snapshot(),
                "buttons": list(self._buttons),
                "armed": self._armed,
                "fault": self._fault,
                "supports_pose_hold": self._supports_pose_hold,
                "control_mode": self._control_mode,
                "log": list(self._log)[-8:],  # deque: copy THEN slice
                "plan_version": self._plan["version"],
                "ident": self._ident,
                "jog": {
                    "axes": list(_JOG_AXES),
                    "joints": len(self._names),
                    "held": None if self._jog is None else list(self._jog),
                    "note": self._jog_note,
                    "speed_m_s": self._jog_speed_m_s,
                    "joint_speed_rad_s": self._jog_joint_speed_rad_s,
                },
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
            if button in ("Plan + preview", "Execute") and self._control_mode == "soft":
                self._log.append("REFUSED: select Track before planning or executing")
                return
            if button == "Execute" and (self._armed is not True or self._fault):
                self._log.append("REFUSED: Execute requires confirmed ARM and no fault")
                return
            if button in ("Stop (hold)", "DISARM"):
                self._jog = None
            if button == "DISARM":
                self._hand_disarm_requested = True
                self._grip_dirty = False
                if self._hand_pending is not None:
                    self._hand_pending = None
                    self._hand_waiting = False
            if button in self._clicks:
                self._clicks[button] += 1

    def set_jog(self, axis, direction, held) -> None:
        """Hold or release one jog direction. Unknown axes are refused loudly."""
        with self._lock:
            if not held:
                self._jog = None
                return
            if self._control_mode == "soft":
                self._jog_note = "jog refused — select Track before jogging"
                return
            axis = str(axis or "")
            if axis not in _JOG_AXES and not (
                axis.startswith("j") and axis[1:].isdigit()
                and int(axis[1:]) < len(self._names)
            ):
                raise ValueError(f"unknown jog axis {axis!r}")
            self._jog = (axis, 1 if float(direction or 0) >= 0 else -1)
            self._jog_at = time.monotonic()

    def jog_held(self) -> tuple[str, int] | None:
        """The held direction, while the page keeps saying it is still held.

        The held jog button IS the deadman: the page re-asserts every 100 ms
        and this goes stale in
        JOG_STALE_S. Releasing stops the motion and leaves the arm armed and
        holding. The controller's own expiry is the second, independent stop --
        this one cannot save an arm from a console that has itself wedged.
        """
        with self._lock:
            if self._jog is None:
                return None
            if time.monotonic() - self._jog_at > JOG_STALE_S:
                self._jog = None
                return None
            return self._jog

    def set_jog_note(self, note: str) -> None:
        with self._lock:
            if note != self._jog_note:
                self._jog_note = note
                if note:
                    self._log.append(note)

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
            if armed is False:
                self._control_mode = "joint"
                self._hand_disarm_requested = False
            if armed is not True or fault:
                self._grip_dirty = False
                if self._hand_pending is not None:
                    self._hand_pending = None
                    self._hand_waiting = False

    def set_control_mode(self, mode: str) -> None:
        with self._lock:
            self._control_mode = mode

    def set_pose_hold_capability(self, supported: bool) -> None:
        with self._lock:
            self._supports_pose_hold = bool(supported)

    def set_measured_grip(self, finger_m: float) -> None:
        with self._lock:
            self._measured_grip = finger_m

    def set_hand_state(self, state: dict) -> None:
        with self._lock:
            if not state.get("available"):
                self._hand_pending = None
                self._hand_request_id = ""
                self._hand_waiting = False
                self._hand_status = "Unavailable: Hand disconnected"
            self._hand_state = dict(state)
            seq = state.get("sample_seq")
            if seq is not None and seq != self._hand_seq and state.get("measured"):
                self._hand_seq = seq
                self._hand_sample_at = time.monotonic()

    def _hand_refusal(self, *, consuming: bool = False) -> str:
        """Called under the panel lock, both at HTTP admission and dispatch."""
        if self._hand_defaults is None or not self._hand_state.get("force_grasp"):
            return "Force grasp unavailable: plant or Hand bridge does not support it"
        if not self._hand_state.get("available") or not self._hand_state.get("measured") or time.monotonic() - self._hand_sample_at > HAND_STALE_S:
            return "Hand feedback unavailable or stale"
        if self._armed is not True or self._fault or self._hand_disarm_requested:
            return "Hand motion requires confirmed ARM and no fault"
        if self._hand_state.get("busy"):
            return "Hand busy: wait for the current action"
        if not consuming and (self._hand_waiting or self._hand_submit_seq == self._hand_seq):
            return "Hand action pending: wait for a fresh result and observation"
        return ""

    def _hand_busy(self) -> bool:
        return (self._hand_waiting or bool(self._hand_state.get("busy"))
                or (self._hand_submit_seq is not None and self._hand_submit_seq == self._hand_seq))

    def _hand_snapshot(self) -> dict:
        if self._hand_waiting and time.monotonic() - self._hand_sent_at > GRASP_TIMEOUT_S + 1:
            self._hand_waiting = False
            self._hand_pending = None
            self._hand_request_id = ""
            self._hand_status = "Unavailable: action result timed out"
        reason = self._hand_refusal()
        fresh = bool(self._hand_state.get("available")) and time.monotonic() - self._hand_sample_at <= HAND_STALE_S
        width = self._hand_state.get("width")
        if (self._hand_status.startswith("Open accepted") and fresh
                and self._hand_state.get("measured") and not self._hand_state.get("busy")
                and self._hand_seq != self._hand_submit_seq and width is not None
                and abs(float(width) - self._hand_defaults["open_width_m"]) <= .002):
            self._hand_status = "Open"
        return {
            "enabled": not reason,
            "reason": reason,
            "status": self._hand_status,
            "busy": self._hand_busy(),
            "defaults": self._hand_defaults,
            "force_min_n": MIN_FORCE_N, "force_max_n": MAX_FORCE_N,
            "width_max_mm": self._grip_hi * 2000,
            "measured_width_mm": float(width) * 1000 if fresh and self._hand_state.get("measured") and width is not None else None,
        }

    def request_hand(self, payload: dict) -> None:
        with self._lock:
            if not isinstance(payload, dict) or set(payload) - {"mode", "width_m", "force_n"}:
                raise ValueError("Hand action accepts mode, width_m and force_n only")
            if payload.get("mode") == "release" and set(payload) != {"mode"}:
                raise ValueError("Open uses the configured open width and speed")
            reason = self._hand_refusal()
            if reason:
                raise ValueError(reason)
            params = resolve_grasp_parameters(self._hand_cfg, payload)
            self._hand_request_id = uuid.uuid4().hex
            self._hand_pending = {
                "request_id": self._hand_request_id, "target_id": "hand",
                "mode": payload.get("mode", "close"),
                "width_m": params["width_m"], "force_n": params["force_n"],
            }
            self._hand_submit_seq = self._hand_seq
            self._hand_waiting = True
            self._hand_sent_at = time.monotonic()
            self._hand_status = "Grasp pending" if payload.get("mode", "close") == "close" else "Open pending"
            self._grip_dirty = False

    def pop_hand(self) -> dict | None:
        with self._lock:
            pending, self._hand_pending = self._hand_pending, None
            if pending is None:
                return None
            reason = self._hand_refusal(consuming=True)
            if time.monotonic() - self._hand_sent_at > HAND_STALE_S:
                reason = "Hand request expired before dispatch"
            if reason:
                self._hand_waiting = False
                self._hand_status = reason
                self._log.append(f"REFUSED: {reason}")
                return None
            return pending

    def set_hand_result(self, result: dict) -> None:
        with self._lock:
            if not self._hand_request_id or result.get("request_id") != self._hand_request_id:
                return
            self._hand_waiting = False
            reason = str(result.get("reason", ""))
            if result.get("ok"):
                self._hand_status = "Open accepted; waiting for finger feedback" if reason == "released" else "Held"
            else:
                self._hand_status = "Lost" if reason == "object lost" else "Missed" if reason == "no object" else f"Unavailable: {reason}"
            self._log.append(f"Hand: {self._hand_status}")

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
            if dirty and (self._armed is not True or self._fault):
                self._log.append("REFUSED: Hand motion requires confirmed ARM and no fault")
                return
            if dirty and (self._hand_busy() or self._hand_disarm_requested):
                self._log.append("REFUSED: Hand action pending; wait before moving the slider")
                return
            self._grip = float(np.clip(value, self._grip_lo, self._grip_hi))
            self._grip_dirty = self._grip_dirty or dirty

    def pop_gripper(self) -> float | None:
        with self._lock:
            if not self._grip_dirty:
                return None
            self._grip_dirty = False
            if self._armed is not True or self._fault or self._hand_disarm_requested or self._hand_busy():
                return None
            return self._grip

    def clicked(self, button: str) -> bool:
        with self._lock:
            fresh = self._clicks[button] > self._seen[button]
            self._seen[button] = self._clicks[button]
            return fresh

    def log(self, message: str) -> None:
        print(f"[arm_console] {message}", flush=True)
        with self._lock:
            self._log.append(message)

    def close(self) -> None:
        self._server.close()


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
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    with ExitStack() as cleanup:
        _run(shutdown, cleanup)


def _run(shutdown: ShutdownFlag, cleanup: ExitStack) -> None:
    import pinocchio as pin
    import rerun as rr

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
    # Gains ship WITH the plan now (pack_plan carries kp/kd), so a plan and the
    # stiffness it was reviewed at cannot be separated in flight. Resolved by
    # the same helper the executor node uses -- one number for one arm, not two
    # resolutions that can disagree. Arm-length: the controller's executor
    # servos the arm joints only, and set_gains checks that shape.
    _gains = resolve_gains(cfg, mode_cfg, names, len(planned))
    plan_kp = _gains["kp"][: len(planned)]
    plan_kd = _gains["kd"][: len(planned)]
    # Gain presets: name -> (kp, kd) over the planned joints. An empty entry
    # means "the configured tracking law", so a config can name `track` without
    # restating numbers that already sit in controller.kp/kd above it.
    # Jog envelope. Every bound is a config number, and the defaults are the
    # conservative ones: 1 cm/s, 20 cm of stroke, a floor at z=0.
    jog_cfg = dict(mode_cfg.get("jog") or {})
    jog_limits = JogLimits(
        max_travel_m=float(jog_cfg.get("max_travel_m", 0.20)),
        speed_m_s=float(jog_cfg.get("speed_m_s", 0.01)),
        rot_speed_rad_s=float(jog_cfg.get("rot_speed_rad_s", 0.10)),
        floor_z=(
            None if jog_cfg.get("floor_z", 0.0) is None
            else float(jog_cfg.get("floor_z", 0.0))
        ),
        floor_clearance_m=float(jog_cfg.get("floor_clearance_m", 0.02)),
        workspace_min=jog_cfg.get("workspace_min"),
        workspace_max=jog_cfg.get("workspace_max"),
        joint_margin_rad=float(jog_cfg.get("joint_margin_rad", 0.05)),
        sigma_min=float(jog_cfg.get("sigma_min", 0.02)),
    )
    jog_joint_speed = float(jog_cfg.get("joint_speed_rad_s", 0.15))

    gain_presets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pose_hold_presets: dict[str, dict] = {}
    for preset_name, spec in (mode_cfg.get("gain_presets") or {}).items():
        spec = dict(spec or {})
        if "pose_hold" in spec:
            pose_hold_values(dict(spec["pose_hold"], id=1))
            pose_hold_presets[str(preset_name)] = dict(spec["pose_hold"])
        gain_presets[str(preset_name)] = (
            plan_kp if spec.get("kp") is None
            else expand_named_values(spec["kp"], names=planned, default=0.0),
            plan_kd if spec.get("kd") is None
            else expand_named_values(spec["kd"], names=planned, default=0.0),
        )
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
    # Detach the sink before interpreter shutdown; an absent viewer must not
    # leave Rerun's implicit final flush waiting after Dora has stopped.
    cleanup.callback(rr.disconnect)
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
    # ARM_ID names the INSTANCE (two arms on one bench are "left"/"right");
    # the config path is the fallback so a single-arm graph needs no env at all.
    arm_id = os.environ.get("ARM_ID", "").strip()
    ident = arm_id or os.environ.get("ARM_CONTROL_CONFIG", "") or "unknown config"
    # The panel's address is a per-INSTANCE fact, so it lives with the robot,
    # not with the mode. It used to sit in the mode config, where it only
    # avoided a collision because the two arms happened to run different modes
    # -- two FR3s would have fought over 7501, and the fix would have been to
    # fork a mode config to change a port. Modes say how to control; the robot
    # config says which robot and which instance.
    console_cfg = dict(cfg.get("console") or {})
    bind = str(console_cfg.get("http_bind", "127.0.0.1"))
    panel = ControlPanel(
        planned,
        world.lower,
        world.upper,
        port=int(console_cfg.get("http_port", 7500)),
        vfk=vfk,
        grip_range=grip_range,
        bind=bind,
        ident=ident,
        extra_buttons=tuple(f"{_GAIN_PREFIX}{n}" for n in gain_presets),
        jog_speed_m_s=jog_limits.speed_m_s,
        jog_joint_speed_rad_s=jog_joint_speed,
        gripper_cfg=dict((cfg.get("franka") or {}).get("gripper") or {}),
    )
    cleanup.callback(panel.close)
    # The legend, stated once, the same on the page and in Rerun. It used to
    # read "solid robot = target, green ghost = live arm, orange = plan",
    # which inverts all three against what both surfaces actually draw
    # (app.js target 0xd99a24 orange / plan 0x55aa7a green; PreviewScene's
    # docstring carries the same spec). A wrong legend printed at startup is
    # worse than no legend -- it is the first thing an operator reads.
    print(
        f"[arm_console] control panel at http://{bind}:{panel.port} "
        f"[{ident}] — page and Rerun share one colour legend: "
        "REAL STL = the live arm, ORANGE = your target, GREEN = planned motion",
        flush=True,
    )
    node = Node()
    measured: np.ndarray | None = None
    # Hardware reports plant health; bare sim graphs report controller authority.
    # Neither an absent topic nor an ARM click is confirmation.
    armed: bool | None = None
    fault = ""
    pending: JointTrajectory | None = None
    # The id the controller knows this plan by; `execute` must name it.
    pending_id = ""
    # Live jog state: the setpoint being walked, the EE position the stroke
    # limit is measured from, and the last tick's clock. All three clear on
    # release, so every press re-anchors.
    jog_q: np.ndarray | None = None
    jog_anchor: np.ndarray | None = None
    jog_t: float | None = None
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
    control_mode = "joint"
    supports_pose_hold = False
    mgrip_f = 0.0     # live measured finger metres (gripper_state topic)
    shown_grip: float | None = None
    # Planning runs on a WORKER thread: the 2.5 s OMPL solve used to run
    # inline on the same loop that services ARM/DISARM/Stop clicks — the
    # authority drop was unresponsive for the whole solve, during live motion.
    plan_thread: threading.Thread | None = None
    plan_box: list = []  # worker appends ("ok", traj, goal_q) | ("err", msg)

    while not shutdown.stop_requested:
        event = next_event_gil_friendly(node, idle_sleep=0.05)
        if shutdown.stop_requested:
            break
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
                    ghost.update(np.append(measured, np.full(len(gj), mgrip_f)))
            elif event["type"] == "INPUT" and event["id"] == "gripper_state":
                hand_state = unpack_json_message(event["value"])
                panel.set_hand_state(hand_state)
                if hand_state.get("width") is None or hand_state.get("available") is False:
                    continue
                width = float(hand_state["width"])
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
            elif event["type"] == "INPUT" and event["id"] == "grasp_result":
                panel.set_hand_result(unpack_grasp_result(event["value"]))
            elif event["type"] == "INPUT" and event["id"] == "plant_capabilities":
                supports_pose_hold = bool(unpack_json_message(event["value"]).get("supports_pose_hold", False))
                panel.set_pose_hold_capability(supports_pose_hold)
            elif event["type"] == "INPUT" and event["id"] == "controller_arm":
                armed = bool(unpack_json_message(event["value"]).get("armed", False))
                panel.set_armed(armed, fault)
                if not armed:
                    control_mode = "joint"
                panel.log("Controller ARMED" if armed else "Controller DISARMED")
            elif event["type"] == "INPUT" and event["id"] == "controller_event":
                result = unpack_json_message(event["value"])
                if result["kind"] == "fault":
                    fault = str(result.get("reason") or "controller fault")
                    panel.set_armed(armed, fault)
                elif result["kind"] == "mode" and result["ok"]:
                    control_mode = str(result["reason"])
                    panel.set_control_mode(control_mode)
                panel.log(f"Controller {result['kind']}: {result.get('reason') or ''}")
            elif event["type"] == "INPUT" and event["id"] == "motor_health":
                health = unpack_json_message(event["value"])
                supports_pose_hold = bool(health.get("supports_pose_hold", False))
                panel.set_pose_hold_capability(supports_pose_hold)
                last_health_t = now
                was, was_fault = armed, fault
                armed = bool(health.get("armed", False))
                if not armed:
                    control_mode = "joint"
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
                    node.send_output(
                        "control",
                        pack_control_update(cancel=True, reason=f"server fault: {fault}"),
                    )
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
        hand_request = panel.pop_hand()
        if hand_request is not None:
            node.send_output("grasp_request", pack_grasp_request(**hand_request))

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
                # Hand the controller the plan GATED and named. It loads it,
                # keeps streaming its hold, and runs nothing until an `execute`
                # names this exact id -- so a plan superseded by a re-plan can
                # never run, which a bare trajectory topic could not express.
                pending_id = f"teleop-{uuid.uuid4().hex[:8]}"
                node.send_output(
                    "plan",
                    pack_plan(
                        plan_id=pending_id,
                        phase="teleop",
                        gated=True,
                        times=pending.times,
                        positions=pending.positions,
                        velocities=pending.velocities,
                        kp=plan_kp,
                        kd=plan_kd,
                    ),
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
            elif armed is not True:
                # Keep the plan: after arming, Execute again without replanning.
                panel.log("REFUSED: DISARMED — press ARM above, then Execute")
            else:
                node.send_output(
                    "control", pack_control_update(execute=pending_id)
                )
                panel.log(f"executing {pending_id}")
                pending = None
                pending_id = ""
                panel.clear_plan()

        # ---- jog: one small step per tick, five gates, then emit -----------
        jog_held = panel.jog_held()
        if jog_held is None:
            jog_q = None          # released: the next press re-anchors
            jog_anchor = None
            jog_t = None
        elif measured is not None:
            axis, direction = jog_held
            now_t = time.monotonic()
            if jog_q is None:
                # Anchor on the PRESS, from the measured pose. The stroke limit
                # is measured from here, so "20 cm" means 20 cm from where the
                # operator started this jog -- not an unbounded walk made of
                # individually-legal millimetres.
                jog_q = np.asarray(measured, dtype=float).copy()
                jog_anchor = ik.fk(jog_q)[:3, 3].copy()
                jog_t = now_t
                panel.set_jog_note(f"jog {axis}{'+' if direction > 0 else '-'} started")
            dt = min(max(now_t - jog_t, 0.0), 0.1)  # a stalled loop must not leap
            jog_t = now_t
            joint_jog = axis.startswith("j")

            if joint_jog:
                q_next = jog_q.copy()
                q_next[int(axis[1:])] += direction * jog_joint_speed * dt
                T_next = ik.fk(q_next)
                p_next = T_next[:3, 3]
            else:
                T = ik.fk(jog_q)
                p_next = T[:3, 3].copy()
                p_next["xyz".index(axis)] += direction * jog_limits.speed_m_s * dt
                T_next = T.copy()
                T_next[:3, 3] = p_next
                q_next = ik.solve(T_next, q0=jog_q)

            if q_next is None:
                panel.set_jog_note(
                    f"jog {axis}: no IK solution — try a joint jog to back out"
                )
            else:
                verdict = check_step(
                    p_next, q_next,
                    anchor=jog_anchor,
                    limits=jog_limits,
                    q_lower=world.lower,
                    q_upper=world.upper,
                    # so a joint already resting on a hard stop can still be
                    # jogged OFF it -- some spawn poses sit exactly there.
                    q_now=jog_q,
                    # A JOINT jog is the escape hatch FROM a singularity, so it
                    # is not gated on one -- refusing it would strand an
                    # operator in the pose they are trying to leave.
                    sigma_min=None if joint_jog else (
                        lambda qq: ik.sigma_min(qq, rows="pos")
                    ),
                    collides=world.in_collision,
                )
                if not verdict.ok:
                    panel.set_jog_note(f"jog {axis} refused — {verdict.reason}")
                else:
                    panel.set_jog_note("")
                    jog_q = q_next
                    node.send_output("jog", pack_jog(q=jog_q, reason=axis))

        for preset_name, (preset_kp, preset_kd) in gain_presets.items():
            if not panel.clicked(f"{_GAIN_PREFIX}{preset_name}"):
                continue
            if preset_name in pose_hold_presets:
                if armed is not True or fault or health_stale_shown or not supports_pose_hold:
                    panel.log("REFUSED: Soft requires confirmed ARM and a Cartesian-capable plant")
                    continue
                if plan_thread is not None:
                    panel.log("REFUSED: wait for planning to finish before selecting Soft")
                    continue
                ident = uuid.uuid4().int & 0xFFFFFFFF or 1
                node.send_output("control", pack_control_update(
                    pose_hold=dict(pose_hold_presets[preset_name], id=ident)))
                panel.set_jog(None, None, False)
                pending = None
                pending_id = ""
                panel.clear_plan()
                panel.log("Soft requested — capture measured EE pose, compliant nullspace")
                continue
            # The controller cancels any running leg before applying these --
            # a plan reviewed at one stiffness must not finish at another.
            node.send_output(
                "control",
                pack_control_update(
                    gains={"kp": preset_kp.tolist(), "kd": preset_kd.tolist()},
                    reason=f"preset {preset_name}",
                ),
            )
            pending = None
            pending_id = ""
            panel.clear_plan()
            panel.log(
                f"gains -> {preset_name} (kp {preset_kp.min():.1f}-"
                f"{preset_kp.max():.1f}, kd {preset_kd.min():.2f}-{preset_kd.max():.2f})"
            )

        if panel.clicked("Stop (hold)"):
            # `cancel`, not `hold` and not `stop`: the arm must come to rest
            # where it is and STAY RUNNABLE. `hold` freezes it forever and
            # `stop` is terminal -- either would need a graph restart to undo.
            node.send_output(
                "control", pack_control_update(cancel=True, reason="operator stop")
            )
            pending = None
            pending_id = ""
            panel.clear_plan()
            panel.log("STOP sent — controller holds the measured pose")

        # The operator gate. This page presses it; the CONTROLLER owns the
        # `arm` topic downstream (it is the single producer of both `arm` and
        # `motor_command`, which is the whole point of the split), so the press
        # travels as a control update rather than as a second `arm` publisher.
        #
        # It is no longer gated on `armed is not None`. That test meant "is
        # there a motor_health wire", which was a proxy for "is there a bridge
        # to arm" -- but the controller must be armed in EVERY graph now,
        # including a sim one with no bridge and no health topic, or it streams
        # nothing and the arm never moves.
        if panel.clicked("ARM"):
            node.send_output("control", pack_control_update(arm=True))
            panel.log("ARM requested — controller enables and holds this pose")
        if panel.clicked("DISARM"):
            # Cancel FIRST: a disarm+rearm inside the runaway window must never
            # resume a stale trajectory. ArmController's own _disarm_reset also
            # clears it, but the ORDERING is the point, so it stays explicit.
            node.send_output(
                "control", pack_control_update(cancel=True, reason="disarm")
            )
            node.send_output("control", pack_control_update(arm=False))
            pending = None
            pending_id = ""
            panel.clear_plan()
            panel.log("DISARM requested — authority drop (bridge verifies)")


def _check_routes() -> None:
    """Every route the page calls must be handled by one of the two panels.

    Each page is checked against ITS OWN panel: console.js against
    ControlPanel, editor.js against GraspEditorPanel. Nothing but this ties the
    JS and the Python together, and the dispatch used to be
    ``self.path.endswith(name)`` -- loose enough that a renamed route could
    still land somewhere. It is exact now, so drift is a 404 at the operator's
    fingertip rather than a failure here. This is that failure, moved earlier.

    One page, one panel. The two pages were ONE file until 2026-09-07, and
    checking against the union of both route tables is precisely what let the
    arm console ship markup calling five grasp-editor routes that its own panel
    answers with 404. The editor's own check went with it to the project that
    owns it.
    """
    import inspect
    import re

    def handles(panel) -> set[str]:
        found: set[str] = set()
        for method in ("_get", "_post"):
            source = inspect.getsource(getattr(panel, method))
            found |= set(re.findall(r'route(?:\[len\()?[^\n]*?[=.]=?\s*"([^"]+)"', source))
            found |= set(re.findall(r'startswith\("([^"]+)"\)', source))
        return found

    from arm_control.console_assets import CONSOLE_DIR

    asset_dir = CONSOLE_DIR

    def calls(*scripts: str) -> set[str]:
        found: set[str] = set()
        for script in scripts:
            text = (asset_dir / script).read_text()
            for pattern in (r'fetch\("(/[^"]*)"', r'post\("(/[^"]*)"',
                            r'getJSON\("(/[^"]*)"'):
                found |= {m.lstrip("/") for m in re.findall(pattern, text)}
        return found

    handled = handles(ControlPanel)
    called = calls("console.js", "core.js")
    missing = sorted(r for r in called if r not in handled)
    assert not missing, (
        f"console.js calls routes ControlPanel does not handle: {missing}"
    )
    assert "mesh/" in handled, "the mesh prefix route vanished"
    print(f"arm_console: {len(called)} page routes all handled by ControlPanel")


def _check_graphs() -> None:
    """Every arm_controller in every graph must be able to stream.

    The controller waits for a bridge's armed health edge before its first
    command, because a bridge publishes motor_state while DISARMED and only a
    state after that edge is provably post-enable. A graph whose plant has no
    bridge publishes no motor_health at all, so the wait never ends and the arm
    simply never moves -- with no error anywhere, which is the worst shape a
    bug can have. Exactly one of the two must hold per graph: a motor_health
    wire, or the ARM_CONTROL_PLANT_HEALTH=0 opt-out.
    """
    import re

    import yaml

    root = REPO_ROOT
    controller = (root / "arm_control" / "control" / "arm_controller.py").read_text()
    handled = set(re.findall(r'"(\w+)": self\.on_', controller))

    checked = 0
    for graph in sorted((root / "dataflows").glob("*.yml")):
        spec = yaml.safe_load(graph.read_text()) or {}
        for console in spec.get("nodes", []):
            if str(console.get("path", "")).endswith("arm_console.py"):
                inputs = console.get("inputs") or {}
                assert ("motor_health" in inputs) != ("controller_arm" in inputs), (
                    f"{graph.name}/{console['id']}: console needs exactly one "
                    "authority feedback source, never an optimistic ARM badge"
                )
        for node in spec.get("nodes", []):
            if not str(node.get("path", "")).endswith("arm_controller.py"):
                continue
            checked += 1
            inputs = set(node.get("inputs") or {})
            unhandled = sorted(inputs - handled)
            assert not unhandled, (
                f"{graph.name}: arm_controller is wired {unhandled}, which it "
                f"has no handler for -- the input would be silently dropped"
            )
            has_health = "motor_health" in inputs
            opts_out = str(
                (node.get("env") or {}).get("ARM_CONTROL_PLANT_HEALTH", "")
            ) in ("0", "false", "no")
            assert has_health != opts_out, (
                f"{graph.name}: arm_controller has motor_health={has_health} and "
                f"ARM_CONTROL_PLANT_HEALTH opt-out={opts_out}. Exactly one is "
                f"needed, or it waits forever for an armed edge and never streams"
            )
    assert checked, "no arm_controller found in any graph -- did paths change?"
    print(f"arm_console: {checked} arm_controller wirings can stream")


def cli() -> None:
    import sys

    # dora runs a node as `python <node>.py`, so __main__ MUST be the node.
    if "--self-check" in sys.argv:
        _check_routes()
        _check_graphs()
    else:
        main()


if __name__ == "__main__":
    cli()
