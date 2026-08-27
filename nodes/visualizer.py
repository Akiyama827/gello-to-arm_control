"""visualizer.py — Dora node: Rerun visualisation replacing RViz2.

Inputs:
    motor_state     packed motor_state float64 array
    can_bus_status  JSON bus telemetry

Initial Rerun layout (blueprint):
    ┌─────────────────────────────────────┐
    │  Positions (measured + commanded)   │  ← visible first, tallest
    ├──────────────────┬──────────────────┤
    │   Velocities     │     Torques      │
    └──────────────────┴──────────────────┘
    (+ Spatial3D panel on the right if URDF + Pinocchio URDF visuals are configured)

To retune the layout: edit _setup_blueprint() below.
To retune colours / line styles: edit _MOTOR_COLORS and _setup_series_style().
"""
from __future__ import annotations

# ruff: noqa: E402

import signal
import time
import os
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dora import Node


from arm_control.config import load_robot_config_dict
from arm_control.joint_motor_map import gripper_motor_to_finger
# Same scene draw as the teleop preview — one implementation, so the two Rerun
# recordings can never disagree about where the dock stands.
from arm_control.planning.preview_rerun import log_static_scene
from arm_control.messages import (
    unpack_motor_state,
    unpack_json_message,
)

try:
    import pinocchio as pin
    _PIN_OK = True
except ImportError:
    _PIN_OK = False

# ---------------------------------------------------------------------------
# Tuning knobs — edit these to change appearance
# ---------------------------------------------------------------------------

# One RGBA colour per motor slot.  Extend the list if you have more than 8 motors.
_MOTOR_COLORS: list[tuple[int, int, int, int]] = [
    (228,  26,  28, 255),  # red
    ( 55, 126, 184, 255),  # blue
    ( 77, 175,  74, 255),  # green
    (152,  78, 163, 255),  # purple
    (255, 127,   0, 255),  # orange
    (166,  86,  40, 255),  # brown
    (247, 129, 191, 255),  # pink
    (153, 153, 153, 255),  # grey
]

# Commanded-position lines use the same hue but lighter (lower alpha).
_CMD_ALPHA = 140

# Line width in pixels
_LINE_WIDTH = 1.5


def _cmd_color(c: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (c[0], c[1], c[2], _CMD_ALPHA)


def _label_with_unit(name: str, unit: str) -> str:
    return f"{name} [{unit}]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    return load_robot_config_dict()


def _resolve_joint_names(cfg: dict, urdf_path: str) -> list[str]:
    render_names = list(cfg.get("render_joint_names") or [])
    if render_names:
        return render_names
    from_cfg = list(cfg.get("joint_names") or [])
    if from_cfg:
        return from_cfg
    if not urdf_path or not Path(urdf_path).is_file():
        return []
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(Path(urdf_path).read_text())
        return [
            j.get("name", "")
            for j in root.findall("joint")
            if j.get("type", "").lower() == "revolute" and j.get("name")
        ]
    except Exception:
        return []


def _mesh_package_dirs(urdf_path: str) -> list[str]:
    urdf_dir = Path(urdf_path).resolve().parent
    package_root = urdf_dir.parent if (urdf_dir.parent / "package.xml").is_file() else urdf_dir
    # Hint Pinocchio's URDF parser at the parent ROS package layout used
    # for our generated combined URDFs: <root>/models/urdf with sibling
    # ``meshes`` and ``meshes_collision`` directories one level up.
    candidates = [urdf_dir, urdf_dir.parent, package_root, package_root.parent]
    return [str(path) for path in candidates if path.is_dir()]


def _build_render_model(urdf_path: str, joint_names: list[str],
                        extra_joints: list[str] | None = None):
    """Returns (model, data, q_indices, visual_model, visual_data, render_names)
    or None.

    ``extra_joints``: non-motor joints (e.g. Franka Hand fingers — driven by
    the gripper_state topic, not a motor slot) articulated on top of the
    motor-driven ones, kept only if the URDF really has them. Without this the
    fingers were filtered out of the articulated set entirely and every live
    width update landed in a no-op (bench: Rerun gripper frozen).
    """
    if not _PIN_OK or not urdf_path or not Path(urdf_path).is_file() or not joint_names:
        return None
    try:
        package_dirs = _mesh_package_dirs(urdf_path)
        model, visual_model = pin.buildModelsFromUrdf(
            urdf_path,
            package_dirs=package_dirs or None,
            geometry_types=[pin.GeometryType.VISUAL],
        )
        data = model.createData()
        visual_data = visual_model.createData()
        render_names = list(joint_names) + [
            j for j in (extra_joints or [])
            if j not in joint_names and model.existJointName(j)
        ]
        q_idx = []
        for jname in render_names:
            jid = model.getJointId(jname)
            if jid >= model.njoints:
                raise ValueError(f"Joint {jname!r} not in URDF")
            q_idx.append(model.joints[jid].idx_q)
        print(f"[viz] URDF render ready: {len(render_names)} joints, {visual_model.ngeoms} meshes")
        return model, data, q_idx, visual_model, visual_data, render_names
    except Exception as exc:
        print(f"[viz] URDF render init failed: {exc}")
        return None


def _rgba8(color: np.ndarray | tuple[float, float, float, float]) -> list[int]:
    rgba = np.asarray(color, dtype=float)
    rgba = np.clip(rgba, 0.0, 1.0)
    return [int(round(channel * 255.0)) for channel in rgba]


def _log_visual_assets(visual_model) -> None:
    for geom in visual_model.geometryObjects:
        mesh_path = Path(geom.meshPath)
        if not mesh_path.is_file():
            print(f"[viz] skipping missing mesh: {mesh_path}")
            continue
        entity_path = f"robot/visuals/{geom.name}"
        rr.log(
            entity_path,
            rr.Asset3D(path=mesh_path, albedo_factor=_rgba8(geom.meshColor)),
            static=True,
        )


def _apply_joint_mimics(
    q: np.ndarray,
    joint_names: list[str],
    motor_names: list[str],
    q_profile: np.ndarray,
    joint_mimics: dict,
) -> None:
    joint_index = {name: i for i, name in enumerate(joint_names)}
    motor_index = {name: i for i, name in enumerate(motor_names)}
    for joint_name, spec in joint_mimics.items():
        source = str(spec.get("source", ""))
        if joint_name not in joint_index or source not in motor_index:
            continue
        raw = float(q_profile[motor_index[source]])
        if "motor_open" in spec and "motor_closed" in spec:
            value = gripper_motor_to_finger(raw, spec)
        else:
            value = raw * float(spec.get("scale", 1.0))
            value += float(spec.get("offset", 0.0))
        if "lower" in spec or "upper" in spec:
            value = float(np.clip(value, spec.get("lower", -np.inf), spec.get("upper", np.inf)))
        q[joint_index[joint_name]] = value


def _log_render_pose(
    render_model,
    joint_names: list[str],
    motor_names: list[str],
    q_profile: np.ndarray,
    joint_mimics: dict | None = None,
    finger_values: dict[str, float] | None = None,
) -> None:
    model, data, q_idx, visual_model, visual_data = render_model[:5]
    q = np.zeros(model.nq)
    # Try name-based match; fall back to positional (joint i ← motor slot i).
    jname_to_slot = {name: i for i, name in enumerate(motor_names)}
    name_matched = False
    for ji, jname in enumerate(joint_names):
        slot = jname_to_slot.get(jname)
        if slot is not None:
            q[q_idx[ji]] = q_profile[slot]
            name_matched = True
    if not name_matched:
        for ji in range(min(len(joint_names), len(q_profile))):
            q[q_idx[ji]] = q_profile[ji]
    if joint_mimics:
        q_sub = np.array([q[index] for index in q_idx])
        _apply_joint_mimics(q_sub, joint_names, motor_names, q_profile, joint_mimics)
        for ji, value in enumerate(q_sub):
            q[q_idx[ji]] = value
    if finger_values:
        # Fingers that are NOT bus motors (Franka Hand): live width arrives on
        # the gripper_state topic instead of a motor slot / mimic.
        jindex = {name: i for i, name in enumerate(joint_names)}
        for jname, value in finger_values.items():
            ji = jindex.get(jname)
            if ji is not None:
                q[q_idx[ji]] = value
    try:
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, visual_model, visual_data, q)
        for geom, placement in zip(visual_model.geometryObjects, visual_data.oMg):
            rr.log(
                f"robot/visuals/{geom.name}",
                rr.Transform3D(
                    translation=placement.translation,
                    mat3x3=placement.rotation,
                    scale=geom.meshScale,
                ),
            )
    except Exception as exc:
        print(f"[viz] render error: {exc}")


# ---------------------------------------------------------------------------
# Blueprint — initial Rerun layout
# ---------------------------------------------------------------------------

def _setup_blueprint(motor_names: list[str], has_3d: bool, time_ranges=None) -> None:
    """Send the initial Rerun panel layout.

    Edit this function to reorganise panels.  The blueprint is sent once
    at startup; the user can still rearrange panels manually in the viewer.
    """
    pos_reading = ["motors/position"]
    pos_cmd     = ["motors/pos_cmd"]
    vel_reading = ["motors/velocity"]
    vel_cmd     = ["motors/vel_cmd"]
    tor_reading = ["motors/torque"]
    tor_cmd     = ["motors/tor_cmd"]

    pos_view = rrb.TimeSeriesView(
        name="Position [rad]", origin="/", contents=pos_reading, time_ranges=time_ranges
    )
    pos_c_view = rrb.TimeSeriesView(
        name="Position Cmd [rad]", origin="/", contents=pos_cmd, time_ranges=time_ranges
    )
    vel_view = rrb.TimeSeriesView(
        name="Velocity [rad/s]", origin="/", contents=vel_reading, time_ranges=time_ranges
    )
    vel_c_view = rrb.TimeSeriesView(
        name="Velocity Cmd [rad/s]", origin="/", contents=vel_cmd, time_ranges=time_ranges
    )
    tor_view = rrb.TimeSeriesView(
        name="Torque [N·m]", origin="/", contents=tor_reading, time_ranges=time_ranges
    )
    tor_c_view = rrb.TimeSeriesView(
        name="Torque Cmd [N·m]", origin="/", contents=tor_cmd, time_ranges=time_ranges
    )
    bus_view = rrb.TimeSeriesView(
        name="CAN Bus Load [%]",
        origin="/",
        contents=["can/bus_load_pct"],
        time_ranges=time_ranges,
    )

    err_view = rrb.TimeSeriesView(
        name="Tracking Error [rad]",
        origin="/",
        contents=["motors/pos_err"],
        time_ranges=time_ranges,
    )

    row1 = rrb.Horizontal(pos_view, pos_c_view)
    row2 = rrb.Horizontal(vel_view, vel_c_view)
    row3 = rrb.Horizontal(tor_view, tor_c_view)
    row4 = rrb.Horizontal(err_view, bus_view)
    right_col = rrb.Vertical(row1, row2, row3, row4)

    if has_3d:
        # origin="/" with explicit contents, NOT origin="/robot": the static
        # scene bodies (the dock/modular base) are logged under /scene, which a
        # view rooted at /robot cannot reach. They streamed in fine and were
        # simply outside the viewport — invisible for the same reason an
        # unplugged monitor is black (bench 2026-08-06).
        spatial_view = rrb.Spatial3DView(
            name="3D (FK)", origin="/", contents=["/robot/**", "/scene/**"]
        )
        root = rrb.Horizontal(right_col, spatial_view, column_shares=[2, 1])
    else:
        root = right_col

    rr.send_blueprint(
        rrb.Blueprint(
            root,
            collapse_panels=False,
        )
    )


# ---------------------------------------------------------------------------
# Series style — per-motor colours logged once as static metadata
# ---------------------------------------------------------------------------

def _setup_series_style(motor_names: list[str]) -> None:
    """Log SeriesLines metadata so each motor gets a distinct colour and label."""
    colors = [_MOTOR_COLORS[i % len(_MOTOR_COLORS)] for i in range(len(motor_names))]
    cmd_colors = [_cmd_color(color) for color in colors]
    rr.log(
        "motors/position",
        rr.SeriesLines(
            colors=colors,
            names=[_label_with_unit(name, "rad") for name in motor_names],
            widths=[_LINE_WIDTH] * len(motor_names),
            interpolation_mode=rr.components.InterpolationMode.Linear,
        ),
        static=True,
    )
    rr.log(
        "motors/pos_cmd",
        rr.SeriesLines(
            colors=cmd_colors,
            names=[_label_with_unit(f"{name} (cmd)", "rad") for name in motor_names],
            widths=[_LINE_WIDTH] * len(motor_names),
            interpolation_mode=rr.components.InterpolationMode.Linear,
        ),
        static=True,
    )
    rr.log(
        "motors/pos_err",
        rr.SeriesLines(
            colors=colors,
            names=[_label_with_unit(f"{name} (err)", "rad") for name in motor_names],
            widths=[_LINE_WIDTH] * len(motor_names),
            interpolation_mode=rr.components.InterpolationMode.Linear,
        ),
        static=True,
    )
    rr.log(
        "motors/velocity",
        rr.SeriesLines(
            colors=colors,
            names=[_label_with_unit(name, "rad/s") for name in motor_names],
            widths=[_LINE_WIDTH] * len(motor_names),
            interpolation_mode=rr.components.InterpolationMode.Linear,
        ),
        static=True,
    )
    rr.log(
        "motors/vel_cmd",
        rr.SeriesLines(
            colors=cmd_colors,
            names=[_label_with_unit(f"{name} (cmd)", "rad/s") for name in motor_names],
            widths=[_LINE_WIDTH] * len(motor_names),
            interpolation_mode=rr.components.InterpolationMode.Linear,
        ),
        static=True,
    )
    rr.log(
        "motors/torque",
        rr.SeriesLines(
            colors=colors,
            names=[_label_with_unit(name, "N*m") for name in motor_names],
            widths=[_LINE_WIDTH] * len(motor_names),
            interpolation_mode=rr.components.InterpolationMode.Linear,
        ),
        static=True,
    )
    rr.log(
        "motors/tor_cmd",
        rr.SeriesLines(
            colors=cmd_colors,
            names=[_label_with_unit(f"{name} (cmd)", "N*m") for name in motor_names],
            widths=[_LINE_WIDTH] * len(motor_names),
            interpolation_mode=rr.components.InterpolationMode.Linear,
        ),
        static=True,
    )


def _fk_publish_period(cfg: dict) -> float:
    rate_hz = float(cfg.get("rerun_fk_rate_hz", cfg.get("viz_publish_rate_hz", 60.0)))
    if rate_hz <= 0:
        raise ValueError("rerun_fk_rate_hz must be > 0")
    return 1.0 / rate_hz


def _fk_min_delta_rad(cfg: dict) -> float:
    value = float(cfg.get("rerun_fk_min_delta_rad", 0.0))
    if value < 0:
        raise ValueError("rerun_fk_min_delta_rad must be >= 0")
    return value


def _plot_time_ranges(cfg: dict):
    window_s = float(cfg.get("rerun_plot_window_s", 10.0))
    if window_s <= 0:
        raise ValueError("rerun_plot_window_s must be > 0")
    return [
        rrb.VisibleTimeRange(
            "elapsed_s",
            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-window_s),
            end=rrb.TimeRangeBoundary.cursor_relative(),
        )
    ]


def _fk_position_changed(previous, current, *, min_delta_rad: float) -> bool:
    if previous is None or min_delta_rad <= 0:
        return True
    return bool(np.max(np.abs(np.asarray(current) - np.asarray(previous))) >= min_delta_rad)


def _log_motor_state_metrics(state: dict[str, np.ndarray], *, rerun_module=rr) -> None:
    rerun_module.log("motors/position", rerun_module.Scalars(state["position"]))
    rerun_module.log("motors/velocity", rerun_module.Scalars(state["velocity"]))
    rerun_module.log("motors/pos_cmd", rerun_module.Scalars(state["position_cmd"]))
    rerun_module.log(
        "motors/pos_err",
        rerun_module.Scalars(state["position_cmd"] - state["position"]),
    )
    rerun_module.log("motors/vel_cmd", rerun_module.Scalars(state["velocity_cmd"]))
    rerun_module.log("motors/torque", rerun_module.Scalars(state["torque"]))
    rerun_module.log("motors/tor_cmd", rerun_module.Scalars(state["torque_cmd"]))


def _init_rerun(app_id: str, *, rerun_module=rr) -> None:
    # CONNECT, never spawn: a viewer child per graph run dies with the graph,
    # piles dead windows on the desk, and splits streams across viewers. The
    # launcher (scripts/view.py) guarantees one persistent viewer beforehand.
    rerun_module.init(app_id, spawn=False)
    rerun_module.connect_grpc()


def _shutdown_from_signal(signum=None, frame=None, *, exit_func=os._exit) -> None:
    exit_func(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg          = _load_cfg()
    n_motors     = int(cfg.get("num_motors", 7))
    motor_names  = list(cfg.get("motor_names") or [f"joint_{i}" for i in range(n_motors)])
    urdf_path    = str(cfg.get("urdf_path") or "")
    app_id       = str(cfg.get("rerun_app_id", "arm_control"))
    fk_period_s  = _fk_publish_period(cfg)
    fk_min_delta = _fk_min_delta_rad(cfg)
    time_ranges  = _plot_time_ranges(cfg)
    log_rates = bool(cfg.get("viz_log_rates", False))
    joint_mimics = dict(cfg.get("joint_mimics") or {})

    joint_names   = _resolve_joint_names(cfg, urdf_path)

    from arm_control.config import gripper_joints as _gj
    render_model = _build_render_model(urdf_path, joint_names, extra_joints=_gj(cfg))
    finger_joints: list[str] = []
    if render_model is not None:
        joint_names = render_model[5]  # motors + URDF-present finger joints
        finger_joints = [j for j in _gj(cfg) if j in joint_names]

    _init_rerun(app_id)

    _setup_series_style(motor_names)
    if render_model is not None:
        _log_visual_assets(render_model[3])
    log_static_scene(cfg)
    _setup_blueprint(motor_names, has_3d=(render_model is not None), time_ranges=time_ranges)

    print(f"[viz] Rerun '{app_id}' started — {n_motors} motors")

    q_profile = np.zeros(n_motors)
    finger_values: dict[str, float] = {}
    node = Node()
    t0   = time.monotonic()
    last_fk_log = 0.0
    last_rate_log = 0.0
    motor_events = 0
    metric_logs = 0
    fk_logs = 0
    changed_samples = 0
    _last_fk_q: np.ndarray | None = None
    _last_stats_q: np.ndarray | None = None

    signal.signal(signal.SIGTERM, _shutdown_from_signal)
    signal.signal(signal.SIGINT, _shutdown_from_signal)

    for event in node:
        if event["type"] not in ("INPUT",):
            if event["type"] == "STOP":
                break
            continue

        now = time.monotonic()
        rr.set_time("elapsed_s", duration=now - t0)
        eid = event["id"]

        if eid == "motor_state":
            motor_events += 1
            state = unpack_motor_state(event["value"], n_motors)
            raw_position = state["position"]
            q_profile[:] = raw_position
            _log_motor_state_metrics(state)
            metric_logs += 1
            if _last_stats_q is None or not np.allclose(raw_position, _last_stats_q, atol=1e-8):
                changed_samples += 1
                _last_stats_q = raw_position.copy()
            if render_model is not None and now - last_fk_log >= fk_period_s:
                if _fk_position_changed(_last_fk_q, q_profile, min_delta_rad=fk_min_delta):
                    _log_render_pose(
                        render_model, joint_names, motor_names, q_profile,
                        joint_mimics, finger_values,
                    )
                    _last_fk_q = q_profile.copy()
                    fk_logs += 1
                last_fk_log = now

        elif eid == "gripper_state":
            payload = unpack_json_message(event["value"], expected_schema="gripper_state")
            half = float(payload.get("width", 0.0)) / 2.0
            finger_values = {j: half for j in finger_joints}
            _last_fk_q = None  # force a re-render even with a motionless arm

        elif eid == "can_bus_status":
            payload = unpack_json_message(event["value"], expected_schema="can_bus_status")
            rr.log("can/bus_load_pct", rr.Scalars(float(payload["bus_load_pct"])))

        if log_rates and now - last_rate_log >= 1.0:
            if last_rate_log > 0:
                dt = now - last_rate_log
                print(
                    f"[viz] rates: input={motor_events / dt:.1f}Hz "
                    f"metric={metric_logs / dt:.1f}Hz fk={fk_logs / dt:.1f}Hz "
                    f"changed={changed_samples / dt:.1f}Hz",
                    flush=True,
                )
            last_rate_log = now
            motor_events = 0
            metric_logs = 0
            fk_logs = 0
            changed_samples = 0

    rr.disconnect()


if __name__ == "__main__":
    main()
