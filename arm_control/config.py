"""Runtime configuration loading and validation for the Control stack."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from arm_control import CONTROL_ROOT  # deployment root (env-overridable seam)


def _as_list(raw: Mapping[str, Any], key: str) -> list:
    value = raw.get(key) or []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return list(value)


def _resolve_control_path(raw_path: Any) -> str:
    if not raw_path:
        return ""
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = CONTROL_ROOT / path
    return str(path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; lists and scalars replace rather than concatenate.

    Replace-not-extend matters: ``motor_ids`` or ``crop_min`` merged elementwise
    would silently produce a config nobody wrote.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config_tree(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict:
    """Read a config YAML, resolving an optional ``include:`` list.

    Split by LIFETIME, not by subsystem — that is the whole point of the
    mechanism. A real arm's identity changes on reassembly, its extrinsics
    change per calibration run, and its scenario changes every bench session;
    keeping those in one 384-line file meant every session's edits collided
    with calibration output. So a config is now a three-line stub::

        include: [assembler/hardware.yaml, assembler/calibration.yaml,
                  assembler/scenario.yaml]

    Includes resolve relative to the including file's directory (then the
    Control root), merge in order, and the including file's own keys win over
    everything it includes — so a stub can override one value without copying
    the block it lives in. Single-file configs keep working untouched.
    """
    config_path = Path(path).resolve()
    if config_path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, config_path))
        raise ValueError(f"circular config include: {chain}")
    with config_path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config YAML must contain a mapping: {config_path}")
    includes = data.pop("include", None) or []
    if isinstance(includes, (str, Path)):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError(f"include must be a list of paths: {config_path}")
    merged: dict[str, Any] = {}
    for entry in includes:
        child = Path(str(entry))
        if not child.is_absolute():
            local = config_path.parent / child
            child = local if local.exists() else CONTROL_ROOT / child
        if not child.exists():
            raise FileNotFoundError(
                f"{config_path}: include {entry!r} not found (looked in "
                f"{config_path.parent} and {CONTROL_ROOT})"
            )
        merged = _deep_merge(merged, load_config_tree(child, _seen | {config_path}))
    return _deep_merge(merged, data)


@dataclass(frozen=True)
class RobotConfig:
    serial_port: str = "/dev/ttyACM0"
    baud_rate: int = 2000000
    num_motors: int = 7
    update_rate_hz: float = 100.0
    motor_ids: list[int] = field(default_factory=list)
    motor_names: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    listen_mode: bool = False
    rx_timeout_ms: float = 200.0
    debug_tx: bool = False
    debug_rx: bool = False
    debug_tx_throttle_hz: float = 0.0
    debug_rx_throttle_hz: float = 0.0
    urdf_path: str = ""
    state_timeout_sec: float = 0.5
    torque_scale: float = 1.0
    torque_limits: list[float] = field(default_factory=list)
    gui_publish_rate_hz: float = 20.0
    gc_alpha: float = 1.0
    torque_ramp_sec: float = 0.5
    rerun_app_id: str = "arm_control"
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotConfig":
        return cls.from_mapping(load_config_tree(path))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RobotConfig":
        data = dict(raw)
        config = cls(
            serial_port=str(data.get("serial_port", "/dev/ttyACM0")),
            baud_rate=int(data.get("baud_rate", 2000000)),
            num_motors=int(data.get("num_motors", 7)),
            update_rate_hz=float(data.get("update_rate_hz", 100.0)),
            motor_ids=[int(v) for v in _as_list(data, "motor_ids")],
            motor_names=[str(v) for v in _as_list(data, "motor_names")],
            joint_names=[str(v) for v in _as_list(data, "joint_names")],
            listen_mode=bool(data.get("listen_mode", False)),
            rx_timeout_ms=float(data.get("rx_timeout_ms", 200.0)),
            debug_tx=bool(data.get("debug_tx", False)),
            debug_rx=bool(data.get("debug_rx", False)),
            debug_tx_throttle_hz=float(data.get("debug_tx_throttle_hz", 0.0)),
            debug_rx_throttle_hz=float(data.get("debug_rx_throttle_hz", 0.0)),
            urdf_path=_resolve_control_path(data.get("urdf_path")),
            state_timeout_sec=float(data.get("state_timeout_sec", 0.5)),
            torque_scale=float(data.get("torque_scale", 1.0)),
            torque_limits=[float(v) for v in _as_list(data, "torque_limits")],
            gui_publish_rate_hz=float(data.get("gui_publish_rate_hz", 20.0)),
            gc_alpha=float(data.get("gc_alpha", 1.0)),
            torque_ramp_sec=float(data.get("torque_ramp_sec", 0.5)),
            rerun_app_id=str(data.get("rerun_app_id", "arm_control")),
            raw=data,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.num_motors <= 0:
            raise ValueError("num_motors must be > 0")
        if self.baud_rate <= 0:
            raise ValueError("baud_rate must be > 0")
        if self.update_rate_hz <= 0:
            raise ValueError("update_rate_hz must be > 0")
        if self.gui_publish_rate_hz <= 0:
            raise ValueError("gui_publish_rate_hz must be > 0")
        if self.rx_timeout_ms < 0:
            raise ValueError("rx_timeout_ms must be >= 0")
        if self.state_timeout_sec <= 0:
            raise ValueError("state_timeout_sec must be > 0")
        if self.torque_ramp_sec < 0:
            raise ValueError("torque_ramp_sec must be >= 0")
        if self.torque_scale < 0:
            raise ValueError("torque_scale must be >= 0")
        if self.gc_alpha < 0:
            raise ValueError("gc_alpha must be >= 0")

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def as_runtime_dict(self) -> dict[str, Any]:
        data = dict(self.raw)
        values = asdict(self)
        values.pop("raw", None)
        data.update(values)
        return data


def load_robot_config(path: str | Path | None = None) -> RobotConfig:
    env_path = os.environ.get("ARM_CONTROL_CONFIG")
    if path is not None:
        config_path = Path(path)
    elif env_path:
        config_path = Path(env_path)
        if not config_path.is_absolute():
            config_path = CONTROL_ROOT / config_path
    else:
        # No silent robot default: loading some particular arm's bench config
        # because an env var was forgotten is how the wrong robot gets driven.
        raise RuntimeError(
            "no robot config: pass a path or set ARM_CONTROL_CONFIG "
            "(a launcher normally exports it for every node it starts)"
        )
    return RobotConfig.from_yaml(config_path)


def load_robot_config_dict(path: str | Path | None = None) -> dict[str, Any]:
    return load_robot_config(path).as_runtime_dict()


# --------------------------------------------------------------------------- #
# Per-arm identity accessors
#
# These live here, next to the loader, rather than in ``planning.stack``: the
# perception node and the FR3 bridge need them but must NOT drag in the planning
# stack (Pinocchio + OMPL + the collision world) it imports at module level.
#
# The ``arm:`` block is the ONLY source. There are deliberately no robot-shaped
# code defaults: a config that forgets its identity must fail here, by name,
# not deep inside Pinocchio wearing another robot's joint list.
# --------------------------------------------------------------------------- #


def _arm_block(cfg: Any) -> dict:
    get = cfg.get if hasattr(cfg, "get") else dict(cfg).get
    return dict(get("arm") or {})


def arm_joints(cfg: Any) -> list[str]:
    joints = _arm_block(cfg).get("joints")
    if not joints:
        raise ValueError("config missing arm.joints (this arm's joint names, in order)")
    return [str(j) for j in joints]


def gripper_joints(cfg: Any) -> list[str]:
    """Finger joints this arm's bridge servos — an explicit EMPTY list means the
    gripper is its own device (the FR3's Franka Hand), so ghosts and plan worlds
    stay arm-only. The key itself is required: absence is indistinguishable from
    "forgot", and the two mean different robots."""
    block = _arm_block(cfg)
    if "gripper_joints" not in block:
        raise ValueError(
            "config missing arm.gripper_joints (list the finger joints, or [] "
            "for an arm whose gripper is its own device)"
        )
    return [str(j) for j in (block["gripper_joints"] or [])]


def ee_frame(cfg: Any) -> str:
    frame = _arm_block(cfg).get("ee_frame")
    if not frame:
        raise ValueError("config missing arm.ee_frame (the frame waypoints target)")
    return str(frame)


def _demo() -> None:
    """Self-check for the include/merge rules, on a throwaway config tree."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "hardware.yaml").write_text(
            "num_motors: 7\nmotor_ids: [1, 2, 3]\nsafety: {deadman_timeout_s: 0.1, temp_limit_c: 85}\n"
        )
        (root / "calibration.yaml").write_text(
            "perception: {static: {world_T_cam: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}}\n"
        )
        (root / "scenario.yaml").write_text(
            "safety: {temp_limit_c: 70}\nperception: {static: {rate_hz: 2.0}}\n"
        )
        (root / "arm.yaml").write_text(
            "include: [hardware.yaml, calibration.yaml, scenario.yaml]\n"
            "num_motors: 8\n"
        )
        merged = load_config_tree(root / "arm.yaml")

        # Nested dicts merge across files instead of the last one winning whole.
        assert merged["safety"]["deadman_timeout_s"] == 0.1, merged["safety"]
        assert merged["safety"]["temp_limit_c"] == 70, merged["safety"]  # later include wins
        assert "world_T_cam" in merged["perception"]["static"], merged["perception"]
        assert merged["perception"]["static"]["rate_hz"] == 2.0, merged["perception"]
        # The including file's own keys beat every include.
        assert merged["num_motors"] == 8, merged["num_motors"]
        # Lists replace; merging them elementwise would invent a config.
        assert merged["motor_ids"] == [1, 2, 3], merged["motor_ids"]
        # `include` is consumed, never left in the runtime config.
        assert "include" not in merged

        # Single-file configs still load untouched.
        (root / "flat.yaml").write_text("num_motors: 6\n")
        assert load_config_tree(root / "flat.yaml") == {"num_motors": 6}

        # A cycle is reported, not a RecursionError.
        (root / "a.yaml").write_text("include: [b.yaml]\n")
        (root / "b.yaml").write_text("include: [a.yaml]\n")
        try:
            load_config_tree(root / "a.yaml")
        except ValueError as exc:
            assert "circular" in str(exc), exc
        else:
            raise AssertionError("circular include was not detected")

        # A missing include names both search paths instead of a bare KeyError.
        (root / "bad.yaml").write_text("include: [nope.yaml]\n")
        try:
            load_config_tree(root / "bad.yaml")
        except FileNotFoundError as exc:
            assert "nope.yaml" in str(exc), exc
        else:
            raise AssertionError("missing include was not detected")
    print("config: ok")


if __name__ == "__main__":
    _demo()
