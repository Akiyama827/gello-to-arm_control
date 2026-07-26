"""Float command owner: gravity feedforward + damping, holding the measured pose."""
from __future__ import annotations

# ruff: noqa: E402

import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from dora import Node

from arm_control import CONTROL_ROOT


from arm_control.config import load_robot_config
from arm_control.dynamics import PinocchioDynamics
from arm_control.messages import (
    pack_motor_command,
    unpack_controller_settings,
    unpack_motor_state,
)
from arm_control.node_utils import _load_mode_config, expand_named_values


def _settings_from_payload(payload: dict, names: list[str], defaults: dict[str, np.ndarray], limits: dict) -> dict:
    return {
        "kp": expand_named_values(
            payload.get("kp"),
            names=names,
            default=0.0,
            clamp_min=0.0,
            clamp_max=float(limits.get("kp_max", 500.0)),
        )
        if "kp" in payload
        else defaults["kp"],
        "kd": expand_named_values(
            payload.get("kd"),
            names=names,
            default=0.0,
            clamp_min=0.0,
            clamp_max=float(limits.get("kd_max", 5.0)),
        )
        if "kd" in payload
        else defaults["kd"],
        "gravity_scale": expand_named_values(
            payload.get("gravity_scale"),
            names=names,
            default=0.0,
            clamp_min=0.0,
            clamp_max=float(limits.get("gravity_scale_max", 1.5)),
        )
        if "gravity_scale" in payload
        else defaults["gravity_scale"],
        "torque_limits": expand_named_values(
            payload.get("torque_limits"),
            names=names,
            default=float(np.max(defaults["torque_limits"]) if len(defaults["torque_limits"]) else 0.0),
            clamp_min=0.0,
            clamp_max=float(limits.get("torque_limit_max", 100.0)),
        )
        if "torque_limits" in payload
        else defaults["torque_limits"],
    }


def apply_setting_update(
    current: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    dt: float,
    ramp_sec: float,
) -> dict[str, np.ndarray]:
    if ramp_sec <= 0:
        return {key: value.copy() for key, value in target.items()}
    alpha = min(1.0, max(0.0, dt / ramp_sec))
    return {key: current[key] + alpha * (target[key] - current[key]) for key in current}


def controller_command(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    gravity: np.ndarray,
    gravity_scale: np.ndarray,
    torque_limits: np.ndarray,
) -> dict[str, np.ndarray]:
    tau_ff = np.asarray(gravity, dtype=float) * np.asarray(gravity_scale, dtype=float)
    if torque_limits.size:
        tau_ff = np.clip(tau_ff, -torque_limits, torque_limits)
    return {
        "position": np.asarray(q_des, dtype=float),
        "velocity": np.asarray(qd_des, dtype=float),
        "torque": tau_ff,
        "kp": np.asarray(kp, dtype=float),
        "kd": np.asarray(kd, dtype=float),
    }


def _make_settings(controller_cfg: dict, names: list[str]) -> dict[str, np.ndarray]:
    return {
        "kp": expand_named_values(controller_cfg.get("kp"), names=names, default=0.0, clamp_min=0.0, clamp_max=500.0),
        "kd": expand_named_values(controller_cfg.get("kd"), names=names, default=0.0, clamp_min=0.0, clamp_max=5.0),
        "gravity_scale": expand_named_values(
            controller_cfg.get("gravity_scale"), names=names, default=0.0, clamp_min=0.0, clamp_max=1.5
        ),
        "torque_limits": expand_named_values(
            controller_cfg.get("torque_limits"), names=names, default=0.0, clamp_min=0.0, clamp_max=100.0
        ),
    }


class _MappedGravityModel:
    def __init__(self, dynamics: PinocchioDynamics, indices: list[int], n: int) -> None:
        self._dynamics = dynamics
        self._indices = np.asarray(indices, dtype=int)
        self._n = n

    def gravity(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros(self._n)
        out[self._indices] = self._dynamics.gravity(np.asarray(q, dtype=float)[self._indices])
        return out


def _urdf_joint_names(urdf_path: str) -> set[str]:
    path = Path(urdf_path)
    if not path.is_absolute():
        path = CONTROL_ROOT / path
    return {
        str(joint.get("name"))
        for joint in ET.parse(path).getroot().findall("joint")
        if joint.get("name")
    }


def make_gravity_model(cfg, names: list[str]):
    if not cfg.urdf_path:
        return None
    try:
        present = _urdf_joint_names(str(cfg.urdf_path))
        indexed_names = [(i, name) for i, name in enumerate(names) if name in present]
        if not indexed_names:
            return None
        indices, model_names = zip(*indexed_names)
        return _MappedGravityModel(PinocchioDynamics(cfg.urdf_path, model_names), list(indices), len(names))
    except Exception as exc:
        print(f"[impedance_controller] gravity disabled: {exc}", flush=True)
        return None


def main() -> None:
    cfg = load_robot_config()
    mode_cfg = _load_mode_config()
    controller_cfg = dict(mode_cfg.get("controller") or {})
    names = list(cfg.joint_names or cfg.motor_names)
    n = cfg.num_motors
    rate_hz = float(controller_cfg.get("command_rate_hz", cfg.update_rate_hz))
    period = 1.0 / rate_hz
    ramp_sec = float(controller_cfg.get("settings_ramp_sec", 0.5))
    state_timeout = float(controller_cfg.get("state_timeout_sec", cfg.state_timeout_sec))

    settings = _make_settings(controller_cfg, names)
    target_settings = {key: value.copy() for key, value in settings.items()}
    limits = dict(controller_cfg.get("limits") or {})
    dynamics = make_gravity_model(cfg, names)

    state: dict[str, np.ndarray] | None = None
    last_state_t = 0.0
    last_step = time.monotonic()
    node = Node()

    while True:
        event = node.next(timeout=period)
        now = time.monotonic()
        if event is not None:
            if event["type"] == "INPUT" and event["id"] == "motor_state":
                state = unpack_motor_state(event["value"], n)
                last_state_t = now
            elif event["type"] == "INPUT" and event["id"] == "controller_settings":
                target_settings = _settings_from_payload(
                    unpack_controller_settings(event["value"]),
                    names,
                    target_settings,
                    limits,
                )
                print("[impedance_controller] accepted controller_settings", flush=True)
            elif event["type"] == "STOP":
                break

        dt = now - last_step
        if dt < period:
            continue
        last_step = now
        settings = apply_setting_update(settings, target_settings, dt=dt, ramp_sec=ramp_sec)

        if state is None or now - last_state_t > state_timeout:
            continue

        q = state["position"]
        qd = state["velocity"]
        # Float hold: command the current measured pose with zero desired velocity.
        # gravity ff carries the weight; kd damps motion; kp (0 for a pure float)
        # only ever acts on the one-cycle q_des-vs-q gap, so it damps rather than
        # dragging to a stale target. Replaces the old behavior.py deadband+follow,
        # whose lagging q_des fought hand-guiding.
        q_des = q.copy()
        qd_des = np.zeros(n)

        gravity = dynamics.gravity(q) if dynamics is not None else np.zeros(n)
        command = controller_command(
            q=q,
            qd=qd,
            q_des=q_des,
            qd_des=qd_des,
            kp=settings["kp"],
            kd=settings["kd"],
            gravity=gravity,
            gravity_scale=settings["gravity_scale"],
            torque_limits=settings["torque_limits"],
        )
        node.send_output(
            "motor_command",
            pack_motor_command(
                command["position"],
                command["velocity"],
                command["torque"],
                command["kp"],
                command["kd"],
            ),
        )


if __name__ == "__main__":
    main()
