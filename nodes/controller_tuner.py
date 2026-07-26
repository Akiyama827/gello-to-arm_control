"""Publishes live controller settings from a watched YAML file."""
from __future__ import annotations

# ruff: noqa: E402

import time
from pathlib import Path

import yaml
from dora import Node

from arm_control import CONTROL_ROOT


from arm_control.messages import pack_controller_settings
from arm_control.node_utils import _load_mode_config


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else CONTROL_ROOT / p


def _load_settings(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    if "controller" in data:
        data = data["controller"] or {}
    return {key: data[key] for key in ("kp", "kd", "gravity_scale", "torque_limits") if key in data}


def main() -> None:
    mode_cfg = _load_mode_config()
    tuner_cfg = dict(mode_cfg.get("tuner") or {})
    settings_path = _resolve(tuner_cfg.get("settings_path", "configs/modes/float_tuning.yaml"))
    poll_period = 1.0 / float(tuner_cfg.get("poll_rate_hz", 2.0))
    node = Node()
    last_mtime: float | None = None

    while True:
        event = node.next(timeout=poll_period)
        if event is not None and event["type"] == "STOP":
            break
        try:
            mtime = settings_path.stat().st_mtime
        except FileNotFoundError:
            continue
        if last_mtime == mtime:
            continue
        last_mtime = mtime
        payload = _load_settings(settings_path)
        if payload:
            node.send_output("controller_settings", pack_controller_settings(payload))
            print(f"[controller_tuner] published {settings_path}", flush=True)
        time.sleep(poll_period)


if __name__ == "__main__":
    main()
