"""Shared node boilerplate for the Dora Control nodes.

Signal-driven shutdown flag, MIT zero-command helpers, and the mode-config
loader/expander that several nodes previously duplicated or imported across the
node boundary. Moving them here keeps one copy and lets sibling nodes import
from ``arm_control`` instead of from each other.
"""
from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import time

import numpy as np
import yaml

from arm_control import CONTROL_ROOT, REPO_ROOT


@dataclass
class ShutdownFlag:
    stop_requested: bool = False

    def request_stop(self) -> None:
        self.stop_requested = True


def install_signal_handlers(flag: ShutdownFlag):
    def _handler(signum, frame) -> None:
        flag.request_stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return _handler


def next_event_gil_friendly(node, *, idle_sleep: float = 0.01):
    """``node.next()`` for a node that also runs a server thread.

    dora's ``next(timeout=T)`` holds the GIL for essentially the whole of T.
    Measured in ``nodes/operator_console.py``: a plain Python thread ticking
    every 10 ms got **2 ticks in 5 seconds** against an ideal of ~500 -- the
    GIL was held 99.6% of the time. Any HTTP server in the same process is
    starved by that: the operator panel's ``/state`` took 3.4 s to answer,
    requests piled up 44 threads deep, and the page stopped responding
    altogether. These panels are the ARM and STOP controls; they do not get to
    be starved by an event loop.

    ``time.sleep()`` DOES release the GIL, so polling with a near-zero dora
    timeout and sleeping explicitly gives the same service rate with the GIL
    free for most of each cycle. Measured after: 3380 ms -> 1.0 ms.

    Only worth using in a node that shares its process with a server thread; a
    pure compute node should keep blocking, which is cheaper.
    """
    event = node.next(timeout=0.001)
    if event is None:
        time.sleep(idle_sleep)
    return event


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def _zero_command(n: int) -> dict[str, np.ndarray]:
    return {
        "position": _zeros(n),
        "velocity": _zeros(n),
        "torque": _zeros(n),
        "kp": _zeros(n),
        "kd": _zeros(n),
    }


def _load_mode_config() -> dict:
    raw = os.environ.get("ARM_CONTROL_MODE_CONFIG")
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_absolute():
        # Deployment root first (a project may override a mode wholesale),
        # then this repo (the shipped modes under configs/modes/).
        for root in (CONTROL_ROOT, REPO_ROOT):
            if (root / path).exists():
                path = root / path
                break
        else:
            path = CONTROL_ROOT / path
    return yaml.safe_load(path.read_text()) or {}


def expand_named_values(
    values,
    *,
    names: list[str],
    default: float,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> np.ndarray:
    if isinstance(values, Mapping):
        arr = np.array([float(values.get(name, default)) for name in names], dtype=float)
    elif isinstance(values, list):
        arr = np.asarray(values, dtype=float)
        if arr.shape != (len(names),):
            raise ValueError(f"expected {len(names)} values, got {arr.size}")
    else:
        arr = np.full(len(names), float(default if values is None else values), dtype=float)
    if clamp_min is not None or clamp_max is not None:
        arr = np.clip(arr, -np.inf if clamp_min is None else clamp_min, np.inf if clamp_max is None else clamp_max)
    return arr


__all__ = [
    "next_event_gil_friendly",
    "ShutdownFlag",
    "install_signal_handlers",
    "_zeros",
    "_zero_command",
    "_load_mode_config",
    "expand_named_values",
]
