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
        # then this repo (the shipped modes under examples/profiles/).
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


def resolve_gains(cfg, mode_cfg: dict, names: list[str], n_arm: int) -> dict | None:
    """Per-motor kp / kd / torque_limits: mode config first, arm table second.

    Returns None when ``names`` is empty -- see below; callers that need the
    vectors should pass that None straight to ``build_executor(gains=...)``,
    which falls back to the arm table.

    Shared by every node that has to state gains -- the trajectory executor
    servos with them, and the console must now ship them WITH each plan
    (``pack_plan`` carries kp/kd, which is how a plan and its stiffness cannot
    be separated in flight). Two copies of this would be two chances to resolve
    a different number for the same arm.

    Without the arm-table fallback a mode config keyed by ANOTHER arm's joint
    names expands to all-zeros -- a limp arm on the bench, silently. The arm
    table (``arm.<key>``) is arm-length and says nothing about non-arm motor
    slots such as the DM gripper, so it is zero-padded up to the full motor
    list; those trailing slots are only ever set by an explicit mode entry.

    Gain CEILINGS are per-arm hardware facts, not policy: 500/5 are the DM MIT
    wire-format limits (``pack_mit_control_frame`` clips above them), while the
    FR3 takes joint stiffness in N*m/rad and needs ~1200. An arm needing other
    bounds states them as ``controller.{kp,kd,torque_limit}_max``.
    """
    import numpy as np

    from arm_control.config import _arm_block

    if not names:
        # No motor-name vector at the top level of this config. That is a
        # legitimate shape, not an error: a composed-SCENE config names its
        # joints in `scene.arm.joint_names` and the plant prefixes them, so
        # the arm table is the only gain source and build_executor already
        # reads it. Returning None hands the caller back to exactly the path
        # every assembly graph used before this function existed.
        #
        # Caught the hard way: wiring arm_controller to this function (Part 2)
        # made it raise "expected 0 values, got 7" on every assembly scenario
        # -- `expand_named_values` against an empty name list -- and killed the
        # node at startup on every assembly scenario (`view.py sim bench`).
        return None

    controller_cfg = dict(mode_cfg.get("controller") or {})
    clamps = {
        "kp": float(controller_cfg.get("kp_max", 500.0)),
        "kd": float(controller_cfg.get("kd_max", 5.0)),
        "torque_limits": float(controller_cfg.get("torque_limit_max", 100.0)),
    }
    arm_keys = {"kp": "kp", "kd": "kd", "torque_limits": "max_tau"}

    out = {}
    for key, clamp_max in clamps.items():
        spec = controller_cfg.get(key)
        if spec is None:
            spec = _arm_block(cfg).get(arm_keys[key])
            if isinstance(spec, list) and len(spec) == n_arm < len(names):
                spec = list(spec) + [0.0] * (len(names) - n_arm)
        out[key] = expand_named_values(
            spec, names=names, default=0.0, clamp_min=0.0, clamp_max=clamp_max
        )

    if not float(np.max(np.abs(out["kp"][:n_arm]))) > 0.0:
        # All-zero arm stiffness is never intentional -- it is a limp arm that
        # holds nothing. The usual cause is a mode config whose per-joint gain
        # keys name a DIFFERENT arm (expand_named_values falls back to 0.0 per
        # missing name), which a dataflow's per-node ARM_CONTROL_MODE_CONFIG can
        # pin behind your back. Fail here, not on the bench.
        raise ValueError(
            f"resolved kp is all zeros for joints {names[:n_arm]}. Check that "
            "controller.kp in the mode config is keyed by THESE joint names "
            "(ARM_CONTROL_MODE_CONFIG, including any per-node env: override in "
            "the dataflow), or drop it to inherit arm.kp from the arm config."
        )
    return out


def entry_point(env_var: str):
    """Resolve a ``module:name`` env var into the object it names, or None.

    The seam that lets a PROJECT inject its own types into a generic node
    without this package importing that project. WORKCELL_LOADER has always
    worked this way; CHAIN_FACTORY and MATE_POLICY joined it.
    """
    import importlib
    import os

    spec = os.environ.get(env_var)
    if not spec:
        return None
    module_name, attribute = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def scene_injections() -> dict:
    """``chain_factory`` and ``mate_policy`` for a composed scene, from env.

    Shared because a scene has more than one consumer: the plant steps it and
    the simulated perception samples truth from the SAME composition. Two
    copies of this resolution would be two chances for the twin the cloud
    tracks to differ from the twin the physics runs.
    """
    chain_cls = entry_point("CHAIN_FACTORY")
    policy_cls = entry_point("MATE_POLICY")
    return {
        "chain_factory": (
            None if chain_cls is None
            else (lambda root_port: chain_cls(root_port=root_port))
        ),
        "mate_policy": None if policy_cls is None else policy_cls(),
    }


def _check_resolve_gains() -> None:
    """The three config shapes that reach resolve_gains, including the empty one.

    The last case is a regression guard. Wiring ``nodes/arm_controller.py`` to
    this function assumed every arm config names its motors at the top level.
    A composed-SCENE config does not -- its joints live under
    ``scene.arm.joint_names`` and the plant prefixes them -- so ``names`` came
    through empty, ``expand_named_values`` raised "expected 0 values, got 7",
    and the controller died at startup on every assembly scenario.
    """
    import numpy as np

    cfg = {"arm": {"kp": [10.0, 11.0], "kd": [1.0, 1.1], "max_tau": [5.0, 5.0]}}
    names = ["j1", "j2"]

    # 1. Mode config wins over the arm table.
    mode = {"controller": {"kp": {"j1": 20.0, "j2": 21.0}}}
    out = resolve_gains(cfg, mode, names, 2)
    assert np.allclose(out["kp"], [20.0, 21.0]), out["kp"]
    assert np.allclose(out["kd"], [1.0, 1.1]), out["kd"]   # falls back

    # 2. No mode config at all: the arm table, exactly as before this existed.
    out = resolve_gains(cfg, {}, names, 2)
    assert np.allclose(out["kp"], [10.0, 11.0]), out["kp"]

    # 3. No motor names: None, not a raise. The caller falls back to the arm
    #    table via build_executor(gains=None).
    assert resolve_gains(cfg, {}, [], 0) is None
    assert resolve_gains(cfg, mode, [], 2) is None

    # 4. An all-zero arm stiffness is still loud -- a limp arm holds nothing.
    try:
        resolve_gains({"arm": {"kp": [0.0, 0.0]}}, {}, names, 2)
    except ValueError as exc:
        assert "all zeros" in str(exc), exc
    else:
        raise AssertionError("all-zero kp resolved silently")
    print("node_utils: resolve_gains OK")


if __name__ == "__main__":
    _check_resolve_gains()
