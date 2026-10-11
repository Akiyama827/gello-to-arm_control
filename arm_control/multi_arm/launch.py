"""A pair's two supervisors, started for the dual console (CTRL-053: every arm on its supervisor, operator 2026-10-09).

Simulation: ONE supervisor process hosts the pair's world (``world.build_pair_world``) and drives the first arm; the second
arm's supervisor is its co-host (``--cohost``), so both arms move in one MuJoCo world, each with its own claim, ARM, policy and
sim Hand (``rt.hand_port``). Hardware: one supervisor per arm over its RT box (``plant rt``), as ``run.py real franka motion``.

``stage_pair`` writes, beside the pair config, each arm's supervisor config and a derived pair config naming them, which is what
the console loads: its Hands then connect where the supervisors serve them. ``ARM_CONTROL_PAIR_SOCKETS`` (``name=path,...``)
tells the console's ``SupervisedHardware`` where each supervisor listens.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import yaml

from arm_control import CONTROL_ROOT
from arm_control.config import _deep_merge, load_config_tree

#: the simulated servo and RT defaults every simulated arm takes (the arm_rt_server's launch values)
SIM_PROFILE = Path(__file__).resolve().parents[2] / "examples" / "profiles" / "sim_supervised.yaml"
#: each simulated arm's Hand port: the first arm's is the profile's, the next ones +10 (ponytail: two arms; a list if more)
SIM_HAND_PORT_STEP = 10
SOCKETS_ENV = "ARM_CONTROL_PAIR_SOCKETS"


def _absolute(raw: dict, base: Path) -> dict:
    """The pair's own relative paths (scene, shared object), made absolute so the staged copy can live elsewhere."""
    out = dict(raw)
    if isinstance(out.get("scene"), str):
        out["scene"] = str((base / out["scene"]).resolve())
    if isinstance(out.get("shared_object"), dict) and out["shared_object"].get("path"):
        out["shared_object"] = dict(out["shared_object"], path=str((base / out["shared_object"]["path"]).resolve()))
    return out


def stage_pair(pair_path: Path, *, plant: str) -> tuple[Path, dict[str, Path]]:
    """Write each arm's supervisor config and the derived pair config; return ``(pair, {arm: config})``."""
    pair_path = Path(pair_path).resolve()
    raw = _absolute(load_config_tree(pair_path), pair_path.parent)
    raw.pop("include", None)
    if plant not in ("sim", "rt"):
        raise ValueError(f"unknown plant {plant!r}: sim or rt")
    sim = load_config_tree(SIM_PROFILE) if plant == "sim" else None
    arms, configs = {}, {}
    for index, (name, entry) in enumerate(raw["arms"].items()):
        cfg = load_config_tree((pair_path.parent / entry["config"]).resolve())
        cfg.pop("include", None)
        if sim is not None:
            port = int(sim["sim_hand"]["port"]) + SIM_HAND_PORT_STEP * index
            cfg = _deep_merge(_deep_merge(cfg, sim), dict(
                rt=dict(instance_id=name, host="127.0.0.1", hand_port=port), sim_hand=dict(port=port), sim_actor=name,
                idle_timeout_sec=0.1))
        configs[name] = cfg
        arms[name] = dict(entry)
    contents = {name: yaml.safe_dump(cfg, sort_keys=False) for name, cfg in configs.items()}
    directory = CONTROL_ROOT / ".cache" / "pair_supervisors" / hashlib.sha256(
        (yaml.safe_dump(raw, sort_keys=True) + "".join(contents.values()) + plant).encode()).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, text in contents.items():
        paths[name] = directory / f"{name}.yaml"
        _write(paths[name], text)
        arms[name]["config"] = str(paths[name])
    staged = directory / "pair.yaml"
    _write(staged, yaml.safe_dump(dict(raw, arms=arms), sort_keys=False))
    return staged, paths


def _write(path: Path, text: str) -> None:
    """Whole or not at all: a concurrent launch of the same pair reads either file, never half of one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def start_pair(pair_path: Path, env: dict, *, plant: str, cwd: Path | None = None, timeout_s: float = 300.0):
    """Start the pair's supervisors; return ``(processes, staged pair config, env with ARM_CONTROL_PAIR_SOCKETS)``. A failure
    stops whatever had started."""
    from arm_control.supervisor.launch import start_supervisor, stop_supervisors

    staged, configs = stage_pair(pair_path, plant=plant)
    names = list(configs)
    procs, sockets = [], {}
    try:
        if plant == "sim":
            first, second = names
            prefix = "PAIR_COHOST"
            world = dict(env, WORKCELL_SCENE=str(staged),
                         WORKCELL_BACKEND_FACTORY="arm_control.multi_arm.world:build_pair_world")
            proc, out = start_supervisor(configs[first], world, plant="sim", cwd=cwd, timeout_s=timeout_s,
                                         cohosts={prefix: configs[second]})
            procs.append(proc)
            sockets = {first: out["ARM_CONTROL_SOCKET"], second: out[f"{prefix}_SOCKET"]}
        else:
            for name in names:
                proc, out = start_supervisor(configs[name], env, plant="rt", cwd=cwd, timeout_s=timeout_s)
                procs.append(proc)
                sockets[name] = out["ARM_CONTROL_SOCKET"]
    except BaseException:
        stop_supervisors(procs, keep_log=True)
        raise
    return procs, staged, dict(env, **{SOCKETS_ENV: ",".join(f"{k}={v}" for k, v in sockets.items())})


@contextmanager
def running_pair(pair_path: Path, env: dict, *, plant: str, cwd: Path | None = None, timeout_s: float = 300.0):
    """``start_pair`` for a ``with`` block: yields ``(staged pair config, env)``; the supervisors stop on the way out."""
    from arm_control.supervisor.launch import stop_supervisors

    procs, staged, env = start_pair(pair_path, env, plant=plant, cwd=cwd, timeout_s=timeout_s)
    try:
        yield staged, env
    finally:
        stop_supervisors(procs, keep_log=plant == "rt")  # a hardware run keeps its supervisors' logs (the bench's evidence)


def pair_sockets(env) -> dict[str, str]:
    """``ARM_CONTROL_PAIR_SOCKETS`` as ``{arm: socket path}`` (empty when unset)."""
    text = env.get(SOCKETS_ENV, "")
    return dict(item.split("=", 1) for item in text.split(",") if item)
