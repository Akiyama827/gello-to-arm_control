#!/usr/bin/env python3
"""Run the operator console against a simulated arm, from a fresh clone.

Deliberately thin. This is NOT a second scenario registry -- a deployment's
launcher (which knows about that project's scenarios, arms and CAD) stays where
it is. All this does is the three things a bare `dora run` cannot do for you:
check the assets are staged, export the two roots, and pick the graph.

    python scripts/run_console.py                 # one arm, console on :7500
    python scripts/run_console.py --dual          # two arms, :7500 and :7510
    python scripts/run_console.py --config <path> # your own entry config

Meshes are not vendored here (this repo owns the code; a project owns its CAD),
so the first run needs the FR3 description staged once:

    git clone https://github.com/frankarobotics/franka_description
    python scripts/setup_fr3_assets.py --source ./franka_description
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SINGLE = REPO_ROOT / "dataflows" / "sim_franka_motion.yml"
DUAL = REPO_ROOT / "dataflows" / "dual_arm_sim.yml"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "entry" / "sim_demo.yaml"


def assets_ready() -> tuple[bool, str]:
    """Is the FR3 description staged? Returns (ready, what to do about it)."""
    urdf = REPO_ROOT / "franka" / "urdf" / "fr3.urdf"
    meshes = REPO_ROOT / "franka" / "meshes"
    if urdf.is_file() and meshes.is_dir() and any(meshes.rglob("*.stl")):
        return True, ""
    return False, (
        f"The FR3 description is not staged at {urdf.parent}.\n"
        f"This repo vendors no meshes -- it owns the code, a project owns its\n"
        f"CAD -- so stage it once:\n\n"
        f"    git clone https://github.com/frankarobotics/franka_description\n"
        f"    python {Path('scripts/setup_fr3_assets.py')} "
        f"--source ./franka_description\n\n"
        f"(`python scripts/setup_fr3_assets.py --check` reports status and\n"
        f"changes nothing.)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"entry config (default: {DEFAULT_CONFIG.name})")
    parser.add_argument("--dual", action="store_true",
                        help="two arms in one graph, on two console ports")
    parser.add_argument("--check", action="store_true",
                        help="report readiness and exit, launching nothing")
    args = parser.parse_args()

    ready, hint = assets_ready()
    config = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    graph = DUAL if args.dual else SINGLE

    if args.check or not ready:
        print(f"config: {config}  {'ok' if config.is_file() else 'MISSING'}")
        print(f"graph:  {graph}  {'ok' if graph.is_file() else 'MISSING'}")
        print(f"assets: {'ok' if ready else 'NOT STAGED'}")
        if not ready:
            print("\n" + hint, file=sys.stderr)
        return 0 if (ready and args.check) else (0 if args.check else 1)
    if not config.is_file():
        print(f"entry config not found: {config}", file=sys.stderr)
        return 1

    env = dict(os.environ)
    # Both roots, because a bare `dora run` rebases each node's cwd to the
    # dataflow's directory: a relative ARM_CONTROL_CONFIG would then resolve
    # against dataflows/ and a relative mesh path against nothing useful.
    env["ARM_CONTROL_ROOT"] = str(REPO_ROOT)
    env["ARM_CONTROL_CONFIG"] = str(config)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    print(f"[run_console] {config.name} -> {graph.name}")
    if args.dual:
        print("[run_console] consoles on http://127.0.0.1:7500 and :7510")
    else:
        print("[run_console] console on http://127.0.0.1:7500")
    print("[run_console] the arm comes up DISARMED — press ARM on the page")
    try:
        return subprocess.call(["dora", "run", str(graph)], env=env, cwd=REPO_ROOT)
    except FileNotFoundError:
        print("`dora` is not on PATH — is the dora-rs CLI installed?",
              file=sys.stderr)
        return 1


def _self_check() -> None:
    """The launcher's own claims: the files it points at have to exist."""
    assert SINGLE.is_file(), SINGLE
    assert DUAL.is_file(), DUAL
    assert DEFAULT_CONFIG.is_file(), DEFAULT_CONFIG
    # The entry config must actually load through the include machinery, and
    # name a console port -- the whole reason it lives with the robot.
    from arm_control.config import load_config_tree

    cfg = load_config_tree(DEFAULT_CONFIG)
    assert cfg["console"]["http_port"], cfg.get("console")
    dual_cfg = load_config_tree(REPO_ROOT / "configs/entry/sim_demo_b.yaml")
    assert dual_cfg["console"]["http_port"] != cfg["console"]["http_port"], (
        "the two demo arms must not share a console port -- that collision is "
        "the exact thing per-instance config exists to prevent"
    )
    ready, hint = assets_ready()
    assert ready or "franka_description" in hint, hint
    print(f"run_console: ok (assets {'staged' if ready else 'not staged'})")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
