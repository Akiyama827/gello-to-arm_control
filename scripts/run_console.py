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
import time
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


def ensure_rerun_viewer(port: int = 9876) -> None:
    """Start ONE persistent viewer if none is listening; nodes only connect.

    Every Rerun producer in this repo uses `spawn=False` + `connect_grpc()` --
    visualizer, the scene mirror, the plan preview -- so nothing here starts a
    viewer. Without one the connect succeeds and the data goes NOWHERE, which
    is the worst failure shape: no error, no visuals, nothing to search for.

    Detached on purpose: a viewer spawned as a child of the graph dies with
    every restart, and the window you are looking at ends up belonging to a
    previous run. (Control's own `view.py` does the same thing for the project
    graphs; this is the standalone launcher's copy, which is why the demo can
    run from a fresh clone with no project around it.)
    """
    import socket

    def listening() -> bool:
        probe = socket.socket()
        probe.settimeout(0.3)
        try:
            probe.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            probe.close()

    if listening():
        print("[run_console] rerun viewer already up — reusing it", flush=True)
        return
    try:
        subprocess.Popen(
            ["rerun"], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("[run_console] no `rerun` on PATH — the console still works, "
              "visuals are just disabled (pip install rerun-sdk)", flush=True)
        return
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        if listening():
            print("[run_console] rerun viewer up (persistent — survives restarts)",
                  flush=True)
            return
        time.sleep(0.3)
    print("[run_console] rerun viewer did not come up — visuals may lag", flush=True)


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

    ensure_rerun_viewer()
    print(f"[run_console] {config.name} -> {graph.name}", flush=True)
    if args.dual:
        print("[run_console] consoles on http://127.0.0.1:7500 and :7510", flush=True)
    else:
        print("[run_console] console on http://127.0.0.1:7500", flush=True)
    print("[run_console] the arm comes up DISARMED — press ARM on the page", flush=True)
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
    _check_entry_keys()
    ready, hint = assets_ready()
    assert ready or "franka_description" in hint, hint
    print(f"run_console: ok (assets {'staged' if ready else 'not staged'})")


def _check_entry_keys() -> None:
    """Every key the entry configs state must be read by somebody.

    A config key nothing consumes is worse than a missing one: it reads as a
    setting, documents itself in a comment, and does nothing. Caught in the
    wild -- `sim_damping_ratio: 0.025` sat in sim_demo.yaml with a paragraph
    explaining why the sim plant needs damping, and no code ever looked it up,
    so the demo shipped kp 1200 against kd 0 (the config's own measurements
    call that "unusable"). A missing key would have failed loudly; this one
    just quietly meant nothing.

    Top-level keys only, and only those the entry file states ITSELF --
    inherited keys are the included fragment's business.
    """
    import re

    sources = "\n".join(
        path.read_text()
        for directory in ("arm_control", "nodes", "scripts")
        for path in sorted((REPO_ROOT / directory).rglob("*.py"))
    )
    for entry in sorted((REPO_ROOT / "configs" / "entry").glob("*.yaml")):
        # The file's own top-level keys, read as text: parsing the tree back
        # would hand us the merged result and lose exactly this distinction.
        own = [
            m.group(1)
            for m in re.finditer(r"^([a-z_][a-z0-9_]*):", entry.read_text(), re.M)
        ]
        for key in own:
            if key in ("include", "arm"):
                continue  # structural: the loader's own, and the shared table
            assert f'"{key}"' in sources or f"'{key}'" in sources, (
                f"{entry.name} sets {key!r}, which no code in arm_control/, "
                f"nodes/ or scripts/ ever reads -- a key that does nothing "
                f"still reads like a setting"
            )


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
