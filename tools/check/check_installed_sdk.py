"""The SDK from an installed wheel, outside the checkout (CTRL-022/060 acceptance), device-free.

    python tools/check/check_installed_sdk.py --server rt/build-offline/arm_rt_server

Builds this library's wheel (no build isolation, no network), installs it with ``--no-deps`` into a scratch virtualenv that
sees the base environment's site-packages (numpy) but not this checkout, and runs check_sdk_supervisor.py with that
interpreter from a scratch directory: its process imports ``arm_control`` from the venv's site-packages (asserted), and the
supervisor it spawns runs from the checkout as a deployment's full environment would. Nothing is installed outside the
scratch directory.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True, help="the fake arm_rt_server binary")
    args = ap.parse_args()
    server = str(Path(args.server).resolve())
    with tempfile.TemporaryDirectory(prefix="installed_sdk_") as tmp:
        tmp = Path(tmp)
        # Build from a copy: setuptools writes build/ and *.egg-info next to the sources.
        src = tmp / "src"
        shutil.copytree(ROOT / "arm_control", src / "arm_control", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "pyproject.toml", src / "pyproject.toml")
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "--no-build-isolation",
                        "--wheel-dir", str(tmp / "dist"), str(src)], check=True)
        wheel = next((tmp / "dist").glob("arm_control-*.whl"))
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(tmp / "venv")], check=True)
        py = str(tmp / "venv" / "bin" / "python")
        subprocess.run([py, "-m", "pip", "install", "--quiet", "--no-deps", "--no-index", str(wheel)], check=True)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ARM_CONTROL_ROOT")}
        where = subprocess.run([py, "-c", "import arm_control; print(arm_control.__file__)"], cwd=tmp, env=env,
                               check=True, capture_output=True, text=True).stdout.strip()
        assert str(tmp / "venv") in where and str(ROOT) not in where, f"arm_control came from {where}"
        # check_sdk_supervisor imports its siblings by file: run it as a copy outside the checkout, so nothing of the
        # checkout is on the client's sys.path.
        script = tmp / "check_sdk_supervisor.py"
        script.write_text((ROOT / "tools/check/check_sdk_supervisor.py").read_text())
        subprocess.run([py, str(script), "--server", server, "--supervisor-python", sys.executable,
                        "--supervisor-path", str(ROOT)], cwd=tmp, env=env, check=True)
        print(f"installed SDK: {wheel.name} into a scratch venv, imported from {where}; SDK -> supervisor -> fake RT: PASS")


if __name__ == "__main__":
    main()
