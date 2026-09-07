"""Compatibility CLI; use tools/bench/rt/timing.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "tools/bench/rt/timing.py"), run_name="__main__")
