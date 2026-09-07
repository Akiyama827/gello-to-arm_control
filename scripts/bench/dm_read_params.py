"""Compatibility CLI; use tools/bench/dm/read_params.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "tools/bench/dm/read_params.py"), run_name="__main__")
