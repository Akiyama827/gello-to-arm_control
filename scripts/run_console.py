"""Compatibility CLI; use examples/run_console.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/run_console.py"), run_name="__main__")
