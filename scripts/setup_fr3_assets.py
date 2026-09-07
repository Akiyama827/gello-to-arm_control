"""Compatibility CLI; use tools/assets/setup_fr3.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/assets/setup_fr3.py"), run_name="__main__")
