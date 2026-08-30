"""Control interfaces, planning and simulation for modular/robot arms.

The top-level surface is deliberately tiny: every caller imports by submodule
path (``from arm_control.config import load_robot_config``), so all that lives
here is the version and the TWO path anchors the split made explicit.
"""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

#: This repository's own root (the directory holding ``arm_control/``,
#: ``nodes/``, ``configs/``, ``dataflows/``). Fixed by the package location —
#: used for things that SHIP WITH THIS REPO: the mode configs under
#: ``configs/modes/`` and the DM vendor shared objects under ``dlls/``.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The DEPLOYMENT root — the directory RELATIVE ASSET PATHS in runtime configs
#: resolve against (URDFs, MJCFs, meshes, the planning cache). Standalone this
#: equals :data:`REPO_ROOT`; embedded as a submodule the launcher exports
#: ``ARM_CONTROL_ROOT=<project>/Control`` so the project's CAD keeps
#: resolving. Before the split this was ``Path(__file__).parents[N]`` computed
#: independently in seven modules — the env var is the one seam.
CONTROL_ROOT = Path(os.environ.get("ARM_CONTROL_ROOT") or REPO_ROOT)

# Scene types live in a separate module so package import remains dependency-light.
