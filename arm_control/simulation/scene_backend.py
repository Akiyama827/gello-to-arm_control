"""Compatibility API for consumers using the injected scene builders."""
from arm_control.plants.mujoco.composer import (
    _resolve as _resolve,
    build_scene_backend as build_scene_backend,
    build_workcell_backend as build_workcell_backend,
)
