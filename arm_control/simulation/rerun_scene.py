"""Mirror the MuJoCo twin's ground truth into Rerun (`sim/…` entities).

Makes Rerun the single visual surface for the shadow run: the TRUE module,
base, and fixtures render live next to the plan-preview overlays — the carried
module and the dock snap are visible without the native MuJoCo viewer (which
stays an optional desk-side physics-debug tool, ``sim_launch_viewer``).

Geometry comes straight from the compiled model (mesh vertices/faces, box and
plane primitives), so no asset paths are needed; per-tick updates are just
world transforms from ``geom_xpos/xmat``. Arm geoms are excluded by default —
the arm is already rendered from ``motor_state`` (measured ghost), and doubling
it only adds clutter.
"""
from __future__ import annotations

import mujoco
import numpy as np
import rerun as rr


def static_mesh_geoms(model, data, *, root_T_world, exclude_prefixes=(), cache_dir):
    """Export a composed snapshot for file-based viewers, in the given frame.

    These are compiled mesh vertices (already scaled/recentered), so geom_xmat
    and geom_xpos apply directly. Cache by content, not model-local mesh IDs.
    No source CAD is rewritten and no simulation step is performed.
    """
    import hashlib
    from pathlib import Path

    import trimesh

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(model.ngeom):
        name = model.geom(i).name or f"geom{i}"
        body = model.body(model.geom_bodyid[i]).name
        if any(body.startswith(p) or name.startswith(p) for p in exclude_prefixes):
            continue
        if model.geom_group[i] == 3:
            continue
        kind = model.geom_type[i]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[i]
            v, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            f, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            mesh = trimesh.Trimesh(model.mesh_vert[v:v + nv], model.mesh_face[f:f + nf], process=False)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * model.geom_size[i])
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            # A finite visual patch represents the infinite collision plane.
            mesh = trimesh.creation.box(extents=[8, 8, 0.002])
            mesh.apply_translation([0, 0, -0.001])
        else:
            raise ValueError(f"static mesh export: unsupported geom {name!r} type {kind}")
        content = mesh.export(file_type='stl')
        path = cache / (hashlib.sha256(content).hexdigest() + '.stl')
        if not path.exists():
            path.write_bytes(content)
        T = np.eye(4)
        T[:3, :3] = data.geom_xmat[i].reshape(3, 3)
        T[:3, 3] = data.geom_xpos[i]
        out.append((name, f"mesh{i}", path, np.asarray(root_T_world) @ T))
    return out


class RerunSceneMirror:
    """Log static geometry once, then stream per-geom world transforms."""

    def __init__(
        self,
        model,
        data,
        prefix: str = "sim",
        exclude_prefixes: tuple[str, ...] = (),
    ) -> None:
        # exclude_prefixes: the ARM bodies (scene.arm_prefixes) — the measured
        # ghost renders the arm; mirroring it twice reads as two robots.
        self._model = model
        self._data = data
        self._prefix = str(prefix).rstrip("/")
        self._geoms: list[tuple[int, str]] = []
        for i in range(model.ngeom):
            name = model.geom(i).name or f"geom{i}"
            body = model.body(model.geom_bodyid[i]).name
            if any(body.startswith(p) or name.startswith(p) for p in exclude_prefixes):
                continue
            if model.geom_group[i] == 3:
                # Collision-class twin of a visual geom (urdf_to_mjcf pairs
                # every body with group-2 visual + group-3 hull) — rendering
                # both doubles every part with a subtly different shape.
                continue
            entity = f"{self._prefix}/{body}/{name}"
            if self._log_asset(i, entity):
                self._geoms.append((i, entity))

    def _log_asset(self, i: int, entity: str) -> bool:
        model = self._model
        rgba = np.clip(model.geom_rgba[i], 0.0, 1.0)
        color = [int(round(c * 255)) for c in rgba]
        gtype = model.geom_type[i]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            did = model.geom_dataid[i]
            v0, nv = model.mesh_vertadr[did], model.mesh_vertnum[did]
            f0, nf = model.mesh_faceadr[did], model.mesh_facenum[did]
            rr.log(
                entity,
                rr.Mesh3D(
                    vertex_positions=model.mesh_vert[v0 : v0 + nv],
                    triangle_indices=model.mesh_face[f0 : f0 + nf],
                    albedo_factor=color,
                ),
                static=True,
            )
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            rr.log(
                entity,
                rr.Boxes3D(half_sizes=[model.geom_size[i]], colors=[color], fill_mode="solid"),
                static=True,
            )
        elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            rr.log(
                entity,
                rr.Boxes3D(
                    half_sizes=[[1.0, 1.0, 0.002]],
                    colors=[[90, 90, 90, 120]],
                    fill_mode="solid",
                ),
                static=True,
            )
        else:  # cylinders/spheres/capsules — none in the current scenes
            return False
        return True

    def update(self) -> None:
        for i, entity in self._geoms:
            rr.log(
                entity,
                rr.Transform3D(
                    translation=self._data.geom_xpos[i],
                    mat3x3=self._data.geom_xmat[i].reshape(3, 3),
                ),
            )


def start_mirror_thread(
    source, data=None, hz: float = 15.0, exclude_prefixes: tuple[str, ...] = ()
) -> None:
    """Run the whole mirror (init, asset upload, updates) on a daemon thread.

    ``source`` is a zero-arg callable returning the CURRENT ``(model, data)``;
    a plain ``(model, data)`` pair is still accepted for scenes that never
    recompile. The callable form matters for any scene that GROWS: a docked
    module rebinds ``backend.model``/``backend.data`` to new objects, and a
    mirror holding the originals keeps streaming the dead model -- the scene
    silently freezes at the graft and never shows the module docked, which is
    worse than showing nothing. Rebuilt on identity change, so the new
    module's geometry is uploaded as static assets exactly once.

    The gRPC sink BLOCKS the calling thread once its channel fills with no
    viewer attached -- on the sim's step thread that freezes the plant and
    starves the whole graph (measured: load-25 stall). Isolating everything
    Rerun-related here means a missing viewer costs one parked thread, never
    the sim. Pose reads are unsynchronized snapshots of ``data`` -- worst case
    a torn frame, acceptable for visualization.
    """
    import threading
    import time

    if data is not None:
        model, fixed = source, data
        source = lambda: (model, fixed)  # noqa: E731

    def _run() -> None:
        from arm_control.planning.preview_rerun import (
            DEFAULT_APP_ID,
            DEFAULT_RECORDING_ID,
        )

        # Same app + recording as the plan preview: the twin's true scene and
        # the orchestrator's overlays merge into ONE viewer recording.
        rr.init(DEFAULT_APP_ID, recording_id=DEFAULT_RECORDING_ID, spawn=False)
        rr.connect_grpc()
        mirror = None
        bound = None
        period = 1.0 / max(1.0, float(hz))
        while True:
            model, data_now = source()
            if model is None or data_now is None:
                # Teardown: the plant drops its model on SIGTERM and this
                # daemon thread outlives it by one tick, so `model.ngeom`
                # raised AttributeError into every shutdown log. Noise that
                # looks exactly like a crash is worse than no log at all --
                # it cost real time reading a clean run as a failed one.
                time.sleep(period)
                continue
            if bound != (id(model), id(data_now)):
                bound = (id(model), id(data_now))
                mirror = RerunSceneMirror(
                    model, data_now, exclude_prefixes=exclude_prefixes
                )
            mirror.update()
            time.sleep(period)

    threading.Thread(target=_run, daemon=True, name="rerun-scene-mirror").start()
