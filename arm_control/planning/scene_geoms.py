"""Scene geometry the PLANNER and the viewers both read, with no viewer dep.

ONE DERIVATION, TWO FAILURE POLICIES. A viewer that cannot find the dock mesh
should still draw the robot; a PLANNER that cannot find it must not plan, because
the alternative is planning through a dock that silently left the collision
world. Same geometry, opposite response to a missing file -- so both callers go
through the same resolver and choose with ``strict``.

These two functions turn the ``scene:`` block into geometry: the same bodies a
viewer draws are the ones the planner must not hit, so they are derived once
here rather than duplicated. They lived in ``preview_rerun`` until 2026-09-10,
which imports ``rerun`` at module level -- so ``planning.stack`` could not be
imported at all without the optional [viz] extra, for two functions that log
nothing. Neither uses rerun; keep it that way.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np



class SceneGeometryError(RuntimeError):
    """A configured scene body could not be resolved. Fatal for planning."""


def static_scene_geoms(cfg, *, strict: bool = False) -> list:
    """[(body, mesh_name, mesh_path, arm_T_geom)] for every non-arm scene body.

    ``strict`` raises SceneGeometryError instead of skipping what it cannot
    resolve. Planning passes it; viewers do not. See the module docstring.

    Reads the SAME `scene:` block the sim composes its MuJoCo model from, so the
    Rerun recordings, the teleop page and the sim can never disagree about where
    the dock stands. Poses come back in the ARM BASE frame — the frame the robot
    meshes and preview ghosts already live in (pinocchio FK, URDF root = arm
    base) — so callers can use them without another transform.

    Returns [] (never raises) when there is no scene, no mujoco, or a missing
    mesh: a viewer that cannot draw the table must still draw the robot.
    """
    # "arm" is the robot itself (self-collision is the planner's own job) and
    # "module" is the pick TARGET, not an obstacle: scene_off does not cover
    # move_to_grasp, so registering it makes its own grasp descent unplannable
    # (2026-08-26 decision). The nest fixture is the obstacle. Inventory
    # modules are a list and are skipped by the Mapping guard below.
    scene = {
        k: v for k, v in dict(cfg.get("scene") or {}).items()
        if k not in ("arm", "module")
    }
    # Inventory modules ARE obstacles — to each other. Only the one being
    # picked is exempt, and that exemption is per-phase and per-module via
    # MuJoCoCollisionWorld.enable_scene_obstacles(exclude=...), keyed on the
    # slot id appearing in the geom name. Poses are the NEST poses: a module
    # already docked is no longer there, so its obstacle is stale — acceptable
    # only because the dock legs run with scene obstacles off anyway.
    for entry in dict(cfg.get("scene") or {}).get("inventory") or []:
        entry = dict(entry)
        scene[str(entry.get("slot"))] = entry
    if not scene:
        return []
    try:
        import xml.etree.ElementTree as ET

        import mujoco

        from arm_control import CONTROL_ROOT, frames

        arm_T_world = frames.invert(frames.world_T_arm(cfg))
    except Exception as exc:  # mujoco is optional on a viewer-only host
        if strict:
            raise SceneGeometryError(f"scene geometry unavailable: {exc}") from exc
        print(f"[scene] static scene skipped: {exc}", flush=True)
        return []

    out = []
    for name, spec in scene.items():
        # Only MODEL SLOTS describe a body to draw. The scene block also
        # carries scalars (ground_z, timestep), lists (static_boxes, and the
        # inventory of pickable modules) and plain sub-dicts (welds,
        # arm_slices); `dict(spec)` on a list of dicts raises
        # "dictionary update sequence element #0 has length N; 2 is required",
        # which took out the whole planning stack for any scenario declaring
        # static_boxes — i.e. the FR3 bench scenes since 2026-08-06.
        if not isinstance(spec, Mapping):
            continue
        spec = dict(spec)
        model_path = spec.get("model_path")
        if not model_path:
            continue
        # Inventory modules are deliberately NOT scene bodies: scene_off does
        # not cover move_to_grasp, so a module registered as planner geometry
        # makes its own grasp descent unplannable. The nest fixture is the
        # obstacle; the module is the target.
        xml_path = Path(str(model_path))
        if not xml_path.is_absolute():
            xml_path = CONTROL_ROOT / xml_path
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            # Pose the body's own joints exactly as the sim holds them, so a
            # rotated dock socket renders rotated here too.
            hold = spec.get("hold_q") or cfg.get("sim_base_hold_q") or []
            for j, value in enumerate(hold[: model.nq]):
                data.qpos[j] = float(value)
            mujoco.mj_forward(model, data)
            # meshdir + file names live in the XML, not the compiled model.
            root = ET.parse(xml_path).getroot()
            compiler = root.find("compiler")
            meshdir = (compiler.get("meshdir") if compiler is not None else "") or ""
            files = {
                m.get("name"): m.get("file")
                for m in root.iter("mesh")
                if m.get("name") and m.get("file")
            }
            world_T_body = frames.T_from_spec(
                {
                    "origin": spec.get("world_pos", [0, 0, 0]),
                    "rpy": spec.get("world_rpy", [0, 0, 0]),
                }
            )
            for gid in range(model.ngeom):
                # group 2 is this model family's VISUAL group (group 3 is the
                # collision copy — drawing both doubles every mesh).
                if model.geom_group[gid] != 2 or model.geom_dataid[gid] < 0:
                    continue
                mesh_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[gid]
                )
                mesh_file = files.get(mesh_name)
                if not mesh_file:
                    continue
                mesh_path = xml_path.parent / meshdir / mesh_file
                if not mesh_path.is_file():
                    if strict:
                        raise SceneGeometryError(f"mesh missing: {mesh_path}")
                    print(f"[scene] mesh missing: {mesh_path}", flush=True)
                    continue
                T_body_geom = np.eye(4)
                T_body_geom[:3, :3] = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
                T_body_geom[:3, 3] = np.asarray(data.geom_xpos[gid])
                # MuJoCo RECENTERS and REORIENTS every mesh asset at compile
                # time; mesh_pos/mesh_quat record what it applied, mapping the
                # stored vertices back to the file's own frame (verified on
                # this model: v_file = R_m v_stored + p_m, residual 2e-4 m).
                # geom_xpos/xmat therefore place the PROCESSED mesh — hand them
                # the raw STL, as Rerun and the teleop page do, and the parts
                # scatter and tumble (bench 2026-08-06). Undo it.
                mesh_id = int(model.geom_dataid[gid])
                R_m = np.zeros(9)
                mujoco.mju_quat2Mat(R_m, model.mesh_quat[mesh_id])
                T_mesh = np.eye(4)
                T_mesh[:3, :3] = R_m.reshape(3, 3)
                T_mesh[:3, 3] = np.asarray(model.mesh_pos[mesh_id])
                out.append(
                    (name, f"{mesh_name}_{gid}", mesh_path,
                     arm_T_world @ world_T_body @ T_body_geom @ frames.invert(T_mesh))
                )
        except SceneGeometryError:
            raise
        except Exception as exc:
            if strict:
                raise SceneGeometryError(f"scene body {name!r} failed: {exc}") from exc
            print(f"[scene] body {name!r} failed: {exc}", flush=True)
    return out


def scene_obstacle_geoms(cfg, *, strict: bool = True) -> list[dict]:
    """`environment`-style MESH obstacles for the static scene bodies.

    The planner only ever knew about `environment:` boxes, so a dock drawn in
    every viewer was still invisible to collision checking — the arm would
    happily plan straight through it. These come from `static_scene_geoms`,
    the SAME call the viewers draw with, so the obstacle and the picture can
    never disagree (they did, by 36 mm, when this computed its own transform).

    One entry per visual geom, pointing at the geom's own STL: MuJoCo collides
    a mesh by its CONVEX HULL — far tighter than a bounding box while never
    optimistic, since a hull contains its mesh. The dock's parts are close to
    convex, so this is nearly exact.

    Marked `toggleable` so `enable_scene_obstacles` can drop them for the
    insertion leg, where the dock stops being an obstacle and becomes the
    target. Poses place the RAW FILE; MuJoCo's own asset recentering is the
    collision world's problem to undo (see MuJoCoCollisionWorld._build).

    STRICT BY DEFAULT, unlike the viewer path: this list IS the collision
    world's knowledge of the scene, so a configured body that cannot be
    resolved must refuse to plan rather than quietly not exist. Dropping one
    here does not degrade a picture -- it invites the arm through a dock.
    """
    try:
        import pinocchio as pin
    except Exception as exc:
        if strict:
            raise SceneGeometryError(f"obstacles unavailable: {exc}") from exc
        print(f"[scene] obstacles skipped: {exc}", flush=True)
        return []
    out = [
        {
            "name": f"scene_{body}_{mesh_name}",
            "mesh": str(mesh_path),
            "pose": [float(v) for v in T[:3, 3]]
            + [float(v) for v in pin.rpy.matrixToRpy(T[:3, :3])],
            "toggleable": True,
        }
        for body, mesh_name, mesh_path, T in static_scene_geoms(cfg, strict=strict)
    ]
    # The plant's static fixtures (module nests, table) are obstacles for the
    # PLANNER too. They live in scene.static_boxes, which only the plant read,
    # so the arm had no reason to avoid the nests it reaches between — and the
    # modules themselves are deliberately not obstacles, so nothing stood in
    # for them. Derived here rather than duplicated into `environment:` so the
    # two can never drift apart.
    try:
        from arm_control import frames

        arm_T_world = frames.invert(frames.world_T_arm(cfg))
        import pinocchio as pin

        for box in (dict(cfg.get("scene") or {}).get("static_boxes") or []):
            T = np.eye(4)
            T[:3, 3] = [float(v) for v in box["pos"]]
            T = arm_T_world @ T
            out.append(
                {
                    "name": f"fixture_{box['name']}",
                    "size": [float(v) for v in box["size"]],
                    "pose": [float(v) for v in T[:3, 3]]
                    + [float(v) for v in pin.rpy.matrixToRpy(T[:3, :3])],
                    "toggleable": True,
                }
            )
    except SceneGeometryError:
        raise
    except Exception as exc:  # never let a fixture stop the robot rendering
        if strict:
            raise SceneGeometryError(f"static fixtures failed: {exc}") from exc
        print(f"[scene] static fixtures skipped: {exc}", flush=True)
    if out:
        print(f"[scene] {len(out)} obstacles from scene", flush=True)
    return out


def _self_check() -> None:
    """A controller host with no [viz] extra must still import the planner.

    This module exists for exactly this: `planning.stack` used to reach
    `scene_obstacle_geoms` through `preview_rerun`, which imports `rerun` at
    module level, so a headless servo host could not import the planning stack
    at all. Blocking `rerun` is the only honest way to check that, and it runs
    in a SUBPROCESS -- unloading half of sys.modules in-process leaves
    pinocchio's C extension partly initialised and the result means nothing.
    """
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "class NoRerun:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'rerun' or name.startswith('rerun.'):\n"
        "            raise ImportError('rerun blocked: host without the [viz] extra')\n"
        "sys.meta_path.insert(0, NoRerun())\n"
        "import arm_control.planning.stack\n"
        "try:\n"
        "    import arm_control.planning.preview_rerun\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('preview_rerun imported with rerun blocked')\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert done.returncode == 0, (
        "planning.stack must import without rerun:\n" + done.stdout + done.stderr
    )

    # ONE derivation, TWO failure policies. A configured body that cannot be
    # resolved must degrade a PICTURE and refuse a PLAN -- because a dropped
    # obstacle does not look wrong, it just quietly stops existing, and the arm
    # is then invited through the dock that was supposed to be in its way.
    broken = {"scene": {"dock": {"model_path": "/nonexistent/dock.xml",
                                 "pos": [0, 0, 0], "rpy": [0, 0, 0]}}}
    assert static_scene_geoms(broken) == [], "viewers must degrade, not raise"
    assert scene_obstacle_geoms(broken, strict=False) == []
    for call in (lambda: static_scene_geoms(broken, strict=True),
                 lambda: scene_obstacle_geoms(broken)):
        try:
            call()
        except SceneGeometryError:
            continue
        raise AssertionError("planning geometry failed open on a missing body")

    from arm_control.planning import preview_rerun, scene_geoms
    assert preview_rerun.scene_obstacle_geoms is scene_geoms.scene_obstacle_geoms
    assert preview_rerun.static_scene_geoms is scene_geoms.static_scene_geoms
    assert scene_obstacle_geoms({}) == [] and static_scene_geoms({}) == []
    print("scene_geoms: planning imports without rerun; preview re-export intact")


if __name__ == "__main__":
    _self_check()
