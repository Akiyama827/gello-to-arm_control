"""Convex-decomposition collision for concave grip parts (CoACD, cached).

The grasp needs TRUE finger/module geometry (slots, ribs) — convex hulls
bloat ~2 cm and stall approaches; MuJoCo mesh-SDFs need watertight input and
our printed-part STLs are open shells (measured: the SDF field reported
-2.3 mm "contact" at 19 mm true clearance). Convex decomposition is the
robust third path: each concave part becomes a set of convex pieces, and
collision runs through MuJoCo's battle-tested convex narrowphase.

Pieces are baked once per (mesh file content, threshold) into ``cache_dir``
as OBJ files and swapped into the spec pre-compile: the original collision
geom is deleted and one geom per piece is added with identical pose, class,
and contact bits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np


def decompose_mesh_file(
    mesh_path: str | Path,
    cache_dir: str | Path,
    threshold: float = 0.02,
) -> list[Path]:
    """CoACD-decompose ``mesh_path`` into convex OBJ pieces (content-cached)."""
    mesh_path = Path(mesh_path)
    cache_dir = Path(cache_dir)
    digest = hashlib.sha256(
        mesh_path.read_bytes() + f":{threshold}".encode()
    ).hexdigest()[:16]
    out_dir = cache_dir / f"{mesh_path.stem}_{digest}"
    done = out_dir / ".done"
    if done.exists():
        return sorted(out_dir.glob("piece_*.obj"))
    import coacd
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    tm = trimesh.load(str(mesh_path), force="mesh")
    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(tm.vertices), np.asarray(tm.faces)),
        threshold=float(threshold),
    )
    paths: list[Path] = []
    kept = 0
    for verts, faces in parts:
        piece = trimesh.Trimesh(vertices=verts, faces=faces).convex_hull
        if piece.volume < 1e-9:  # sub-mm^3 sliver: no collision value, and
            continue             # the compiler rejects near-zero-volume meshes
        p = out_dir / f"piece_{kept:03d}.obj"
        piece.export(str(p))
        paths.append(p)
        kept += 1
    done.touch()
    print(
        f"[convex_decomp] {mesh_path.name}: {len(paths)} convex pieces "
        f"(threshold {threshold}) -> {out_dir}",
        flush=True,
    )
    return paths


def replace_with_decomposition(
    spec: mujoco.MjSpec,
    bodies_containing: list[str],
    cache_dir: str | Path,
    mesh_search_dirs: list[str | Path],
    threshold: float = 0.02,
) -> int:
    """Swap matching bodies' collision MESH geoms for their convex pieces.

    Must run BEFORE ``spec.compile()``. ``mesh_search_dirs``: the attached
    child models' meshdirs — spec.attach keeps asset file paths relative to
    the child model that owned them, so the composed spec cannot resolve
    them alone. Returns the number of geoms replaced.
    """

    def _resolve_mesh(rel: str) -> Path | None:
        cand = Path(rel)
        if cand.is_absolute() and cand.exists():
            return cand
        for d in mesh_search_dirs:
            hit = Path(d) / rel
            if hit.exists():
                return hit
        return None

    mesh_files = {mesh.name: mesh.file for mesh in spec.meshes}
    replaced = 0
    for body in spec.bodies:
        name = body.name or ""
        if not any(needle in name for needle in bodies_containing):
            continue
        for geom in list(body.geoms):
            if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if not (geom.contype or geom.conaffinity):
                continue  # visual-class geom stays untouched
            mesh_file = _resolve_mesh(mesh_files.get(geom.meshname) or "")
            if mesh_file is None:
                print(
                    f"[convex_decomp] WARNING: cannot resolve mesh for geom "
                    f"{geom.name or geom.meshname!r} — left as hull",
                    flush=True,
                )
                continue
            pieces = decompose_mesh_file(mesh_file, cache_dir, threshold)
            for i, piece in enumerate(pieces):
                asset_name = f"{geom.meshname}_cvx{i:03d}"
                spec.add_mesh(name=asset_name, file=str(piece))
                g = body.add_geom(
                    name=f"{(geom.name or geom.meshname)}_cvx{i:03d}",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=asset_name,
                )
                g.pos = list(geom.pos)
                g.quat = list(geom.quat)
                g.contype = geom.contype
                g.conaffinity = geom.conaffinity
                g.group = geom.group
                g.friction = list(geom.friction)
                g.condim = geom.condim
            spec.delete(geom)
            replaced += 1
    return replaced
