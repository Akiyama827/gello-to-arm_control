"""Stage a URDF for MuJoCo: the ``<mujoco>`` compiler extension, cached.

Neither planning nor plant. Both load the SAME staged URDF -- the planner
collision-only, the sim with visuals (``keep_visual``) -- and it lived in
``planning.mujoco_collision`` until 2026-09-10, so ``plants.mujoco`` imported
the planning package to build its own model. The old name
``build_planning_model`` says planning about a function whose own docstring
offers a sim mode; it stays as an alias for callers that have not moved.
"""
from __future__ import annotations

import hashlib
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def _explicit_geom_names(text: str) -> str:
    """Keep unnamed URDF geoms unnamed in MuJoCo's generated input copy.

    MuJoCo 3.12 can inherit the preceding visual's name for an unnamed
    collision, then warn and discard the duplicate name. An explicit empty
    name avoids that parser path and produces the same compiled geom names.
    Geometry and named geoms remain unchanged; the source URDF is not edited.
    """
    root = ET.fromstring(text)
    changed = False
    for geom in (*root.iter('visual'), *root.iter('collision')):
        if 'name' not in geom.attrib:
            geom.set('name', '')
            changed = True
    return ET.tostring(root, encoding='unicode') if changed else text


def build_mujoco_model(
    urdf_path: str | Path, cache_dir: str | Path, *, keep_visual: bool = False
) -> Path:
    """Write a URDF copy carrying the ``<mujoco>`` compiler extension, cached.

    MuJoCo's URDF loader strips mesh paths to basenames by default, so it needs a
    meshdir. Two cases:

    - all refs resolve into ONE directory -> ``meshdir`` + ``strippath="true"``
      (the DM assembler arm, and how this always worked)
    - refs span several directories AND are all relative -> keep the paths
      (``strippath="false"``) with ``meshdir`` at the URDF's own directory, so
      ``../meshes/{collision,visual}/...`` resolves as written. The FR3 needs
      this: Franka splits collision STLs from visual meshes.

    ``package://`` refs cannot use the second form (MuJoCo does not resolve ROS
    package URIs), so a multi-directory package URDF is still an error.
    The default ``discardvisual`` keeps the plan model collision-only;
    ``keep_visual`` keeps visual geometry for sim/viewer use of the same loader.
    """
    urdf_path = Path(urdf_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "sim" if keep_visual else "planning"
    text = _explicit_geom_names(urdf_path.read_text())
    identity = hashlib.sha256((str(urdf_path) + '\0' + text).encode()).hexdigest()[:20]
    out = cache_dir / f"{urdf_path.stem}_{identity}_mj_{suffix}.urdf"
    if out.exists():
        return out
    package_root = urdf_path.parent.parent
    mesh_refs = set(re.findall(r'filename="([^"]+\.(?:STL|stl|obj|OBJ))"', text))
    mesh_dirs = set()
    for ref in mesh_refs:
        rel = re.sub(r"^package://[^/]+/", "", ref)
        path = (
            (package_root / rel) if ref.startswith("package://") else (urdf_path.parent / rel)
        )
        # Keep the package path here instead of resolving symlinks. Modular
        # packages intentionally symlink common dock plates from one shared
        # directory; MuJoCo can open those links from the package meshdir.
        mesh_dirs.add(path.absolute().parent)
    keep_paths = False
    if len(mesh_dirs) > 1:
        if any(ref.startswith("package://") for ref in mesh_refs):
            raise ValueError(
                f"{urdf_path}: meshes span {sorted(map(str, mesh_dirs))} and use "
                "package:// refs — MuJoCo resolves neither. Stage them into one "
                "directory or rewrite the refs relative to the URDF."
            )
        # Relative refs across several dirs: keep them and anchor meshdir at the
        # URDF, rather than flattening the tree just to satisfy strippath.
        keep_paths = True
    discard = "false" if keep_visual else "true"
    # MuJoCo decodes STL/OBJ/MSH only, so a URDF whose visuals are e.g. COLLADA
    # would fail the compile with "no decoder found for mesh file". Fall back to
    # collision-only geometry — the engine convex-hulls it, which is what the DM
    # arm already relies on. (For the FR3 this no longer triggers:
    # tools/assets/setup_fr3.py converts Franka's .dae visuals to .stl at
    # staging time, because Rerun cannot read COLLADA either.)
    if keep_visual:
        undecodable = sorted(
            {
                Path(ref).suffix.lower()
                for ref in re.findall(r'filename="([^"]+)"', text)
                if Path(ref).suffix and Path(ref).suffix.lower() not in
                (".stl", ".obj", ".msh")
            }
        )
        if undecodable:
            discard = "true"
            print(
                f"[mujoco] {urdf_path.name}: visual meshes are "
                f"{', '.join(undecodable)} which MuJoCo cannot decode — loading "
                "collision geometry only",
                flush=True,
            )
    ext = f'<mujoco><compiler balanceinertia="true" discardvisual="{discard}"'
    if keep_paths:
        ext += f' meshdir="{urdf_path.parent}" strippath="false"'
    elif mesh_dirs:
        ext += f' meshdir="{mesh_dirs.pop()}" strippath="true"'
    ext += "/></mujoco>"
    m = re.search(r"<robot[^>]*>", text)
    if m is None:
        raise ValueError(f"{urdf_path}: no <robot> element")
    # Several planner/viewer processes may stage the same source at startup.
    # Publish only a complete file; exists() must never expose a partial URDF.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=cache_dir, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text[: m.end()] + "\n  " + ext + text[m.end() :])
        temporary.replace(out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return out


# Callers that predate the move. The name is wrong (this is not planning-only)
# but it is load-bearing in mujoco_collision's own history.
build_planning_model = build_mujoco_model

