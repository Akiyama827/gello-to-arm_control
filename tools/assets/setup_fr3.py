#!/usr/bin/env python3
"""Stage the FR3 description (URDF + meshes) into Control/franka/.

The FR3 model is not vendored: Pinocchio needs only the URDF for IK/dynamics,
but the MuJoCo collision world, the sim plant, and the Rerun ghost all need the
link meshes. This script copies them from whichever source is already on the
machine, in preference order:

  1. an explicit ``--source`` directory (a franka_description checkout —
     https://github.com/frankarobotics/franka_description, Apache-2.0)
  2. the Isaac Sim install's FR3 motion-policy assets, if present

It rewrites the URDF's mesh paths to plain relative ``meshes/...`` so the result
is self-contained and does not depend on ROS package resolution.

    python tools/assets/setup_fr3.py                 # auto-detect
    python tools/assets/setup_fr3.py --source ~/franka_description
    python tools/assets/setup_fr3.py --source-urdf model.urdf --dest output
    python tools/assets/setup_fr3.py --check         # report status, change nothing
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Stage into the DEPLOYMENT root (env-overridable seam): configs resolve
# franka/urdf/fr3.urdf against it. Standalone that is this repo; embedded
# it is the project's Control/ (launcher exports ARM_CONTROL_ROOT).
from arm_control import CONTROL_ROOT
DEST = CONTROL_ROOT / "franka"
DEST_URDF = DEST / "urdf" / "fr3.urdf"
DEST_MESHES = DEST / "meshes"

ISAAC_FR3 = (
    Path.home()
    / "miniconda3/envs/env_isaacsim/lib/python3.12/site-packages/isaacsim"
    / "extsDeprecated/isaacsim.robot_motion.motion_generation"
    / "motion_policy_configs/FR3"
)


def find_urdf(source: Path | None) -> Path | None:
    """Locate an fr3 URDF in an explicit source or the known Isaac location."""
    candidates: list[Path] = []
    if source is not None:
        candidates += sorted(source.rglob("fr3*.urdf"))
    if ISAAC_FR3.exists():
        candidates += sorted(ISAAC_FR3.glob("*.urdf"))
    return candidates[0] if candidates else None


def mesh_references(urdf_text: str) -> list[str]:
    """The ``meshes/...`` paths the staged URDF will try to open."""
    refs = re.findall(r'filename="\.\./(meshes/[^"]+)"', urdf_text)
    return sorted(set(refs))


def _score(candidate: Path, want: Path) -> tuple:
    """Rank source files matching a wanted mesh path; higher sorts first.

    Upstream ships several robot families and two hand colours under the same
    basenames, so ``link0.stl`` alone is ambiguous. Prefer a candidate whose
    path mentions the fr3 (not the older fer/fp3), whose collision/visual
    subdirectory matches, and — for the hand — the black shell the FR3 wears.
    """
    parts = {p.lower() for p in candidate.parts}
    return (
        "fr3" in parts,
        want.parent.name in parts,          # collision vs visual
        "franka_hand_black" in parts,
        -len(candidate.parts),              # shallowest wins ties
    )


def stage_meshes(urdf_text: str, source: Path) -> tuple[int, list[str]]:
    """Copy every mesh the URDF references into ``franka/meshes/...``.

    Resolves BY BASENAME instead of assuming an upstream directory layout:
    franka_description splits arm links (``meshes/robots/fr3/``) from the
    end-effector (``meshes/robot_ee/franka_hand_*/``), while the Isaac export
    uses one flat ``meshes/fr3/`` tree. Copying to the exact path the URDF asks
    for makes the staged pair self-consistent whatever the source looked like.
    Returns (copied, unresolved).
    """
    by_name: dict[str, list[Path]] = {}
    for path in source.rglob("*"):
        if path.suffix.lower() in (".stl", ".dae", ".obj") and path.is_file():
            by_name.setdefault(path.name.lower(), []).append(path)

    copied, unresolved = 0, []
    for ref in mesh_references(urdf_text):
        want = Path(ref)
        exact = source / want
        candidates = [exact] if exact.is_file() else by_name.get(want.name.lower(), [])
        if not candidates:
            unresolved.append(ref)
            continue
        best = max(candidates, key=lambda c: _score(c, want))
        dest = DEST / want
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        copied += 1
    return copied, unresolved


def convert_collada(urdf_text: str) -> tuple[str, int, list[str]]:
    """Convert staged ``.dae`` visual meshes to ``.stl`` and repoint the URDF.

    Franka ships its visual meshes as COLLADA and NOTHING downstream reads it:
    Rerun's ``Asset3D`` rejects it ("text/xml files are not supported"), MuJoCo
    has no COLLADA decoder, the teleop page's three.js ``STLLoader`` cannot parse
    it, and open3d does not know the extension. Converting once at staging serves
    every consumer, and it is the only way to render the FINGERS at all — they
    have a visual mesh but no collision STL to fall back to.

    STL specifically, because it is the ONE format all four consumers already
    accept (OBJ fails the teleop page). Losing STL's lack of materials costs
    nothing: geometry colour comes from the URDF's ``meshColor``, which is what
    Rerun's ``albedo_factor`` and the page's per-geom colour already use — the
    same reason the DM assembler arm ships STL.

    Returns (rewritten urdf text, converted count, failures).
    """
    refs = [r for r in mesh_references(urdf_text) if r.lower().endswith(".dae")]
    if not refs:
        return urdf_text, 0, []
    try:
        import trimesh
    except ImportError:
        return urdf_text, 0, ["trimesh is not installed"]

    converted, failures = 0, []
    for ref in refs:
        src = DEST / ref
        if not src.is_file():
            failures.append(f"{ref}: not staged")
            continue
        dst = src.with_suffix(".stl")
        try:
            # force='mesh' concatenates the scene's parts: a COLLADA file is a
            # scene graph, and per-part OBJs would need per-part URDF entries.
            trimesh.load(src, force="mesh").export(dst)
        except Exception as exc:  # noqa: BLE001 - report, do not abort staging
            failures.append(f"{ref}: {type(exc).__name__}: {exc}")
            continue
        src.unlink()  # leaving it invites a stale reference
        urdf_text = urdf_text.replace(ref, ref[: -len(".dae")] + ".stl")
        converted += 1
    return urdf_text, converted, failures


def rewrite_mesh_paths(text: str) -> str:
    """Point every mesh reference at ``meshes/...`` relative to the URDF.

    Handles both the ROS ``package://franka_description/meshes/...`` form and
    the Isaac ``./meshes/fr3/...`` form, so the staged URDF loads with no
    package resolution.
    """
    text = re.sub(r'filename="package://[^/]+/meshes/', 'filename="../meshes/', text)
    text = re.sub(r'filename="\./meshes/', 'filename="../meshes/', text)
    text = re.sub(r'filename="meshes/', 'filename="../meshes/', text)
    return text


def status() -> tuple[bool, str]:
    if not DEST_URDF.exists():
        return False, f"missing URDF: {_display(DEST_URDF)}"
    meshes = [
        m for m in DEST_MESHES.rglob("*")
        if m.suffix.lower() in (".stl", ".obj", ".dae") and m.is_file()
    ]
    if not meshes:
        return False, f"missing meshes: {_display(DEST_MESHES)} is empty"
    return True, f"fr3 assets ready ({len(meshes)} meshes)"


def _display(path: Path) -> Path:
    try:
        return path.relative_to(CONTROL_ROOT)
    except ValueError:
        return path


def main(argv: list[str] | None = None) -> int:
    global DEST, DEST_URDF, DEST_MESHES

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, help="franka_description checkout")
    ap.add_argument("--source-urdf", type=Path, help="exact standalone URDF")
    ap.add_argument("--dest", type=Path, default=DEST, help="output package root")
    ap.add_argument("--check", action="store_true", help="report status only")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite a staged F/T-sensor model with the plain description",
    )
    args = ap.parse_args(argv)
    DEST = args.dest.resolve()
    DEST_URDF = DEST / "urdf" / "fr3.urdf"
    DEST_MESHES = DEST / "meshes"

    if args.check:
        ok, message = status()
        print(("[fr3] " if ok else "[fr3] NOT READY — ") + message)
        return 0 if ok else 1

    # The staged model may be the FR3 + F/T sensor variant (hand-staged from
    # franka_ft_sensor_description; this script cannot regenerate it). Silently
    # replacing it with the plain description would desync sim/planning from the
    # real arm's mass and 37.5 mm TCP offset.
    if not args.force and DEST_URDF.exists() and "fr3_ft_sensor" in DEST_URDF.read_text():
        print(
            f"[fr3] {_display(DEST_URDF)} is the F/T-sensor "
            "variant, which this script cannot restage. Refusing to overwrite "
            "it with the plain FR3; pass --force if you really want that.",
            file=sys.stderr,
        )
        return 1

    urdf = args.source_urdf.resolve() if args.source_urdf else find_urdf(args.source)
    if urdf is not None and not urdf.is_file():
        urdf = None
    if urdf is None:
        print(
            "[fr3] no fr3 URDF found. Clone the description and retry:\n"
            "  git clone https://github.com/frankarobotics/franka_description\n"
            "  python tools/assets/setup_fr3.py --source ./franka_description",
            file=sys.stderr,
        )
        return 1
    print(f"[fr3] urdf source: {urdf}")

    DEST_URDF.parent.mkdir(parents=True, exist_ok=True)
    urdf_text = rewrite_mesh_paths(urdf.read_text())
    DEST_URDF.write_text(urdf_text)
    print(f"[fr3] wrote {_display(DEST_URDF)}")

    # Search the explicit source first, then the URDF's own neighbourhood (the
    # Isaac export keeps its meshes beside it when it has any).
    sources = [s for s in (args.source, urdf.parent, urdf.parent.parent) if s and s.exists()]
    if DEST_MESHES.exists():
        shutil.rmtree(DEST_MESHES)
    copied, unresolved = 0, mesh_references(urdf_text)
    for src in sources:
        copied, unresolved = stage_meshes(urdf_text, src)
        if not unresolved:
            print(f"[fr3] meshes from {src} -> {_display(DEST_MESHES)}")
            break
    if unresolved:
        print(
            f"[fr3] WARNING: {len(unresolved)} mesh reference(s) unresolved "
            f"(e.g. {', '.join(unresolved[:3])}).\n"
            "      Pinocchio IK/dynamics will still work, but the MuJoCo "
            "collision world, the sim plant and the Rerun ghost will not.\n"
            "      Fetch the description: git clone "
            "https://github.com/frankarobotics/franka_description\n"
            "      then: python tools/assets/setup_fr3.py --source ./franka_description",
            file=sys.stderr,
        )
        return 1

    # 许可/版权文件（Apache-2.0）一并带到 DEST：这些资产会随仓库分发，需要保留
    # 上游的 LICENSE / NOTICE。
    for name in ("LICENSE", "NOTICE"):
        for src in sources:
            candidate = src / name
            same = False
            try:
                same = candidate.resolve() == (DEST / name).resolve()
            except OSError:
                same = False
            if candidate.is_file() and not same:
                shutil.copy2(candidate, DEST / name)
                print(f"[fr3] copied {name}")
                break

    # COLLADA -> OBJ, and repoint the URDF at the results. Must run AFTER the
    # meshes are staged (it converts the staged copies in place) and the URDF is
    # rewritten again below with the new extensions.
    urdf_text, converted, failures = convert_collada(urdf_text)
    if converted:
        DEST_URDF.write_text(urdf_text)
        print(f"[fr3] converted {converted} COLLADA visual meshes to .stl")
    for problem in failures:
        print(f"[fr3] WARNING: COLLADA conversion — {problem}", file=sys.stderr)
    if failures:
        print(
            "[fr3]   .dae is unreadable by Rerun, MuJoCo and open3d alike. "
            "Install the reader:  pip install pycollada",
            file=sys.stderr,
        )

    ok, message = status()
    print(f"[fr3] staged {copied} meshes")
    print(f"[fr3] {message}")
    return 0 if ok else 1


def _demo() -> None:
    """Self-check for the path rewriting (the only non-trivial logic here)."""
    ros = '<mesh filename="package://franka_description/meshes/visual/link0.dae"/>'
    isaac = '<mesh filename="./meshes/fr3/collision/link0.stl"/>'
    bare = '<mesh filename="meshes/fr3/visual/finger.dae"/>'
    assert rewrite_mesh_paths(ros) == '<mesh filename="../meshes/visual/link0.dae"/>'
    assert rewrite_mesh_paths(isaac) == '<mesh filename="../meshes/fr3/collision/link0.stl"/>'
    assert rewrite_mesh_paths(bare) == '<mesh filename="../meshes/fr3/visual/finger.dae"/>'
    # Already-relative paths must not be rewritten twice.
    once = rewrite_mesh_paths(isaac)
    assert rewrite_mesh_paths(once) == once, "rewrite is not idempotent"
    # Non-mesh attributes are untouched.
    other = '<xacro:include filename="package://franka_description/robots/x.xacro"/>'
    assert rewrite_mesh_paths(other) == other

    # Reference extraction reads the REWRITTEN paths (what the staged URDF opens).
    text = rewrite_mesh_paths(
        '<mesh filename="./meshes/fr3/collision/link0.stl"/>'
        '<mesh filename="./meshes/fr3/visual/hand.dae"/>'
        '<mesh filename="./meshes/fr3/collision/link0.stl"/>'  # duplicate
    )
    assert mesh_references(text) == [
        "meshes/fr3/collision/link0.stl",
        "meshes/fr3/visual/hand.dae",
    ], mesh_references(text)

    # Source ranking: fr3 over fer, matching collision/visual, black hand.
    want = Path("meshes/fr3/collision/link0.stl")
    fr3 = Path("/d/meshes/robots/fr3/collision/link0.stl")
    fer = Path("/d/meshes/robots/fer/collision/link0.stl")
    visual = Path("/d/meshes/robots/fr3/visual/link0.stl")
    assert max([fer, fr3], key=lambda c: _score(c, want)) is fr3
    assert max([visual, fr3], key=lambda c: _score(c, want)) is fr3
    hand = Path("meshes/fr3/visual/hand.dae")
    black = Path("/d/meshes/robot_ee/franka_hand_black/visual/hand.dae")
    white = Path("/d/meshes/robot_ee/franka_hand_white/visual/hand.dae")
    assert max([white, black], key=lambda c: _score(c, hand)) is black

    # convert_collada is a no-op when there is nothing COLLADA to convert.
    stl_only = rewrite_mesh_paths('<mesh filename="./meshes/fr3/collision/link0.stl"/>')
    assert convert_collada(stl_only) == (stl_only, 0, []), convert_collada(stl_only)

    print("setup_fr3_assets: ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _demo()
    else:
        raise SystemExit(main())
