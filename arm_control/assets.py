"""Portable source-asset identity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET


def mesh_package_dirs(urdf_path: str | Path) -> list[str]:
    """Search path for Pinocchio's URDF parser to resolve mesh references.

    Hints at the parent ROS-package layout our generated combined URDFs use:
    ``<root>/models/urdf`` with sibling ``meshes``/``meshes_collision`` one
    level up. Extra entries only add fallbacks -- first match wins, and the
    order here puts the URDF's own directory first.

    There were two of these. ``nodes/visualizer`` additionally probed for a
    ``package.xml`` to pick a package root, which read as the more careful
    version but was dead: every directory that branch could select was already
    in the list, so its candidate set was a strict SUBSET of this one -- it
    just dropped the grandparent whenever no ``package.xml`` was present. This
    is the union, which is to say the simpler one.

    Lives here rather than in ``planning.preview_rerun``, where the surviving
    copy was, because that module imports ``rerun`` at module scope and
    ``rerun-sdk`` is an optional ``[viz]`` extra: reaching in here for a path
    helper made ``grasp_visual`` -- and so the whole web console, which draws
    no Rerun at all -- unimportable on a headless install.
    """
    urdf_dir = Path(urdf_path).resolve().parent
    package_root = urdf_dir.parent
    candidates = [urdf_dir, package_root, package_root.parent]
    return [str(path) for path in candidates if path.is_dir()]


def _resolve_mesh(urdf: Path, reference: str) -> Path:
    if reference.startswith("package://"):
        package_ref = reference.removeprefix("package://")
        package, separator, relative = package_ref.partition("/")
        if not separator or package != urdf.parent.parent.name:
            raise ValueError(f"unsupported package mesh reference: {reference}")
        path = urdf.parent.parent / relative
    else:
        path = Path(reference)
        if path.is_absolute():
            raise ValueError(f"absolute mesh reference is not portable: {reference}")
        path = urdf.parent / path
        if not path.exists():
            path = urdf.parent.parent / reference
    if not path.is_file():
        raise FileNotFoundError(f"URDF mesh not found: {reference}")
    return path


def asset_fingerprint(urdf_path: str | Path) -> str:
    """Hash a URDF and every referenced mesh, independent of its root path."""
    urdf = Path(urdf_path).resolve(strict=True)
    references = sorted(
        {
            mesh.get("filename")
            for mesh in ET.parse(urdf).getroot().findall(".//mesh")
            if mesh.get("filename")
        }
    )
    digest = hashlib.sha256()

    def add(label: str, data: bytes) -> None:
        encoded = label.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

    add(f"urdf/{urdf.name}", urdf.read_bytes())
    for reference in references:
        add(reference, _resolve_mesh(urdf, reference).read_bytes())
    return digest.hexdigest()


def _self_check() -> None:
    """The search path, and the import boundary that made it live here."""
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        urdf_dir = root / "pkg" / "models" / "urdf"
        urdf_dir.mkdir(parents=True)
        urdf = urdf_dir / "arm.urdf"
        urdf.write_text("<robot name='x'/>")
        dirs = mesh_package_dirs(urdf)
        # The URDF's own directory first, then out two levels -- which is where
        # the sibling ``meshes`` of a generated combined URDF actually sits.
        assert dirs == [str(urdf_dir), str(urdf_dir.parent), str(urdf_dir.parent.parent)], dirs
        # A ``package.xml`` changes nothing: the branch that used to look for
        # one could only ever select a directory already in this list.
        (urdf_dir.parent / "package.xml").write_text("<package/>")
        assert mesh_package_dirs(urdf) == dirs, mesh_package_dirs(urdf)

    # The boundary IS the feature (same argument as execution/factory.py): the
    # web console draws no Rerun, so it must import with the optional [viz]
    # extra absent. Poisoning sys.modules makes any `import rerun` raise, so
    # the import SUCCEEDING is the proof that none was needed.
    probe = (
        "import sys, importlib;"
        "sys.modules['rerun'] = None;"
        "importlib.import_module('arm_control.grasp_visual');"
        "assert 'arm_control.planning.preview_rerun' not in sys.modules;"
        "print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0 and "clean" in out.stdout, (
        "grasp_visual must import with rerun absent -- it reached through "
        "planning.preview_rerun for a path helper:\n" + out.stdout + out.stderr
    )
    print("assets self-check ok")


if __name__ == "__main__":
    _self_check()
