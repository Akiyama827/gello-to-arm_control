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

    Lives here rather than in ``viz.preview``, where the surviving
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


def asset_fingerprint(urdf_path: str | Path, *, root_link: str | None = None) -> str:
    """Hash an asset, or an explicitly scoped link subtree, plus its meshes.

    The default retains the historical whole-file identity. A tool-local grasp
    can bind to ``root_link`` and its descendants: upstream arm calibration
    does not change that grasp, but tool geometry, internal transforms, limits
    and mesh contents still invalidate it. The subtree is expressed in the
    root link's frame, so the joint attaching it to its parent is excluded.
    """
    urdf = Path(urdf_path).resolve(strict=True)
    root = ET.parse(urdf).getroot()
    if root_link is None:
        label, content = f"urdf/{urdf.name}", urdf.read_bytes()
    else:
        from copy import deepcopy

        links = {link.get("name"): link for link in root.findall("link")}
        if root_link not in links:
            raise ValueError(f"subtree root link absent: {root_link}")
        selected = {root_link}
        joints = []
        pending = list(root.findall("joint"))
        while True:
            children = [j for j in pending if j.find("parent").get("link") in selected]
            if not children:
                break
            for joint in children:
                child = joint.find("child").get("link")
                if child not in links or child in selected:
                    raise ValueError("tool subtree is not a valid link tree")
                selected.add(child)
                joints.append(joint)
                pending.remove(joint)
        subtree = ET.Element("robot", {"name": root_link})
        for link in sorted(selected):
            subtree.append(deepcopy(links[link]))
        for joint in sorted(joints, key=lambda j: j.get("name", "")):
            subtree.append(deepcopy(joint))
        material_names = {m.get("name") for m in subtree.findall(".//material")}
        for material in root.findall("material"):
            if material.get("name") in material_names:
                subtree.append(deepcopy(material))
        root = subtree
        label = f"urdf-subtree-v1/{root_link}"
        content = ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True).encode()
    references = sorted(
        {
            mesh.get("filename")
            for mesh in root.findall(".//mesh")
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

    add(label, content)
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

        mesh = urdf_dir / "tool.stl"
        mesh.write_bytes(b"tool mesh revision one")
        source = """<robot name="arm">
          <link name="base"/><link name="tool"/>
          <link name="tcp"><visual><geometry>
            <mesh filename="tool.stl"/>
          </geometry></visual></link>
          <joint name="arm_joint" type="fixed"><parent link="base"/>
            <child link="tool"/><origin xyz="0 0 1"/>
          </joint>
          <joint name="tcp_joint" type="fixed"><parent link="tool"/>
            <child link="tcp"/><origin xyz="0 0 0.1"/>
          </joint>
        </robot>"""
        urdf.write_text(source)
        whole = asset_fingerprint(urdf)
        scoped = asset_fingerprint(urdf, root_link="tool")
        moved = urdf_dir / "calibrated.urdf"
        moved.write_text(source.replace('xyz="0 0 1"', 'xyz="0 0 1.001"'))
        assert asset_fingerprint(moved) != whole
        assert asset_fingerprint(moved, root_link="tool") == scoped
        moved.write_text(source.replace('xyz="0 0 0.1"', 'xyz="0 0 0.101"'))
        assert asset_fingerprint(moved, root_link="tool") != scoped
        mesh.write_bytes(b"tool mesh revision two")
        assert asset_fingerprint(urdf, root_link="tool") != scoped
        try:
            asset_fingerprint(urdf, root_link="missing")
        except ValueError:
            pass
        else:
            raise AssertionError("missing subtree root accepted")

    # The boundary IS the feature (same argument as execution/factory.py): the
    # web console draws no Rerun, so it must import with the optional [viz]
    # extra absent. Poisoning sys.modules makes any `import rerun` raise, so
    # the import SUCCEEDING is the proof that none was needed.
    probe = (
        "import sys, importlib;"
        "sys.modules['rerun'] = None;"
        "importlib.import_module('arm_control.grasp_visual');"
        "assert 'arm_control.viz.preview' not in sys.modules;"
        "print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0 and "clean" in out.stdout, (
        "grasp_visual must import with rerun absent -- it reached through "
        "viz.preview for a path helper:\n" + out.stdout + out.stderr
    )
    print("assets self-check ok")


if __name__ == "__main__":
    _self_check()
