"""Portable source-asset identity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET


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
