from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/setup_fr3_assets.py"


def test_explicit_urdf_and_destination_use_exact_mesh_path(tmp_path):
    source = tmp_path / "source"
    exact = source / "meshes/visual/link.stl"
    duplicate = source / "other/link.stl"
    exact.parent.mkdir(parents=True)
    duplicate.parent.mkdir(parents=True)
    exact.write_bytes(b"exact")
    duplicate.write_bytes(b"wrong")
    urdf = source / "urdf/model.urdf"
    urdf.parent.mkdir()
    urdf.write_text(
        '<robot name="x"><link name="base"><visual><geometry>'
        '<mesh filename="../meshes/visual/link.stl"/>'
        "</geometry></visual></link></robot>"
    )
    dest = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--source-urdf",
            str(urdf),
            "--dest",
            str(dest),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (dest / "urdf/fr3.urdf").is_file()
    assert (dest / "meshes/visual/link.stl").read_bytes() == b"exact"
