from arm_control.planning.mujoco_collision import build_planning_model
from arm_control.simulation.mujoco_backend import _load_model_spec


def test_package_mesh_symlinks_share_one_syntactic_meshdir(tmp_path):
    package = tmp_path / "Widget"
    meshes = package / "meshes"
    urdf_dir = package / "urdf"
    shared = tmp_path / "shared"
    meshes.mkdir(parents=True)
    urdf_dir.mkdir()
    shared.mkdir()
    for name in ("a.STL", "b.STL"):
        target = shared / name
        target.write_bytes(b"unused by this conversion test")
        (meshes / name).symlink_to(target)
    urdf = urdf_dir / "Widget.urdf"
    urdf.write_text(
        '<robot name="Widget"><link name="root"><visual><geometry>'
        '<mesh filename="package://Widget/meshes/a.STL"/>'
        '</geometry></visual><collision><geometry>'
        '<mesh filename="package://Widget/meshes/b.STL"/>'
        "</geometry></collision></link></robot>"
    )

    generated = build_planning_model(urdf, tmp_path / "cache", keep_visual=True)

    assert f'meshdir="{meshes}"' in generated.read_text()


def test_model_loader_accepts_an_external_cache_directory(tmp_path):
    urdf = tmp_path / "plain.urdf"
    urdf.write_text(
        '<robot name="plain"><link name="root"><inertial>'
        '<origin xyz="0 0 0"/><mass value="1"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>'
        "</inertial></link></robot>"
    )
    cache = tmp_path / "elsewhere"

    spec = _load_model_spec(urdf, cache_dir=cache)

    assert spec is not None
    assert (cache / "plain_mj_sim.urdf").is_file()
