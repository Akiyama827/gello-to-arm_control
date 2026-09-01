import struct

import mujoco
import pytest

from arm_control.scene import load_scene
from arm_control.simulation.mujoco_backend import compose_workcell_scene


def _write_tetrahedron(path):
    vertices = ((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1))
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    data = bytearray(80) + struct.pack("<I", len(faces))
    for face in faces:
        coords = [value for i in face for value in vertices[i]]
        data += struct.pack("<12fH", 0, 0, 0, *coords, 0)
    path.write_bytes(data)


def test_mesh_obstacle_loads_and_compiles(tmp_path):
    (tmp_path / "actor.xml").write_text(
        "<mujoco><worldbody><body name='arm'><joint name='axis'/><geom size='.01' mass='.1'/></body></worldbody></mujoco>"
    )
    _write_tetrahedron(tmp_path / "bench.stl")
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(
        """version: 1
scene:
  actors:
    arm: {path: actor.xml, joints: [axis]}
  obstacles:
    bench: {shape: mesh, path: bench.stl, pos: [1, 2, 3], rpy: [0, 0, 1.57]}
"""
    )

    scene = load_scene(scene_path)
    model = compose_workcell_scene(scene, scene.state(), ground_z=None).compile()

    assert scene.obstacles[0].path == (tmp_path / "bench.stl").resolve()
    assert model.geom("obstacle__bench").type == mujoco.mjtGeom.mjGEOM_MESH


def test_fixed_model_loads_once_without_runtime_joints(tmp_path):
    (tmp_path / "actor.xml").write_text(
        "<mujoco><worldbody><body name='arm'><joint name='axis' type='hinge'/>"
        "<geom type='sphere' size='.01' mass='.1'/></body></worldbody></mujoco>"
    )
    (tmp_path / "fixture.xml").write_text(
        "<mujoco><worldbody><body name='holder' pos='0 0 .02'>"
        "<geom type='box' size='.01 .02 .03' mass='.1'/></body></worldbody></mujoco>"
    )
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(
        """version: 1
scene:
  actors:
    arm: {path: actor.xml, joints: [axis]}
  fixtures:
    storage_2:
      path: fixture.xml
      pos: [1.0, 2.0, 3.0]
      rpy: [0.0, 0.0, 0.0]
"""
    )

    scene = load_scene(scene_path)
    model = compose_workcell_scene(
        scene,
        scene.state(),
        ground_z=None,
    ).compile()

    assert scene.fixtures[0].name == "storage_2"
    assert model.body("storage_2__holder").jntnum == 0
    assert model.body("storage_2__holder").pos.tolist() == [0.0, 0.0, 0.02]


def test_fixed_model_rejects_runtime_joints(tmp_path):
    (tmp_path / "actor.xml").write_text(
        "<mujoco><worldbody><body name='arm'><joint name='axis' type='hinge'/>"
        "<geom type='sphere' size='.01' mass='.1'/></body></worldbody></mujoco>"
    )
    (tmp_path / "fixture.xml").write_text(
        "<mujoco><worldbody><body name='holder'><joint name='hinge' type='hinge'/>"
        "<geom type='box' size='.01 .02 .03' mass='.1'/></body></worldbody></mujoco>"
    )
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(
        """version: 1
scene:
  actors:
    arm: {path: actor.xml, joints: [axis]}
  fixtures:
    storage_2:
      path: fixture.xml
      pos: [1.0, 2.0, 3.0]
      rpy: [0.0, 0.0, 0.0]
"""
    )

    scene = load_scene(scene_path)
    with pytest.raises(ValueError, match="fixed model storage_2 must have zero DoF"):
        compose_workcell_scene(scene, scene.state(), ground_z=None)
