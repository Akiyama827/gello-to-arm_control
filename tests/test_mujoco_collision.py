import numpy as np
import pytest

from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.scene import load_scene


def test_scene_fixture_contact_is_environment_collision(tmp_path):
    (tmp_path / "actor.xml").write_text(
        "<mujoco><worldbody><body name='arm'>"
        "<joint name='axis' type='slide' axis='1 0 0' range='-1 1'/>"
        "<geom name='tip' type='sphere' size='.01' mass='.1'/>"
        "</body></worldbody></mujoco>"
    )
    (tmp_path / "fixture.xml").write_text(
        "<mujoco><worldbody><body name='holder' pos='0 0 .019'>"
        "<geom type='box' size='.01 .01 .01' mass='.1'/>"
        "</body></worldbody></mujoco>"
    )
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(
        """version: 1
scene:
  actors:
    arm: {path: actor.xml, joints: [axis]}
  fixtures:
    storage_2: {path: fixture.xml}
"""
    )
    scene = load_scene(scene_path)
    world = MuJoCoCollisionWorld.from_scene(
        scene,
        scene.state(),
        "arm",
        self_collision_padding_m=-0.002,
        ground_z=None,
    )

    assert world.in_collision(np.array([0.0]))


def test_held_positions_update_only_non_planned_qpos(tmp_path):
    (tmp_path / "robot.urdf").write_text(
        """<robot name="r">
<link name="base"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="a"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><box size="0.01 0.01 0.01"/></geometry></collision></link>
<link name="b"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><box size="0.01 0.01 0.01"/></geometry></collision></link>
<joint name="axis" type="prismatic"><parent link="base"/><child link="a"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
<joint name="finger" type="prismatic"><parent link="a"/><child link="b"/><axis xyz="0 1 0"/><limit lower="0" upper="0.04" effort="1" velocity="1"/></joint>
</robot>"""
    )
    world = MuJoCoCollisionWorld(
        tmp_path / "robot.urdf",
        ["axis"],
        cache_dir=tmp_path / "cache",
    )
    planned_addr = int(world.model.joint("axis").qposadr[0])
    held_addr = int(world.model.joint("finger").qposadr[0])

    world.set_held_positions({"finger": 0.02})

    assert world._held_qpos[held_addr] == 0.02
    assert world._held_qpos[planned_addr] == 0.0
    with pytest.raises(ValueError, match="planned joint"):
        world.set_held_positions({"axis": 0.1})


def test_scene_held_positions_preserve_non_planned_actor_q(tmp_path):
    (tmp_path / "arm.xml").write_text(
        "<mujoco><worldbody><body name='arm'>"
        "<joint name='axis' type='slide' axis='1 0 0' range='-1 1'/>"
        "<geom type='sphere' size='.01' mass='.1'/>"
        "</body></worldbody></mujoco>"
    )
    (tmp_path / "hand.xml").write_text(
        "<mujoco><worldbody><body name='hand'>"
        "<joint name='finger' type='slide' axis='1 0 0' range='0 .04'/>"
        "<geom type='sphere' size='.01' mass='.1'/>"
        "</body></worldbody></mujoco>"
    )
    scene_path = tmp_path / "scene.yaml"
    scene_path.write_text(
        """version: 1
scene:
  actors:
    arm: {path: arm.xml, joints: [axis], q: [0.1]}
    hand: {path: hand.xml, joints: [finger], q: [0.03]}
"""
    )
    scene = load_scene(scene_path)
    world = MuJoCoCollisionWorld.from_scene(
        scene,
        scene.state(),
        "arm",
        ground_z=None,
    )
    planned_addr = int(world.model.joint("arm__axis").qposadr[0])
    held_addr = int(world.model.joint("hand__finger").qposadr[0])

    assert world._held_qpos[held_addr] == 0.03
    before = world._held_qpos[planned_addr]

    world.set_held_positions({"hand__finger": 0.02})

    assert world._held_qpos[held_addr] == 0.02
    assert world._held_qpos[planned_addr] == before
