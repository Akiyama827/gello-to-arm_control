import numpy as np

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
