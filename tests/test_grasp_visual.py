from pathlib import Path
import sys

import numpy as np
import pytest

from arm_control.grasp_visual import FixedHalfspace, FixedMesh, FixedUrdf, GraspVisual
from arm_control.planning.preview_rerun import _mesh_package_dirs

sys.path.append(str(Path(__file__).resolve().parents[1] / "nodes"))
from motion_teleop import VisualFK  # noqa: E402


_STL = """solid triangle
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 0.01 0 0
    vertex 0 0.01 0
  endloop
endfacet
endsolid triangle
"""


def _link(name, mesh, collision=""):
    return f"""<link name="{name}">
  <visual><geometry><mesh filename="{mesh.name}"/></geometry></visual>
  {collision}
</link>"""


def _fixed(name, parent, child, xyz="0 0 0"):
    return f"""<joint name="{name}" type="fixed">
  <parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/>
</joint>"""


def _assets(tmp_path: Path):
    mesh = tmp_path / "triangle.stl"
    mesh.write_text(_STL)
    box = '<collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision>'
    module = tmp_path / "module.urdf"
    module.write_text(
        f'<robot name="module">{_link("Passive", mesh, box)}</robot>'
    )
    fixture = tmp_path / "fixture.urdf"
    fixture.write_text(
        f'<robot name="fixture">{_link("Holder", mesh, box)}</robot>'
    )
    tool = tmp_path / "tool.urdf"
    links = "".join(
        [
            _link("arm", mesh),
            _link("tool_root", mesh),
            _link("sensor", mesh),
            _link(
                "hand",
                mesh,
                '<collision><origin xyz="-0.05 0 0"/>'
                '<geometry><box size="0.02 0.02 0.02"/></geometry></collision>',
            ),
            _link("tcp", mesh),
            _link("left_finger", mesh, box),
            _link("right_finger", mesh, box),
            _link("camera_holder", mesh),
            _link("camera", mesh),
        ]
    )
    joints = "".join(
        [
            _fixed("mount", "arm", "tool_root", "0.1 0 0"),
            _fixed("sensor_fixed", "tool_root", "sensor"),
            _fixed("hand_fixed", "sensor", "hand"),
            _fixed("tcp_fixed", "hand", "tcp"),
            """<joint name="left_joint" type="prismatic">
  <parent link="hand"/><child link="left_finger"/>
  <origin xyz="0.03 0 0"/><axis xyz="0 1 0"/>
  <limit lower="0" upper="0.04" effort="1" velocity="1"/>
</joint>""",
            """<joint name="right_joint" type="prismatic">
  <parent link="hand"/><child link="right_finger"/>
  <origin xyz="0.03 0 0"/><axis xyz="0 -1 0"/>
  <limit lower="0" upper="0.04" effort="1" velocity="1"/>
</joint>""",
            _fixed("camera_mount", "hand", "camera_holder"),
            _fixed("camera_fixed", "camera_holder", "camera"),
        ]
    )
    tool.write_text(f'<robot name="tool">{links}{joints}</robot>')
    return module, fixture, tool


def _translation(x=0.0, y=0.0, z=0.0):
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


def _scene(tmp_path, *, halfspaces=()):
    module, fixture, tool = _assets(tmp_path)
    return GraspVisual(
        module=FixedUrdf("module", module, _translation(0.03)),
        fixture=FixedUrdf("fixture", fixture, _translation(1.0)),
        tool_urdf=tool,
        tool_root_link="tool_root",
        ee_frame="tcp",
        finger_joints=("left_joint", "right_joint"),
        intended_tool_links=frozenset({"left_finger", "right_finger"}),
        halfspaces=halfspaces,
    )


def test_tool_subtree_and_geometry_are_expressed_from_tcp(tmp_path):
    scene = _scene(tmp_path)

    assert set(scene.tool_links) == {
        "tool_root",
        "sensor",
        "hand",
        "tcp",
        "left_finger",
        "right_finger",
        "camera_holder",
        "camera",
    }
    assert "arm" not in scene.tool_links
    np.testing.assert_allclose(scene.frame_T("tool", "tcp"), np.eye(4))
    assert {item.group for item in scene.visuals()} == {
        "fixture",
        "module",
        "tool",
    }


def test_finger_width_moves_both_real_joint_frames(tmp_path):
    scene = _scene(tmp_path)

    scene.set_finger_width(0.04)

    assert scene.frame_T("tool", "left_finger")[1, 3] == 0.02
    assert scene.frame_T("tool", "right_finger")[1, 3] == -0.02


def test_contacts_distinguish_fingers_from_forbidden_tool_parts(tmp_path):
    scene = _scene(tmp_path)

    assert scene.contacts(np.eye(4)) == {
        "intended_tool_links": ["left_finger", "right_finger"],
        "forbidden_tool_links": [],
        "ok": True,
    }
    hand_contact = scene.contacts(_translation(0.08))
    assert hand_contact["ok"] is False
    assert "hand" in hand_contact["forbidden_tool_links"]
    fixture_contact = scene.contacts(_translation(0.97))
    assert fixture_contact["ok"] is False
    assert fixture_contact["forbidden_tool_links"]


def test_contacts_include_fixed_halfspace(tmp_path):
    scene = _scene(
        tmp_path,
        halfspaces=(FixedHalfspace("bench", (0.0, 0.0, 1.0), 0.0),),
    )

    report = scene.contacts(_translation(z=-0.02))

    assert report["ok"] is False
    assert "hand" in report["forbidden_tool_links"]


def test_context_models_and_meshes_keep_world_pose_and_emphasis(tmp_path):
    module, fixture, tool = _assets(tmp_path)
    mesh = tmp_path / "triangle.stl"
    scene = GraspVisual(
        module=FixedUrdf("module", module, _translation(0.03), emphasized=True),
        fixture=FixedUrdf("fixture", fixture, _translation(1.0), emphasized=True),
        tool_urdf=tool,
        tool_root_link="tool_root",
        ee_frame="tcp",
        finger_joints=("left_joint", "right_joint"),
        intended_tool_links=frozenset({"left_finger", "right_finger"}),
        context=(FixedUrdf("storage_2", fixture, _translation(0.5)),),
        meshes=(FixedMesh("bench", mesh, _translation(z=-0.02)),),
    )

    groups = {item.group for item in scene.visuals()}
    assert {"module", "fixture", "storage_2", "bench", "tool"} <= groups
    assert all(
        item.emphasized
        for item in scene.visuals()
        if item.group in {"module", "fixture"}
    )
    assert all(
        not item.emphasized
        for item in scene.visuals()
        if item.group in {"storage_2", "bench"}
    )


def test_fixed_urdf_rejects_unknown_declared_joint(tmp_path):
    module, fixture, tool = _assets(tmp_path)

    with pytest.raises(ValueError, match="unknown fixed joint: missing"):
        GraspVisual(
            module=FixedUrdf(
                "module",
                module,
                _translation(0.03),
                joints=(("missing", 0.0),),
            ),
            fixture=FixedUrdf("fixture", fixture, _translation(1.0)),
            tool_urdf=tool,
            tool_root_link="tool_root",
            ee_frame="tcp",
            finger_joints=("left_joint", "right_joint"),
            intended_tool_links=frozenset({"left_finger", "right_finger"}),
        )


def test_fixed_mesh_rejects_directory_path(tmp_path):
    module, fixture, tool = _assets(tmp_path)

    with pytest.raises(ValueError, match="dir_mesh.mesh must be a file"):
        GraspVisual(
            module=FixedUrdf("module", module, _translation(0.03)),
            fixture=FixedUrdf("fixture", fixture, _translation(1.0)),
            tool_urdf=tool,
            tool_root_link="tool_root",
            ee_frame="tcp",
            finger_joints=("left_joint", "right_joint"),
            intended_tool_links=frozenset({"left_finger", "right_finger"}),
            meshes=(FixedMesh("dir_mesh", tmp_path, _translation()),),
        )


def test_visual_fk_applies_declared_world_mount(tmp_path):
    _module, _fixture, tool = _assets(tmp_path)
    visual = VisualFK(
        tool,
        ["left_joint"],
        "tcp",
        ["right_joint"],
        world_T_root=_translation(1.0, 2.0, 3.0),
    )

    result = visual.poses([0.0], 0.0)

    assert result["geoms"]
    np.testing.assert_allclose(result["geoms"][0]["p"], [1.0, 2.0, 3.0])
    assert visual.scene_json("arm-mesh")[0]["mesh"].startswith("arm-mesh/")


def test_package_search_includes_parent_without_package_manifest(tmp_path):
    urdf_dir = tmp_path / "Storage" / "urdf"
    urdf_dir.mkdir(parents=True)
    urdf = urdf_dir / "Storage.urdf"
    urdf.write_text('<robot name="storage"><link name="root"/></robot>')

    assert str(tmp_path) in _mesh_package_dirs(urdf)
