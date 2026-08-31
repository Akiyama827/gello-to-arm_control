from pathlib import Path

import numpy as np
import pytest

from arm_control.calibration_console import (
    ActorCatalog,
    CalibrationStore,
    ConsoleAuthority,
    validate_grasp_profile,
)
from arm_control.assets import asset_fingerprint
from nodes.motion_teleop import console_asset


def test_deadman_expiry_requests_hold_then_disarm():
    authority = ConsoleAuthority(("assembler", "base"), deadman_timeout_s=0.25)
    authority.select("assembler", now=10.0)
    authority.set_deadman(True, now=10.0)

    assert authority.may_move("assembler", now=10.2)
    assert authority.expire(now=10.3) == [
        ("hold", "assembler"),
        ("disarm", "assembler"),
    ]
    assert not authority.may_move("assembler", now=10.3)
    assert authority.expire(now=10.4) == []


def test_actor_switch_holds_and_disarms_previous_actor():
    authority = ConsoleAuthority(("assembler", "base"), deadman_timeout_s=0.25)
    authority.select("assembler", now=1.0)

    assert authority.select("base", now=2.0) == [
        ("hold", "assembler"),
        ("disarm", "assembler"),
    ]


def test_calibration_store_backs_up_and_rejects_outside_root(tmp_path):
    target = tmp_path / "row.yaml"
    target.write_text("version: 1\nvalue: old\n")
    store = CalibrationStore((tmp_path,))

    store.save_yaml(
        target,
        {"version": 1, "value": "new"},
        expected_revision=store.revision(target),
    )

    assert "value: new" in target.read_text()
    assert len(list(tmp_path.glob("row.yaml.*.bak"))) == 1
    with pytest.raises(ValueError, match="allowed roots"):
        store.save_yaml(
            Path(tmp_path).parent / "escape.yaml",
            {"version": 1},
            expected_revision="missing",
        )


def test_stale_revision_does_not_change_file(tmp_path):
    target = tmp_path / "profile.yaml"
    target.write_text("value: current\n")
    before = target.read_bytes()

    with pytest.raises(ValueError, match="stale revision"):
        CalibrationStore((tmp_path,)).save_yaml(
            target,
            {"value": "wrong"},
            expected_revision="0" * 64,
        )

    assert target.read_bytes() == before
    assert list(tmp_path.glob("*.bak")) == []


def test_actor_catalog_routes_without_robot_name_branches():
    catalog = ActorCatalog.from_mapping(
        {
            "assembler": {
                "joints": ["j1", "j2"],
                "capabilities": ["cartesian", "gripper", "wrench"],
                "urdf": "robot.urdf",
                "ee_frame": "tool",
            },
            "base": {
                "joints": ["b1", "b2"],
                "capabilities": ["joint"],
            },
        }
    )

    assert catalog["assembler"].command_port == "assembler_command"
    assert catalog["base"].arm_port == "base_arm"
    assert catalog["base"].supports("joint")
    assert not catalog["base"].supports("cartesian")


def test_cartesian_actor_requires_kinematic_model():
    with pytest.raises(ValueError, match="urdf and ee_frame"):
        ActorCatalog.from_mapping(
            {"arm": {"joints": ["j1"], "capabilities": ["cartesian"]}}
        )


def test_console_page_is_offline_and_has_functional_sections():
    content_type, body, _cache = console_asset("")
    html = body.decode()

    assert content_type == "text/html; charset=utf-8"
    assert "http://" not in html and "https://" not in html
    assert "/static/vendor/three.module.js" in html
    for ident in (
        "control-panel",
        "telemetry-panel",
        "calibration-panel",
        "scene-panel",
        "deadman",
    ):
        assert f'id="{ident}"' in html
    assert console_asset("static/vendor/three.module.js")[0] == "text/javascript"


def _grasp_profile_assets(tmp_path):
    module_urdf = tmp_path / "part.urdf"
    module_urdf.write_text("<robot name='part'><link name='Passive'/></robot>")
    tool_urdf = tmp_path / "tool.urdf"
    tool_urdf.write_text(
        """<robot name="tool">
  <link name="upstream"/>
  <link name="tool_root"/>
  <link name="tcp"/>
  <link name="left_finger"/>
  <link name="right_finger"/>
  <link name="other"/>
  <joint name="mount" type="fixed">
    <parent link="upstream"/><child link="tool_root"/>
  </joint>
  <joint name="tcp_fixed" type="fixed">
    <parent link="tool_root"/><child link="tcp"/>
  </joint>
  <joint name="left_joint" type="prismatic">
    <parent link="tool_root"/><child link="left_finger"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
  <joint name="right_joint" type="prismatic">
    <parent link="tool_root"/><child link="right_finger"/>
    <axis xyz="0 -1 0"/><limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
</robot>"""
    )
    raw = {
        "version": 2,
        "module_type": "Part",
        "calibration_status": "draft",
        "source_sha256": asset_fingerprint(module_urdf),
        "grasp": {
            "reference_link": "Passive",
            "ee_frame": "tcp",
            "tool_root_link": "tool_root",
            "tool_source_sha256": asset_fingerprint(tool_urdf),
            "finger_width_m": 0.04,
            "link_T_ee": {"pos": [0, 0, 0], "quat": [2, 0, 0, 0]},
            "approach_offset_m": [0, 0, 0.1],
            "retreat_offset_m": [0, 0, -0.1],
        },
    }
    return module_urdf, tool_urdf, raw


def _validate_grasp(raw, module_urdf, tool_urdf):
    return validate_grasp_profile(
        raw,
        module_urdf=module_urdf,
        tool_urdf=tool_urdf,
        finger_joints=("left_joint", "right_joint"),
    )


def test_grasp_profile_is_normalized_and_bound_to_both_assets(tmp_path):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)

    grasp = _validate_grasp(raw, module_urdf, tool_urdf)

    assert grasp.reference_link == "Passive"
    assert grasp.ee_frame == "tcp"
    assert grasp.tool_root_link == "tool_root"
    assert grasp.finger_width_m == 0.04
    np.testing.assert_allclose(grasp.link_T_ee, np.eye(4))


def test_grasp_profile_rejects_stale_tool_asset(tmp_path):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)
    raw["grasp"]["tool_source_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="tool fingerprint"):
        _validate_grasp(raw, module_urdf, tool_urdf)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ee_frame", "missing", "ee_frame"),
        ("tool_root_link", "missing", "tool_root_link"),
        ("tool_root_link", "other", "ancestor"),
    ],
)
def test_grasp_profile_rejects_invalid_tool_frames(
    tmp_path, field, value, message
):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)
    raw["grasp"][field] = value

    with pytest.raises(ValueError, match=message):
        _validate_grasp(raw, module_urdf, tool_urdf)


def test_grasp_profile_rejects_width_beyond_joint_limits(tmp_path):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)
    raw["grasp"]["finger_width_m"] = 0.081

    with pytest.raises(ValueError, match="finger_width_m"):
        _validate_grasp(raw, module_urdf, tool_urdf)


def test_grasp_profile_rejects_version_one_and_stale_module_asset(tmp_path):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)
    raw["version"] = 1

    with pytest.raises(ValueError, match="version must be 2"):
        _validate_grasp(raw, module_urdf, tool_urdf)

    raw["version"] = 2
    raw["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_grasp(raw, module_urdf, tool_urdf)
