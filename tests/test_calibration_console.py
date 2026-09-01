from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from arm_control.calibration_console import (
    ActorCatalog,
    CalibrationStore,
    ConsoleAuthority,
    validate_grasp_profile,
)
from arm_control.assets import asset_fingerprint
from arm_control.grasp_visual import VisualGeometry

sys.path.append(str(Path(__file__).resolve().parents[1]))

import nodes.calibration_console as calibration_console_node
from nodes.calibration_console import (
    GraspCheck,
    GraspEditorPanel,
    GraspEditorWorkspace,
)
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


def test_grasp_page_uses_real_geometry_and_finger_control():
    app = console_asset("static/app.js")[1].decode()
    html = console_asset("")[1].decode()

    assert "new THREE.BoxGeometry" not in app
    assert 'id="finger-opening"' in html
    assert "forbidden_tool_links" in app
    assert '$("plan-summary").textContent = "Editor collision preview only"' not in app
    assert "editorPregraspTool.group.visible = !samePose" in app
    assert "editorRetreatTool.group.visible = !samePose" in app
    assert app.count('$("grip-readout").value = `${value.toFixed(3)} m`') == 2
    for ident in (
        "module-select",
        "storage-select",
        "context-select",
        "show-arm",
        "pose-x",
        "pose-y",
        "pose-z",
        "pose-roll",
        "pose-pitch",
        "pose-yaw",
        "storage-ik",
        "dock-ik",
    ):
        assert f'id="{ident}"' in html
    assert "setTimeout(runEditorChecks, 150)" in app
    assert "result.edit_revision !== editorEditRevision" in app
    assert 'new THREE.Euler(roll, pitch, yaw, "ZYX")' in app
    assert "for (const name of editorContexts)" in app
    assert 'for (const name of ["storage", "dock"])' not in app


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


class _FakeGraspVisual:
    def __init__(self, module_urdf, tool_urdf, mesh):
        self.module_urdf = module_urdf
        self.tool_urdf = tool_urdf
        self.finger_joints = ("left_joint", "right_joint")
        self.finger_width_limits = (0.0, 0.08)
        self.finger_width_m = 0.04
        self.mesh = mesh

    def visuals(self):
        return tuple(
            VisualGeometry(
                group,
                "Passive" if group == "module" else "part",
                self.mesh,
                (0.5, 0.5, 0.5, 1.0),
                (1.0, 1.0, 1.0),
                np.eye(4),
            )
            for group in ("fixture", "module", "tool")
        )

    def frame_T(self, group, frame):
        assert (group, frame) in {("module", "Passive"), ("tool", "tcp")}
        return np.eye(4)

    def set_finger_width(self, width_m):
        self.finger_width_m = float(width_m)

    def contacts(self, _world_T_tcp):
        return {
            "intended_tool_links": [],
            "forbidden_tool_links": [],
            "ok": True,
        }


class _FakeArmVisual:
    def __init__(self, mesh):
        self.mesh = mesh

    def scene_json(self, mesh_prefix="mesh"):
        return [
            {
                "mesh": f"{mesh_prefix}/0",
                "color": [0.2, 0.3, 0.4, 1.0],
                "scale": [1.0, 1.0, 1.0],
            }
        ]

    def mesh_path(self, index):
        return self.mesh if index == 0 else None

    def poses(self, q, finger_m):
        return {
            "geoms": [
                {
                    "p": [float(q[0]), finger_m, 0.0],
                    "q": [0.0, 0.0, 0.0, 1.0],
                }
            ]
        }


class _NamedGraspVisual(_FakeGraspVisual):
    def __init__(self, name, module_urdf, tool_urdf, mesh):
        super().__init__(module_urdf, tool_urdf, mesh)
        self.name = name
        self.trigger = None

    def visuals(self):
        return tuple(
            VisualGeometry(
                group,
                self.name,
                self.mesh,
                (0.5, 0.5, 0.5, 1.0),
                (1.0, 1.0, 1.0),
                np.eye(4),
            )
            for group in ("fixture", "module", "tool")
        )

    def frame_T(self, group, frame):
        if self.trigger is not None:
            trigger = self.trigger
            self.trigger = None
            trigger()
        return super().frame_T(group, frame)


class _FakeServer:
    def __init__(self, address, _handler):
        self.server_address = (address[0], 12345)

    def serve_forever(self):
        return None

    def shutdown(self):
        return None

    def server_close(self):
        return None


def test_grasp_panel_exposes_context_and_three_pose_reports(tmp_path, monkeypatch):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)
    profile = tmp_path / "Part.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False))
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"solid part\nendsolid part\n")
    visual = _FakeGraspVisual(module_urdf, tool_urdf, mesh)
    monkeypatch.setattr(calibration_console_node, "ThreadingHTTPServer", _FakeServer)
    panel = GraspEditorPanel(profile, visual, bind="127.0.0.1", port=0)
    try:
        scene = panel.scene()
        assert set(scene) >= {"fixture", "module", "tool", "frames"}
        editor = panel.update(
            {
                "pos": [0.01, 0.02, 0.03],
                "quat": [1, 0, 0, 0],
                "finger_width_m": 0.03,
                "approach_offset_m": [0, 0, 0.1],
                "retreat_offset_m": [0, 0, -0.1],
            }
        )
        assert set(editor["contacts"]) == {
            "pregrasp",
            "grasp",
            "retreat",
        }
        assert visual.finger_width_m == 0.03
    finally:
        panel.close()


def test_grasp_panel_workspace_selection_stale_checks_and_save(tmp_path, monkeypatch):
    module_urdf, tool_urdf, raw_a = _grasp_profile_assets(tmp_path)
    raw_b = yaml.safe_load(yaml.safe_dump(raw_a, sort_keys=False))
    raw_a["module_type"] = "part_a"
    raw_a["calibration_status"] = "approved"
    raw_b["module_type"] = "part_b"
    profile_a = tmp_path / "part_a.yaml"
    profile_b = tmp_path / "part_b.yaml"
    profile_a.write_text(yaml.safe_dump(raw_a, sort_keys=False))
    profile_b.write_text(yaml.safe_dump(raw_b, sort_keys=False))
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"solid part\nendsolid part\n")
    built = []

    def build_visual(target, placement, context):
        built.append((target, placement, context))
        return _FakeGraspVisual(module_urdf, tool_urdf, mesh)

    workspace = GraspEditorWorkspace(
        targets={"part_a": profile_a, "part_b": profile_b},
        default_placements={"part_a": "storage_1", "part_b": "storage_2"},
        placements=("storage_1", "storage_2"),
        contexts=("storage", "dock"),
        build_visual=build_visual,
        evaluate=lambda _target, _placement, _grasp_raw: {
            "storage": GraspCheck("valid", "ok", (0.1,)),
            "dock": GraspCheck("unreachable", "too far"),
        },
        arm_visual=_FakeArmVisual(mesh),
    )
    monkeypatch.setattr(calibration_console_node, "ThreadingHTTPServer", _FakeServer)
    panel = GraspEditorPanel(workspace, bind="127.0.0.1", port=0)
    try:
        state = panel.state()["module"]
        assert state["choices"] == {
            "modules": ["part_a", "part_b"],
            "storages": ["storage_1", "storage_2"],
            "contexts": ["storage", "dock"],
        }
        assert state["selection"] == {
            "module": "part_a",
            "storage": "storage_1",
            "context": "storage",
        }

        revision = state["edit_revision"]
        panel.update(
            {
                "pos": [0.01, 0.02, 0.03],
                "quat": [1, 0, 0, 0],
                "finger_width_m": 0.03,
                "approach_offset_m": [0, 0, 0.1],
                "retreat_offset_m": [0, 0, -0.1],
            }
        )
        assert panel.check({"edit_revision": revision}) == {
            "stale": True,
            "edit_revision": panel.state()["module"]["edit_revision"],
        }
        assert panel.state()["module"]["status"] == "draft"
        assert yaml.safe_load(profile_a.read_text())["calibration_status"] == "approved"
        with pytest.raises(
            ValueError,
            match="unsaved grasp; save or discard before switching module",
        ):
            panel.select({"module": "part_b"})
        panel.select({"storage": "storage_2"})
        panel.select({"context": "dock"})
        assert yaml.safe_load(profile_a.read_text())["calibration_status"] == "approved"
        current = panel.state()["module"]
        before_b = profile_b.read_bytes()
        panel.save({"confirm": True, "expected_revision": current["revision"]})
        assert profile_b.read_bytes() == before_b
        saved = yaml.safe_load(profile_a.read_text())
        assert saved["calibration_status"] == "draft"
        assert saved["grasp"]["link_T_ee"]["pos"] == [0.01, 0.02, 0.03]
        assert len(list(tmp_path.glob("part_a.yaml.*.bak"))) == 1
        assert list(tmp_path.glob("part_b.yaml.*.bak")) == []
        assert built[-1] == ("part_a", "storage_2", "dock")
    finally:
        panel.close()


def test_grasp_panel_noop_update_and_save_preserve_approved_profile(
    tmp_path, monkeypatch
):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)
    raw["calibration_status"] = "approved"
    profile = tmp_path / "part_a.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False))
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"solid part\nendsolid part\n")
    monkeypatch.setattr(calibration_console_node, "ThreadingHTTPServer", _FakeServer)
    panel = GraspEditorPanel(
        profile,
        _FakeGraspVisual(module_urdf, tool_urdf, mesh),
        bind="127.0.0.1",
        port=0,
    )
    try:
        state = panel.state()["module"]
        panel.update(
            {
                "pos": [0, 0, 0],
                "quat": [1, 0, 0, 0],
                "finger_width_m": 0.04,
                "approach_offset_m": [0, 0, 0.1],
                "retreat_offset_m": [0, 0, -0.1],
            }
        )
        assert panel.state()["module"]["status"] == "approved"
        assert panel.state()["module"]["edit_revision"] == state["edit_revision"]
        panel.save({"confirm": True, "expected_revision": state["revision"]})
        assert yaml.safe_load(profile.read_text())["calibration_status"] == "approved"
        assert list(tmp_path.glob("part_a.yaml.*.bak")) == []
    finally:
        panel.close()


def test_grasp_panel_scene_uses_one_visual_snapshot(tmp_path, monkeypatch):
    module_urdf, tool_urdf, raw_a = _grasp_profile_assets(tmp_path)
    raw_b = yaml.safe_load(yaml.safe_dump(raw_a, sort_keys=False))
    raw_b["module_type"] = "part_b"
    profile_a = tmp_path / "part_a.yaml"
    profile_b = tmp_path / "part_b.yaml"
    profile_a.write_text(yaml.safe_dump(raw_a, sort_keys=False))
    profile_b.write_text(yaml.safe_dump(raw_b, sort_keys=False))
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"solid part\nendsolid part\n")
    visuals = {
        "part_a": _NamedGraspVisual("part_a", module_urdf, tool_urdf, mesh),
        "part_b": _NamedGraspVisual("part_b", module_urdf, tool_urdf, mesh),
    }
    workspace = GraspEditorWorkspace(
        targets={"part_a": profile_a, "part_b": profile_b},
        default_placements={"part_a": "storage_1", "part_b": "storage_1"},
        placements=("storage_1",),
        contexts=("storage",),
        build_visual=lambda target, _placement, _context: visuals[target],
        evaluate=lambda _target, _placement, _grasp_raw: {
            "storage": GraspCheck("checking")
        },
        arm_visual=_FakeArmVisual(mesh),
    )
    monkeypatch.setattr(calibration_console_node, "ThreadingHTTPServer", _FakeServer)
    panel = GraspEditorPanel(workspace, bind="127.0.0.1", port=0)
    visuals["part_a"].trigger = lambda: panel.select(
        {"module": "part_b", "discard": True}
    )
    try:
        scene = panel.scene()
        assert {item["link"] for item in scene["fixed"]} == {"part_a"}
        assert {item["link"] for item in scene["tool"]} == {"part_a"}
        assert panel.state()["module"]["selection"]["module"] == "part_b"
    finally:
        panel.close()


def test_grasp_panel_accepts_storage_initial_argument(tmp_path, monkeypatch):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)
    profile = tmp_path / "part_a.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False))
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"solid part\nendsolid part\n")
    workspace = GraspEditorWorkspace(
        targets={"part_a": profile},
        default_placements={"part_a": "storage_1"},
        placements=("storage_1", "storage_2"),
        contexts=("storage",),
        build_visual=lambda _target, _placement, _context: _FakeGraspVisual(
            module_urdf, tool_urdf, mesh
        ),
        evaluate=lambda _target, _placement, _grasp_raw: {
            "storage": GraspCheck("checking")
        },
        arm_visual=_FakeArmVisual(mesh),
    )
    monkeypatch.setattr(calibration_console_node, "ThreadingHTTPServer", _FakeServer)
    panel = GraspEditorPanel(
        workspace,
        bind="127.0.0.1",
        port=0,
        storage="storage_2",
    )
    try:
        assert panel.state()["module"]["selection"]["storage"] == "storage_2"
    finally:
        panel.close()


def test_grasp_panel_rejects_directory_profile_path(tmp_path):
    workspace = GraspEditorWorkspace(
        targets={"part_a": tmp_path},
        default_placements={"part_a": "storage_1"},
        placements=("storage_1",),
        contexts=("storage",),
        build_visual=lambda _target, _placement, _context: None,
        evaluate=lambda _target, _placement, _grasp_raw: {},
        arm_visual=_FakeArmVisual(tmp_path),
    )

    with pytest.raises(ValueError, match="profile path must be a file: part_a"):
        GraspEditorPanel(workspace, bind="127.0.0.1", port=0)


def test_grasp_panel_rejects_non_loopback_bind(tmp_path):
    module_urdf, tool_urdf, raw = _grasp_profile_assets(tmp_path)
    profile = tmp_path / "Part.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False))
    mesh = tmp_path / "part.stl"
    mesh.write_bytes(b"solid part\nendsolid part\n")

    with pytest.raises(ValueError, match="loopback"):
        GraspEditorPanel(
            profile,
            _FakeGraspVisual(module_urdf, tool_urdf, mesh),
            bind="0.0.0.0",
            port=0,
        )
