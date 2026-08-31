from pathlib import Path

import pytest

from arm_control.calibration_console import CalibrationStore, ConsoleAuthority


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
