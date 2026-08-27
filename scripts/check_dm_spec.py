#!/usr/bin/env python3
"""Guard: the base's --dm-spec derives from config AND matches the systemd unit.

The RT server takes its motor list as a CLI string while the same facts live in
configs/real/base/hardware.yaml. This asserts the two agree, so a motor
reflashed to a new Master ID can't leave a stale literal in the unit file —
which fails as "motor N never replied", a message that names only the LAST
silent motor and reads like a wiring fault.

    PYTHONPATH=. python scripts/check_dm_spec.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from arm_control.config import RobotConfig
from arm_control.hardware.dm_backend import dm_spec_from_config

ROOT = Path(__file__).resolve().parents[1]
EXPECT = "can0;0x01:4340:0x11,0x02:4340p:0x12"  # MEASURED 2026-08-27
RESERVED_PORTS = {47800, 47801, 47802, 47803}  # FR3 server + hand_bridge


def main() -> int:
    cfg = RobotConfig.from_yaml(ROOT / "configs/real/base/hardware.yaml")
    spec = dm_spec_from_config(cfg)
    print(f"derived : {spec}")
    assert spec == EXPECT, f"{spec!r} != measured {EXPECT!r}"

    # Grammar parity with rt/src/backend_dm.cpp parse_spec ("%i:%15[^:]:%i").
    iface, rest = spec.split(";", 1)
    assert iface == "can0", iface
    for item in rest.split(","):
        fields = item.split(":")
        assert len(fields) == 3, item
        can_id, motor_type, mst = int(fields[0], 0), fields[1], int(fields[2], 0)
        assert 1 <= can_id <= 15, f"payload-nibble routed, got {can_id}"
        assert motor_type in ("4310", "4310p", "4340", "4340p"), motor_type
        assert mst == 0x10 + can_id, f"motor {can_id} master_id 0x{mst:02X}"

    unit = (ROOT / "rt/systemd/arm-base-server.service").read_text()
    match = re.search(r'--dm-spec\s+"([^"]+)"', unit)
    assert match, "no --dm-spec in arm-base-server.service"
    print(f"unit    : {match.group(1)}")
    assert match.group(1) == spec, f"unit {match.group(1)!r} != config {spec!r}"

    # This unit must never be enable-able: motors would go live at boot with
    # nobody at the bench, and this plant has no safety layer of its own.
    assert not re.search(r"^\[Install\]", unit, re.M), "must have no [Install] section"
    assert not re.search(r"^Restart=always", unit, re.M), "use Restart=on-failure"

    rt = cfg.raw["rt"]
    ports = {rt["udp_port"], rt["tcp_port"]}
    assert ports.isdisjoint(RESERVED_PORTS), f"{ports} collides with {RESERVED_PORTS}"
    for port in ports:
        assert str(port) in unit, f"unit does not launch on port {port}"

    print("dm_spec: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
