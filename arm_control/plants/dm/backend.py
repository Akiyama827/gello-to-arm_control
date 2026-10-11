"""DM motor wire facts for the base's RT server: MIT codec, per-type limits, and ``--dm-spec`` from a config.

The PC-side DM driver is gone (the DM arm left the stack, 2026-10-09); ``arm_rt_server --backend dm`` drives the base,
and this module keeps the Python half of its wire format (parity: ``tools/check/rt/torque_cap.py``).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any

from arm_control.control.gains import validate_hardware_gains


@dataclass(frozen=True)
class DmMotorLimits:
    position_min: float = -12.5
    position_max: float = 12.5
    velocity_min: float = -30.0
    velocity_max: float = 30.0
    kp_min: float = 0.0
    kp_max: float = 500.0
    kd_min: float = 0.0
    kd_max: float = 5.0
    torque_min: float = -10.0
    torque_max: float = 10.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None, motor_type: str) -> "DmMotorLimits":
        defaults = DEFAULT_LIMITS_BY_TYPE.get(motor_type, DEFAULT_LIMITS_BY_TYPE["4310"])
        if not raw:
            return defaults
        values = defaults.__dict__ | {str(key): float(value) for key, value in raw.items()}
        return cls(**values)


DEFAULT_LIMITS_BY_TYPE: dict[str, DmMotorLimits] = {
    # Override in YAML if your firmware or motor table uses different ranges.
    # Limits sourced from createra-robotics/rust_motorcom (MIT protocol per-motor
    # constants verified against Damiao datasheets).
    #
    # DM4310:  10:1 reduction, rated 3 N·m / peak 7 N·m, 120 rpm
    "4310": DmMotorLimits(),  # position ±12.5, velocity ±30, torque ±10
    "4310p": DmMotorLimits(velocity_min=-50.0, velocity_max=50.0),
    # DM4340:  40:1 reduction, rated 9 N·m / peak 27 N·m, 36 rpm
    "4340": DmMotorLimits(
        velocity_min=-10.0, velocity_max=10.0,
        torque_min=-28.0, torque_max=28.0,
    ),
    # DM4340P: same electrical specs as DM4340; cross-roller bearing variant
    "4340p": DmMotorLimits(
        velocity_min=-10.0, velocity_max=10.0,
        torque_min=-28.0, torque_max=28.0,
    ),
}


@dataclass(frozen=True)
class DmMotorConfig:
    name: str
    joint: str
    can_id: int
    motor_type: str
    master_id: int | None = None
    module: str | None = None
    bank: str | None = None
    enabled_on_open: bool = False
    limits: DmMotorLimits = field(default_factory=DmMotorLimits)


@dataclass
class DmMotorState:
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    position_cmd: float = 0.0
    velocity_cmd: float = 0.0
    torque_cmd: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    motor_id: int = 0
    error: int = 0
    t_mos: float = 0.0
    t_rotor: float = 0.0


def _uint_to_float(raw: int, lo: float, hi: float, bits: int) -> float:
    span = hi - lo
    if span <= 0:
        raise ValueError("invalid motor limit span")
    return (float(raw) * span) / float((1 << bits) - 1) + lo


def decode_mit_reply(data: bytes, limits: DmMotorLimits) -> DmMotorState | None:
    """Decode an 8-byte DM motor MIT-mode reply frame.

    Wire format (createra-robotics/rust_motorcom):
      byte 0: [error(4)] [motor_id(4)]
      byte 1-2: position uint16
      byte 3-4: [velocity uint12 (high 8 then low 4)] [torque uint12 (high 4)]
      byte 5: torque low
      byte 6: T_mos (°C)
      byte 7: T_rotor (°C)
    """
    if data is None or len(data) < 8:
        return None
    motor_id = data[0] & 0x0F
    error = (data[0] >> 4) & 0x0F
    pos_u = (data[1] << 8) | data[2]
    vel_u = (data[3] << 4) | (data[4] >> 4)
    tau_u = ((data[4] & 0x0F) << 8) | data[5]
    return DmMotorState(
        position=_uint_to_float(pos_u, limits.position_min, limits.position_max, 16),
        velocity=_uint_to_float(vel_u, limits.velocity_min, limits.velocity_max, 12),
        torque=_uint_to_float(tau_u, limits.torque_min, limits.torque_max, 12),
        motor_id=motor_id,
        error=error,
        t_mos=float(data[6]),
        t_rotor=float(data[7]),
    )


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


def _float_to_uint(value: float, lo: float, hi: float, bits: int) -> int:
    value = _clip(value, lo, hi)
    span = hi - lo
    if span <= 0:
        raise ValueError("invalid motor limit span")
    return int((value - lo) * ((1 << bits) - 1) / span)


def pack_mit_control_frame(
    *,
    position: float,
    velocity: float,
    kp: float,
    kd: float,
    torque: float,
    limits: DmMotorLimits,
) -> bytes:
    # Reject out-of-range gains here (the single encode choke point) so a stale
    # Isaac 600/60 command fails loudly instead of clipping to 500/5.
    validate_hardware_gains(kp, kd, torque, limits)
    p = _float_to_uint(position, limits.position_min, limits.position_max, 16)
    v = _float_to_uint(velocity, limits.velocity_min, limits.velocity_max, 12)
    k_p = _float_to_uint(kp, limits.kp_min, limits.kp_max, 12)
    k_d = _float_to_uint(kd, limits.kd_min, limits.kd_max, 12)
    t = _float_to_uint(torque, limits.torque_min, limits.torque_max, 12)
    return bytes(
        [
            (p >> 8) & 0xFF,
            p & 0xFF,
            (v >> 4) & 0xFF,
            ((v & 0x0F) << 4) | ((k_p >> 8) & 0x0F),
            k_p & 0xFF,
            (k_d >> 4) & 0xFF,
            ((k_d & 0x0F) << 4) | ((t >> 8) & 0x0F),
            t & 0xFF,
        ]
    )


def _parse_motor_configs(cfg: Any) -> list[DmMotorConfig]:
    raw_motors = list(cfg.raw.get("motors") or [])
    if not raw_motors:
        raw_motors = [
            {
                "name": cfg.motor_names[i] if i < len(cfg.motor_names) else f"motor_{i}",
                "joint": cfg.joint_names[i] if i < len(cfg.joint_names) else None,
                "can_id": cfg.motor_ids[i] if i < len(cfg.motor_ids) else i + 1,
                "motor_type": (cfg.raw.get("motor_types") or ["4310"] * cfg.num_motors)[i],
            }
            for i in range(cfg.num_motors)
        ]
    motors: list[DmMotorConfig] = []
    for index, raw in enumerate(raw_motors):
        item = dict(raw)
        name = str(item.get("name") or item.get("motor_name") or f"motor_{index}")
        joint = str(item.get("joint") or item.get("joint_name") or name)
        motor_type = str(item.get("motor_type") or item.get("type") or "4310")
        limits = DmMotorLimits.from_mapping(item.get("limits"), motor_type)
        motors.append(
            DmMotorConfig(
                name=name,
                joint=joint,
                can_id=int(item.get("can_id", item.get("id", index + 1))),
                motor_type=motor_type,
                master_id=_optional_int(item.get("master_id")),
                module=_optional_str(item.get("module")),
                bank=_optional_str(item.get("bank")),
                enabled_on_open=bool(item.get("enabled_on_open", False)),
                limits=limits,
            )
        )
    return motors


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def dm_spec_from_config(cfg: Any, iface: str | None = None) -> str:
    """Build ``arm_rt_server --backend dm``'s ``--dm-spec`` from a hardware config.

    The RT server takes its motor list as a CLI string
    (``iface;id:type[:mst[:pmax:vmax:tmax]],...``)
    while the same facts live in ``hardware.yaml``'s ``motors:`` block. Deriving
    the string here keeps the two from drifting: a motor reflashed to a new
    Master ID is a config edit, not a systemd-unit edit.

    ``master_id`` matters more than it looks — the RT backend installs a KERNEL
    CAN FILTER on it (``rt/src/backend_dm.cpp:240``), so a wrong value drops
    every reply before the backend sees it and surfaces as "motor N never
    replied". ``None`` means the DM factory default 0x00 and is emitted as a
    bare ``id:type`` pair, matching the server's own default.

    Carry non-default motor ``limits`` as symmetric wire half-ranges. These
    are codec parameters, not task speed limits or the RT operating torque
    cap. An explicit mapping needs all three values and a Master ID (zero
    if absent); partial/asymmetric ranges cannot be represented by this CLI.
    """
    motors = _parse_motor_configs(cfg)
    if iface is None:
        iface = str((getattr(cfg, "raw", {}) or {}).get("bus", {}).get("interface", "can0"))
    parts = []
    for m in motors:
        defaults = DEFAULT_LIMITS_BY_TYPE[m.motor_type]
        ranges = []
        overridden = False
        for axis in ("position", "velocity", "torque"):
            lo, hi = getattr(m.limits, f"{axis}_min"), getattr(m.limits, f"{axis}_max")
            levels = 65535 if axis == "position" else 4095
            if not (math.isfinite(lo) and math.isfinite(hi) and hi >= sys.float_info.min
                    and lo == -hi and math.isfinite((2 * hi) * levels)):
                raise ValueError(f"motor {m.name}: {axis} wire range must be finite, "
                                 "positive, symmetric and encodable")
            ranges.append(hi)
            overridden |= hi != getattr(defaults, f"{axis}_max")
        item = f"0x{m.can_id:02X}:{m.motor_type}"
        if m.master_id is not None or overridden:
            item += f":0x{m.master_id if m.master_id is not None else 0:02X}"
        if overridden:
            item += ":" + ":".join(format(value, ".17g") for value in ranges)
        parts.append(item)
    return f"{iface};" + ",".join(parts)


__all__ = [
    "DEFAULT_LIMITS_BY_TYPE",
    "DmMotorConfig",
    "DmMotorLimits",
    "decode_mit_reply",
    "dm_spec_from_config",
    "pack_mit_control_frame",
]


def _self_check() -> None:
    from types import SimpleNamespace

    # Configured wire ranges must survive the Python -> RT launch boundary.
    item = {"can_id": 1, "motor_type": "4340", "master_id": 0x11,
            "limits": {"velocity_min": -20.0, "velocity_max": 20.0}}
    cfg = SimpleNamespace(raw={"motors": [item]})
    assert dm_spec_from_config(cfg) == "can0;0x01:4340:0x11:12.5:20:28"
    item.pop("master_id")
    assert dm_spec_from_config(cfg) == "can0;0x01:4340:0x00:12.5:20:28"
    item["limits"] = {}
    assert dm_spec_from_config(cfg) == "can0;0x01:4340"
    item["master_id"] = 0x11
    assert dm_spec_from_config(cfg, "can1") == "can1;0x01:4340:0x11"
    other = {"can_id": 2, "motor_type": "4310p", "master_id": 0x12,
             "limits": {"position_min": -6.0, "position_max": 6.0,
                        "velocity_min": -5.0, "velocity_max": 5.0,
                        "torque_min": -3.0, "torque_max": 3.0}}
    cfg.raw["motors"].append(other)
    assert dm_spec_from_config(cfg) == (
        "can0;0x01:4340:0x11,0x02:4310p:0x12:6:5:3")
    cfg.raw["motors"].pop()
    for axis in ("position", "velocity", "torque"):
        for lo, hi in ((-10.0, 20.0), (0.0, 0.0), (20.0, -20.0),
                       (-float("inf"), float("inf")),
                       (float("nan"), 20.0), (-20.0, float("nan")),
                       (-1e308, 1e308), (-8e307, 8e307), (-1e-310, 1e-310)):
            item["limits"] = {f"{axis}_min": lo, f"{axis}_max": hi}
            try:
                dm_spec_from_config(cfg)
            except ValueError as exc:
                assert axis in str(exc), exc
            else:
                raise AssertionError(f"invalid {axis} wire mapping accepted: {lo}, {hi}")
    item["limits"] = {"velocity_min": -20.0, "velocity_max": 20.0}
    limits = _parse_motor_configs(cfg)[0].limits
    # Velocity code 2457 = +4 rad/s at VMAX=20, not +2 at the legacy default.
    state = decode_mit_reply(bytes.fromhex("0180009998001922"), limits)
    assert state is not None and abs(state.velocity - 4.0) < 1e-12
    assert DEFAULT_LIMITS_BY_TYPE["4340"].velocity_max == 10.0
    assert DEFAULT_LIMITS_BY_TYPE["4340p"].velocity_max == 10.0

    print("dm_backend: OK")


if __name__ == "__main__":
    _self_check()
