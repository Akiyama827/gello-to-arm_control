import pytest

from arm_control import rt_protocol as rtp
from arm_control.hardware.rt_backend import RtBackend, RtConfig


def test_online_mask_uses_reserved_state_flag_bits():
    flags = rtp.with_online_mask(rtp.FLAG_ARMED, 0b101)

    assert rtp.online_mask(flags) == 0b101
    assert flags & rtp.FLAG_ARMED


def test_active_mask_only_grows_while_armed(monkeypatch):
    backend = RtBackend(RtConfig(), ["a", "b", "c"])
    backend._active_mask = 0b011
    backend._state = rtp.State(n=3, flags=rtp.FLAG_ARMED)
    monkeypatch.setattr(
        backend,
        "_control_roundtrip",
        lambda _kind, *, arg=0: rtp.Control(
            rtp.CTL_STATUS, 0, arg, 0, "active mask applied"
        ),
    )

    backend.set_active_mask(0b111)
    assert backend.motor_health()["active_mask"] == 0b111
    with pytest.raises(RuntimeError, match="remove"):
        backend.set_active_mask(0b011)
