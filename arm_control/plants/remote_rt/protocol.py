"""Python mirror of the RT wire protocol (``rt/include/arm_rt/protocol.hpp``).

Hand-mirrored, byte-exact, enforced: ``python -m arm_control.plants.remote_rt.protocol
--hex`` prints the canonical golden packets and the C++ ``protocol_selfcheck``
prints the same lines — ``rt_backend``'s ``_demo`` diffs them automatically.
Any field change edits BOTH files in one commit and bumps ``VERSION``.

This module is a peer of ``messages.py``, not part of it: ``messages`` is the
Dora-graph Arrow contract, this is the PC<->RT-machine UDP/TCP contract.
"""
from __future__ import annotations

import struct
import math
from dataclasses import dataclass, field

MAGIC_CMD = 0x444D4341  # bytes "ACMD" on the wire
MAGIC_STATE = 0x41545341  # "ASTA"
MAGIC_CTL = 0x4C544341  # "ACTL"
VERSION = 1
POSE_HOLD_VERSION = 2
MAX_JOINTS = 16

FLAG_ARMED = 1 << 0
FLAG_FAULTED = 1 << 1
FLAG_HOLDING = 1 << 2
FLAG_WRENCH_VALID = 1 << 3
ONLINE_MASK_SHIFT = 16
ONLINE_MASK_BITS = 0xFFFF << ONLINE_MASK_SHIFT

CTL_HELLO = 1
CTL_ARM = 2
CTL_DISARM = 3
CTL_PING = 4
CTL_PONG = 5
CTL_STATUS = 6
CTL_FAULT = 7
CTL_SET_ACTIVE = 8

FAULT_CMD_LOST = 1
FAULT_CTL_LOST = 2
FAULT_PLANT = 3

_CMD_FMT = "<IHHIIQ" + "80d"  # header 24 + 5*16 doubles = 664
_STATE_FMT = "<IHHIIQII" + "88d"  # header 32 + (5*16 + 6 + 2) doubles = 736
_CTL_FMT = "<IHHIIQ104s"  # header 24 + text 104 = 128

CMD_SIZE = struct.calcsize(_CMD_FMT)
POSE_HOLD_CMD_SIZE = CMD_SIZE + 15 * 8
STATE_SIZE = struct.calcsize(_STATE_FMT)
CTL_SIZE = struct.calcsize(_CTL_FMT)
assert (CMD_SIZE, STATE_SIZE, CTL_SIZE) == (664, 736, 128)


def with_online_mask(flags: int, mask: int) -> int:
    if not 0 <= int(mask) <= 0xFFFF:
        raise ValueError("online mask must fit 16 slots")
    return (int(flags) & ~ONLINE_MASK_BITS) | (int(mask) << ONLINE_MASK_SHIFT)


def online_mask(flags: int) -> int:
    return (int(flags) & ONLINE_MASK_BITS) >> ONLINE_MASK_SHIFT


def _padded(values, n_total: int = MAX_JOINTS) -> list[float]:
    out = [float(v) for v in values]
    if len(out) > n_total:
        raise ValueError(f"at most {n_total} joints, got {len(out)}")
    return out + [0.0] * (n_total - len(out))


def pack_command(
    *, n: int, seq: int, t_mono_ns: int, q_des, qd_des, tau_ff, kp, kd, pose_hold=None
) -> bytes:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_JOINTS:
        raise ValueError("command n must be an integer in 1..16")
    arrays = [[float(v) for v in a] for a in (q_des, qd_des, tau_ff, kp, kd)]
    if any(len(a) != n or not all(math.isfinite(v) for v in a) for a in arrays):
        raise ValueError("command arrays must be finite and length n")
    if any(v < 0 for a in arrays[3:] for v in a):
        raise ValueError("command gains must be nonnegative")
    tail = b""
    if pose_hold is not None:
        from arm_control.contracts.impedance import pose_hold_values

        tail = struct.pack("<15d", *pose_hold_values(pose_hold))
    return struct.pack(
        _CMD_FMT,
        MAGIC_CMD,
        POSE_HOLD_VERSION if pose_hold is not None else VERSION,
        n,
        seq & 0xFFFFFFFF,
        0,
        t_mono_ns,
        *(v for a in arrays for v in _padded(a)),
    ) + tail


@dataclass
class State:
    n: int = 0
    state_seq: int = 0
    last_cmd_seq: int = 0
    t_mono_ns: int = 0
    flags: int = 0
    fault_code: int = 0
    q: list[float] = field(default_factory=list)
    dq: list[float] = field(default_factory=list)
    tau: list[float] = field(default_factory=list)
    tau_cmd: list[float] = field(default_factory=list)
    q_cmd: list[float] = field(default_factory=list)
    wrench: list[float] = field(default_factory=list)

    @property
    def armed(self) -> bool:
        return bool(self.flags & FLAG_ARMED)

    @property
    def faulted(self) -> bool:
        return bool(self.flags & FLAG_FAULTED)

    @property
    def holding(self) -> bool:
        return bool(self.flags & FLAG_HOLDING)

    @property
    def wrench_valid(self) -> bool:
        """True when ``wrench`` is a real estimate, not the reserved zeros.

        The field and the flag have existed since v1; the franka backend
        started filling them in 2026-08-02 (the fake backend never will — it
        has no geometry). NOT a wire change: same layout, same VERSION.
        """
        return bool(self.flags & FLAG_WRENCH_VALID)

    @property
    def online_mask(self) -> int:
        return online_mask(self.flags)


def unpack_state(data: bytes) -> State:
    if len(data) != STATE_SIZE:
        raise ValueError(f"state packet must be {STATE_SIZE} bytes, got {len(data)}")
    vals = struct.unpack(_STATE_FMT, data)
    magic, version, n = vals[0], vals[1], vals[2]
    if magic != MAGIC_STATE or version != VERSION:
        raise ValueError(f"bad state packet (magic {magic:#x}, version {version})")
    d = vals[8:]
    j = MAX_JOINTS
    return State(
        n=n,
        state_seq=vals[3],
        last_cmd_seq=vals[4],
        t_mono_ns=vals[5],
        flags=vals[6],
        fault_code=vals[7],
        q=list(d[0:n]),
        dq=list(d[j : j + n]),
        tau=list(d[2 * j : 2 * j + n]),
        tau_cmd=list(d[3 * j : 3 * j + n]),
        q_cmd=list(d[4 * j : 4 * j + n]),
        wrench=list(d[5 * j : 5 * j + 6]),
    )


def pack_control(*, ctl_type: int, seq: int, arg: int, t_mono_ns: int, text: str = "") -> bytes:
    return struct.pack(
        _CTL_FMT, MAGIC_CTL, VERSION, ctl_type, seq, arg, t_mono_ns, text.encode()
    )


@dataclass
class Control:
    ctl_type: int
    seq: int
    arg: int
    t_mono_ns: int
    text: str


def unpack_control(data: bytes) -> Control:
    if len(data) != CTL_SIZE:
        raise ValueError(f"control packet must be {CTL_SIZE} bytes, got {len(data)}")
    magic, version, ctl_type, seq, arg, t, text = struct.unpack(_CTL_FMT, data)
    if magic != MAGIC_CTL or version != VERSION:
        raise ValueError(f"bad control packet (magic {magic:#x})")
    return Control(ctl_type, seq, arg, t, text.split(b"\x00", 1)[0].decode(errors="replace"))


# --------------------------------------------------------------------------- #
# Golden packets — the exact bytes protocol_selfcheck prints. One field, one
# formula, both languages.
# --------------------------------------------------------------------------- #
_GOLDEN_T = 1234567890123456789


def golden_lines() -> list[str]:
    n = 7
    cmd = pack_command(
        n=n,
        seq=42,
        t_mono_ns=_GOLDEN_T,
        q_des=[0.1 * j for j in range(n)],
        qd_des=[0.01 * j for j in range(n)],
        tau_ff=[1.0 * j for j in range(n)],
        kp=[100.0 + j for j in range(n)],
        kd=[0.5 * j for j in range(n)],
    )
    state = struct.pack(
        _STATE_FMT,
        MAGIC_STATE,
        VERSION,
        n,
        1000,
        42,
        _GOLDEN_T,
        with_online_mask(FLAG_ARMED, 0b101),
        0,
        *_padded([0.1 * j + 0.01 for j in range(n)]),
        *_padded([0.02 * j for j in range(n)]),
        *_padded([0.5 * j for j in range(n)]),
        *_padded([0.25 * j for j in range(n)]),
        *_padded([0.1 * j for j in range(n)]),
        *([0.0] * 8),  # wrench + reserved
    )
    ctl = pack_control(ctl_type=CTL_STATUS, seq=7, arg=1, t_mono_ns=_GOLDEN_T, text="ok")
    soft = bytearray(cmd)
    struct.pack_into("<H", soft, 4, POSE_HOLD_VERSION)
    soft += struct.pack("<15d", 3, *([100.0] * 3 + [10.0] * 3),
                        *([20.0] * 3 + [2.0] * 3), 1.0, 3.0)
    return [
        f"CMD {cmd.hex()}",
        f"STATE {state.hex()}",
        f"CTL {ctl.hex()}",
        f"SIZES {CMD_SIZE} {STATE_SIZE} {CTL_SIZE}",
        f"POSE_HOLD {soft.hex()}",
        f"POSE_HOLD_SIZE {POSE_HOLD_CMD_SIZE}",
    ]


def _demo() -> None:
    # Round trips.
    st = unpack_state(bytes.fromhex(golden_lines()[1].split()[1]))
    assert st.n == 7 and st.armed and not st.faulted and st.last_cmd_seq == 42
    assert st.online_mask == 0b101
    assert abs(st.q[3] - 0.31) < 1e-12 and abs(st.tau_cmd[4] - 1.0) < 1e-12
    ctl = unpack_control(bytes.fromhex(golden_lines()[2].split()[1]))
    assert ctl.ctl_type == CTL_STATUS and ctl.arg == 1 and ctl.text == "ok"
    # Spot-check golden bytes against a constant CAPTURED from the C++ side
    # (protocol_selfcheck, 2026-07-27) — guards this file against silent
    # re-ordering even when the binary is not around to diff against.
    assert golden_lines()[2].split()[1] == (
        "4143544c0100060007000000010000001581e97df4102211"
        "6f6b" + "00" * 102
    ), "ControlPacket layout drifted from the captured C++ golden"
    print("rt_protocol: ok (sizes", CMD_SIZE, STATE_SIZE, CTL_SIZE, ")")


if __name__ == "__main__":
    import sys

    if "--hex" in sys.argv:
        print("\n".join(golden_lines()))
    else:
        _demo()
