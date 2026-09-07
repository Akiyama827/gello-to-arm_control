#!/usr/bin/env python3
"""Read DM motor parameter registers over SocketCAN. READ-ONLY.

Sends ONLY 0x7FF register READS (D[2] = 0x33). It never writes (0x55), never
saves (0xAA), never enables (0xFC) and never sends a MIT frame — so it cannot
make a motor move or persist a change, and it needs no enable to work.

Protocol (DM-J4340-2EC manual V1.1 p.13):
    read  -> id 0x7FF STD, data [CANID_L, CANID_H, 0x33, RID]
    reply -> id MST_ID  STD, data [CANID_L, CANID_H, 0x33, RID, d0..d3]
with d0..d3 a little-endian float32 or uint32.

Why it exists: some numbers the models need are NOT in the datasheets. Rotor
inertia in particular is exposed only as register 0x0C, so ``armature`` in the
MJCFs (Gr^2 * J_rotor) can only be grounded by asking the motor. Measured
2026-08-29 on the bench base: 0x01 J=1.8014e-5 Gr=40, 0x02 J=1.6810e-5 Gr=40.

Run it ON the machine that owns the bus (the RT box):

    python3 scripts/dm_read_params.py --ids 1 2
    python3 scripts/dm_read_params.py --scan
"""
from __future__ import annotations

import argparse
import ctypes
import socket
import struct
import time


class canfd_frame(ctypes.Structure):
    _fields_ = [
        ("can_id", ctypes.c_uint32),
        ("len", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("res0", ctypes.c_uint8),
        ("res1", ctypes.c_uint8),
        ("data", ctypes.c_uint8 * 64),
    ]


# linux/can.h — same values (and the same hard-won ordering) as
# arm_control/hardware/socketcan_transport.py. DM motors reply FD/BRS, so a
# reader without CAN_RAW_FD_FRAMES sees silence, not an error.
CANFD_BRS, CANFD_ESI, CANFD_FDF = 0x01, 0x02, 0x04

# RID -> (label, "f" float32 | "u" uint32). Appendix <寄存器列表及范围>.
REGISTERS: dict[int, tuple[str, str]] = {
    0x00: ("UV_Value under-volt", "f"),
    0x0B: ("Damp viscous", "f"),
    0x0C: ("Inertia (rotor)", "f"),
    0x0E: ("sw_ver", "u"),
    0x10: ("NPP pole pairs", "u"),
    0x11: ("Rs phase resistance", "f"),
    0x14: ("Gr gear ratio", "f"),
    0x15: ("PMAX", "f"),
    0x16: ("VMAX", "f"),
    0x17: ("TMAX", "f"),
}


def _frame(can_id: int, payload: bytes) -> bytes:
    frame = canfd_frame()
    frame.can_id = can_id & 0x1FFFFFFF
    frame.len = len(payload)
    frame.flags = CANFD_FDF | CANFD_BRS
    for i, byte in enumerate(payload):
        frame.data[i] = byte
    return bytes(frame)


def read_register(sock, motor_id: int, rid: int, timeout: float = 0.35):
    """One register, or None if the motor does not answer."""
    sock.send(
        _frame(0x7FF, bytes([motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0x33, rid,
                             0, 0, 0, 0]))
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = sock.recv(72)
        except socket.timeout:
            return None
        frame = canfd_frame.from_buffer_copy(raw.ljust(72, b"\0"))
        data = bytes(frame.data)[: max(frame.len, 8)]
        if (
            len(data) >= 8
            and data[2] == 0x33
            and data[3] == rid
            and data[0] == (motor_id & 0xFF)
        ):
            kind = REGISTERS.get(rid, ("", "f"))[1]
            return (
                struct.unpack("<f", data[4:8])[0]
                if kind == "f"
                else struct.unpack("<I", data[4:8])[0]
            )
    return None


def open_bus(interface: str):
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((interface,))
    sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
    sock.settimeout(0.2)
    return sock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interface", default="can0")
    ap.add_argument("--ids", type=lambda s: int(s, 0), nargs="*", default=[1, 2])
    ap.add_argument(
        "--scan", action="store_true", help="probe CAN ids 1..15 instead of --ids"
    )
    args = ap.parse_args()

    sock = open_bus(args.interface)
    ids = range(1, 16) if args.scan else args.ids
    found = []
    for motor_id in ids:
        gear = read_register(sock, motor_id, 0x14, timeout=0.15)
        if gear is None:
            if not args.scan:
                print(f"\n=== motor 0x{motor_id:02X}: NO REPLY")
            continue
        found.append(motor_id)
        print(f"\n=== motor 0x{motor_id:02X} ===")
        for rid, (label, _) in sorted(REGISTERS.items()):
            value = read_register(sock, motor_id, rid)
            print(f"  0x{rid:02X} {label:22s} = "
                  f"{'--' if value is None else value!r}")
        inertia = read_register(sock, motor_id, 0x0C)
        if inertia is not None and gear:
            # What the MJCF <joint armature="..."> wants: rotor inertia seen
            # from the OUTPUT shaft.
            print(f"  -> MJCF armature = Gr^2 * J_rotor = {gear**2 * inertia:.6g}")
    if args.scan:
        print(f"\nmotors present: {[hex(i) for i in found]}")
    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
