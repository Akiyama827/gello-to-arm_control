#!/usr/bin/env python3
"""Test all 6 motors: enable each, check for reply."""
# ruff: noqa: E402
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

CONTROL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL_ROOT))

_dlls = CONTROL_ROOT / "dlls"
for name in ("libusb-1.0.so.0", "libdm_device.so"):
    p = _dlls / name
    if p.is_file():
        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)

from arm_control.hardware.dm_backend import (
    DM_DISABLE_FRAME,
    DM_ENABLE_FRAME,
    DmMotorLimits,
    pack_mit_control_frame,
)
from dmcan import DmCanContext, dmcan_channel_can_info, dmcan_device_type

ctx = DmCanContext()
n = ctx.find_devices(dmcan_device_type.USB2CANFD)
print(f"Devices found: {n}")
dev = ctx.get_device(0)
dev.open()

info = dmcan_channel_can_info()
info.channel = 0
info.canfd = False
info.can_baudrate = 1_000_000
info.canfd_baudrate = 1_000_000
info.can_sp = 0.75
info.canfd_sp = 0.80
dev.set_channel_baudrate(0, info)
dev.enable_channel(0, True)

replies_by_id = {i: [] for i in range(1, 7)}

def on_recv(dev_obj, frame):
    f = frame
    if f.head.dlc >= 8 and f.head.dir == 0:
        p = bytes(f.payload[i] for i in range(8))
        mid = p[0] & 0x0F
        pos_u = (p[1] << 8) | p[2]
        replies_by_id.get(mid, []).append(pos_u)

dev.hook_recv_callback(on_recv)

# Use DM4310 limits for the test (all motors share same position range)
limits = DmMotorLimits()

for can_id in range(1, 7):
    print(f"\n--- Motor 0x0{can_id} ---", flush=True)
    replies_by_id[can_id] = []

    # Enable
    dev.send_can(0, can_id, 8, DM_ENABLE_FRAME, False, False, False, False)
    time.sleep(0.02)

    # Send MIT zero-torque frames
    zf = pack_mit_control_frame(position=0.0, velocity=0.0, kp=0.0, kd=0.0, torque=0.0, limits=limits)
    for _ in range(5):
        dev.send_can(0, can_id, 8, zf, False, False, False, False)
        time.sleep(0.005)

    time.sleep(0.1)
    n_replies = len(replies_by_id[can_id])
    if n_replies > 0:
        avg_pos = sum(replies_by_id[can_id]) / n_replies
        print(f"  OK — {n_replies} replies, pos_raw=0x{int(avg_pos):04X}", flush=True)
    else:
        print("  NO REPLY", flush=True)

    # Disable
    dev.send_can(0, can_id, 8, DM_DISABLE_FRAME, False, False, False, False)
    time.sleep(0.01)

dev.enable_channel(0, False)
dev.close()
ctx._ctx = None
print("\nDone", flush=True)
