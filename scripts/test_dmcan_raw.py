#!/usr/bin/env python3
"""Standalone DM motor read test — uses the vendor dmcan SDK directly.

Listens for MIT replies from motor 6 (CAN ID 0x06) and prints raw hex,
decoded position/velocity/torque, and inter-frame timing.

Usage:
  python scripts/test_dmcan_raw.py [motor_id] [--enable]
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

# ---------- locate and load the vendor dmcan SDK ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DLLS = REPO_ROOT / "dlls"
if str(DLLS) not in sys.path:
    sys.path.insert(0, str(DLLS))
# dmcan needs libdm_device.so and libusb-1.0.so.0 in dlls/
import ctypes
_ctypes_dll = ctypes.CDLL(str(DLLS / "libusb-1.0.so.0"), mode=ctypes.RTLD_GLOBAL)
_ctypes_dll = ctypes.CDLL(str(DLLS / "libdm_device.so"), mode=ctypes.RTLD_GLOBAL)

# monkey-patch so dmcan finds the .so (it usually searches ./dlls/ from CWD)
try:
    import dmcan.dmcan_context as _ctx
    _ctx.find_backend_dll_path = lambda: str(DLLS / "libdm_device.so")
except Exception:
    pass

from dmcan import (
    DmCanContext,
    DmCanDevice,
    dmcan_channel_can_info,
    dmcan_device_type,
    usb_rx_frame,
)

# ---------- MIT protocol (same as dm_backend.py) ----------
DM_ENABLE  = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
DM_DISABLE = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])

# Motor 6 is DM4310: ±12.5 rad pos, ±30 rad/s vel, ±10 N·m torque
POS_MIN, POS_MAX = -12.5, 12.5
VEL_MIN, VEL_MAX = -30.0, 30.0
TAU_MIN, TAU_MAX = -10.0, 10.0
KP_MIN,  KP_MAX  = 0.0, 500.0
KD_MIN,  KD_MAX  = 0.0, 5.0


def uint_to_float(raw: int, lo: float, hi: float, bits: int) -> float:
    span = hi - lo
    return (float(raw) * span) / float((1 << bits) - 1) + lo


def float_to_uint(value: float, lo: float, hi: float, bits: int) -> int:
    v = max(lo, min(hi, value))
    span = hi - lo
    return int((v - lo) * ((1 << bits) - 1) / span)


def decode_mit(data: bytes) -> dict | None:
    if len(data) < 8:
        return None
    motor_id  = data[0] & 0x0F
    error     = (data[0] >> 4) & 0x0F
    pos_u     = (data[1] << 8) | data[2]
    vel_u     = (data[3] << 4) | (data[4] >> 4)
    tau_u     = ((data[4] & 0x0F) << 8) | data[5]
    return {
        "motor_id": motor_id,
        "error":    error,
        "position": uint_to_float(pos_u, POS_MIN, POS_MAX, 16),
        "velocity": uint_to_float(vel_u, VEL_MIN, VEL_MAX, 12),
        "torque":   uint_to_float(tau_u, TAU_MIN, TAU_MAX, 12),
        "t_mos":    float(data[6]),
        "t_rotor":  float(data[7]),
    }


def pack_mit_cmd(position: float, velocity: float, kp: float,
                 kd: float, torque: float) -> bytes:
    p = float_to_uint(position, POS_MIN, POS_MAX, 16)
    v = float_to_uint(velocity, VEL_MIN, VEL_MAX, 12)
    k_p = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    k_d = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    t = float_to_uint(torque, TAU_MIN, TAU_MAX, 12)
    return bytes([
        (p >> 8) & 0xFF, p & 0xFF,
        (v >> 4) & 0xFF, ((v & 0x0F) << 4) | ((k_p >> 8) & 0x0F),
        k_p & 0xFF,
        (k_d >> 4) & 0xFF, ((k_d & 0x0F) << 4) | ((t >> 8) & 0x0F),
        t & 0xFF,
    ])

# ---------- global state ----------
g_can_id: int = 6
g_enable: bool = False
g_count: int = 0
g_last_cb_time: float = 0.0
g_start_time: float = 0.0
g_recent_dts: deque[float] = deque(maxlen=20)
g_prev_pos: float | None = None
g_pos_changes: int = 0


def recv_callback(dev_handle: DmCanDevice, rx_frame: usb_rx_frame) -> None:
    """Called by the SDK receive thread for every CAN frame."""
    global g_count, g_last_cb_time, g_pos_changes, g_prev_pos

    now = time.perf_counter()
    g_count += 1

    # Show ALL frames for debugging — tag RX vs TX
    direction = "RX" if rx_frame.head.dir == 0 else "TX"

    # Extract payload
    dlc = rx_frame.head.dlc
    dlen = dlc if dlc <= 8 else {9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64}.get(dlc, 0)
    payload = bytes(rx_frame.payload[i] for i in range(min(dlen, 64)))

    # DM motor replies have CAN frame ID = 0; the motor ID is in payload[0].
    motor_id = payload[0] & 0x0F if dlen >= 1 else 0
    hex_str = " ".join(f"{b:02X}" for b in payload[:min(dlen, 8)])

    # Only process RX matching our target motor
    if rx_frame.head.dir != 0 or motor_id != g_can_id:
        if g_count % 200 == 0:
            decoded = decode_mit(payload) if dlen >= 8 else None
            pos_str = f" pos={decoded['position']:+9.6f}" if decoded else ""
            print(f"[{g_count:5d}] {direction} motor_id={motor_id} "
                  f"(can_hdr=0x{rx_frame.head.can_id:08X}) "
                  f"dlc={dlc} data=[{hex_str}]{pos_str}  <SKIPPED>",
                  flush=True)
        return

    dt = now - g_last_cb_time if g_last_cb_time > 0 else 0.0
    g_last_cb_time = now
    g_recent_dts.append(dt)

    decoded = decode_mit(payload)

    if decoded:
        pos = decoded["position"]
        if g_prev_pos is not None and abs(pos - g_prev_pos) > 1e-9:
            g_pos_changes += 1
        g_prev_pos = pos

        print(
            f"[{g_count:5d}] dt={dt*1e3:7.2f}ms "
            f"motor={motor_id} "
            f"data=[{hex_str}] "
            f"pos={pos:+9.6f} vel={decoded['velocity']:+8.4f} "
            f"tau={decoded['torque']:+7.4f} "
            f"Tmos={decoded['t_mos']:5.1f}C Trot={decoded['t_rotor']:5.1f}C "
            f"err={decoded['error']}",
            flush=True,
        )
    else:
        print(
            f"[{g_count:5d}] dt={dt*1e3:7.2f}ms "
            f"motor={motor_id} "
            f"data=[{hex_str}]  <DECODE FAILED>",
            flush=True,
        )


def send_mit(device: DmCanDevice, channel: int, can_id: int,
             pos: float = 0.0) -> None:
    """Send one MIT keepalive frame (zero torque, hold position)."""
    cmd = pack_mit_cmd(position=pos, velocity=0.0, kp=0.0, kd=0.0, torque=0.0)
    device.send_can(channel, can_id, len(cmd), cmd,
                    canfd=True, ext=False, rtr=False, brs=True)


# ---------- main ----------
def main() -> None:
    global g_can_id, g_enable, g_pos_changes, g_prev_pos, g_start_time, g_last_cb_time

    parser = argparse.ArgumentParser(description="Standalone DM motor read test")
    parser.add_argument("motor_id", nargs="?", type=int, default=6,
                        help="CAN ID of the motor (default: 6)")
    parser.add_argument("--enable", action="store_true",
                        help="Send MIT ENABLE frame before listening")
    parser.add_argument("--hz", type=int, default=200,
                        help="Keepalive send rate in Hz (default: 200)")
    args = parser.parse_args()

    g_can_id = args.motor_id
    g_enable = args.enable
    channel = 0

    if g_can_id < 1 or g_can_id > 0xFE:
        print("motor_id must be 1–254", file=sys.stderr)
        sys.exit(1)

    # ---------- open device ----------
    ctx = DmCanContext()
    ctx.print_version()

    count = ctx.find_devices(dmcan_device_type.USB2CANFD)
    if count == 0:
        print("No DM USB2FDCAN device found!", file=sys.stderr)
        sys.exit(1)

    device = ctx.get_device(0)
    if not device.open():
        print("Failed to open device", file=sys.stderr)
        sys.exit(1)

    device.print_version()

    # ---------- configure channel ----------
    info = dmcan_channel_can_info()
    info.channel = channel
    info.canfd = True
    info.can_baudrate = 1_000_000
    info.canfd_baudrate = 5_000_000
    info.can_sp = 0.75
    info.canfd_sp = 0.75

    if not device.set_channel_baudrate(channel, info):
        print("WARNING: set_channel_baudrate failed — using existing config",
              file=sys.stderr)

    device.enable_channel(channel, True)

    # Show actual baudrate
    actual = device.get_channel_baudrate(channel)
    if actual:
        print(f"Channel {channel}: canfd={actual.canfd} "
              f"can_baud={actual.can_baudrate} canfd_baud={actual.canfd_baudrate} "
              f"can_sp={actual.can_sp:.2f} canfd_sp={actual.canfd_sp:.2f}")

    # ---------- register callback ----------
    device.hook_recv_callback(recv_callback)

    print(f"Listening for motor 0x{g_can_id:02X} on channel {channel} "
          f"(send keepalive at {args.hz} Hz)...")
    print(f"{'COUNT':>5s}  {'dt(ms)':>8s}  {'CAN_ID':>6s}  "
          f"{'HEX':>23s}  {'pos(rad)':>10s}  {'vel(rad/s)':>9s}  "
          f"{'tau(Nm)':>8s}  {'Tmos':>6s}  {'Trot':>6s}  err")
    print("-" * 120)

    # ---------- optionally enable motor ----------
    if g_enable:
        print("Sending MIT ENABLE frames (x3)...")
        for _ in range(3):
            device.send_can(channel, g_can_id, 8, DM_ENABLE,
                            canfd=True, ext=False, rtr=False, brs=True)
            time.sleep(0.02)

    # ---------- main loop: send keepalive, let callback handle RX ----------
    g_start_time = time.perf_counter()
    g_last_cb_time = g_start_time
    # Test 1: send ONE frame, wait for reply, measure round-trip
    print("Test 1: single send, measuring round-trip...")
    for trial in range(5):
        t0 = time.perf_counter()
        cmd = pack_mit_cmd(position=0.0, velocity=0.0, kp=0.0, kd=0.0, torque=0.0)
        device.send_can(channel, g_can_id, 8, cmd,
                        canfd=True, ext=False, rtr=False, brs=True)
        # Wait for reply (up to 1 second)
        time.sleep(1.0)
        t1 = time.perf_counter()
        print(f"  trial {trial+1}: slept 1.0s, actual={t1-t0:.3f}s, "
              f"callbacks received (total)={g_count}")
    # Test 2: send at 8 Hz (125ms, just below 10 Hz threshold)
    print("\nTest 2: sending at 8 Hz for 5 seconds...")
    g_pos_changes = 0
    g_prev_pos = None
    g_start_8hz = time.perf_counter()
    g_last_cb_time = g_start_8hz
    next_t = g_start_8hz
    deadline = g_start_8hz + 5.0
    period_8hz = 1.0 / 8.0  # 125ms
    while time.perf_counter() < deadline:
        cmd = pack_mit_cmd(position=0.0, velocity=0.0, kp=0.0, kd=0.0, torque=0.0)
        device.send_can(channel, g_can_id, 8, cmd,
                        canfd=True, ext=False, rtr=False, brs=True)
        next_t += period_8hz
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_t = time.perf_counter()
    elapsed_8hz = time.perf_counter() - g_start_8hz
    print(f"  8 Hz done. {g_count} total callbacks in {elapsed_8hz:.1f}s "
          f"(~{g_count/elapsed_8hz:.0f} callbacks/s)")

    # ---------- summary ----------
    elapsed = time.perf_counter() - g_start_time
    print("\n--- summary ---")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"Total callbacks: {g_count}")
    print(f"Total RX frames (dir=0, matching CAN ID): {g_pos_changes} "
          f"({g_pos_changes / elapsed:.1f}/s)")
    if g_recent_dts:
        dts = sorted(g_recent_dts)
        print(f"Inter-frame dt (last {len(dts)}): "
              f"min={min(dts)*1e3:.2f}ms "
              f"median={dts[len(dts)//2]*1e3:.2f}ms "
              f"max={max(dts)*1e3:.2f}ms")

    # ---------- cleanup ----------
    print("Shutting down...")
    try:
        device.send_can(channel, g_can_id, 8, DM_DISABLE,
                        canfd=True, ext=False, rtr=False, brs=True)
    except Exception:
        pass
    time.sleep(0.1)
    try:
        device.close()
    except Exception:
        pass
    time.sleep(0.05)
    try:
        ctx.destroy()
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
