"""SocketCAN transport for DM motors via the gsusb firmware.

Replaces the vendor DM SDK (libdm_device.so) entirely.  The DM USB2FDCAN
device, flashed with the gsusb firmware, appears as a native Linux ``can0``
interface.  CAN FD frames are sent and received through the kernel's
SocketCAN stack — no userspace library, no callbacks, no GIL issues.
"""
from __future__ import annotations

import ctypes
import socket
import threading
import time
from typing import Any, Callable


class canfd_frame(ctypes.Structure):
    _fields_ = [
        ("can_id", ctypes.c_uint32),
        ("len",    ctypes.c_uint8),
        ("flags",  ctypes.c_uint8),
        ("res0",   ctypes.c_uint8),
        ("res1",   ctypes.c_uint8),
        ("data",   ctypes.c_uint8 * 64),
    ]


# Values from linux/can.h — the previous labels were permuted (FDF=0x01,
# BRS=0x02, ESI=0x04), so TX flags went out as real BRS|ESI: bit-rate switch
# by luck, ERROR STATE INDICATOR spuriously asserted, FDF absent (the kernel
# inferred it from the 72-byte canfd_frame write, which is why it worked).
CANFD_BRS = 0x01      # bit-rate switch
CANFD_ESI = 0x02      # error state indicator
CANFD_FDF = 0x04      # FD frame


def _build_canfd_frame(can_id: int, payload: bytes, brs: bool = True) -> bytes:
    f = canfd_frame()
    f.can_id = can_id & 0x1FFFFFFF
    f.len = len(payload)
    f.flags = CANFD_FDF | (CANFD_BRS if brs else 0)
    for i, b in enumerate(payload):
        f.data[i] = b
    return bytes(f)


class ParsedCanFrame:
    __slots__ = ("can_id", "flags", "data")
    def __init__(self, can_id: int, flags: int, data: bytes) -> None:
        self.can_id = can_id
        self.flags = flags
        self.data = data


class SocketCanTransport:
    """Minimal SocketCAN transport matching the DmCanTransport protocol."""

    def __init__(self, interface: str = "can0") -> None:
        self._interface = interface
        self._sock: socket.socket | None = None
        self._callback: Callable[[Any, Any], None] | None = None
        self._rx_thread: threading.Thread | None = None
        self._rx_stop = threading.Event()

    # ------------------------------------------------------------------
    # Public API (matches DmCanTransport protocol)
    # ------------------------------------------------------------------

    def open(self, bus: Any) -> None:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((self._interface,))
        # Enable CAN FD reception
        s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
        s.setblocking(False)
        self._sock = s

    def configure_channel(self, bus: Any) -> None:
        # Channel/baudrate configured via 'ip link set can0 ...' before launch.
        pass

    def enable_channel(self, channel: int) -> None:
        pass

    def disable_channel(self, channel: int) -> None:
        pass

    def send_can(
        self,
        channel: int,
        can_id: int,
        payload: bytes,
        *,
        canfd: bool = True,
        extended_id: bool = False,
        remote_frame: bool = False,
        brs: bool = True,
    ) -> None:
        if self._sock is None:
            raise RuntimeError("SocketCAN device not open")
        frame = _build_canfd_frame(can_id, payload, brs=brs)
        self._sock.send(frame)

    def hook_recv_callback(self, callback: Callable[[Any, Any], None]) -> None:
        self._callback = callback
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(
            target=self._rx_loop, daemon=True, name="socketcan-rx",
        )
        self._rx_thread.start()

    def close(self) -> None:
        self._rx_stop.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # ------------------------------------------------------------------
    # RX loop
    # ------------------------------------------------------------------

    def _rx_loop(self) -> None:
        sock = self._sock
        callback = self._callback
        if sock is None or callback is None:
            return

        buf = bytearray(72)  # sizeof(struct canfd_frame)
        while not self._rx_stop.is_set():
            try:
                n = sock.recv_into(buf, 72)
                if n > 0:
                    rx = canfd_frame.from_buffer_copy(buf[:n])
                    payload = bytes(rx.data[: rx.len])
                    pf = ParsedCanFrame(can_id=rx.can_id, flags=rx.flags, data=payload)
                    _proxy = _FrameProxy(pf)
                    callback(self, _proxy)
            except BlockingIOError:
                time.sleep(0.0002)  # 200 µs — no data, brief yield
            except OSError:
                if self._rx_stop.is_set():
                    break
                time.sleep(0.001)


class _FrameProxy:
    """Minimal proxy so the existing _on_can_recv callback works unchanged."""
    __slots__ = ("head", "payload")
    def __init__(self, pf: ParsedCanFrame) -> None:
        self.head = _HeadProxy(pf)
        self.payload = (ctypes.c_uint8 * 64)(*pf.data, *([0] * (64 - len(pf.data))))


class _HeadProxy:
    __slots__ = ("can_id", "dlc", "dir", "ack")
    def __init__(self, pf: ParsedCanFrame) -> None:
        self.can_id = pf.can_id
        # DLC is the actual frame data length (0-8 for CAN 2.0, 0-64 for FD).
        self.dlc = len(pf.data)
        self.dir = 0   # always RX
        self.ack = 0
