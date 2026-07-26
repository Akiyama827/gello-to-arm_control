"""DM motor backend using the DM USB2FDCAN device SDK."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from arm_control.config import RobotConfig
from arm_control.hardware.gains import validate_hardware_gains

from arm_control import CONTROL_ROOT, REPO_ROOT

DM_ENABLE_FRAME = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
DM_DISABLE_FRAME = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])


class DmBackendUnavailableError(RuntimeError):
    """Raised when the DM SDK or USB2FDCAN device cannot be opened."""


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
class DmCanBusConfig:
    adapter: str = "USB2FDCAN"
    device_index: int = 0
    device_type: str = "USB2CANFD"
    channel: int = 0
    canfd: bool = True
    can_baudrate: int = 500000
    canfd_baudrate: int = 2000000
    can_sp: float = 0.75
    canfd_sp: float = 0.80
    brs: bool = True
    extended_id: bool = False
    remote_frame: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "DmCanBusConfig":
        data = dict(raw or {})
        return cls(
            adapter=str(data.get("adapter", "USB2FDCAN")),
            device_index=int(data.get("device_index", 0)),
            device_type=str(data.get("device_type", "USB2CANFD")),
            channel=int(data.get("channel", 0)),
            canfd=bool(data.get("canfd", True)),
            can_baudrate=int(data.get("can_baudrate", 500000)),
            canfd_baudrate=int(data.get("canfd_baudrate", 2000000)),
            can_sp=float(data.get("can_sp", 0.75)),
            canfd_sp=float(data.get("canfd_sp", 0.80)),
            brs=bool(data.get("brs", True)),
            extended_id=bool(data.get("extended_id", False)),
            remote_frame=bool(data.get("remote_frame", False)),
        )


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


class DmCanTransport(Protocol):
    def open(self, bus: DmCanBusConfig) -> None: ...
    def configure_channel(self, bus: DmCanBusConfig) -> None: ...
    def enable_channel(self, channel: int) -> None: ...
    def disable_channel(self, channel: int) -> None: ...

    def send_can(
        self,
        channel: int,
        can_id: int,
        payload: bytes,
        *,
        canfd: bool,
        extended_id: bool,
        remote_frame: bool,
        brs: bool,
    ) -> None: ...

    def hook_recv_callback(self, callback) -> None: ...
    def close(self) -> None: ...


class DmCanSdkTransport:
    """Thin adapter around the DM ``dmcan`` Python SDK.

    The SDK is imported lazily so unit tests and sim-only machines do not need
    the vendor wheel or shared library installed.
    """

    def __init__(self) -> None:
        self._context = None
        self._device = None

    def open(self, bus: DmCanBusConfig) -> None:
        _preload_dm_runtime_libs()
        try:
            from dmcan import DmCanContext, dmcan_device_type
        except Exception as exc:  # pragma: no cover - depends on vendor install
            raise DmBackendUnavailableError(
                "DM dmcan SDK is not importable; install dmcan_sdk and libdm_device.so"
            ) from exc

        type_map = {
            "USB2CANFD": dmcan_device_type.USB2CANFD,
            "USB2FDCAN": dmcan_device_type.USB2CANFD,
            "USB2CANFD_DUAL": dmcan_device_type.USB2CANFD_DUAL,
            "DUAL": dmcan_device_type.USB2CANFD_DUAL,
            "LINKX4C": dmcan_device_type.LinkX4C,
        }
        device_type = type_map.get(bus.device_type.upper())
        self._context = DmCanContext()
        count = self._context.find_devices(device_type)
        if count <= bus.device_index:
            self.close()
            raise DmBackendUnavailableError(
                f"DM device index {bus.device_index} not found; discovered {count} device(s)"
            )
        self._device = self._context.get_device(bus.device_index)
        if not self._device.open():
            self.close()
            raise DmBackendUnavailableError(f"failed to open DM device index {bus.device_index}")

    def configure_channel(self, bus: DmCanBusConfig) -> None:
        if self._device is None:
            raise DmBackendUnavailableError("DM device is not open")
        info = self._device.get_channel_baudrate(bus.channel)
        if info is None:
            from dmcan import dmcan_channel_can_info

            info = dmcan_channel_can_info()
        info.channel = bus.channel
        info.canfd = bus.canfd
        info.can_baudrate = bus.can_baudrate
        info.canfd_baudrate = bus.canfd_baudrate
        info.can_sp = bus.can_sp
        info.canfd_sp = bus.canfd_sp
        if not self._device.set_channel_baudrate(bus.channel, info):
            raise DmBackendUnavailableError(f"failed to configure CAN channel {bus.channel}")

    def enable_channel(self, channel: int) -> None:
        if self._device is None:
            raise DmBackendUnavailableError("DM device is not open")
        self._device.enable_channel(channel, True)

    def disable_channel(self, channel: int) -> None:
        if self._device is not None:
            self._device.enable_channel(channel, False)

    def send_can(
        self,
        channel: int,
        can_id: int,
        payload: bytes,
        *,
        canfd: bool,
        extended_id: bool,
        remote_frame: bool,
        brs: bool,
    ) -> None:
        if self._device is None:
            raise DmBackendUnavailableError("DM device is not open")
        ok = self._device.send_can(
            channel,
            can_id,
            len(payload),
            payload,
            canfd,
            extended_id,
            remote_frame,
            brs,
        )
        if not ok:
            raise DmBackendUnavailableError(f"failed to send CAN frame to id {can_id}")

    def hook_recv_callback(self, callback) -> None:
        """Register a callback for incoming CAN frames.

        The callback receives ``(device, usb_rx_frame)`` where ``usb_rx_frame``
        has ``head.can_id``, ``head.dlc``, ``head.ack``, ``head.dir``, and a
        64-byte ``payload`` buffer.
        """
        if self._device is None:
            raise DmBackendUnavailableError("DM device is not open")
        self._device.hook_recv_callback(callback)

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
        if self._context is not None:
            # The DM Python SDK's context destroy path currently aborts on
            # Ubuntu 22.04 after a valid device close when used with the
            # bundled libusb 1.0.27. Device/channel close above is the safety
            # operation; null the private handle so the SDK __del__ skips its
            # broken destroy call during interpreter shutdown.
            if hasattr(self._context, "_ctx"):
                self._context._ctx = None
            self._context = None


def _preload_dm_runtime_libs() -> None:
    """Load bundled runtime libs before the vendor SDK dlopens libdm_device.

    The dmcan SDK's DmCanContext.__init__ calls find_backend_dll_path() which
    searches CWD-relative paths (``./dlls/``, ``./``) that don't exist when
    dora spawns the node from the dataflows/ directory.  We preload both libs
    with absolute paths + RTLD_GLOBAL, then monkey-patch the finder so the
    SDK's CDLL call picks up our absolute path.
    """
    import ctypes

    for name in ("libusb-1.0.so.0", "libdm_device.so"):
        path = next((r / "dlls" / name for r in (REPO_ROOT, CONTROL_ROOT) if (r / "dlls" / name).exists()), REPO_ROOT / "dlls" / name)
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    # Patch the SDK's path finder so CDLL receives our absolute path.
    try:
        import dmcan.dmcan_context as _ctx
        _device_so = str(next((r / "dlls" / "libdm_device.so" for r in (REPO_ROOT, CONTROL_ROOT) if (r / "dlls" / "libdm_device.so").exists()), REPO_ROOT / "dlls" / "libdm_device.so"))
        _ctx.find_backend_dll_path = lambda: _device_so
    except ImportError:
        pass


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


class DmHardwareBackend:
    def __init__(
        self,
        *,
        bus: DmCanBusConfig,
        motors: deque[DmMotorConfig],
        modules: list[dict[str, Any]] | None = None,
        transport: DmCanTransport | None = None,
    ) -> None:
        self.bus = bus
        self.motors = motors
        self.modules = list(modules or [])
        self._transport = transport or DmCanSdkTransport()
        self._states: list[DmMotorState] = [DmMotorState() for _ in self.motors]
        self._is_open = False
        self._channel_enabled = False
        self._motors_enabled = False
        self.last_close_errors: list[Exception] = []
        # CAN id → motor index for routing feedback frames
        self._motor_index_by_can_id: dict[int, int] = {
            m.can_id: i for i, m in enumerate(self.motors)
        }
        self._states_lock = threading.Lock()
        # Frame counters for bus-load estimation
        self._tx_frame_count: int = 0
        self._rx_frame_count: int = 0
        self._bus_load_last_t: float = -1.0

    @classmethod
    def from_config(
        cls,
        cfg: RobotConfig,
        *,
        transport: DmCanTransport | None = None,
    ) -> "DmHardwareBackend":
        raw = cfg.raw
        if transport is None:
            real_backend = str(raw.get("real_backend", "") or raw.get("socketcan_interface", ""))
            if real_backend and not real_backend.startswith("dm_"):
                # SocketCAN path
                from arm_control.hardware.socketcan_transport import SocketCanTransport
                transport = SocketCanTransport(interface=real_backend)
                bus = DmCanBusConfig()
            else:
                bus = DmCanBusConfig.from_mapping(raw.get("bus") or raw.get("dm_bus"))
        else:
            bus = DmCanBusConfig.from_mapping(raw.get("bus") or raw.get("dm_bus"))
        motors = deque(_parse_motor_configs(cfg))
        modules = list(raw.get("modules") or [])
        return cls(bus=bus, motors=motors, modules=modules, transport=transport)

    @property
    def num_motors(self) -> int:
        return len(self.motors)

    def open(self) -> None:
        if self._is_open:
            return
        try:
            self._transport.open(self.bus)
            self._transport.configure_channel(self.bus)
            self._transport.enable_channel(self.bus.channel)
            self._channel_enabled = True
            self._is_open = True
            # Hook CAN receive so motor feedback populates _states.
            self._transport.hook_recv_callback(self._on_can_recv)
            for motor in self.motors:
                if motor.enabled_on_open:
                    self._send_frame(motor, DM_ENABLE_FRAME)
        except Exception:
            self.close()
            raise

    def enable_all(self) -> None:
        """Send ENABLE frames to all motors, putting them in MIT mode."""
        if not self._is_open:
            raise DmBackendUnavailableError("DM backend is not open")
        for motor in self.motors:
            self._send_frame(motor, DM_ENABLE_FRAME)
        self._motors_enabled = True

    def listen_step(self) -> None:
        """Send one MIT zero-torque frame per motor to solicit feedback replies.

        Call this in a loop at your desired feedback rate (e.g. 100 Hz).
        Received replies are processed asynchronously by ``_on_can_recv`` and
        written into ``_states``.  Motors must have been enabled first via
        ``enable_all()``.

        The caller **must** periodically release the GIL (e.g. ``time.sleep(0)``
        or block on I/O) so that the libusb callback thread can deliver
        received CAN frames to ``_on_can_recv``.
        """
        if not self._is_open:
            raise DmBackendUnavailableError("DM backend is not open")
        for i, motor in enumerate(self.motors):
            with self._states_lock:
                pos = self._states[i].position
            frame = pack_mit_control_frame(
                position=pos,
                velocity=0.0,
                kp=0.0,
                kd=0.0,
                torque=0.0,
                limits=motor.limits,
            )
            self._send_frame(motor, frame)
        self._tx_frame_count += len(self.motors)

    @property
    def bus_load_pct(self) -> float:
        """Estimated CAN bus load as a percentage of bandwidth.

        Computed from the frame rate since the previous call, normalised to
        bits/sec at the configured baud rate.  Call periodically; the first
        call after ``open()`` returns 0.0.
        """
        now = time.monotonic()
        if self._bus_load_last_t <= 0:
            self._bus_load_last_t = now
            self._tx_frame_count = 0
            self._rx_frame_count = 0
            return 0.0
        dt = now - self._bus_load_last_t
        self._bus_load_last_t = now
        if dt <= 0:
            return 0.0
        tx = self._tx_frame_count
        rx = self._rx_frame_count
        self._tx_frame_count = 0
        self._rx_frame_count = 0
        bits_per_frame = 120
        bits_per_sec = (tx + rx) * bits_per_frame / dt
        baud = (
            self.bus.canfd_baudrate
            if (self.bus.brs and self.bus.canfd and self.bus.canfd_baudrate > 0)
            else self.bus.can_baudrate
        )
        if baud <= 0:
            return 0.0
        return (bits_per_sec / float(baud)) * 100.0


    def _on_can_recv(self, _device, frame) -> None:
        """Callback: decode incoming CAN frame and update matching motor state.

        Routes by the ``motor_id`` field embedded in the MIT reply payload
        (byte 0, low nibble), not by the CAN frame ID, since motors may
        reply on a shared or different bus ID than their command ID.
        """
        try:
            payload = bytes(frame.payload[i] for i in range(min(frame.head.dlc, 8)))
        except Exception:
            return
        # Only process RX frames (dir=0), not echoes of our own TX.
        if frame.head.dir:
            return
        self._rx_frame_count += 1
        if len(payload) < 6:
            return
        # Extract motor_id from the MIT reply before full decode, so we can
        # pick the correct per-motor limits for decoding.
        motor_id = payload[0] & 0x0F
        motor_idx = self._motor_index_by_can_id.get(motor_id)
        if motor_idx is None:
            return
        motor = self.motors[motor_idx]
        decoded = decode_mit_reply(payload, motor.limits)
        if decoded is None:
            return
        state = self._states[motor_idx]
        with self._states_lock:
            state.position = decoded.position
            state.velocity = decoded.velocity
            state.torque = decoded.torque
            state.motor_id = decoded.motor_id
            state.error = decoded.error
            state.t_mos = decoded.t_mos
            state.t_rotor = decoded.t_rotor

    def apply_command(self, command: dict[str, Any]) -> None:
        if not self._is_open:
            raise DmBackendUnavailableError("DM backend is not open")
        arrays = {
            key: np.asarray(command[key], dtype=np.float64)
            for key in ("position", "velocity", "torque", "kp", "kd")
        }
        for key, values in arrays.items():
            if values.size != self.num_motors:
                raise ValueError(f"{key} command length {values.size} != {self.num_motors}")
        # Pack (and validate) every frame BEFORE sending any, so an out-of-range
        # gain on motor i raises here — with nothing on the bus — instead of after
        # motors 0..i-1 are already commanded, leaving the arm half-commanded.
        frames = [
            pack_mit_control_frame(
                position=float(arrays["position"][i]),
                velocity=float(arrays["velocity"][i]),
                kp=float(arrays["kp"][i]),
                kd=float(arrays["kd"][i]),
                torque=float(arrays["torque"][i]),
                limits=motor.limits,
            )
            for i, motor in enumerate(self.motors)
        ]
        for i, (motor, frame) in enumerate(zip(self.motors, frames)):
            self._send_frame(motor, frame)
            state = self._states[i]
            state.position_cmd = float(arrays["position"][i])
            state.velocity_cmd = float(arrays["velocity"][i])
            state.torque_cmd = float(arrays["torque"][i])
            state.kp = float(arrays["kp"][i])
            state.kd = float(arrays["kd"][i])

    def motor_state(self) -> dict[str, np.ndarray]:
        with self._states_lock:
            return {
                "position": np.array([s.position for s in self._states], dtype=np.float64),
                "velocity": np.array([s.velocity for s in self._states], dtype=np.float64),
                "position_cmd": np.array([s.position_cmd for s in self._states], dtype=np.float64),
                "velocity_cmd": np.array([s.velocity_cmd for s in self._states], dtype=np.float64),
                "torque_cmd": np.array([s.torque_cmd for s in self._states], dtype=np.float64),
                "kp": np.array([s.kp for s in self._states], dtype=np.float64),
                "kd": np.array([s.kd for s in self._states], dtype=np.float64),
                "torque": np.array([s.torque for s in self._states], dtype=np.float64),
            }

    def motor_health(self) -> dict[str, np.ndarray]:
        """Per-motor fault/temperature snapshot from the latest MIT replies.

        Surfaces the DM error nibble and T_mos/T_rotor that ``motor_state``
        decodes then drops, so the safety layer can act on faults/over-temp.
        """
        with self._states_lock:
            return {
                "motor_id": np.array([s.motor_id for s in self._states], dtype=np.int64),
                "error": np.array([s.error for s in self._states], dtype=np.int64),
                "t_mos": np.array([s.t_mos for s in self._states], dtype=np.float64),
                "t_rotor": np.array([s.t_rotor for s in self._states], dtype=np.float64),
            }

    def safe_stop(self) -> None:
        """Zero-torque + DISABLE every motor WITHOUT tearing down the channel.

        The mid-run safe-stop for the software safety layer (disarm / deadman /
        fault reflex).  Unlike ``close()`` the transport stays open so the bridge
        can re-arm.  Per-frame errors are swallowed so a failed disable can never
        crash the real-time loop — the latched disarm is the backstop.
        """
        if not self._is_open:
            return
        errors: list[Exception] = []
        for motor in list(self.motors):
            self._zero_torque_and_disable(motor, errors)
        self._motors_enabled = False

    def close(self) -> None:
        if not self._is_open and not self._channel_enabled:
            return
        self.last_close_errors.clear()
        if self._is_open:
            # Send per-motor zero-torque + disable using each motor's own limits
            # so the encoding is correct regardless of motor type.
            for motor in list(self.motors):
                self._zero_torque_and_disable(motor, self.last_close_errors)
            self._motors_enabled = False
            time.sleep(0.05)  # let the CAN frames flush before tearing down
        if self._channel_enabled:
            try:
                self._transport.disable_channel(self.bus.channel)
            except Exception as exc:  # pragma: no cover
                self.last_close_errors.append(exc)
            self._channel_enabled = False
        try:
            self._transport.close()
        except Exception as exc:  # pragma: no cover
            self.last_close_errors.append(exc)
        self._is_open = False

    def _zero_torque_and_disable(self, motor: DmMotorConfig, errors: list) -> None:
        """Send this motor a zero-gain hold then DISABLE, collecting send errors.

        Shared by ``close()`` and ``safe_stop()`` so both put a motor into the
        identical safe state (kp=kd=torque=0 then DM_DISABLE) at its own limits.
        """
        zt = pack_mit_control_frame(
            position=0.0, velocity=0.0, kp=0.0, kd=0.0, torque=0.0,
            limits=motor.limits,
        )
        for payload in (zt, DM_DISABLE_FRAME):
            try:
                self._send_frame(motor, payload)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    def _send_frame(self, motor: DmMotorConfig, payload: bytes) -> None:
        self._transport.send_can(
            self.bus.channel,
            motor.can_id,
            payload,
            canfd=self.bus.canfd,
            extended_id=self.bus.extended_id,
            remote_frame=self.bus.remote_frame,
            brs=self.bus.brs,
        )



def _parse_motor_configs(cfg: RobotConfig) -> list[DmMotorConfig]:
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


__all__ = [
    "DM_DISABLE_FRAME",
    "DM_ENABLE_FRAME",
    "DmBackendUnavailableError",
    "DmCanBusConfig",
    "DmCanSdkTransport",
    "DmHardwareBackend",
    "DmMotorConfig",
    "DmMotorLimits",
    "pack_mit_control_frame",
]
