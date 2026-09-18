"""宇树 S288 无刷数字舵机：规格、MIT 协议编解码、总线与关节读取。

S288 规格（来源：Unitree 官网 DigitalServo 页面 / YS-342026-S288 数据，2026 查得）
--------------------------------------------------------------------------
  减速比 gear_ratio           288.35:1
  力矩常数 torque_constant    0.554 N·m/A
  最大堵转扭矩                0.6 N·m（与转速反向）
  最大空载速度                16.5 rad/s @12V
  空载电流                    0.27 A @12V
  输入电压                    6.4V ~ 12.6V（推荐 12.6V）
  编码器                      双绝对值：转子端 15bit（8192 步）+ 输出端
  通信                        半双工异步串口，8N1，6,000,000 bps，TTL 多点总线
  协议                        Unitree Custom Protocol（宇树私有），ID 0~14
  控制模式                    混合模式（转子 q / dq / tau / kp / kd，即 MIT 模式）
  反馈                        转子扭矩、转子角度、输出端角度、转子角速度、
                              温度、电压、错误状态
  尺寸/重量                   20 x 34 x 26 mm / 19.5 g

单位与换算约定（这是"电机参数不同"要处理的核心）
--------------------------------------------------------------------------
宇树电机的 `q/dq/tau/kp/kd` 都是**转子侧**量，而机械臂关节角是**输出侧**量。
记减速比 r = 288.35：

    q_out   = q_rotor / r
    dq_out  = dq_rotor / r
    tau_out = tau_rotor * r            （理想无损；实际有摩擦/效率损失）
    kp_rotor = kp_out / r^2
    kd_rotor = kd_out / r^2

反向（要让输出端到达 q_out，命令转子）：

    q_rotor_cmd = q_out * r
    tau_rotor_cmd = tau_out / r
    kp_rotor_cmd = kp_out / r^2
    kd_rotor_cmd = kd_out / r^2

反馈里同时给了"转子角度"和"输出端角度"，优先用**输出端角度**，避免自己除 r
引入的累积误差；若固件未上报输出端角度，则退回 `q_rotor / r`。

串口通信走宇树官方 SDK
--------------------------------------------------------------------------
真实硬件默认用官方 `unitree_actuator_sdk`（编译后的 pybind 扩展）通信：

    from unitree_actuator_sdk import SerialPort, MotorCmd, MotorData
    serial = SerialPort('/dev/ttyUSB0')
    cmd, data = MotorCmd(), MotorData()
    cmd.motorType = data.motorType = MotorType.<型号>
    cmd.mode = queryMotorMode(...); cmd.id = ...; cmd.q/dq/kp/kd/tau = ...
    serial.sendRecv(cmd, data)   # data.q/dq/temp/merror 为反馈

SDK 的 `q/dq/tau/kp/kd` 全是**转子侧**量，而本模块对外统一用**输出端**语义；
`UnitreeSdkS288Bus` 在边界用 `S288Spec` 做换算（这是"电机参数不同"的核心）。

`SerialS288Bus`（自实现的 0xFE 0xEE MIT 帧 + CRC16-CCITT）保留作离线/无 SDK
时的后备，并用于自检。

    ★ 若官方 SDK 的 `MotorType` 里没有 S288 名称，`UnitreeSdkS288Bus` 会列出
      可选项并在构造时报错，避免悄无声息地发错协议。

对外主接口
--------------------------------------------------------------------------
    S288Spec                     规格与单位换算
    S288Codec                    MIT 命令/反馈帧编解码（CRC16-CCITT，后备）
    S288Bus(Protocol)            总线抽象：read / write / close
    UnitreeSdkS288Bus            官方 unitree_actuator_sdk 串口实现（首选）
    SerialS288Bus                 自实现帧的串口实现（后备）
    FakeS288Bus                   仿真实现，无需硬件
    S288JointChain               多电机链：读输出端关节角/速度，写输出端位置
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

# MIT 混合模式里，刚度/阻尼都为 0 时相当于纯力矩控制；这里给出保守默认。
_MODE_FOC = 0x01  # 宇树定义：0x01 = FOC/伺服工作模式，0x00 = 停止/待机


# --------------------------------------------------------------------------- #
# 规格与单位换算
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class S288Spec:
    """S288 的物理规格与转子<->输出端换算。"""

    gear_ratio: float = 288.35
    torque_constant: float = 0.554          # N·m/A（转子侧）
    stall_torque_nm: float = 0.6            # 输出端堵转
    max_speed_rad_s: float = 16.5           # 输出端空载 @12V
    voltage_min_v: float = 6.4
    voltage_max_v: float = 12.6
    voltage_nominal_v: float = 12.6
    encoder_steps: int = 8192               # 输出角度分辨率 8192 步
    baudrate: int = 6_000_000
    rotor_encoder_bits: int = 15

    @property
    def rad_per_step(self) -> float:
        """输出端每步对应角度（rad）。"""
        return 2.0 * np.pi / self.encoder_steps

    # -- 输出端 -> 转子端（下发命令用） --
    def output_to_rotor_angle(self, q_out: float | np.ndarray):
        return np.asarray(q_out, dtype=float) * self.gear_ratio

    def output_to_rotor_velocity(self, dq_out: float | np.ndarray):
        return np.asarray(dq_out, dtype=float) * self.gear_ratio

    def output_to_rotor_torque(self, tau_out: float | np.ndarray):
        return np.asarray(tau_out, dtype=float) / self.gear_ratio

    def output_to_rotor_kp(self, kp_out: float | np.ndarray):
        return np.asarray(kp_out, dtype=float) / (self.gear_ratio**2)

    def output_to_rotor_kd(self, kd_out: float | np.ndarray):
        return np.asarray(kd_out, dtype=float) / (self.gear_ratio**2)

    # -- 转子端 -> 输出端（读反馈用） --
    def rotor_to_output_angle(self, q_rotor: float | np.ndarray):
        return np.asarray(q_rotor, dtype=float) / self.gear_ratio

    def rotor_to_output_velocity(self, dq_rotor: float | np.ndarray):
        return np.asarray(dq_rotor, dtype=float) / self.gear_ratio

    def rotor_to_output_torque(self, tau_rotor: float | np.ndarray):
        return np.asarray(tau_rotor, dtype=float) * self.gear_ratio

    def torque_to_current(self, tau_rotor_nm: float | np.ndarray):
        """转子扭矩 -> 相电流估计（A），仅用于诊断/限流参考。"""
        return np.asarray(tau_rotor_nm, dtype=float) / self.torque_constant


# --------------------------------------------------------------------------- #
# 帧编解码
# --------------------------------------------------------------------------- #
def crc16_ccitt(data: bytes, crc: int = 0x0000) -> int:
    """CRC16-CCITT/XMODEM（poly 0x1021，init 0x0000，宇树电机协议用）。"""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


@dataclass(frozen=True)
class S288Command:
    """一条输出端语义的 MIT 命令；由 codec 换算成转子侧字节。"""

    q_out: float = 0.0
    dq_out: float = 0.0
    tau_out: float = 0.0
    kp_out: float = 0.0
    kd_out: float = 0.0


@dataclass(frozen=True)
class S288State:
    """一台电机的反馈（转子侧原始量 + 已换算的输出端量）。"""

    motor_id: int
    q_rotor: float
    dq_rotor: float
    tau_rotor: float
    q_out: float
    dq_out: float
    tau_out: float
    temperature_c: float = float("nan")
    voltage_v: float = float("nan")
    error: int = 0
    # 固件是否真的上报了"输出端角度"；False 时 q_out 由 q_rotor/r 推出。
    has_output_angle: bool = False


@dataclass
class S288Codec:
    """宇树 MIT 协议帧编解码。

    默认布局（**需按 S288 固件核对**）：

        命令帧（主机 -> 电机），共 26 字节：
            [0:2]   header      0xFE 0xEE
            [2]     mode        0x01=FOC / 0x00=stop
            [3]     motor_id
            [4:8]   q_rotor     float32 LE
            [8:12]  dq_rotor    float32 LE
            [12:16] kp_rotor    float32 LE
            [16:20] kd_rotor    float32 LE
            [20:24] tau_rotor   float32 LE
            [24:26] crc16       LE，覆盖 [0:24]

        反馈帧（电机 -> 主机），默认按"输出端角度可选"解析：
            [0:2]   header
            [2]     mode/status
            [3]     motor_id
            [4:8]   q_rotor
            [8:12]  dq_rotor
            [12:16] tau_rotor
            [16:20] q_out       （若固件上报；否则该 4 字节为温度/电压）
            [20]    temperature
            [21]    error
            ...
    """

    header: tuple[int, int] = (0xFE, 0xEE)
    mode_foc: int = _MODE_FOC
    mode_stop: int = 0x00
    has_crc: bool = True
    response_len: int = 24
    # 反馈里是否有独立的输出端角度字段；不确定时保持 False，用 q_rotor/r 推导。
    feedback_has_output_angle: bool = False
    # 反馈字段偏移（字节），便于按实测调整。
    fb_id_offset: int = 3
    fb_q_offset: int = 4
    fb_dq_offset: int = 8
    fb_tau_offset: int = 12
    fb_qout_offset: int = 16
    fb_temp_offset: int = 20
    fb_error_offset: int = 21

    def pack_command(self, motor_id: int, command: S288Command, spec: S288Spec) -> bytes:
        q_r = float(spec.output_to_rotor_angle(command.q_out))
        dq_r = float(spec.output_to_rotor_velocity(command.dq_out))
        kp_r = float(spec.output_to_rotor_kp(command.kp_out))
        kd_r = float(spec.output_to_rotor_kd(command.kd_out))
        tau_r = float(spec.output_to_rotor_torque(command.tau_out))
        mode = self.mode_foc
        body = struct.pack(
            "<BBfffff",
            mode,
            int(motor_id) & 0xFF,
            q_r,
            dq_r,
            kp_r,
            kd_r,
            tau_r,
        )
        frame = bytes(self.header) + body
        if self.has_crc:
            frame += struct.pack("<H", crc16_ccitt(frame))
        return frame

    def pack_stop(self, motor_id: int) -> bytes:
        """停止/待机帧（只保留头、mode、id 与 CRC）。"""
        frame = bytes(self.header) + bytes([self.mode_stop, int(motor_id) & 0xFF])
        if self.has_crc:
            frame += struct.pack("<H", crc16_ccitt(frame))
        return frame

    def unpack_state(self, frame: bytes, spec: S288Spec) -> S288State:
        if len(frame) < self.fb_tau_offset + 4:
            raise ValueError(f"S288 反馈帧过短：{len(frame)} 字节")
        if tuple(frame[0:2]) != tuple(self.header):
            raise ValueError(f"S288 反馈帧头不符：{frame[0:2]!r}")
        motor_id = frame[self.fb_id_offset]
        q_r = struct.unpack_from("<f", frame, self.fb_q_offset)[0]
        dq_r = struct.unpack_from("<f", frame, self.fb_dq_offset)[0]
        tau_r = struct.unpack_from("<f", frame, self.fb_tau_offset)[0]
        has_qout = self.feedback_has_output_angle and len(
            frame
        ) >= self.fb_qout_offset + 4
        if has_qout:
            q_out = struct.unpack_from("<f", frame, self.fb_qout_offset)[0]
        else:
            q_out = float(spec.rotor_to_output_angle(q_r))
        temperature = float(frame[self.fb_temp_offset]) if len(frame) > self.fb_temp_offset else float("nan")
        error = int(frame[self.fb_error_offset]) if len(frame) > self.fb_error_offset else 0
        return S288State(
            motor_id=motor_id,
            q_rotor=q_r,
            dq_rotor=dq_r,
            tau_rotor=tau_r,
            q_out=q_out,
            dq_out=float(spec.rotor_to_output_velocity(dq_r)),
            tau_out=float(spec.rotor_to_output_torque(tau_r)),
            temperature_c=temperature,
            error=error,
            has_output_angle=has_qout,
        )


# --------------------------------------------------------------------------- #
# 总线抽象
# --------------------------------------------------------------------------- #
class S288Bus(Protocol):
    """一条多点总线上的若干 S288。"""

    def write_commands(self, commands: dict[int, S288Command]) -> None:
        """（可选）下发命令；实现可以在这里顺带取回状态。"""
        ...

    def read_states(self) -> dict[int, S288State]:
        """读取总线上各电机的最新状态。"""
        ...

    def close(self) -> None:
        ...


class FakeS288Bus:
    """纯软件的 S288 总线：一阶跟踪 + 速度估计，用于无硬件开发。

    每台电机维护一个输出端角度，按一阶响应逼近最近的命令目标：

        q <- q + (q_cmd - q) * (dt / tau)
    """

    def __init__(
        self,
        motor_ids: Sequence[int],
        spec: S288Spec | None = None,
        time_constant_s: float = 0.1,
        seed: int | None = 0,
    ) -> None:
        self._ids = [int(i) for i in motor_ids]
        self.spec = spec or S288Spec()
        self._tau = max(float(time_constant_s), 1e-3)
        self._rng = np.random.default_rng(seed)
        self._q = {i: 0.0 for i in self._ids}
        self._dq = {i: 0.0 for i in self._ids}
        self._q_cmd = {i: 0.0 for i in self._ids}
        self._last_t = time.monotonic()
        self._lock = threading.Lock()

    def write_commands(self, commands: dict[int, S288Command]) -> None:
        with self._lock:
            for mid, cmd in commands.items():
                if mid in self._q_cmd:
                    self._q_cmd[mid] = float(cmd.q_out)

    def read_states(self) -> dict[int, S288State]:
        now = time.monotonic()
        with self._lock:
            dt = max(now - self._last_t, 0.0)
            self._last_t = now
            states: dict[int, S288State] = {}
            for mid in self._ids:
                alpha = min(dt / self._tau, 1.0) if dt > 0 else 0.0
                prev = self._q[mid]
                q = prev + (self._q_cmd[mid] - prev) * alpha
                dq = (q - prev) / dt if dt > 1e-9 else 0.0
                dq += float(self._rng.normal(0.0, 1e-4))  # 轻微编码器噪声
                self._q[mid] = q
                self._dq[mid] = dq
                states[mid] = S288State(
                    motor_id=mid,
                    q_rotor=float(self.spec.output_to_rotor_angle(q)),
                    dq_rotor=float(self.spec.output_to_rotor_velocity(dq)),
                    tau_rotor=0.0,
                    q_out=q,
                    dq_out=dq,
                    tau_out=0.0,
                    temperature_c=25.0,
                    voltage_v=self.spec.voltage_nominal_v,
                    error=0,
                    has_output_angle=True,
                )
            return states

    def close(self) -> None:
        pass


class SerialS288Bus:
    """真实串口总线（半双工 TTL 多点，默认 6 Mbps）。

    半双工时序：主机发一帧给某 ID，该电机在同一总线上回一帧状态；因此
    `exchange_one` 逐台发命令并读取响应。多台时按 `motor_ids` 顺序轮询。

    依赖 pyserial（arm_control 未强制依赖，故懒加载；未安装时给出清晰报错）。
    """

    def __init__(
        self,
        motor_ids: Sequence[int],
        port: str,
        spec: S288Spec | None = None,
        codec: S288Codec | None = None,
        response_timeout_s: float = 0.005,
        inter_frame_s: float = 0.0002,
    ) -> None:
        try:
            import serial  # noqa: F401  (懒加载，避免无硬件时 import 失败)
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise RuntimeError(
                "SerialS288Bus 需要 pyserial：pip install pyserial"
            ) from exc
        self.spec = spec or S288Spec()
        self.codec = codec or S288Codec()
        self._ids = [int(i) for i in motor_ids]
        self._port = port
        self._response_timeout_s = float(response_timeout_s)
        self._inter_frame_s = float(inter_frame_s)
        self._serial = None
        self._last_states: dict[int, S288State] = {}
        self._lock = threading.Lock()
        self._open()

    def _open(self) -> None:
        import serial

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self.spec.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self._response_timeout_s,
            write_timeout=self._response_timeout_s,
        )
        self._serial.reset_input_buffer()

    def _exchange_one(self, motor_id: int, command: S288Command | None) -> None:
        assert self._serial is not None
        frame = (
            self.codec.pack_stop(motor_id)
            if command is None
            else self.codec.pack_command(motor_id, command, self.spec)
        )
        self._serial.reset_input_buffer()
        self._serial.write(frame)
        self._serial.flush()
        deadline = time.monotonic() + self._response_timeout_s
        buf = bytearray()
        while time.monotonic() < deadline and len(buf) < self.codec.response_len:
            chunk = self._serial.read(self.codec.response_len - len(buf))
            if chunk:
                buf.extend(chunk)
            else:
                time.sleep(self._inter_frame_s)
        if len(buf) < self.codec.fb_tau_offset + 4:
            raise TimeoutError(f"S288 id={motor_id} 无有效反馈（收到 {len(buf)} 字节）")
        self._last_states[motor_id] = self.codec.unpack_state(bytes(buf), self.spec)

    def write_commands(self, commands: dict[int, S288Command]) -> None:
        with self._lock:
            for mid in self._ids:
                self._exchange_one(mid, commands.get(mid))

    def read_states(self) -> dict[int, S288State]:
        with self._lock:
            for mid in self._ids:
                if mid not in self._last_states:
                    self._exchange_one(mid, None)
            return dict(self._last_states)

    def close(self) -> None:
        if self._serial is not None:
            try:
                for mid in self._ids:
                    self._serial.write(self.codec.pack_stop(mid))
            except Exception:
                pass
            self._serial.close()
            self._serial = None


# --------------------------------------------------------------------------- #
# 官方 SDK 串口实现（首选）
# --------------------------------------------------------------------------- #
class UnitreeSdkS288Bus:
    """宇树官方 `unitree_actuator_sdk` 的串口总线。

    对外仍是输出端语义的 `S288Command`/`S288State`，边界处按 `S288Spec`
    在转子端与输出端之间换算——SDK 的 `q/dq/tau/kp/kd` 都是转子侧量。

    参数
    ----
    motor_type: SDK `MotorType` 里的枚举名。S288 若不在官方枚举里，构造时会
        列出全部可选项并报错；把 `motor_type` 设成实际可用的名称即可。
    port:       串口设备，如 `/dev/ttyUSB0`（S288 为半双工 TTL 多点总线）。

    依赖：编译好的 `unitree_actuator_sdk`（pybind 扩展）在 PYTHONPATH 上。
    """

    def __init__(
        self,
        motor_ids: Sequence[int],
        port: str,
        spec: S288Spec | None = None,
        motor_type: str = "S288",
    ) -> None:
        self.spec = spec or S288Spec()
        self._ids = [int(i) for i in motor_ids]
        self._port = port
        self._motor_type_name = motor_type
        self._sdk = _import_unitree_actuator_sdk()
        self._motor_type = self._resolve_motor_type(motor_type)
        self._mode = self._sdk.queryMotorMode(
            self._motor_type, self._sdk.MotorMode.FOC
        )
        self._serial = self._open_serial(port)
        self._lock = threading.Lock()
        self._last_states: dict[int, S288State] = {}
        self._last_target: dict[int, float] = {i: 0.0 for i in self._ids}

    # -- 初始化辅助 ---------------------------------------------------------
    def _resolve_motor_type(self, name: str):
        mt = getattr(self._sdk.MotorType, name, None)
        if mt is None:
            options = [n for n in dir(self._sdk.MotorType) if not n.startswith("_")]
            raise RuntimeError(
                f"官方 SDK 的 MotorType 里没有 {name!r}；可用：{options}。"
                "若是较新的 S288/J288，请按实际枚举名设置 motor_type。"
            )
        return mt

    def _open_serial(self, port: str):
        # 不同版本的 pybind 签名可能是 (port) 或 (port, baudrate)。
        try:
            return self._sdk.SerialPort(port, self.spec.baudrate)
        except TypeError:
            return self._sdk.SerialPort(port)

    # -- 单次收发 -----------------------------------------------------------
    def _exchange(self, motor_id: int, command: S288Command | None) -> None:
        sdk = self._sdk
        cmd = sdk.MotorCmd()
        data = sdk.MotorData()
        cmd.motorType = data.motorType = self._motor_type
        cmd.mode = self._mode
        cmd.id = int(motor_id)

        if command is None:
            q_out = self._last_target.get(motor_id, 0.0)
            dq_out = tau_out = kp_out = kd_out = 0.0
        else:
            q_out, dq_out, tau_out, kp_out, kd_out = (
                command.q_out, command.dq_out, command.tau_out,
                command.kp_out, command.kd_out,
            )
        spec = self.spec
        cmd.q = float(spec.output_to_rotor_angle(q_out))
        cmd.dq = float(spec.output_to_rotor_velocity(dq_out))
        cmd.kp = float(spec.output_to_rotor_kp(kp_out))
        cmd.kd = float(spec.output_to_rotor_kd(kd_out))
        cmd.tau = float(spec.output_to_rotor_torque(tau_out))

        self._serial.sendRecv(cmd, data)

        q_r, dq_r, tau_r = float(data.q), float(data.dq), float(data.tau)
        temp = float(getattr(data, "temp", float("nan")))
        err = int(getattr(data, "merror", 0))
        self._last_states[motor_id] = S288State(
            motor_id=motor_id,
            q_rotor=q_r,
            dq_rotor=dq_r,
            tau_rotor=tau_r,
            q_out=float(spec.rotor_to_output_angle(q_r)),
            dq_out=float(spec.rotor_to_output_velocity(dq_r)),
            tau_out=float(spec.rotor_to_output_torque(tau_r)),
            temperature_c=temp,
            error=err,
            has_output_angle=False,
        )
        self._last_target[motor_id] = float(q_out)

    # -- S288Bus 接口 -------------------------------------------------------
    def write_commands(self, commands: dict[int, S288Command]) -> None:
        with self._lock:
            for mid in self._ids:
                self._exchange(mid, commands.get(mid))

    def read_states(self) -> dict[int, S288State]:
        with self._lock:
            for mid in self._ids:
                if mid not in self._last_states:
                    self._exchange(mid, None)
            return dict(self._last_states)

    def close(self) -> None:
        with self._lock:
            try:
                for mid in self._ids:  # 零刚度零阻尼，撤销主动控制
                    self._exchange(mid, S288Command())
            except Exception:
                pass
        close = getattr(self._serial, "close", None)
        if callable(close):
            close()


def _import_unitree_actuator_sdk():
    """懒加载官方 SDK，缺失时给出可操作的报错。"""
    try:
        import unitree_actuator_sdk as sdk  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于部署环境
        raise RuntimeError(
            "UnitreeSdkS288Bus 需要官方 unitree_actuator_sdk。请编译该 SDK 并把 "
            "生成的扩展（如 lib/unitree_actuator_sdk*.so）加入 PYTHONPATH，"
            "或改用 bus=serial_raw / fake。"
        ) from exc
    required = ("SerialPort", "MotorCmd", "MotorData", "MotorType", "MotorMode", "queryMotorMode")
    missing = [name for name in required if not hasattr(sdk, name)]
    if missing:
        raise RuntimeError(f"unitree_actuator_sdk 缺少接口：{missing}")
    return sdk


# --------------------------------------------------------------------------- #
# 多电机链：面向关节角/速度的读写
# --------------------------------------------------------------------------- #
@dataclass
class S288JointChain:
    """一挂 S288 关节：读输出端角度/速度，写输出端位置目标。

    这是"电机层"到"关节层"的最后一跳，之后由 LeaderArm 施加标定与平滑。
    """

    motor_ids: Sequence[int]
    bus: S288Bus
    spec: S288Spec = field(default_factory=S288Spec)
    codec: S288Codec = field(default_factory=S288Codec)

    def __post_init__(self) -> None:
        self.motor_ids = [int(i) for i in self.motor_ids]
        self._lock = threading.Lock()

    @property
    def n(self) -> int:
        return len(self.motor_ids)

    def read_positions(self) -> np.ndarray:
        states = self.bus.read_states()
        out = np.empty(self.n, dtype=float)
        for i, mid in enumerate(self.motor_ids):
            st = states.get(mid)
            if st is None:
                raise RuntimeError(f"S288 id={mid} 缺少状态反馈")
            out[i] = st.q_out
        return out

    def read_positions_and_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        states = self.bus.read_states()
        pos = np.empty(self.n, dtype=float)
        vel = np.empty(self.n, dtype=float)
        for i, mid in enumerate(self.motor_ids):
            st = states.get(mid)
            if st is None:
                raise RuntimeError(f"S288 id={mid} 缺少状态反馈")
            pos[i] = st.q_out
            vel[i] = st.dq_out
        return pos, vel

    def command_positions(
        self,
        q_out: Sequence[float],
        kp_out: Sequence[float] | float = 0.0,
        kd_out: Sequence[float] | float = 0.0,
        dq_out: Sequence[float] = (),
        tau_out: Sequence[float] = (),
    ) -> None:
        """按输出端语义下发位置目标（kp/kd 也是输出端量，内部按 r^2 换算）。"""
        q = np.asarray(q_out, dtype=float)
        if q.shape != (self.n,):
            raise ValueError(f"q_out 长度应为 {self.n}，得到 {q.shape}")
        kp = np.broadcast_to(np.asarray(kp_out, dtype=float), (self.n,))
        kd = np.broadcast_to(np.asarray(kd_out, dtype=float), (self.n,))
        dq = (
            np.asarray(dq_out, dtype=float)
            if len(dq_out)
            else np.zeros(self.n, dtype=float)
        )
        tau = (
            np.asarray(tau_out, dtype=float)
            if len(tau_out)
            else np.zeros(self.n, dtype=float)
        )
        commands = {
            mid: S288Command(
                q_out=float(q[i]),
                dq_out=float(dq[i]),
                tau_out=float(tau[i]),
                kp_out=float(kp[i]),
                kd_out=float(kd[i]),
            )
            for i, mid in enumerate(self.motor_ids)
        }
        self.bus.write_commands(commands)

    def release(self) -> None:
        """撤销所有电机的使能（发停止帧），让机械臂可被手动拖动。"""
        self.bus.write_commands({mid: S288Command() for mid in self.motor_ids})

    def close(self) -> None:
        self.bus.close()
