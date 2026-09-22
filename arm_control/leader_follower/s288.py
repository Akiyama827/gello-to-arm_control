"""宇树 S288/J288 无刷数字舵机：规格、官方协议编解码、总线与关节读取。

S288 规格（来源：Unitree 官网 DigitalServo 页面 / YS-342026-S288 数据）
----------------------------------------------------------------------
  减速比 gear_ratio           288.35:1（官方 RATIO = 70070 / 243）
  力矩常数 torque_constant    0.554 N·m/A
  最大堵转扭矩                0.6 N·m（与转速反向）
  最大空载速度                16.5 rad/s @12V（J288 @25.2V 为 35 rad/s）
  输入电压                    6.4V ~ 12.6V（推荐 12.6V；J288 推荐 25.2V）
  编码器                      双绝对值：转子端 15bit + 输出端 13bit
  通信                        半双工 TTL 单线多点，8N1，6,000,000 bps（固定）
  总线容量                    最多 15 台（ID 0~14），ID 15 = 广播（无回包）
  控制模式                    0 = 停止并锁力；1 = 混合闭环（转子 q/dq/tau/kp/kd）

协议来源（重要）
----------------------------------------------------------------------
S288/J288 **不在** 官方 ``unitree_actuator_sdk`` 里（该 SDK 的 ``MotorType`` 只有
A1 / B1 / GO-M8010-6）。宇树为 J288/S288 单独开源了通信协议与参考实现：

    https://github.com/unitreerobotics/digital_servo
    specs/protocol.md（协议）、python/servo_demo.py（PC 参考）、
    stm32/（嵌入式参考）

本模块的帧格式、CRC32、定点换算与上述官方参考**逐字节对齐**，不依赖任何预编译
SDK，只用 pyserial 直接收发。注意 S288 是**半双工 TTL 单线**多点总线，需要 USB-TTL
收发一体适配器，不能把 TX/RX 分别接到总线上。

帧格式
----------------------------------------------------------------------
  控制包（主机 -> 舵机）20 字节：
      [0:2]    head       0xFE 0xEE
      [2]      mode_byte  [3:0]=id, [6:4]=mode, [7]=timeout
      [3]      reserved    0x00
      [4:16]   comd        12 字节定点（int16 tor, int16 spd, int32 pos,
                           int16 k_pos, int16 k_spd，小端）
      [16:20]  CRC32       小端，覆盖 [0:16]
  反馈包（舵机 -> 主机）26 字节：
      [0:2]    head       0xFC 0xEE
      [2]      mode_byte
      [3:22]   fbk         19 字节（布局见 ``S288Codec``）
      [22:26]  CRC32       小端，覆盖 [2:22]

CRC32：多项式 0x04C11DB7，初值 0xFFFFFFFF；按 32 位**小端字**、每字节查表推进
（与官方 ``crc32_lookup_byte_by_byte`` 完全一致）。

单位与换算（"电机参数不同"的核心）
----------------------------------------------------------------------
协议把输出端物理量换算成定点原始值（官方公式，RATIO = 288.35）：

    k_pos   = Kp   / RATIO^2 * 1_280_000       (0 .. 2128.523 -> 0..32767)
    k_spd   = Kd   / RATIO^2 * 128_000_000     (0 .. 21.285   -> 0..32767)
    pos_des = q_out  * RATIO * 32768 / (2π)    (±1428.019 rad -> ±2^31)
    spd_des = dq_out * RATIO * 2.560 / (2π)    (±278.901 rad/s -> ±32767)
    tor_des = tau_out/ RATIO * 256_000          (±36.909 N·m -> ±32767)

反馈反向：

    q_out   = 2π * pos    / 32768 / RATIO      （转子端多圈位置 -> 输出端）
    dq_out  = (speed / 2.56) * 2π / RATIO
    tau_out = torque / 256000 * RATIO
    ExPos   = 2π * OutPos / 8192               （输出端单圈绝对角，0..2π）

关节角优先用**多圈**的 ``q_out``（由转子 pos 推得）；``ExPos`` 只给单圈绝对角，
用于上电找零 / 诊断。

对外主接口
----------------------------------------------------------------------
    S288Spec                     规格与定点换算
    S288Codec                    官方 20B/26B 帧编解码（CRC32）
    crc32_unitree                宇树 CRC32
    S288Bus(Protocol)            总线抽象：read / write / close
    SerialS288Bus                 真实串口实现（官方协议，pyserial，首选）
    FakeS288Bus                   仿真实现，无需硬件
    S288JointChain               多电机链：读输出端关节角/速度，写输出端位置
    UnitreeSdkS288Bus            已废弃：官方 SDK 不支持 S288，构造即报错
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

# 协议里 mode 字段的取值（官方 RIS_Mode_t.status）
_MODE_STOP = 0x0  # 停止并锁力
_MODE_FOC = 0x1   # 混合闭环（位置/速度/力矩）


# --------------------------------------------------------------------------- #
# 规格与单位换算
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class S288Spec:
    """S288 的物理规格，以及输出端物理量 <-> 协议定点原始值的换算。"""

    gear_ratio: float = 70070.0 / 243.0     # = 288.35...
    torque_constant: float = 0.554          # N·m/A（转子侧）
    stall_torque_nm: float = 0.6            # 输出端堵转
    max_speed_rad_s: float = 16.5           # 输出端空载 @12V
    voltage_min_v: float = 6.4
    voltage_max_v: float = 12.6
    voltage_nominal_v: float = 12.6
    encoder_steps: int = 8192               # 输出端绝对编码器 13bit
    rotor_encoder_bits: int = 15
    baudrate: int = 6_000_000

    # -- 输出端 <-> 转子端（解析用，SDK 风格）--
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

    def rotor_to_output_angle(self, q_rotor: float | np.ndarray):
        return np.asarray(q_rotor, dtype=float) / self.gear_ratio

    def rotor_to_output_velocity(self, dq_rotor: float | np.ndarray):
        return np.asarray(dq_rotor, dtype=float) / self.gear_ratio

    def rotor_to_output_torque(self, tau_rotor: float | np.ndarray):
        return np.asarray(tau_rotor, dtype=float) * self.gear_ratio

    def torque_to_current(self, tau_rotor_nm: float | np.ndarray):
        """转子扭矩 -> 相电流估计（A），仅用于诊断/限流参考。"""
        return np.asarray(tau_rotor_nm, dtype=float) / self.torque_constant

    # -- 输出端物理量 <-> 协议定点原始值（官方 digital_servo 公式）--
    def output_pos_to_raw(self, q_out: float | np.ndarray):
        return np.asarray(q_out, dtype=float) * self.gear_ratio * (32768.0 / (2.0 * np.pi))

    def raw_to_output_pos(self, raw: float | np.ndarray):
        return np.asarray(raw, dtype=float) * (2.0 * np.pi) / 32768.0 / self.gear_ratio

    def raw_to_rotor_pos(self, raw: float | np.ndarray):
        return np.asarray(raw, dtype=float) * (2.0 * np.pi) / 32768.0

    def output_spd_to_raw(self, dq_out: float | np.ndarray):
        return np.asarray(dq_out, dtype=float) * self.gear_ratio * (2.560 / (2.0 * np.pi))

    def raw_to_output_spd(self, raw: float | np.ndarray):
        return np.asarray(raw, dtype=float) * (2.0 * np.pi) / 2.560 / self.gear_ratio

    def raw_to_rotor_spd(self, raw: float | np.ndarray):
        return np.asarray(raw, dtype=float) * (2.0 * np.pi) / 2.560

    def output_torque_to_raw(self, tau_out: float | np.ndarray):
        return np.asarray(tau_out, dtype=float) / self.gear_ratio * 256000.0

    def raw_to_output_torque(self, raw: float | np.ndarray):
        return np.asarray(raw, dtype=float) / 256000.0 * self.gear_ratio

    def raw_to_rotor_torque(self, raw: float | np.ndarray):
        return np.asarray(raw, dtype=float) / 256000.0

    def output_kp_to_raw(self, kp_out: float | np.ndarray):
        return np.asarray(kp_out, dtype=float) / (self.gear_ratio**2) * 1_280_000.0

    def raw_to_output_kp(self, raw: float | np.ndarray):
        return np.asarray(raw, dtype=float) / 1_280_000.0 * (self.gear_ratio**2)

    def output_kd_to_raw(self, kd_out: float | np.ndarray):
        return np.asarray(kd_out, dtype=float) / (self.gear_ratio**2) * 128_000_000.0

    def raw_to_output_kd(self, raw: float | np.ndarray):
        return np.asarray(raw, dtype=float) / 128_000_000.0 * (self.gear_ratio**2)

    def ex_pos_to_rad(self, out_pos: float | np.ndarray):
        """输出端单圈绝对编码器（13bit）-> rad（0..2π）。"""
        return np.asarray(out_pos, dtype=float) * (2.0 * np.pi) / float(self.encoder_steps)


# --------------------------------------------------------------------------- #
# CRC32（宇树 S288/J288 专用；与官方 digital_servo 逐字节一致）
# --------------------------------------------------------------------------- #
def _build_crc32_table() -> tuple[int, ...]:
    """生成官方那张 256 项 CRC32 表（poly 0x04C11DB7，MSB-first）。"""
    table = []
    for i in range(256):
        crc = (i << 24) & 0xFFFFFFFF
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return tuple(table)


CRC32_TABLE: tuple[int, ...] = _build_crc32_table()


def crc32_unitree(data: bytes) -> int:
    """宇树 S288/J288 的 CRC32。

    多项式 0x04C11DB7，初值 0xFFFFFFFF；按 32 位**小端字**、每字节查表推进。
    与官方 ``crc32_lookup_byte_by_byte`` 完全一致：每个小端字内先处理最高字节。
    只处理 4 字节对齐的部分（官方实现如此）；控制包 16B、反馈包 20B 都是 4 的倍数。
    """
    crc = 0xFFFFFFFF
    n = len(data)
    i = 0
    while i + 3 < n:
        b0, b1, b2, b3 = data[i + 3], data[i + 2], data[i + 1], data[i]
        crc = CRC32_TABLE[(crc >> 24) ^ b0] ^ ((crc << 8) & 0xFFFFFFFF)
        crc = CRC32_TABLE[(crc >> 24) ^ b1] ^ ((crc << 8) & 0xFFFFFFFF)
        crc = CRC32_TABLE[(crc >> 24) ^ b2] ^ ((crc << 8) & 0xFFFFFFFF)
        crc = CRC32_TABLE[(crc >> 24) ^ b3] ^ ((crc << 8) & 0xFFFFFFFF)
        i += 4
    return crc & 0xFFFFFFFF


def _clip16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


def _clip32(value: int) -> int:
    return max(-2147483648, min(2147483647, int(value)))


# --------------------------------------------------------------------------- #
# 帧编解码
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class S288Command:
    """一条输出端语义的混合（MIT）命令；由 codec 换算成定点字节。"""

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
    # 固件是否上报了"输出端角度"；本协议恒为 True（ExPos）。
    has_output_angle: bool = False
    # 输出端单圈绝对角（0..2π，来自 13bit 输出编码器）
    ex_pos_rad: float = float("nan")
    # 绕组温度（℃，来自 sensor 字节）
    sensor_c: float = float("nan")
    # 扩展告警位（ExFlag）
    warn: int = 0


@dataclass
class S288Codec:
    """宇树 S288/J288 官方协议帧编解码（20B 控制 / 26B 反馈，CRC32）。

    与 https://github.com/unitreerobotics/digital_servo 的
    ``specs/protocol.md`` 和 ``python/servo_demo.py`` 逐字节一致。
    """

    header_cmd: tuple[int, int] = (0xFE, 0xEE)
    header_fb: tuple[int, int] = (0xFC, 0xEE)
    mode_foc: int = _MODE_FOC
    mode_stop: int = _MODE_STOP
    # 默认启用超时保护：主机停发命令约 1s 后电机自行卸力。
    timeout_default: bool = True

    control_len: int = 20
    response_len: int = 26

    # 反馈 fbk 字段相对 fbk 起点（即整帧 offset 3）的偏移，对照官方 RIS_Fbk_t
    fb_temp_offset: int = 0     # int8
    fb_sensor_offset: int = 1   # uint8
    fb_vol_offset: int = 2      # uint8（value/2 = V）
    fb_torque_offset: int = 3   # int16
    fb_speed_offset: int = 5    # int16
    fb_pos_offset: int = 7      # int32（转子多圈位置）
    fb_merror_offset: int = 11  # uint32
    fb_outpos_offset: int = 15  # uint16（低 13 位 OutPos，高 3 位 ExFlag）
    fb_excom_offset: int = 18   # uint8

    # -- 组包 ---------------------------------------------------------------
    @staticmethod
    def mode_byte(motor_id: int, mode: int, timeout: bool) -> int:
        return (
            (int(motor_id) & 0x0F)
            | ((int(mode) & 0x07) << 4)
            | ((1 if timeout else 0) << 7)
        )

    def pack_command(
        self,
        motor_id: int,
        command: S288Command,
        spec: S288Spec,
        *,
        mode: Optional[int] = None,
        timeout: Optional[bool] = None,
    ) -> bytes:
        """组一条 20 字节控制包（默认 mode=混合闭环、开启超时保护）。"""
        mode = self.mode_foc if mode is None else int(mode)
        timeout = self.timeout_default if timeout is None else bool(timeout)

        tor_raw = _clip16(round(float(spec.output_torque_to_raw(command.tau_out))))
        spd_raw = _clip16(round(float(spec.output_spd_to_raw(command.dq_out))))
        pos_raw = _clip32(round(float(spec.output_pos_to_raw(command.q_out))))
        kp_raw = _clip16(round(float(spec.output_kp_to_raw(command.kp_out))))
        kd_raw = _clip16(round(float(spec.output_kd_to_raw(command.kd_out))))

        comd = struct.pack("<hhihh", tor_raw, spd_raw, pos_raw, kp_raw, kd_raw)
        body = bytes([self.mode_byte(motor_id, mode, timeout), 0x00]) + comd
        frame = bytes(self.header_cmd) + body
        frame += struct.pack("<I", crc32_unitree(frame))
        return frame

    def pack_stop(self, motor_id: int, *, timeout: bool = False) -> bytes:
        """停止并锁力帧（mode=0，零目标），用于退出/失能。"""
        return self.pack_command(
            motor_id, S288Command(), S288Spec(), mode=self.mode_stop, timeout=timeout
        )

    # -- 解包 ---------------------------------------------------------------
    def unpack_state(self, frame: bytes, spec: S288Spec) -> S288State:
        """解析一条 26 字节反馈包；CRC 或帧头不符时抛 ``ValueError``。"""
        if len(frame) < self.response_len:
            raise ValueError(f"S288 反馈帧过短：{len(frame)} < {self.response_len}")
        if tuple(frame[0:2]) != tuple(self.header_fb):
            raise ValueError(f"S288 反馈帧头不符：{frame[0:2]!r}")
        crc_rx = struct.unpack_from("<I", frame, self.response_len - 4)[0]
        crc_calc = crc32_unitree(frame[2 : self.response_len - 4])
        if crc_rx != crc_calc:
            raise ValueError(f"S288 反馈 CRC 校验失败：收到 0x{crc_rx:08X}，算出 0x{crc_calc:08X}")

        base = 3
        mode_byte = frame[2]
        motor_id = mode_byte & 0x0F
        temp = struct.unpack_from("<b", frame, base + self.fb_temp_offset)[0]
        sensor = frame[base + self.fb_sensor_offset]
        vol_raw = frame[base + self.fb_vol_offset]
        torque_raw = struct.unpack_from("<h", frame, base + self.fb_torque_offset)[0]
        speed_raw = struct.unpack_from("<h", frame, base + self.fb_speed_offset)[0]
        pos_raw = struct.unpack_from("<i", frame, base + self.fb_pos_offset)[0]
        merror = struct.unpack_from("<I", frame, base + self.fb_merror_offset)[0]
        outpos_exflag = struct.unpack_from("<H", frame, base + self.fb_outpos_offset)[0]
        out_pos = outpos_exflag & 0x1FFF
        ex_flag = (outpos_exflag >> 13) & 0x07

        return S288State(
            motor_id=motor_id,
            q_rotor=float(spec.raw_to_rotor_pos(pos_raw)),
            dq_rotor=float(spec.raw_to_rotor_spd(speed_raw)),
            tau_rotor=float(spec.raw_to_rotor_torque(torque_raw)),
            q_out=float(spec.raw_to_output_pos(pos_raw)),
            dq_out=float(spec.raw_to_output_spd(speed_raw)),
            tau_out=float(spec.raw_to_output_torque(torque_raw)),
            temperature_c=float(temp),
            voltage_v=float(vol_raw) / 2.0,
            error=int(merror),
            has_output_angle=True,
            ex_pos_rad=float(spec.ex_pos_to_rad(out_pos)),
            sensor_c=float(sensor),
            warn=int(ex_flag),
        )

    # -- 测试/自检辅助：伪造一条合法反馈包 -----------------------------------
    def build_feedback_frame(
        self,
        motor_id: int,
        spec: S288Spec,
        *,
        q_out: float = 0.0,
        dq_out: float = 0.0,
        tau_out: float = 0.0,
        temperature_c: int = 25,
        sensor_c: int = 30,
        voltage_v: float = 12.6,
        error: int = 0,
        ex_pos_rad: float = 0.0,
        warn: int = 0,
        mode: int = _MODE_FOC,
        timeout: bool = True,
    ) -> bytes:
        """按协议拼一条反馈包（仅用于离线自检 / 回环测试）。"""
        pos_raw = _clip32(round(float(spec.output_pos_to_raw(q_out))))
        spd_raw = _clip16(round(float(spec.output_spd_to_raw(dq_out))))
        tor_raw = _clip16(round(float(spec.output_torque_to_raw(tau_out))))
        out_pos = int(round(float(ex_pos_rad) / (2.0 * np.pi) * spec.encoder_steps)) & 0x1FFF
        outpos_exflag = (out_pos & 0x1FFF) | ((int(warn) & 0x07) << 13)
        vol_raw = max(0, min(255, int(round(float(voltage_v) * 2.0))))
        fbk = struct.pack(
            "<bBBhh i I H BB".replace(" ", ""),
            int(temperature_c),
            int(sensor_c) & 0xFF,
            vol_raw & 0xFF,
            tor_raw,
            spd_raw,
            pos_raw,
            int(error) & 0xFFFFFFFF,
            outpos_exflag,
            0,
            0,
        )
        assert len(fbk) == 19, len(fbk)
        body = bytes([self.mode_byte(motor_id, mode, timeout)]) + fbk
        frame = bytes(self.header_fb) + body
        frame += struct.pack("<I", crc32_unitree(frame[2:22]))
        assert len(frame) == 26, len(frame)
        return frame


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
                    ex_pos_rad=float(q % (2.0 * np.pi)),
                    sensor_c=30.0,
                    warn=0,
                )
            return states

    def close(self) -> None:
        pass


class SerialS288Bus:
    """真实串口总线（半双工 TTL 单线多点，固定 6 Mbps，官方协议）。

    半双工时序：主机发一帧给某 ID，该电机在同一总线上回一帧状态；因此逐台发命令
    并读取响应，按 ``motor_ids`` 顺序轮询。pyserial 只是字节通道——务必使用
    USB-TTL **收发一体**适配器接到单线总线，不要把 TX/RX 分别接上去。

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
        # 记住每台最后一次命令，读状态时原样重发（对应 timeout=1 的"持续发帧"语义）
        self._last_command: dict[int, S288Command] = {i: S288Command() for i in self._ids}
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

    def _read_response(self) -> Optional[bytes]:
        """读满一条 26 字节反馈包；用 0xFC 0xEE 做帧同步，失败返回 None。"""
        assert self._serial is not None
        need = self.codec.response_len
        deadline = time.monotonic() + self._response_timeout_s
        buf = bytearray()
        while time.monotonic() < deadline and len(buf) < need:
            chunk = self._serial.read(need - len(buf))
            if chunk:
                buf.extend(chunk)
            else:
                time.sleep(self._inter_frame_s)
        idx = bytes(buf).find(bytes(self.codec.header_fb))
        if idx < 0:
            return None
        while time.monotonic() < deadline and len(buf) < idx + need:
            chunk = self._serial.read(idx + need - len(buf))
            if chunk:
                buf.extend(chunk)
            else:
                time.sleep(self._inter_frame_s)
        if len(buf) < idx + need:
            return None
        return bytes(buf[idx : idx + need])

    def _exchange_one(self, motor_id: int, command: S288Command) -> None:
        assert self._serial is not None
        frame = self.codec.pack_command(motor_id, command, self.spec)
        self._serial.reset_input_buffer()
        self._serial.write(frame)
        self._serial.flush()
        payload = self._read_response()
        if payload is None:
            raise TimeoutError(f"S288 id={motor_id} 无有效反馈（超时/CRC/帧头不符）")
        self._last_states[motor_id] = self.codec.unpack_state(payload, self.spec)

    def write_commands(self, commands: dict[int, S288Command]) -> None:
        with self._lock:
            for mid in self._ids:
                cmd = commands.get(mid, self._last_command.get(mid, S288Command()))
                self._last_command[mid] = cmd
                self._exchange_one(mid, cmd)

    def read_states(self) -> dict[int, S288State]:
        with self._lock:
            for mid in self._ids:
                self._exchange_one(mid, self._last_command.get(mid, S288Command()))
            return dict(self._last_states)

    def close(self) -> None:
        if self._serial is not None:
            try:
                for mid in self._ids:  # 停止并锁力
                    self._serial.write(self.codec.pack_stop(mid))
                self._serial.flush()
            except Exception:
                pass
            self._serial.close()
            self._serial = None


# --------------------------------------------------------------------------- #
# 官方 SDK：已废弃（不支持 S288）
# --------------------------------------------------------------------------- #
class UnitreeSdkS288Bus:
    """已废弃：官方 ``unitree_actuator_sdk`` **不支持** S288/J288。

    该 SDK 的 ``MotorType`` 只有 A1 / B1 / GO-M8010-6。S288/J288 的官方协议与参考
    实现见 https://github.com/unitreerobotics/digital_servo，已由本模块的
    ``SerialS288Bus`` 逐字节实现。请把配置里的 ``leader.bus`` 设为 ``serial``。

    保留此类名只为在旧配置下给出**清晰的报错**，而不是静默发错协议。
    """

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "unitree_actuator_sdk 不支持 S288/J288（其 MotorType 只有 A1/B1/GO-M8010-6）。"
            "请把 leader.bus 设为 'serial'，使用本模块按官方 digital_servo 协议实现的 "
            "SerialS288Bus（需要 pip install pyserial）。"
        )


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

    def read_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """一次轮询取回 ``(输出端角度, 输出端速度, 输出端单圈绝对角 ExPos)``。

        ExPos 不可用时对应元素为 ``nan``。三种读取方法都走这里，保证每 tick 只
        轮询总线一次。
        """
        states = self.bus.read_states()
        pos = np.empty(self.n, dtype=float)
        vel = np.empty(self.n, dtype=float)
        ex = np.full(self.n, dtype=float, fill_value=float("nan"))
        for i, mid in enumerate(self.motor_ids):
            st = states.get(mid)
            if st is None:
                raise RuntimeError(f"S288 id={mid} 缺少状态反馈")
            pos[i] = st.q_out
            vel[i] = st.dq_out
            ex[i] = st.ex_pos_rad
        return pos, vel, ex

    def read_positions(self) -> np.ndarray:
        return self.read_arrays()[0]

    def read_positions_and_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        pos, vel, _ = self.read_arrays()
        return pos, vel

    def read_positions_and_ex_positions(self) -> tuple[np.ndarray, np.ndarray]:
        pos, _, ex = self.read_arrays()
        return pos, ex

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
        """撤销所有电机的主动控制（零刚度零阻尼），让机械臂可被手动拖动。"""
        self.bus.write_commands({mid: S288Command() for mid in self.motor_ids})

    def close(self) -> None:
        self.bus.close()
