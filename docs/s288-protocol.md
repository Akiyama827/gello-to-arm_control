# 宇树 J288 / S288 通信协议参考

本文件记录宇树 J288 / S288 数字舵机的**官方通信协议**，以及本仓库
`arm_control/leader_follower/s288.py` 的实现依据。用于离线查阅、代码评审、
以及真机联调时对照。

## 0. 协议来源

S288/J288 **不在** 官方 `unitree_actuator_sdk` 里——该 SDK 的 `MotorType` 只有
`A1` / `B1` / `GO_M8010_6`。宇树为 J288/S288 单独开源了协议与参考实现：

- 仓库：<https://github.com/unitreerobotics/digital_servo>
  - `specs/protocol.md` —— 协议说明
  - `python/servo_demo.py` —— PC（pyserial）参考实现
  - `stm32/` —— STM32 HAL 参考实现（`App/protocol.c/.h`, `App/crc_ccitt.h`）

本仓库的 `SerialS288Bus` / `S288Codec` / `crc32_unitree` 与上述官方参考
**逐字节对齐**（已用随机数据 + 样例帧与官方 Python 参考做过比对）。

## 1. 物理层

| 项目 | 值 |
|---|---|
| 接口 | 半双工 **TTL 单线**多点总线 |
| 波特率 | **6,000,000 bps（固定，不可改）** |
| 帧格式 | 8N1（8 数据位 / 1 停止位 / 无校验） |
| 总线容量 | 最多 15 台（ID 0~14）；ID 15 = 广播（无回包） |
| 减速比 | `RATIO = 70070 / 243 ≈ 288.35` |
| 接线 | 主机 TX/RX 经 **USB-TTL 半双工收发一体适配器**接到单线总线，**必须共地** |

> ⚠️ 不能把 TX、RX 分别接到总线；必须用收发一体的半双工适配器，否则发出去的字
> 会被自己收回来（回显）导致反馈错乱。

## 2. 控制包（主机 → 舵机）：20 字节

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 2 | `head` | `0xFE 0xEE` |
| 2 | 1 | `mode_byte` | 位域：`[3:0]=id`，`[6:4]=mode`，`[7]=timeout` |
| 3 | 1 | `reserved` | 固定 `0x00` |
| 4 | 12 | `comd` | 控制参数（见下） |
| 16 | 4 | `CRC32` | 小端；覆盖 `[0:16)` |

### mode_byte

| 位 | 字段 | 取值 |
|---|---|---|
| `[3:0]` | id | 0~15（15 = 广播） |
| `[6:4]` | mode | `0` = 停止并锁力；`1` = 混合闭环（FOC） |
| `[7]` | timeout | `0` = 关闭超时保护；`1` = 开启（主机停发约 1s 后电机自行卸力） |

### comd（12 字节，小端）

| 偏移 | 类型 | 名字 | 说明 |
|---|---|---|---|
| 0 | int16 | `tor_des` | 目标转子扭矩（原始值） |
| 2 | int16 | `spd_des` | 目标转子速度（原始值） |
| 4 | int32 | `pos_des` | 目标转子位置（原始值） |
| 8 | int16 | `k_pos` | 位置刚度（原始值） |
| 10 | int16 | `k_spd` | 速度阻尼（原始值） |

## 3. 反馈包（舵机 → 主机）：26 字节

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 2 | `head` | `0xFC 0xEE` |
| 2 | 1 | `mode_byte` | 同控制包的位域 |
| 3 | 19 | `fbk` | 反馈数据（见下） |
| 22 | 4 | `CRC32` | 小端；覆盖 `[2:22)` |

### fbk（19 字节，小端；对应官方 `RIS_Fbk_t`）

| fbk 偏移 | 整帧偏移 | 类型 | 名字 | 说明 |
|---|---|---|---|---|
| 0 | 3 | int8 | `temp` | 驱动器温度（℃，-128~127） |
| 1 | 4 | uint8 | `sensor` | 绕组温度（℃） |
| 2 | 5 | uint8 | `vol` | 母线电压原始值，`vol / 2 = V`（255 → 127.5V） |
| 3 | 6 | int16 | `torque` | 转子扭矩（原始值） |
| 5 | 8 | int16 | `speed` | 转子速度（原始值） |
| 7 | 10 | int32 | `pos` | 转子**多圈**位置（原始值） |
| 11 | 14 | uint32 | `MError` | 错误位（报警） |
| 15 | 18 | uint16 | `OutPos:13 | ExFlag:3` | 低 13 位 = 输出端单圈绝对角；高 3 位 = 扩展标志 |
| 17 | 20 | uint8 | `ExSensor2` | 扩展位 |
| 18 | 21 | uint8 | `ExCom` | 扩展通信位 |

> 官方 `specs/protocol.md` 的字节表把 `vol` 写成 offset 2 处的 uint16，与 STM32
> 结构体 / Python 参考不一致；**以结构体 `RIS_Fbk_t` 和 `python/servo_demo.py`
> 为准**：`vol` 是 offset 2 的**单字节**。本实现遵循后者。

## 4. CRC32

- 多项式 `0x04C11DB7`，初值 `0xFFFFFFFF`，MSB-first。
- 官方按 **32 位小端字**、每字节查表推进（`crc32_lookup_byte_by_byte`）。
- 控制包对 16 字节、反馈包对 20 字节校验，都是 4 的倍数。

```python
# 与官方等价的实现（见 s288.py: crc32_unitree）
crc = 0xFFFFFFFF
for each little-endian 32-bit word, byte order b3 b2 b1 b0 (MSB-first):
    for b in (b3, b2, b1, b0):
        crc = TABLE[(crc >> 24) ^ b] ^ ((crc << 8) & 0xFFFFFFFF)
```

参考向量（由官方 `servo_demo.py` 算得）：

| 输入 | CRC32 |
|---|---|
| `00 00 00 00` | `0xC704DD7B` |
| `00 01 .. 0F` | `0x081B46CA` |
| `FE EE 21 00` | `0xC408C219` |

空闲命令帧 `id=1, mode=1, timeout=1, comd=全零` = `feee91000000000000000000000000009ce9c752`。

## 5. 物理量 ↔ 定点原始值换算

`RATIO = 70070/243 ≈ 288.35`，输出端物理量 → 原始值：

```
k_pos   = Kp    / RATIO^2 * 1_280_000       (0 .. 2128.523   -> 0..32767)
k_spd   = Kd    / RATIO^2 * 128_000_000     (0 .. 21.285     -> 0..32767)
pos_des = q_out * RATIO * 32768 / (2π)      (±1428.019 rad  -> ±2^31)
spd_des = dq_out* RATIO * 2.560 / (2π)      (±278.901 rad/s -> ±32767)
tor_des = tau_out / RATIO * 256_000          (±36.909 N·m   -> ±32767)
```

反馈反向：

```
q_out   = 2π * pos    / 32768 / RATIO      （转子端多圈位置 -> 输出端）
dq_out  = (speed / 2.56) * 2π / RATIO
tau_out = torque / 256000 * RATIO
ExPos   = 2π * OutPos / 8192               （输出端单圈绝对角，0..2π）
vol     = vol_raw / 2                      （V）
```

- 关节角用**多圈** `q_out`（由转子 `pos` 推得）；`ExPos` 只给单圈绝对角，用于
  上电找零 / 诊断。
- 换算成对实现：`S288Spec.output_pos_to_raw` / `raw_to_output_pos`、
  `output_kp_to_raw` / `raw_to_output_kp` 等（共 5 组）。

## 6. 与 `unitree_actuator_sdk` 的关系

| | `unitree_actuator_sdk` | 本仓库 `SerialS288Bus` |
|---|---|---|
| 支持电机 | A1 / B1 / GO-M8010-6 | **J288 / S288** |
| 传输 | `SerialPort` + `sendRecv` | pyserial 直接收发 |
| 帧 | A1/B1 用 `MasterComdV3`；GO 用 `ControlData_t` | 官方 digital_servo 20B/26B |
| CRC | A1/B1 用 CRC32 core；GO 用 CRC-CCITT | CRC32（poly 0x04C11DB7） |

**结论**：驱动 S288/J288 **不能**用 `unitree_actuator_sdk`。本仓库的
`leader.bus: serial` 直接实现官方 digital_servo 协议，只依赖 `pyserial`。
配置里若仍写旧的 `leader.bus: unitree_sdk`，构造函数会**立刻抛出可操作的报错**
并提示改用 `serial`，避免静默发错协议。

## 7. 排障速查

| 症状 | 可能原因 | 处理 |
|---|---|---|
| 完全无反馈 | 波特率非 6M / TX-RX 接法错 / 未共地 | 核对 6 Mbps、半双工一体适配器、共地 |
| 反馈乱码 / CRC 失败 | 适配器回显了自己的发送 | 换收发一体半双工适配器 |
| `timeout` 持续增长 | 接线/电压/波特率问题 | 同上；确认电机供电（J288 25.2V / S288 12.6V） |
| 电机不回 | `mode` 没设为 1 | 用混合闭环模式（`mode=1`） |
| 电机发烫/顶死 | `timeout=0` 且命令持续 | 用 `timeout=1` 让主机停发后自动卸力（本实现默认开启） |
