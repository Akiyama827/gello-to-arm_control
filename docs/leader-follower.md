# 小臂 -> 大臂 实时主从遥操作

人手动操控小臂（leader），实时采集其关节状态，经换算后驱动大臂（follower），
使小臂动作映射到大臂。

硬件构成：

- **小臂（leader）**：**8 个宇树 S288** = 7 个臂关节 + 1 个夹爪；串口通信走
  **官方 `unitree_actuator_sdk`**。
- **大臂（follower）**：**FR3** = `fr3_joint1..7` + Franka Hand。臂关节走
  `arm_controller` 的 `jog`/`control`，夹爪走 `franka_gripper` 的 `gripper`。

小臂 7 个臂关节与 FR3 的 7 个关节**天然一一对应**，关节空间直连即可。

## 数据流

```
S288LeaderArm / GelloLeaderAdapter / FakeLeaderArm   （读小臂）
        │  get_joint_state() -> [7 臂关节(rad), 夹爪(0..1)]
        ▼
Retargeter.map()                                     （换算）
        │  θ_fr3_i = offset_i + scale_i * sign_i * θ_small[src_i]
        │  夹爪 [0,1] -> FR3 单指位移(m)；平滑/限幅/限位/auto_align
        ▼
SafetyMonitor                                        （安全门）
        │  leader 合法性/断流、关节限位、目标跳变、两臂碰撞、跟踪误差…
        ▼
DryRunFollower / DoraJogFollower / RtFollower / FakeFollower （下发 FR3）
```

## 模块

| 文件 | 作用 |
|---|---|
| `arm_control/leader_follower/s288.py` | S288 规格、各单位换算；官方 SDK 总线 `UnitreeSdkS288Bus`；自实现帧总线 `SerialS288Bus`（后备）；仿真总线 |
| `leader.py` | 小臂接口 + S288 / gello 适配 / fake 三种实现 |
| `mapping.py` | 关节空间换算（sign/offset/scale/限位/夹爪/auto_align） |
| `follower.py` | 大臂四种下发后端 |
| `safety.py` | 碰撞守卫 + 安全监视器 + `SafetyStop` |
| `loop.py` | 实时主循环 `TeleopLoop` |
| `dora_feedback.py` | 大臂回读 `DoraFollowerFeedback`（motor_state/health/controller_event/gripper_state） |
| `node.py` | Dora 节点入口：一个 `Node` 同时发 jog/control/gripper、收回读 |
| `config.py` | YAML 配置与装配 |
| `nodes/leader_teleop.py` | 上述节点的 Dora 可执行薄壳 |
| `dataflows/leader_teleop_franka.yml` | 真机 FR3 遥操作 Dora 图（本节点作为唯一运动源） |
| `examples/leader_follower_teleop.py` | 可运行示例（默认全仿真） |
| `examples/configs/leader_follower.yaml` | 示例配置 |
| `tools/bench/check_leader_follower.py` | 离线自检 |

## 串口通信：官方 unitree_actuator_sdk

真实硬件默认 `leader.bus: unitree_sdk`，即用官方 SDK 的 `SerialPort` +
`MotorCmd`/`MotorData` + `serial.sendRecv(cmd, data)`：

```python
from unitree_actuator_sdk import SerialPort, MotorCmd, MotorData
serial = SerialPort('/dev/ttyUSB0')
cmd, data = MotorCmd(), MotorData()
cmd.motorType = data.motorType = MotorType.<型号>
cmd.mode = queryMotorMode(MotorType.<型号>, MotorMode.FOC)
cmd.id = 0
cmd.q, cmd.dq, cmd.kp, cmd.kd, cmd.tau = ...
serial.sendRecv(cmd, data)      # data.q/dq/tau/temp/merror 为反馈
```

**关键点**：SDK 的 `q/dq/tau/kp/kd` 全是**转子侧**量，而本模块对外统一用**输出端**
语义；`UnitreeSdkS288Bus` 在边界用 `S288Spec` 换算（`q_out=q_rotor/r`、
`tau_out=tau_rotor*r`、`kp_rotor=kp_out/r²`、`kd_rotor=kd_out/r²`）。这是"电机参数
不同"要处理的核心。

`bus` 三选一：`unitree_sdk`（首选，需编译好的扩展在 `PYTHONPATH`）/
`serial_raw`（自实现 0xFE 0xEE MIT 帧 + CRC16-CCITT，无 SDK 时用）/
`fake`（纯软件）。若官方 SDK 的 `MotorType` 里没有 `S288` 名称，
`UnitreeSdkS288Bus` 会**列出全部可选项并在构造时报错**，避免悄无声息地发错协议。

## FR3 下发通道（Dora）

FR3 在 `real_franka_motion.yml` 里由 `arm_console` 产出三个 topic，我们的遥操作
节点就按同样格式产出来替代/并行这个运动源：

| topic | 消费者 | 内容 |
|---|---|---|
| `jog` | `arm_controller` | `pack_jog(q=[7 关节], reason=...)`；单帧设定点，0.2s 不刷新自动停 |
| `control` | `arm_controller` | `pack_control_update(arm=True)` 使能；`cancel=True`/`arm=False` 撤销 |
| `gripper` | `franka_gripper` | `pack_motor_command([finger, finger], 0,0,0,0)`，**单指位移（米）** |

要点：

- FR3 的 `arm_controller` **不**消费 `gripper`（`num_motors: 7`，臂上无夹爪电机）；
  夹爪由独立的 `franka_gripper` 节点经它自己的 TCP 服务驱动。
- `franka_gripper` 取 `gripper` 消息的 `position[0]` 作为**单指位移**，内部
  `width = 2 * finger`；FR3 行程是 `gripper_range_m: [0.0, 0.04]`（单指）。
  所以 `GripperMapping.open_finger_m` 用 **0.04**，不是整手 0.075/0.08。
- 部署时把 `real_franka_motion.yml` 里 `arm_controller`/`franka_gripper` 的
  `jog`/`control`/`gripper` 输入从 `arm_console/*` 改接到遥操作节点。已经落成
  `dataflows/leader_teleop_franka.yml`：`leader_teleop` 作**唯一运动源**，
  产出的三个 topic 与 `arm_console` 完全同格式；`arm_console` 不再入图。
- RT 直连后端（`RtFollower`）只发 7 个臂关节，**不驱动 Franka Hand**；要连夹爪
  一起遥操作必须走 Dora 的 `DoraJogFollower`。

### 大臂回读（跟踪误差门的前提）

`jog` 是单向通道，所以 `DoraFollowerFeedback`（`dora_feedback.py`）在同一个
Dora `Node` 上收反馈，喂给安全层：

| topic | 用途 |
|---|---|
| `plant_interface/motor_state` | 实测关节位置/速度 -> 跟踪误差、速度异常、启动对齐基准 |
| `plant_interface/motor_health` | `armed` / `latched_fault` -> 失能/故障位停机 |
| `arm_controller/controller_event` | `kind=fault` -> 控制器停止，粘住不自动清除 |
| `franka_gripper/gripper_state` | 夹爪开度（仅记录；`width` 为整手，单指 = `width/2`） |

要点：

- 每 tick `feedback()` 用 `try_recv()` 非阻塞吸干已到达事件，只留最新一条。
- **新鲜度用样本自带的到达时刻 `timestamp`，不是"这一 tick 调用过"。** 否则
  调用方每 tick 递回同一个缓存样本就能骗过 `feedback_timeout_s`。
- 启动先 `prime()` 等到首帧 `motor_state`，拿**实测**位形做 `auto_align`；
  FR3 的零位不合法（`fr3_joint4` 约 `[-3.0421, -0.1518]`），不能拿零点对齐。
- 图配置里 `follower.auto_arm: true` 时节点 `open()` 发 `control(arm=True)`、
  退出/安全停机发 `control(cancel, arm=False)`；RT 服务器保留正常 deadman，
  节点一停大臂即被驻停。**停节点就是 DISARM**。若要保留操作台 ARM/DISARM 门，
  跑 `real_franka_motion.yml` 并把 `follower.auto_arm` 设为 `false`。

## S288 参数（来源：Unitree 官网 DigitalServo 页，2026 查得）

- 减速比 **288.35:1**，力矩常数 **0.554 N·m/A**，堵转 **0.6 N·m**，空载 **16.5 rad/s@12V**
- 双绝对值编码器（转子 15bit / 输出端），半双工串口 **8N1 @ 6 Mbps**，ID 0–14
- 控制模式：混合 MIT（q/dq/tau/kp/kd）；反馈含转子/输出端角度、速度、扭矩、温度、电压、错误位

> 走官方 SDK 后，协议细节由 SDK 负责；仍需在整机联调时核对两点：官方是否已支持
> S288（`MotorType` 枚举名），以及半双工 6 Mbps 总线的端口/收发时序。

## 安全机制（宁停勿撞）


分两层，每 tick 都跑便宜的门，昂贵的碰撞查询可降频：

**运动学层**：关节限位（含余量）、单 tick 目标跳变、两臂碰撞（预警区限速、逼近即停）。

**跟踪/健康层（"大臂没按预期运行"）**：命令-实测跟踪误差持续超阈、反馈陈旧/丢失、
控制器故障位、plant 失能、实测关节速度异常、leader 数据非法/断流。

任一触发 -> `SafetyStop` -> `TeleopLoop` 捕获后**立即 `follower.safe_stop()`**。

碰撞守卫三种，部署侧任选：
- `NoCollisionGuard`：显式关闭（需说明原因）
- `CallableCollisionGuard`：包装 arm_control 的 `MuJoCoCollisionWorld`（自碰撞 + 环境 +
  把小臂作为场景 actor = 高保真两臂碰撞）
- `SphereCollisionGuard`：纯 numpy 两臂球体近似，开箱可用

> 注意：`DoraJogFollower` 的 `jog` 通道本身不回读，但 Dora 节点用
> `DoraFollowerFeedback` 另接 `plant_interface/motor_state` +
> `plant_interface/motor_health` + `arm_controller/controller_event`，跟踪误差、
> 反馈陈旧、故障位、失能都能判。若确实没有回读，必须设 `feedback_required: false`
> 并明确接受该保护缺失。

## 运行

```bash
# 离线自检（无需硬件）
PYTHONPATH=. python -B tools/bench/check_leader_follower.py
# 全仿真跑 10s
PYTHONPATH=. python -B examples/leader_follower_teleop.py --duration 10
# 带玩具碰撞守卫，演示两臂碰撞停机
PYTHONPATH=. python -B examples/leader_follower_teleop.py --collision-demo --duration 10
```

真机 Dora 图（需部署配置：`leader.kind: s288` + `follower.kind: dora`）：

```bash
LEADER_FOLLOWER_CONFIG=$PWD/deploy/leader_follower.yaml \
ARM_CONTROL_ROOT=$PWD ARM_CONTROL_CONFIG=$PWD/configs/entries/real_franka.yaml \
  dora run libs/arm_control/dataflows/leader_teleop_franka.yml
```

## 进度

已完成：

- S288 规格与官方 SDK 总线、关节空间换算、FR3 三种下发后端、安全层与主循环。
- FR3 回读 `DoraFollowerFeedback` + Dora 节点入口 + 真机图
  `dataflows/leader_teleop_franka.yml`（本节点作唯一运动源，含跟踪误差/陈旧/故障/失能停机）。

上线前仍需：

- 按实际 URDF 核对 FR3 关节限位；标定 `joint_signs`/`joint_offsets`。
- 实测确认官方 SDK 的 `MotorType` 枚举名（无 `S288` 则需换名/自实现后备总线）。
- 高保真两臂碰撞：把 `MuJoCoCollisionWorld`（小臂作为场景 actor）包成
  `CallableCollisionGuard` 注入 `build_dora_node(collision_guard=...)`。

## 大臂不必逐关节复刻，可做路径优化（用户 2026-09-18 需求）

需求：**在"效果一样"（末端位姿/任务等价）的前提下，大臂可以走优化过的路径，
不必严格复刻小臂的逐关节动作。**

小臂 7 关节与 FR3 7 关节已天然对齐，关节空间直连即可用；路径优化是在此之上的
增强，计划在 `mapping.py` 增加任务空间模式，与现有关节空间模式并存、可配置切换：

1. 小臂 FK -> 末端位姿，经标定的基座变换 + 工作空间缩放映射到大臂目标位姿。
2. 用 `arm_control.planning.ik` 求大臂关节解；IK 失败时保持上一位形并告警。
3. **路径优化**（"效果一样"的判据 = 末端误差容差，如 <5mm / <2°）：
   - 冗余零空间姿态规整（preferred posture / null-space regulation）
   - 关节空间平滑（min-jerk / 限 jerk），避免小臂抖动直接传入
   - 碰撞/奇异感知的局部优化
   - 必要时用 `planning/ompl_planner.py` / `planning/trajopt.py` 在保持末端路点
     的前提下重规划整段路径
4. 安全层不变（限位、碰撞、跟踪误差）继续生效；优化后的路径同样逐帧过安全门。

配置预留：`mapping.mode: joint | pose | pose_optimized`，`mapping.pose` 段放
末端位姿容差、缩放、姿态选项。
