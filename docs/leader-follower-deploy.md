# 小臂 → FR3 遥操作：交付文件清单与部署步骤

本文档是 [leader-follower.md](leader-follower.md)（设计与原理）的**操作版**：一份
照着做就能上线的清单。硬件约定见设计文档——小臂 = 8 个宇树 S288（7 臂关节 +
1 夹爪，官方 `unitree_actuator_sdk`），大臂 = FR3（`fr3_joint1..7` + Franka Hand）。

---

## 1. 交付文件清单

### 1.1 本仓库（`libs/arm_control`）内

| 文件 | 作用 | 上线需改 |
|---|---|---|
| `arm_control/leader_follower/s288.py` | S288 规格换算；`UnitreeSdkS288Bus`（官方 SDK）/`SerialS288Bus`（后备）/`FakeS288Bus`；关节链读取 | 否（`motor_type` 走配置） |
| `arm_control/leader_follower/leader.py` | 小臂抽象：`S288LeaderArm` / `GelloLeaderAdapter` / `FakeLeaderArm` | 否 |
| `arm_control/leader_follower/mapping.py` | 关节/夹爪换算、平滑、限幅、`auto_align` | 否（参数走配置） |
| `arm_control/leader_follower/follower.py` | 下发后端：`DryRunFollower`/`FakeFollower`/`DoraJogFollower`/`RtFollower` | 否 |
| `arm_control/leader_follower/safety.py` | `SafetyMonitor` + 碰撞守卫 + `SafetyStop` | 否 |
| `arm_control/leader_follower/loop.py` | 主循环 `TeleopLoop` | 否 |
| `arm_control/leader_follower/dora_feedback.py` | FR3 回读：`motor_state`/`motor_health`/`controller_event`/`gripper_state` → 安全层 | 否 |
| `arm_control/leader_follower/node.py` | Dora 节点入口：单 `Node` 同时下发 + 收回读，启动 `prime` | 否 |
| `arm_control/leader_follower/config.py` | YAML 解析与装配（新增 `follower.auto_arm`） | 否 |
| `nodes/leader_teleop.py` | 上述节点的 Dora 可执行薄壳 | 否 |
| `dataflows/leader_teleop_franka.yml` | 真机 Dora 图：遥操作节点作唯一运动源接 `arm_controller`/`franka_gripper` | 否（env 走部署侧） |
| `examples/configs/leader_follower.yaml` | 全仿真示例配置（fake/fake） | 否 |
| `examples/configs/leader_follower_real.example.yaml` | **真机配置模板**，复制后按现场标定 | **是** |
| `examples/leader_follower_teleop.py` | 单机可运行示例（默认全仿真） | 否 |
| `examples/leader_follower_viewer.py` | MuJoCo 3D 可视化（两条胶囊臂实时跟随，无需资产） | 否 |
| `tools/bench/check_leader_follower.py` | 离线自检（无硬件） | 否 |
| `docs/leader-follower.md` | 设计与原理 | 否 |
| `docs/leader-follower-deploy.md` | 本文档 | 否 |

### 1.2 部署侧需新增

在第 4 节会给出一份完整模板。至少：

1. `configs/entries/leader_follower.yaml` —— 真机配置（由上面的 `.example.yaml` 复制而来），用环境变量 `LEADER_FOLLOWER_CONFIG` 指向它。
2. RT 机器上的 `arm_rt_server`（`--backend franka`）——本仓库 `rt/` 编译产出，不由 Dora 图拉起，需**先**启动。

---

## 2. 前提与依赖

### 2.1 硬件 / 总线

| 项 | 要求 |
|---|---|
| 小臂 | 8 × 宇树 S288，半双工 TTL 多点总线，8N1 @ 6 Mbps，ID 在 0–14 内且互不相同 |
| 串口 | 一个 USB 转半双工适配器，Linux 设备如 `/dev/ttyUSB0`；当前用户需在 `dialout` 组（`sudo usermod -aG dialout $USER` 后重新登录） |
| 大臂 | FR3 + Franka Hand，RT 机器与 FR3 同网段，Hand 有自己的 TCP 服务 |
| Dora | 部署根与节点进程都能 `import dora`；`dora` CLI 在 PATH |

### 2.2 软件

- **官方 SDK**：编译 `unitree_actuator_sdk`（pybind 扩展）并把产物放到 `PYTHONPATH`。
  未完成时可先用 `leader.bus: serial_raw`（后备自实现帧）或 `fake` 跑通链路。
- **Python 环境**：需 `numpy`、`pyarrow`、`dora-rs`（本仓库 `dependencies`）。本机验证用的解释器是
  `/home/akiyama0827/miniconda3/envs/arm_control/bin/python`。
- **RT 服务器**：FR3 需用 `-DWITH_FRANKA=ON` 编译，详见 `rt/README.md`。

---

## 3. 启动前的一次性核对

```bash
# 1) 串口能看到
ls -l /dev/ttyUSB0

# 2) 官方 SDK 是否在 PYTHONPATH（无 S288 枚举会在此报错并列出可选项）
python - <<'PY'
from unitree_actuator_sdk import MotorType
print([n for n in dir(MotorType) if not n.startswith('_')])
PY

# 3) 离线自检全绿（不接硬件）
PYTHONPATH=. python -B tools/bench/check_leader_follower.py
```

若第 2 步没有 `S288`：把配置里的 `leader.motor_type` 改成实际枚举名；若官方确实
不支持 S288，改用 `leader.bus: serial_raw`，或先用 `fake` 验证其余链路。

---

## 4. 真机配置

复制模板并填写：

```bash
cp examples/configs/leader_follower_real.example.yaml \
   $DEPLOY_ROOT/configs/entries/leader_follower.yaml
```

需要按现场标定的字段（见下一节）：`port`、`motor_ids`、`motor_type`、
`joint_offsets`/`joint_signs`、`gripper_open_rad`/`gripper_close_rad`，以及
`mapping.joints[*].lower/upper`（FR3 官方限位，按实际 URDF 核对）、`auto_arm`。

`auto_arm` 决定安全模型：

- `true`（默认）：节点 `open()` 发 `control(arm=True)`，退出/安全停机发
  `control(cancel, arm=False)`。**停节点即 DISARM**，RT 服务器正常 deadman 会驻停大臂。
- `false`：保留操作台的 ARM/DISARM 门（跑 `dataflows/real_franka_motion.yml`，
  由 `arm_console` 使能）。此时本节点只做 jog 流。

---

## 5. 标定（上线前必做）

目标：小臂静止时大臂也静止，小臂动 1 rad 大臂方向/幅度符合预期，夹爪开合对应。

1. **关节符号 `joint_signs`**：单个小臂关节缓慢正向转一点，看对应 FR3 关节是否
   同向。反向就把它置 `-1.0`。默认已给一组常见值，仍需实机确认。
2. **零位偏移 `joint_offsets`**：把两臂摆到机械上一致的姿态，读小臂角 `θ_s`、FR3
   角 `θ_f`，则 `offset = θ_f - sign·scale·θ_s`。`auto_align: true` 会吸收启动
   时刻的偏差，但 `offset` 决定**运动中**的一致性，仍要标。
3. **夹爪 `gripper_open_rad`/`gripper_close_rad`**：小臂夹爪全开/全闭时的 S288
   输出端角度（rad）。映射会把 `[close, open]` 线性归一到 `[0, 1]`，再变成 FR3
   单指位移 `[0, 0.04] m`。
4. **FR3 限位**：`mapping.joints[*].lower/upper` 与 `safety.joint_lower/upper`
   必须和实际 URDF 一致（尤其 `fr3_joint4` 约为 `[-3.0421, -0.1518]`，零位越限）。
5. **SDK 枚举 / 波特率**：`motor_type` 与总线参数按实机核对。

标定顺序建议：先 `leader.bus: fake` + `follower.kind: dry_run` 验证 mapping
方向，再 `bus: s288` + `follower.kind: dora` 真机。

---

## 6. 启动步骤

### 6.1 先起 RT 服务器（RT 机器）

按 `rt/README.md` 编译后，在 RT 机器上起 franka 后端（参数以 README 为准，
关键是**正常 deadman**，不要用 handguide 的 `--fault-ms 3600000`）：

```bash
build/arm_rt_server --backend franka --franka-ip <FR3-IP> --n 7 \
  --hold-ms 200 --fault-ms 1000
```

服务器起来时是 **DISARMED**。

### 6.2 再起 Dora 图（控制 PC）

在部署根（含 `configs/` 和 `libs/arm_control`）执行：

```bash
export ARM_CONTROL_ROOT=$PWD
export ARM_CONTROL_CONFIG=$PWD/configs/entries/real_franka.yaml
export LEADER_FOLLOWER_CONFIG=$PWD/configs/entries/leader_follower.yaml

dora run libs/arm_control/dataflows/leader_teleop_franka.yml
```

> 注意：图里每个节点的 `env:` 会**覆盖**启动器环境，所以 `LEADER_FOLLOWER_CONFIG`
> 故意不写进 `leader_teleop` 的 `env:`，让它从启动器继承。若省略该变量，节点会
> 回退到 fake/fake 示例并**拒绝启动**（报错提示），不会静默空跑。

本地只有本仓库、没有部署根时，可直接：

```bash
ARM_CONTROL_ROOT=$PWD ARM_CONTROL_CONFIG=<你的 real_franka 配置> \
LEADER_FOLLOWER_CONFIG=$PWD/examples/configs/leader_follower_real.example.yaml \
  dora run dataflows/leader_teleop_franka.yml
```

### 6.3 启动时观察

节点会先 `prime()` 等首帧 `motor_state`（默认 10s，可用
`LEADER_FOLLOWER_PRIME_S` 调整）；拿不到就拒绝启动。随后打印
`[teleop] 已 auto_align...` 和周期日志：

```
[teleop] 启动：100 Hz，Ctrl-C 停止
[teleop] tick=... 实际≈100.0Hz 最大间隔=...ms 最近距离=n/a
```

`最近距离=n/a` 表示当前是 `NoCollisionGuard`（未接碰撞守卫）——真机建议接上
（见第 8 节第 3 条）。

---

## 7. 安全操作

### 7.1 正常停止

在跑 Dora 图的终端按 **Ctrl-C**：主循环捕获后 `follower.safe_stop()`，`finally`
里发 `control(cancel, arm=False)` 并关闭小臂串口。`auto_arm: false` 模式下停节点
不会 DISARM，需在操作台点 DISARM。

### 7.2 触发安全停机后

会看到类似：

```
[teleop][安全停机] source=feedback 原因：命令-实测误差 0.21rad 持续 0.31s
```

处理流程：本 tick 起停止下发 `jog`（RT 服务器 0.2s 后 HOLD），节点发
`control(cancel)`，并 `arm=False` 退出。恢复：排除原因（线缆/使能/碰撞风险/限位）
→ 重新起 RT 服务器会话（若有故障锁存需 DISARM→ARM）→ 重跑第 6 节。

### 7.3 各安全门含义（配置在 `safety.limits`）

| 字段 | 触发条件 | 处置 |
|---|---|---|
| `leader_timeout_s` | 小臂读数断流超时 | 停 |
| `joint_margin_rad` | 目标进入硬限位余量内 | 停 |
| `max_step_rad` | 单 tick 目标跳变超阈 | 停 |
| `collision_stop_m` / `collision_warn_m` | 两臂最近距离 ≤ 1cm 停 / ≤ 5cm 限速 | 停 / 降速 |
| `track_err_rad` + `track_err_hold_s` | 命令-实测误差持续超阈 | 停 |
| `feedback_timeout_s` | 回读陈旧（按样本到达时刻判） | 停 |
| `track_vel_rad_s` | 实测关节速度异常 | 停 |
| 故障位 / 失能 | `motor_health.latched_fault` / `armed=false` / `controller_event=fault` | 停 |

**宁停勿撞**：拿不准就停；只有 `feedback_required: false`（且明知缺保护）才不要求回读。

---

## 8. 验证顺序（强烈建议逐级）

1. **离线自检**：`PYTHONPATH=. python -B tools/bench/check_leader_follower.py` 全绿。
2. **单机仿真**：`PYTHONPATH=. python -B examples/leader_follower_teleop.py --duration 10`，
   观察 mapping 方向与周期。想直观看动作就开
   `PYTHONPATH=. python -B examples/leader_follower_viewer.py --duration 20`（MuJoCo 3D
   窗口，两条臂实时跟随；角标有跟踪误差/最近距离；`--collision-demo` 可看到逼近后停机）。
3. **碰撞演示**：`... examples/leader_follower_teleop.py --collision-demo --duration 10`，
   确认逼近时安全停机。真机高保真两臂碰撞建议把 `MuJoCoCollisionWorld`（小臂作
   场景 actor）包成 `CallableCollisionGuard`，在 `node.py` 的
   `build_dora_node(cfg, node=..., collision_guard=...)` 注入。
4. **真机空载**：`auto_arm` 先设 `false`，只观察 `motor_state` 回读与 auto_align
   日志；确认无跳变后再在操作台 ARM。
5. **真机低速**：把 `mapping.alpha` 调小（如 0.6）、`max_rate` 调小，小臂缓慢
   动作，确认跟踪正常后再恢复。
6. **接入 `motor_state_logger`**：图中已包含，产出 CSV，用 `qdes - pos` 复核跟踪误差。

---

## 9. 故障排查

| 症状 | 可能原因 | 处理 |
|---|---|---|
| 节点报「找不到配置文件」 | 未设 `LEADER_FOLLOWER_CONFIG` | 指向真机配置（见 6.2） |
| 节点报「follower.kind 不是 dora」 | 用了 fake/fake 示例 | 改配置或换 `LEADER_FOLLOWER_CONFIG` |
| 启动超时未收到 `motor_state` | `plant_interface` 未起 / RT 服务器没连上 / 图接线错 | 先看 RT 服务器日志，再确认 Dora 图里 `plant_interface` 节点起来了 |
| 构造时报 MotorType 无 `S288` | 官方枚举名不同或未编 S288 | 改 `motor_type` 或 `bus: serial_raw` |
| 打开串口失败 | 端口错 / 权限不足 | 查 `ls -l /dev/ttyUSB0`、加入 `dialout` |
| 一启动就安全停机（越限） | `fr3_joint4` 零位越限 / 初始位形非法 | 确保 `auto_align: true` 且 FR3 处于合法位形；核对限位 |
| 一启动就安全停机（跟踪误差） | 大臂没使能 / 增益 0 / `jog` 未接上 | 检查 `arm_controller` 的 `jog` 输入与 ARM 状态 |
| 运行中报「反馈陈旧」 | `motor_state` 断流 | 查 `plant_interface` 与 RT 链路 |
| 小臂动、大臂不动 | `jog` 未接线 / 目标被限幅掉 / DISARMED | 看 `arm_controller` 日志的 IGNORING 与 armed 状态 |
| 方向相反 | `joint_signs` 标定错 | 按第 5 节重标 |
| 夹爪不动 | `gripper` 未接 `franka_gripper` / 单指米超 `[0,0.04]` | 查图接线与 `open_finger_m` |

---

## 10. 上线核对清单

- [ ] `check_leader_follower.py` 全绿
- [ ] 官方 SDK/`MotorType` 已核对，或已切 `serial_raw`
- [ ] 串口端口与 `dialout` 权限确认
- [ ] `joint_signs` / `joint_offsets` 实机标定
- [ ] `gripper_open_rad` / `gripper_close_rad` 标定，夹爪开合方向正确
- [ ] FR3 限位按 URDF 核对（`mapping.joints` 与 `safety.joint_lower/upper` 一致）
- [ ] `auto_arm` 策略确定（自使能 / 操作台门）
- [ ] 两臂碰撞守卫已接（或书面接受 `NoCollisionGuard`）
- [ ] RT 服务器 deadman 为正常值（非 3600000）
- [ ] 逐级验证（第 8 节）完成
