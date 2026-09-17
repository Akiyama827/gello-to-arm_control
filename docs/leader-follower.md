# 小臂 -> 大臂 实时主从遥操作

人手动操控小臂（leader），实时采集其关节状态，经换算后驱动大臂（follower），
使小臂动作映射到大臂。小臂关节电机为宇树 **S288**，大臂为 `arm_control` 里的机械臂。

## 数据流

```
S288LeaderArm / GelloLeaderAdapter / FakeLeaderArm   （读小臂）
        │  get_joint_state() -> [6 臂关节(rad), 夹爪(0..1)]
        ▼
Retargeter.map()                                     （换算）
        │  θ_big_i = offset_i + scale_i * sign_i * θ_small[src_i]
        │  夹爪 [0,1] -> 大臂手指宽度(m)；平滑/限幅/限位/auto_align
        ▼
SafetyMonitor                                        （安全门）
        │  leader 合法性/断流、关节限位、目标跳变、两臂碰撞、跟踪误差…
        ▼
DryRunFollower / DoraJogFollower / RtFollower / FakeFollower （下发大臂）
```

## 模块

| 文件 | 作用 |
|---|---|
| `arm_control/leader_follower/s288.py` | S288 规格、MIT 协议编解码、串口/仿真总线、关节链读写 |
| `leader.py` | 小臂接口 + S288 / gello 适配 / fake 三种实现 |
| `mapping.py` | 关节空间换算（sign/offset/scale/限位/夹爪/auto_align） |
| `follower.py` | 大臂四种下发后端 |
| `safety.py` | 碰撞守卫 + 安全监视器 + `SafetyStop` |
| `loop.py` | 实时主循环 `TeleopLoop` |
| `config.py` | YAML 配置与装配 |
| `examples/leader_follower_teleop.py` | 可运行示例（默认全仿真） |
| `examples/configs/leader_follower.yaml` | 示例配置 |
| `tools/bench/check_leader_follower.py` | 离线自检 |

## S288 参数（来源：Unitree 官网，2026 查得）

- 减速比 **288.35:1**，力矩常数 **0.554 N·m/A**，堵转 **0.6 N·m**，空载 **16.5 rad/s@12V**
- 双绝对值编码器（转子 15bit / 输出端），半双工串口 **8N1 @ 6 Mbps**，ID 0–14
- 控制模式：混合 MIT（q/dq/tau/kp/kd）；反馈含转子/输出端角度、速度、扭矩、温度、电压、错误位

单位换算是"电机参数不同"的核心：`q_out=q_rot/r`、`tau_out=tau_rot*r`、
`kp_rot=kp_out/r²`、`kd_rot=kd_out/r²`。

> **协议不确定性**：宇树官方 `unitree_actuator_sdk` 只覆盖 GO-M8010-6/A1/B1，不含 S288。
> `S288Codec` 的帧布局按宇树公开 MIT 协议通用形式实现，**上线前须用 Unitree Motor
> Assistant 或逻辑分析仪核对帧长/字段/CRC**。未确认时用 `use_fake_bus: true` 开发。

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

> 注意：`DoraJogFollower` 的 jog 通道不回读。要启用跟踪误差门，必须另接订阅
> `motor_state` 的反馈实现；否则应设置 `feedback_required: false` 并明确接受该保护缺失。

## 运行

```bash
# 离线自检（无需硬件）
PYTHONPATH=. python -B tools/bench/check_leader_follower.py
# 全仿真跑 10s
PYTHONPATH=. python -B examples/leader_follower_teleop.py --duration 10
# 带玩具碰撞守卫，演示两臂碰撞停机
PYTHONPATH=. python -B examples/leader_follower_teleop.py --collision-demo --duration 10
```

## 待办：大臂不必逐关节复刻，可做路径优化（用户 2026-09-18 需求）

需求：**在"效果一样"（末端位姿/任务等价）的前提下，大臂可以走优化过的路径，
不必严格复刻小臂的逐关节动作。**

计划在 `mapping.py` 增加任务空间模式，与现有关节空间模式并存、可配置切换：

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
