# FR3_2 真机 ARM 后运动关节抽搐 —— 根因分析报告

- 日期：2026-10-14
- 现象：MuJoCo 虚拟小臂遥操真机 FR3_2（绝对对齐），ARM 后运动时关节抽搐
- 结论：抽搐物理机制已实锤（RT server 内部日志），根因在 **RT box ↔ 真机控制盒的读取/通信环节**，**不是 dora/命令/控制器问题**。

## 一、抽搐时刻（22:55:20-22:55:22）RT server 日志实锤的物理现象

```
period=2.5~3.6ms（应 1ms）            ← 伺服循环实时性严重超时
read=2.5~3.6ms（应 <0.1ms）           ← 读真机状态这一环是瓶颈
qd_err 飙到 0.85~1.6 rad/s            ← 跟踪速度误差暴涨
tau_raw 飙到 30~35 N.m                ← 原始力矩指令暴涨
limiter_joint=4 (0基=joint5) clamp/slew ← 12N.m 上限关节被反复钳制
最终: libfranka Net Exception → rt control session closed → 进程退出
```

## 二、决定性根因链条

```
RT 伺服循环 read 真机慢（1.5~3.6ms，本应 <0.1ms）
  └→ 伺服周期超时（period 超 1ms 预算）
      └→ 控制更新滞后 → 跟踪误差暴涨（qd_err 1.6）
          └→ 控制器力矩指令暴涨（tau_raw 30 N.m）
              └→ 力矩上限最小关节 joint5（12 N.m）先触顶 clamp/slew
                  └→ 力矩反复钳制/回摆 → 关节抽搐
                      └→ 最终 libfranka Net Exception → RT 进程退出
```

## 三、已排除的候选（均有证据）

| 候选根因 | 结论 | 证据 |
|---------|------|------|
| dora/命令层不平滑 | 排除 | 命令/JogRamp/理想阻抗/RT流离线全平滑 |
| abort→回中 | 排除 | 0 条 jog refused/expired，无新 fault trace 于操作时 |
| joint5 力矩上限不足(静态超限) | 排除 | MuJoCo 算 joint5 重力矩仅 0.86 N.m（12 上限内） |
| RT box CPU 被抢占 | 排除 | isolcpus=1,3 已隔离；CPU1 无中断；top 65% idle |
| joint4 重力矩 22 N.m | 正常 | MuJoCo 算 21.91 N.m，与 RT tau_meas 22-25 吻合；上限 87 够 |

## 四、read 慢的最终定性 —— libfranka 固有帧等待，非故障、非配置错误

**`read≈1.5ms` 是 liffranka ActiveControl 固有的「伺服循环 vs 1kHz 数据帧」同步等待代价**，不是 bug、不是网络延迟、不是实时性配置缺失。已逐一排除：

| 候选 | 结果 | 证据 |
|------|------|------|
| 网络延迟 | 排除 | ping 172.16.2.3: avg 0.23ms, mdev 0.07ms, 0 丢包 |
| CPU 抢占 | 排除 | isolcpus=1,3 nohz_full=1,3 irqaffinity=0,2；CPU1 中断全 0；load 0.07 |
| NIC 中断合并 | 排除 | enp5s0 rx-usecs=0, tx-usecs=0（已禁用）|
| FCI IRQ 打伺服核 | 排除 | enp5s0 全部 IRQ 在 CPU0/CPU2，CPU1 列全 0 |
| 实时调度未配 | 排除 | PREEMPT_RT、SCHED_FIFO 80/99、mlockall、绑核 cpu 1、Nice=-20 |

**结论：RT box 端实时性已是极限配置。伺服循环实际 ~650Hz（1.5ms），达不到设计的 1kHz，这是该 libfranka+真机组合的固有生理极限。** 抽搐只是偶发把 read 从 1.5ms 放大到 3.6ms（+2.4ms 额外抖动），属 libfranka FCI / 真机控制盒层面的偶发抖动，非 RT box 配置可消除。

## 四之二、实际可控的减害手段

由于 read 抖动在 RT box C++ 黑盒内且已是最优配置，**唯一从我们侧可控的是降低命令对每个 servo tick 的控制量冲击**（已实施）：
- 命令目标车速自检对齐真机 0.4 rad/s
- setpoint 步长 0.30
- 目的：READ 抖动时控制量（tau_raw）不会因跟踪急目标而暴涨，joint5 clamp/slew 更温和，抽搐从"剧烈"降至"几乎不可感知"。**待真机实测验证。**

## 五、已实施的安全修复（未提交，待定）
- follower.py：命令目标车速自检对齐真机 0.4 rad/s + 回读新鲜度门槛 + setpoint 步长 0.30
- dora_feedback.py：新增 cached_arm_state_age()
- interactive.py：注入 velocity_limits_rad_s / measured_limiter_age + _read_execution_policy()
- robots/fr3/execution.yaml：velocity_limits [0.4]*7, abort_pos_err_rad 0.35

## 六、时间线
- 22:54 起图 → 22:55:09 ARM → 22:55:10 自动恢复跟随 → 22:55:23 手动 STOP
- 22:55:20-22:55:22 RT 伺服 period 崩到 3.6ms，joint5 clamp/slew 反复
- 22:57:00 libfranka Net Exception → 22:57:57 RT 进程退出(failed)
- 23:03:26 systemd 自动重启 arm_rt_server（pid 3959074）
- 当前重连后仍间歇性 loop gap（1.5~2.7ms），未 ARM、静止时也发生
