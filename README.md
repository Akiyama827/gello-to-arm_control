# arm_control

Reusable robot control, planning, plant backends, and Dora process adapters.
The consuming project supplies robot instances, measured calibration, workcell
geometry, task policy, and hardware deployment. This library does not require
the consuming project or a perception package to import its generic APIs.

Start with [Architecture](docs/ARCHITECTURE.md) for ownership and dependency
boundaries, and [Runbook](docs/RUNBOOK.md) for current commands. The
[historical manuals](docs/history/README.md) preserve previous designs and bench
records; they are not current operating instructions.

> **小臂 → 大臂 主从遥操作**：怎么跑、怎么看可视化、常见问题，见
> **[使用说明书 docs/leader-follower-usage.md](docs/leader-follower-usage.md)**。

## Quickstart（新机器：仿真 + 3D 可视化）

```bash
git clone https://github.com/Akiyama827/gello-to-arm_control.git arm_control   # 换成你的仓库地址
cd arm_control
python3 -m venv .venv && source .venv/bin/activate            # fish: source .venv/bin/activate.fish
pip install -e '.[sim,viz,assets]'                            # 想锁版本：pip install -r requirements.txt
PYTHONPATH=. python -B tools/bench/check_leader_follower.py   # 离线自检，应打印“全部通过”
PYTHONPATH=. python -B examples/leader_follower_rerun.py      # 真实 FR3 + 缩小孪生小臂 + 曲线
```

- 需要 **Python ≥ 3.10**（3.12 验证）和图形界面；无显示环境用 `--headless` / `--save`。
- **真实 FR3 资产已随仓库分发**（`franka/`，约 12MB），克隆后**无需联网**即可跑真实 FR3 的
  3D 查看器；需要更新时再跑 `python tools/assets/fetch_fr3_description.py`。
- 小臂外观默认是**缩小的 FR3 孪生**（与 FR3 同构，关节同号可对应）。
  另提供 **Franka 官方 GELLO 真实零件**外观（`gello_leader/franka_fr3/*.STL`，随仓库分发），
  用 `--leader-appearance gello` 开启；其零件间装配位姿是**反求近似**
  （上游只发布打印用 STL，没有装配体/CAD），待拿到官方 CAD 再精确对齐。查看/调参：
  `PYTHONPATH=. python -B examples/gello_leader_preview.py`。
- 交互拖拽版：`PYTHONPATH=. python -B examples/leader_follower_interactive.py`。
- 胶囊示意臂（无需任何 FR3 资产）：`PYTHONPATH=. python -B examples/leader_follower_viewer.py`。
- **真 S288 → 仿真 FR3**（插上硬件就能看真小臂带动 Rerun 里的真 FR3）：
  `PYTHONPATH=. python -B examples/leader_follower_rerun.py --config examples/configs/leader_follower_s288_sim.yaml --leader-source config`
  （先 `pip install pyserial`）。S288 走官方 `digital_servo` 协议（20B/26B + CRC32），
  已由 `leader_follower/s288.py` 逐字节实现，**不需要** `unitree_actuator_sdk`。
- **标定真 S288 小臂**（读数 / 找零 / 定方向 / 标夹爪，可写回 YAML）：
  `PYTHONPATH=. python -B examples/s288_calibrate.py --config examples/configs/leader_follower_s288_sim.yaml read`
  子命令 `read|zero|signs|gripper`；无硬件加 `--bus fake` 演练，`--apply` 写回配置。
  用 `zero --from-ex` 可借**绝对单圈编码器 ExPos** 做上电即绝对零位（`use_ex_pos: true`）。
  单关节（如夹爪）也可单独切 ExPos：配置里 `leader.gripper_use_ex_pos: true`。
- **直接驱动 FR3（不经小臂/映射）自检**：`PYTHONPATH=. python -B examples/franka_direct_demo.py`
  （7 关节摆动 + 夹爪 0↔40mm 自动演示，用来验证"程序能不能命令 FR3"）。
- **S288 单电机探针**（零刚度反驱 + 小幅主动驱动，排查"反馈冻结"）：
  `PYTHONPATH=. python -B examples/s288_gripper_probe.py --config <真机配置> --id 8`。
- 接**真机**还需厂商工具链（`pyserial` 读 S288 / `pylibfranka` / `dmcan`）与 C++ 实时核心，
  见 [部署文档 docs/leader-follower-deploy.md](docs/leader-follower-deploy.md)。

### 已知问题（当前真机现场）

- **S288 夹爪电机（第 8 个，ID 8）转子多圈反馈故障**：`q_out` 冻结（手动扳动时不变）
  且 `err=256`（`0x100`）重上电不消，但**输出端的绝对单圈编码器 `ExPos` 仍正常**
  （随扳机变化）。结论是**转子多圈计数坏**，不是扳机没接、也不是程序问题。
  该电机因此**不能再主动驱动**（FOC 依赖转子位置），但**仍可读**。
  - **已用 ExPos 规避**：配置里 `leader.gripper_use_ex_pos: true`，只把**夹爪**这一个
    关节的角度源切到 `ExPos`（臂关节仍用 `q_out`），并重标了夹爪 → 夹爪遥操作恢复正常。
  - 诊断脚本：`examples/s288_gripper_probe.py`（零刚度反驱 + 小幅主动驱动）。
  - 详见[部署文档 §9.1](docs/leader-follower-deploy.md)；彻底修复需换/修该电机。

## Layout

| Path | Owner/responsibility |
|---|---|
| `arm_control/motion/` | Shared joint-state, command, and trajectory values |
| `arm_control/control/` | Execution, controller state, safety, gain validation |
| `arm_control/planning/` | IK, collision checking, planning, retiming, preview builders |
| `arm_control/contracts/` | Central Dora/Arrow wire contracts |
| `arm_control/plants/` | Plant protocol; DM, Franka, remote RT, MuJoCo adapters |
| `arm_control/end_effectors/` | Generic grasp and Hand behavior |
| `arm_control/ui/`, `arm_control/viz/` | Operator adapters, packaged web assets, telemetry |
| `arm_control/simulation/` | Compatibility imports, generic convex decomposition and scene mirroring |
| `nodes/` | Dora executable wrappers; package modules contain implementations |
| `examples/` | Standalone console application, example configs, reusable mode profiles |
| `tools/` | Generic DM/RT diagnostics and asset staging |
| `dataflows/` | Generic process compositions |
| `rt/` | Reusable C++ servo core, backends, protocol, and Franka Hand bridge |

## Quick offline checks

Planning requires `pip install -e '.[planning]'`, which pins TOPPRA 0.6.10.
The shared retimer blends corners locally, solves per-joint velocity and
acceleration constraints, and bounds the returned Hermite curve. Planning
callers validate the blended geometry for collisions. Jerk limits are not
provided. `PYTHONPATH=. python tools/bench/check_retiming.py` exercises the
corner regression, short moves, reversals, collision retries and replay caps.

From this repository, in an environment containing the required dependencies:

```bash
PYTHONPATH=. python -B -m arm_control.config
PYTHONPATH=. python -B -m arm_control.frames
PYTHONPATH=. python -B -m arm_control.messages
PYTHONPATH=. python -B examples/run_console.py --check
```

`--check` reports local example assets and launches nothing. It is not a
simulation pass or hardware-readiness check. Installation and optional
dependencies are declared in `pyproject.toml`; see the runbook before launching.

The duplicate trajectory-executor process is retired; execution guards are an
explicit consumer-selected controller policy, and the reusable executor class
remains. The relocated MuJoCo backend still retains coupled legacy fixture/contact policy.
Project scene construction now
belongs to the consumer; this does not make the physics core fully generic.
