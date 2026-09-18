# 主从遥操作 · 使用说明书

这套程序把**小臂（leader，8×宇树 S288）**的动作实时映射到**大臂（follower，FR3）**，
带安全监控，并提供可视化。本文是"怎么用"；原理见
[leader-follower.md](leader-follower.md)，真机部署见
[leader-follower-deploy.md](leader-follower-deploy.md)。

---

## 1. 这套程序由什么组成

| 类别 | 文件 | 作用 |
| --- | --- | --- |
| 核心 | `arm_control/leader_follower/*.py` | 采集→换算→下发→安全的主从逻辑 |
| 示例 | `examples/leader_follower_teleop.py` | 命令行跑一遍（默认全仿真） |
| 可视化 | `examples/leader_follower_viewer.py` | MuJoCo 窗口：两条胶囊臂（无需资产） |
| 可视化 | `examples/leader_follower_rerun.py` | Rerun 单窗口：真实 FR3 网格 + FR3 孪生小臂 + 曲线 |
| 可视化 | `examples/leader_follower_interactive.py` | MuJoCo 窗口：**鼠标拖拽小臂**，真实 FR3 跟随 |
| 模型 | `arm_control/simulation/leader_arm_model.py` | 小臂模型（**等比缩小的 FR3 孪生**，网格+关节复用 FR3） |
| 安全 | `arm_control/simulation/mj_collision_guard.py` | 同场景真实几何的两臂碰撞守卫 |
| 配置 | `examples/configs/leader_follower.yaml` | 仿真配置（fake/fake） |
| 配置 | `examples/configs/leader_follower_real.example.yaml` | 真机配置模板 |
| 自检 | `tools/bench/check_leader_follower.py` | 无硬件离线自检 |
| 资产 | `tools/assets/fetch_fr3_description.py` | 获取并 staging FR3 描述到 `franka/` |
| 真机 | `dataflows/leader_teleop_franka.yml` | 真机 Dora 图 |

四种运行形态，按需选：

| 我想…… | 用哪个 | 需要 FR3 资产？ |
| --- | --- | --- |
| 快速确认逻辑没坏 | `leader_follower_teleop.py` | 否 |
| 看动作，机器上没 FR3 资产 | `leader_follower_viewer.py`（MuJoCo） | 否 |
| 看**真实 FR3 外形 + FR3 孪生小臂 + 曲线** | `leader_follower_rerun.py`（Rerun） | **是** |
| **用鼠标拖小臂**、看真实 FR3 实时跟随 | `leader_follower_interactive.py`（MuJoCo） | **是** |
| 接真机 | `dora run dataflows/leader_teleop_franka.yml` | 是 |

---

## 2. 环境准备（一次性）

```bash
# 0) 克隆仓库到任意目录（示例目录名用 arm_control）
git clone https://github.com/Akiyama827/gello-to-arm_control.git arm_control
cd arm_control

# 1) 虚拟环境（bash/zsh 用 activate；fish 用 activate.fish）
python3 -m venv .venv
source .venv/bin/activate

# 2) 依赖：仿真 + 可视化 + 资产工具
pip install -e '.[sim,viz,assets]'
#    想锁版本复现：pip install -r requirements.txt
#    （用 requirements.txt 时未安装本包，运行时记得带 PYTHONPATH=.）

# 3) staging 真实 FR3 描述（约 27MB，落在 franka/，已被 gitignore）
python tools/assets/fetch_fr3_description.py
#    之后可用 --check 复查：
#    python tools/assets/fetch_fr3_description.py --check
```

> 第 3 步只有看**真实 FR3 外形**的两个查看器（`leader_follower_rerun.py`、
> `leader_follower_interactive.py`）需要；命令行示例和胶囊版
> `leader_follower_viewer.py` 都不需要。该步会联网 `git clone`
> `franka_description`，离线机器请提前把 `franka/` 放到仓库根目录。

---

## 3. 五分钟上手

```bash
cd arm_control            # 换成你的克隆目录
source .venv/bin/activate # fish: source .venv/bin/activate.fish
```

**第 1 步 · 自检**（不碰硬件，应打印"全部通过"）

```fish
PYTHONPATH=. python -B tools/bench/check_leader_follower.py
```

**第 2 步 · 纯仿真跑一遍**（默认 FakeLeader + FakeFollower）

```fish
PYTHONPATH=. python -B examples/leader_follower_teleop.py --duration 10
```

看到 `[teleop] tick=... 实际≈100Hz ...` 就对了。

**第 3 步 · 看可视化**（三选一）

```fish
# A. 真实 FR3 + FR3 孪生小臂 + 曲线（Rerun 窗口）
PYTHONPATH=. python -B examples/leader_follower_rerun.py

# B. 胶囊示意臂（MuJoCo 窗口，无需 FR3 资产）
PYTHONPATH=. python -B examples/leader_follower_viewer.py

# C. 鼠标拖拽小臂、真实 FR3 实时跟随（MuJoCo 窗口，需 FR3 资产）
PYTHONPATH=. python -B examples/leader_follower_interactive.py
```

---

## 4. 可视化怎么用

### 4.1 Rerun：真实 FR3 外形 + FR3 孪生小臂 + 时间序列曲线

```fish
PYTHONPATH=. python -B examples/leader_follower_rerun.py
```

窗口里会看到：

- **3D 视图**
  - 右侧 = **真实 FR3**（视觉网格），跟随大臂实测关节角，手指跟夹爪开合。
  - 左侧 = **小臂模型**（**等比缩小的 FR3 孪生**；宇树 S288 无公开网格，直接复用
    真实 FR3 网格 + 同一套关节坐标系缩小而成）。
  - 两条臂**同构**：`leader_fr3_joint_i ↔ fr3_joint_i` 一一对应，**同号连杆同色**
    （J1..J7 七彩、夹爪灰），每个关节位置标 `J1..J7`，对应关系一眼可对。
- **Time series 视图**（Rerun 自动聚合）：每个关节的
  `plots/leader/qN`、`plots/follower_cmd/qN`（指令）、`plots/follower_meas/qN`（实测）、
  `plots/track_err/qN`（跟踪误差），以及 `plots/gripper_m`、`plots/collision_distance_m`。

操作：3D 里**左键拖=旋转、右键拖=平移、滚轮=缩放**；底部时间轴可**暂停/拖动回放**；
左侧实体树可点选高亮。停止：终端 `Ctrl-C`；Rerun 窗口单独关即可。

两条臂的**底座都落在世界原点所在的同一水平面（z=0）**上，只是小臂等比缩小了
（默认 `--leader-scale 0.75`），所以能直接比姿态。

常用参数：

```fish
PYTHONPATH=. python -B examples/leader_follower_rerun.py --duration 20       # 跑 20s
PYTHONPATH=. python -B examples/leader_follower_rerun.py --separation 1.5    # 两臂分得更开
PYTHONPATH=. python -B examples/leader_follower_rerun.py --leader-scale 0.6  # 小臂缩得更小
PYTHONPATH=. python -B examples/leader_follower_rerun.py --collision-demo    # 演示碰撞停机
PYTHONPATH=. python -B examples/leader_follower_rerun.py --no-spawn          # 初始化但不弹窗
```

### 4.2 MuJoCo：胶囊示意臂（最快，无需资产）

```fish
PYTHONPATH=. python -B examples/leader_follower_viewer.py
```

左蓝 = 小臂，右橙 = 大臂（实测）；左上角显示 tick、跟踪误差、两臂最近距离，
安全停机时显示 `STOPPED` 和原因。

```fish
PYTHONPATH=. python -B examples/leader_follower_viewer.py --duration 20
PYTHONPATH=. python -B examples/leader_follower_viewer.py --separation 2.2       # 两臂分得更开
PYTHONPATH=. python -B examples/leader_follower_viewer.py --collision-demo
PYTHONPATH=. python -B examples/leader_follower_viewer.py --headless --duration 3   # 无窗口自检
```

### 4.3 MuJoCo：鼠标拖拽小臂，真实 FR3 实时跟随（交互式）

```fish
PYTHONPATH=. python -B examples/leader_follower_interactive.py
```

窗口里：

- 左 = **小臂模型**（**等比缩小的 FR3 孪生**，与右侧同构、同号连杆同色、标 `J1..J7`），
  **可用鼠标拖拽**；
- 右 = **真实 FR3 网格**，按 `Retargeter` 实时跟动。仿真里小臂是 FR3 孪生，所以
  关节映射按**直连**（`sign=+1`）走，拖哪一节、大臂同号关节就同向跟动。

拖拽方式（MuJoCo 原生扰动 = 施加弹簧力，小臂关节在阻尼下运动）：

- **默认已经选中小臂末端**，直接按住 **Ctrl + 鼠标右键拖动 = 平移施力**（推荐），
  **Ctrl + 鼠标左键拖动 = 绕选中点旋转施力**；
- 想拖别的连杆：先 **鼠标左键双击** 选中那一节，再按上面的方式拖动；
- 左键拖动 = 旋转视角，右键 = 平移视角，滚轮 = 缩放。

拖出来的小臂关节角会喂给**真正的 `TeleopLoop`**（与真机同一条安全链路：`auto_align`、
关节限位/跳变、跟踪误差、leader 超时、以及用**同场景真实几何**算的两臂最近距离碰撞
守卫）。所以越限 / 两臂将碰都会**安全停机并冻结大臂**，左上角显示 `[安全停机]`。

```fish
PYTHONPATH=. python -B examples/leader_follower_interactive.py --separation 1.4   # 分得更开
PYTHONPATH=. python -B examples/leader_follower_interactive.py --leader-scale 0.6 # 小臂缩得更小
PYTHONPATH=. python -B examples/leader_follower_interactive.py --duration 60      # 60s 后停止下发
PYTHONPATH=. python -B examples/leader_follower_interactive.py --no-collision-guard
PYTHONPATH=. python -B examples/leader_follower_interactive.py --rerun             # 边拖边看曲线
```

> 小臂外形是**视觉替身**：宇树 S288 没有公开网格/URDF，这里直接把真实 FR3 网格
> 等比缩小后作为小臂，好处是关节一一对应、颜色也对得上；它**不代表 S288 的真实
> 外观**。真机小臂的零位/转向仍以 `S288LeaderArm` 的 `joint_offsets / joint_signs`
> 标定为准（仿真里的"直连"映射只针对这个 FR3 孪生，真机标定配置不受影响）。

### 4.4 存盘 / 回放（无显示环境或存档）

```fish
# Rerun 版存成 .rrd（不弹窗）
PYTHONPATH=. python -B examples/leader_follower_rerun.py --save /tmp/teleop.rrd --duration 10
# 回放
.venv/bin/rerun /tmp/teleop.rrd
```

### 4.5 小臂模型（FR3 孪生）怎么来的

宇树 S288 没有公开的网格/URDF。为了让"小臂和大臂的关节对应关系"一眼可见，本仓库
**不再用 capsule 拼近似外形**，而是直接复用真实 FR3 的描述做一条**等比缩小的 FR3**：

- 同一个 `MjSpec`（`arm_control/simulation/leader_arm_model.py`）里，用
  `mujoco.MjSpec.attach` 把 FR3 整体复制一份并缩放（`--leader-scale`），网格自动
  加 `leader_` 前缀，挂到大臂基座左侧 `(-separation, 0, 0)` 处；
- 于是 `leader_fr3_joint_i ↔ fr3_joint_i` **一一对应**，夹爪用 Franka Hand 双指；
- `color_arm_links()` 按连杆编号给**两条臂同号连杆上同一颜色**（J1..J7 七彩、夹爪灰），
  Rerun 里再在每个关节位置标 `J1..J7`；
- 两臂底座都在 **z=0**，只是缩放比不同。

> 因为小臂是 FR3 孪生，仿真的关节映射会调用 `force_identity_arm_mapping()` 改成
> **直连**（`sign=+1`、`scale=1`），这样拖小臂时大臂是"同形跟动"而不是"镜像"。
> 这**只影响仿真查看器**；真机 S288 的 `sign/offset` 标定仍在 YAML 里，不受影响。

想改配色/缩放：颜色在 `leader_arm_model.py` 的 `JOINT_COLORS`；缩放用
`--leader-scale`（Rerun 默认 0.75、交互式默认 0.8）。

---

## 5. 配置文件（YAML）

仿真配置 `examples/configs/leader_follower.yaml` 关键项：

| 段落 | 字段 | 含义 |
| --- | --- | --- |
| `leader` | `kind` | `fake` / `s288`（真机） / `gello` |
| `leader` | `n_arm_joints` | 臂关节数（7；加夹爪共 8 个 S288） |
| `follower` | `kind` | `dry_run` / `fake`（仿真） / `dora`（真机） / `rt` |
| `follower` | `initial` | FR3 合法初始位形（`fr3_joint4` 零位越限，必须给合法值） |
| `mapping` | `auto_align` | 启动时把大臂对齐到小臂当前姿态，无跳变接管 |
| `mapping` | `joints[]` | 每个关节 `sign/offset/scale/lower/upper/max_rate` |
| `mapping` | `gripper` | 夹爪→单指位移（米）的换算 |
| `safety` | `limits` | 各类安全门阈值（见第 6 节） |
| `loop` | `hz` | 控制频率（默认 100Hz） |

真机：复制 `leader_follower_real.example.yaml`，按现场标定 `sign/offset` 与总线/端口。

---

## 6. 安全机制（为什么它会自己停）

`SafetyMonitor` 在每个 tick 检查若干"门"，任一不过就抛 `SafetyStop` →
立即 `follower.safe_stop()` 并停止。门包括：

| 门 | 触发条件（默认） |
| --- | --- |
| leader 超时 | 小臂数据超过 `leader_timeout_s`（0.3s）没更新 |
| 关节限位 | 目标超出 `joint_lower/upper`（含 `joint_margin_rad`） |
| 目标跳变 | 单 tick 变化 > `max_step_rad`（0.30 rad） |
| 目标速度 | 超过 `max_target_vel_rad_s`（2.0 rad/s） |
| 回读必需 | `feedback_required: true` 时无回读/回读陈旧（> `feedback_timeout_s`） |
| 跟踪误差 | 命令-实测误差持续超 `track_err_rad`（0.15 rad）达 `track_err_hold_s` |
| 反馈故障 | 大臂报故障位/失能 |
| 两臂碰撞 | 最近距离 ≤ `collision_stop_m`（1cm）停机；≤ `collision_warn_m`（5cm）限速 |

演示：加 `--collision-demo` 会装一个玩具球体守卫（两臂各 6 球、沿 y 分开），
跑到逼近时你会看到它自动停机并打印原因。

交互式查看器 `leader_follower_interactive.py` 默认用更保真的
`MjGeomDistanceGuard`：在两臂所在的同一个 MuJoCo 模型里，用 `mj_geomDistance`
逐对算 leader 几何与 FR3 几何的最近有符号距离，直接作为上面这道碰撞门。
想关掉可用 `--no-collision-guard`。

---

## 7. 真机运行（简述）

前提：部署侧已按 [leader-follower-deploy.md](leader-follower-deploy.md) 配好
`arm_controller` / `franka_gripper` 与真机配置。

```bash
LEADER_FOLLOWER_CONFIG=$PWD/configs/entries/leader_follower.yaml \
ARM_CONTROL_ROOT=$PWD ARM_CONTROL_CONFIG=$PWD/configs/entries/real_franka.yaml \
  dora run libs/arm_control/dataflows/leader_teleop_franka.yml
```

- `dora run` 必须在**已激活 venv** 的 shell 里跑（否则节点用系统 python）。
- 首次接真机建议 `follower.auto_arm: false`，只观察回读与 `auto_align`，确认无跳变后再 ARM。

---

## 8. 常见问题（FAQ）

**Q：Rerun 里只看到小臂线框、没有大臂 FR3？**
已在 `feat/leader-follower-teleop` 修复（网格改为 `static` 记录）；小臂也已从线框
换成**等比缩小的 FR3 孪生**（网格 + 关节坐标系复用真实 FR3）。先 `git pull` 到最新，
再重跑。若仍无，检查 `franka/urdf/fr3.urdf` 是否存在。

**Q：两臂离得太近 / 想分得更开？**
三个可视化都支持 `--separation <米>`（两臂基座间距），例如
`--separation 1.5`（Rerun）或 `--separation 2.2`（MuJoCo 胶囊版）。

**Q：交互式窗口里拖不动小臂？**
脚本**默认已经选中小臂末端**，直接按住 **Ctrl + 右键拖动 = 平移施力**（推荐），
**Ctrl + 左键拖动 = 旋转施力**。想拖别的连杆：先**鼠标左键双击**选中那一节，
再按上面的方式拖。直接左键拖是旋转视角，不会拖小臂。

**Q：小臂为什么长得和大臂一样，不像宇树 S288？**
因为 S288 没有公开网格。为让关节对应关系直观，小臂直接用**等比缩小的 FR3**
作视觉替身（见 4.5 节）；它**不代表 S288 的真实外观**，真机标定仍按 YAML 里的
`sign/offset` 走。

**Q：两臂底座不在同一水平面 / 想调小臂大小？**
两臂底座都已固定在 **z=0** 同一水平面。小臂大小用 `--leader-scale`（Rerun 默认
0.75、交互式默认 0.8）；间距用 `--separation`。

**Q：`.rrd` 里小臂网格缺失 / 只落了一部分？**
已在 `feat/leader-follower-teleop` 修复：脚本结束会主动 `rr.disconnect()` 落盘
（MuJoCo 退出时的段错误会跳过析构、丢缓冲）。`git pull` 到最新再重跑。

**Q：交互式窗口里大臂突然不动了、显示 `[安全停机]`？**
这是安全门生效（例如把小臂拖到让大臂目标越限，或两臂最近距离 ≤ 1cm）。
重新运行脚本即可恢复；这是预期行为，不是崩溃。

**Q：脚本说找不到 viewer / 不弹窗？**
几乎都是没激活 venv（`rerun` 不在 PATH）。`source .venv/bin/activate.fish` 后再跑；
或改用 `--save` 存盘，再用 `.venv/bin/rerun file.rrd` 打开。

**Q：报 `franka/urdf/fr3.urdf` 不存在？**
```fish
pip install -e '.[assets]'
python tools/assets/fetch_fr3_description.py
```

**Q：MuJoCo 窗口有一堆 `libdecor` / `OpenGL error 0x502` 告警？**
Wayland 会话的已知无害提示，窗口照常工作。起不来就换 X11 会话或 `--headless`。

**Q：fish 里 `source .venv/bin/activate` 报错？**
fish 要用 `source .venv/bin/activate.fish`。

**Q：`dora run` 报 `No module named arm_control`？**
没激活 venv（`dora` 靠 `VIRTUAL_ENV` 找节点 python）。

**Q：大臂 `fr3_joint4` 第一 tick 就安全停机？**
初始位形越限（合法范围约 `[-3.0421, -0.1518]`）。给 `follower.initial` 一个合法 home。

---

## 9. 命令速查

```fish
source .venv/bin/activate.fish

# 自检
PYTHONPATH=. python -B tools/bench/check_leader_follower.py

# 纯仿真
PYTHONPATH=. python -B examples/leader_follower_teleop.py --duration 10
PYTHONPATH=. python -B examples/leader_follower_teleop.py --collision-demo --duration 10

# 可视化：真实 FR3 + FR3 孪生小臂 + 曲线
PYTHONPATH=. python -B examples/leader_follower_rerun.py
PYTHONPATH=. python -B examples/leader_follower_rerun.py --save /tmp/teleop.rrd --duration 10
.venv/bin/rerun /tmp/teleop.rrd

# 可视化：胶囊示意臂
PYTHONPATH=. python -B examples/leader_follower_viewer.py
PYTHONPATH=. python -B examples/leader_follower_viewer.py --headless --duration 3

# 可视化：鼠标拖拽小臂 -> 真实 FR3 跟随（默认选中末端，Ctrl+右键平移）
PYTHONPATH=. python -B examples/leader_follower_interactive.py
PYTHONPATH=. python -B examples/leader_follower_interactive.py --separation 1.4 --rerun

# FR3 资产
python tools/assets/fetch_fr3_description.py
python tools/assets/fetch_fr3_description.py --check
```
