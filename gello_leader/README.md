# GELLO 小臂零件（Franka FR3）

本目录是从 **`wuphilipp/gello_mechanical`** 仓库的 `franka_fr3/` 子目录**原样复制**的
Franka 官方 GELLO 3D 打印件 STL，用于本仓库可视化里的小臂外观（避免克隆后再联网下载）。
单位是 **毫米**，每个零件都在**自身局部坐标系**里（没有装配变换）。

- 来源：`https://github.com/wuphilipp/gello_mechanical`（子目录 `franka_fr3/`）
- 许可：**MIT**，见同目录 [`franka_fr3/LICENSE`](franka_fr3/LICENSE)（© Franka Robotics GmbH）
- 上游说明：见同目录 [`franka_fr3/README.md`](franka_fr3/README.md)（装配拓扑、"电机法兰有
  4 种安装朝向"等）

## 装配位姿在哪

本目录只放**零件**。零件相对每条连杆坐标系的位姿写在
`arm_control/simulation/leader_arm_model.py` 的 `GELLO_LINK_PARTS` / `GELLO_BASE_PARTS`
里（`位置(m) + rpy(rad)`）。那些数值是按零件孔轴 + 包围盒**自动摆出来的近似装配**；
上游没有公开装配体/URDF/STEP，所以**绕各关节轴的法兰朝向（4 选 1，即 90° 的整数倍）
需要按实物微调**。用下面这个窗口边看边调：

```bash
PYTHONPATH=. python -B examples/gello_leader_preview.py --appearance gello
PYTHONPATH=. python -B examples/gello_leader_preview.py --appearance gello --save /tmp/gello.png  # 无显示环境
```

## 已随仓库分发的零件

`01_BASE`、`02_BASE_BEARING`、`03_A12_CONNECTOR`、`04_A23_MOTOR_FLANGE`、
`05_A34_CORNER_LINK`、`06_A45_CORNER_LINK`、`07_A56_MOTOR_FLANGE`、`08_A67_CONNECTOR`、
`09_LIMIT_STOP_BACK`、`11_MOTOR_SHAFT_UNIVERSAL`、`12_BEARING_UNIVERSAL`、
`20_TRIGGER_BODY`、`21_TRIGGER_FINGER`、`00_TABLE_BASE_MOUNT_SINGLE`。

（`00_TABLE_BASE_MOUNT_DUO`、`22_TRIGGER_STORAGE`、`23_TRIGGER_STORAGE_45` 未分发；
需要时从上游仓库取，或参考 `arm_control/simulation/leader_arm_model.py` 的
`GELLO_TABLE_MOUNT` 把安装板加进 `extra_base_parts`。）
