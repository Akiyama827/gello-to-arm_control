# franka/ — 随仓库分发的 FR3 描述（vendored）

本目录是从上游 **franka_description** 生成并**随仓库一起分发**的真实 FR3 描述，
供 3D 可视化与碰撞几何使用。上游许可为 **Apache-2.0**，
Copyright 2023 Franka Robotics GmbH（见本目录 `LICENSE` / `NOTICE`）。

- `urdf/fr3.urdf` — 由 `xacro` 展开生成，含 Franka Hand。
- `meshes/robots/fr3/visual/`、`meshes/robots/fr3/collision/` — FR3 各连杆网格（STL）。
- `meshes/robot_ee/franka_hand_white/` — Franka Hand 网格。

> 之所以把这些资产放**进仓库**（而不是让使用者联网拉取），是为了让新机器
> **克隆后无需联网即可直接运行真实 FR3 的 3D 查看器**
> （`examples/leader_follower_rerun.py`、`examples/leader_follower_interactive.py`）。
> 全部约 12MB。

## 重新生成 / 更新

```bash
pip install xacro trimesh pycollada
python tools/assets/fetch_fr3_description.py            # 重新 staging（会覆盖本目录）
python tools/assets/fetch_fr3_description.py --check    # 只检查是否就绪
```

重新生成后请确认 `LICENSE` 与 `NOTICE` 仍在；若上游更新了许可，请同步替换。
