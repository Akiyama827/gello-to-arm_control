#!/usr/bin/env python3
"""小臂外观 = **Franka 官方 GELLO 零件** 的查看 / 调参工具。

小臂的骨架仍是等比缩小的 FR3（保证 ``leader_fr3_joint_i <-> fr3_joint_i`` 一一
对应），但每一节的视觉网格换成了 ``gello_leader/franka_fr3/`` 里的**真实 3D 打印件**。
零件的相对位姿写在 ``arm_control/simulation/leader_arm_model.py`` 的
``GELLO_LINK_PARTS`` / ``GELLO_BASE_PARTS`` 里（每个连杆坐标系下的 ``位置(m) + rpy(rad)``）。

那个表是"按零件孔轴 + 包围盒自动摆出来的近似装配"——**尤其是绕关节轴的法兰 roll
需要按实物微调**。改完表后用这个窗口看效果、迭代即可。

窗口操作（MuJoCo 原生）：左键拖动 = 转视角，右键 = 平移，滚轮 = 缩放；Esc/q 退出。

运行（fish）：
    source .venv/bin/activate.fish
    PYTHONPATH=. python -B examples/gello_leader_preview.py
    PYTHONPATH=. python -B examples/gello_leader_preview.py --appearance twin   # 看旧孪生
    PYTHONPATH=. python -B examples/gello_leader_preview.py --save /tmp/gello.png
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402,F401  # 子模块不会随 import mujoco 自动加载

from arm_control.simulation.leader_arm_model import (  # noqa: E402
    GELLO_LEADER_SCALE,
    build_combined_spec,
)
from arm_control.simulation.mujoco_model import build_mujoco_model  # noqa: E402

FR3_URDF = ROOT / "franka" / "urdf" / "fr3.urdf"


def _render_sheet(
    model: mujoco.MjModel, data: mujoco.MjData, *, separation: float = 1.15, size: int = 460
) -> np.ndarray:
    """离屏渲染四视角，返回拼好的 RGB 图（用于无显示环境调参）。"""
    from PIL import Image

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # 取景：两臂整体
    cam.lookat[:] = [-0.5 * separation, 0.0, 0.2]
    cam.distance = 1.4 * separation + 0.6
    renderer = mujoco.Renderer(model, size, size)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0  # 关掉碰撞网格，露出按连杆着色的视觉网格
    tiles = []
    for az, el in ((135, -10), (180, 0), (90, 0), (45, 20)):
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(data, cam, opt)
        tiles.append(renderer.render().copy())
    del renderer
    top = np.concatenate(tiles[:2], axis=1)
    bottom = np.concatenate(tiles[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="小臂 GELLO 零件外观 查看/调参")
    parser.add_argument("--separation", type=float, default=1.15, help="两臂基座间距（米）")
    parser.add_argument("--leader-scale", type=float, default=None,
                        help=f"小臂缩放（默认 gello {GELLO_LEADER_SCALE}，twin 0.8）")
    parser.add_argument("--appearance", choices=("gello", "twin"), default="twin",
                        help="twin=缩小 FR3 孪生（默认，连贯）；"
                             "gello=GELLO 零件外观（装配位姿为反求近似，待官方 CAD 精确对齐）")
    parser.add_argument("--save", default=None, help="离屏渲染到 PNG（无显示环境）")
    args = parser.parse_args(argv)

    if not FR3_URDF.exists():
        print(f"[gello-preview] 找不到 FR3 URDF：{FR3_URDF}", file=sys.stderr)
        print("[gello-preview] 先运行: python tools/assets/fetch_fr3_description.py", file=sys.stderr)
        return 2

    cache = Path(tempfile.gettempdir()) / "arm_control_fr3_mjcache"
    staged = build_mujoco_model(FR3_URDF, cache_dir=cache, keep_visual=True)
    use_gello = args.appearance == "gello"
    scale = args.leader_scale
    if scale is None:
        scale = GELLO_LEADER_SCALE if use_gello else 0.8

    spec, _refs = build_combined_spec(
        str(staged),
        leader_position=(-args.separation, 0.0, 0.0),
        leader_scale=scale,
        leader_gello_parts=use_gello,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.save:
        from PIL import Image

        img = _render_sheet(model, data, separation=args.separation)
        Image.fromarray(img).save(args.save)
        print(f"[gello-preview] 渲染写入 {args.save}（外观={args.appearance}，缩放={scale}）")
        return 0

    print(f"[gello-preview] 外观={args.appearance}，缩放={scale}；关窗退出。", flush=True)
    with mujoco.viewer.launch_passive(model, data) as handle:
        handle.opt.geomgroup[0] = 0  # 关掉碰撞网格，露出按连杆着色的视觉网格
        while handle.is_running():
            handle.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
