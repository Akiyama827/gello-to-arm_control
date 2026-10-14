#!/usr/bin/env python3
"""MUKCa 几何运动学标定 —— FR3 端到端 CLI。

用「双插座夹具」标定 FR3 大臂 URDF 里每个关节 ``<origin>`` 的 xyz/rpy 参数，
在校正 FK/运控的几何精度。无需外部测量设备。

原理（MUKCa，arXiv:2503.12584）：
    机械臂带着末端球/销，把法兰球心插到地面夹具的两个孔里（间距 d 已知）。
    每个插座位姿记录多组不同构型的关节角。利用两个物理约束反求几何参数：
      1) 同一孔内不同构型算出的球心应聚成一个点（一致性 C）；
      2) 两孔球心距离 = 已知夹具间距 d（距离畸变 D）。
    最小化 J = C + D + 正则化，用 CasADi/IPOPT 拟合每个被选关节的 origin 参数。

本脚本流程：
    采样目录（每个插座位姿一个子目录） -> 读 Placement -> 拟合 fr3_joint1..7
    -> admission 校验 -> 写回调校后的新 URDF（绝不覆盖原文件）。

前置依赖（运行时用，不在仓库默认依赖里）：
    pip install casadi forwardkinematics pinocchio pyyaml

用法：
    # 采样目录布局（示例）：
    #   <data>/front/manifest.yaml   hole_0.csv   hole_1.csv
    #   <data>/side/ manifest.yaml   hole_0.csv   hole_1.csv
    #   <data>/top/  manifest.yaml   hole_0.csv   hole_1.csv
    #
    # manifest.yaml 内容：
    #   arm_id: fr3
    #   session: 2026-xx-xx
    #   placement: front          # 唯一，区分不同插座位姿
    #   joints: [fr3_joint1, ..., fr3_joint7]   # CSV 列名/顺序
    #   fixture:
    #     spacing_m: 0.05         # 实测两孔中心距（米）
    #
    # 运行：
    PYTHONPATH=. python -B examples/mukca_fr3_calibrate.py \
        --data <采样目录> \
        --urdf franka/urdf/fr3.urdf \
        --out franka/urdf/fr3.calibrated.urdf \
        --root-link base --ball-link mukca_ball_link --parent-link fr3_link8 \
        --spacing 0.05            # reference 模式共用间距；缺省从 manifest 取

    # 只看流程但用假数据演练（不依赖真机/CasADi 之外的可选）：
    PYTHONPATH=. python -B examples/mukca_fr3_calibrate.py --self-check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_control.mukca import (  # noqa: E402
    Placement,
    add_ball_joint,
    admission_errors,
    load_placement,
    read_origins,
    rewrite_origins,
)

# FR3 的 7 个旋转关节（干 line：fr3_joint1..7），固定/手指关节不拟合。
FR3_ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FR3_BALL_LINK = "mukca_ball_link"
FR3_BALL_JOINT = "mukca_ball_joint"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MUKCa FR3 几何标定端到端 CLI。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=None,
                   help="采样数据根目录（每个插座位姿一个子目录，含 manifest.yaml / hole_0.csv / hole_1.csv）")
    p.add_argument("--urdf", type=Path, default=Path("franka/urdf/fr3.urdf"),
                   help="目标 URDF 路径")
    p.add_argument("--out", type=Path, default=None,
                   help="写回调校后 URDF 的路径；缺省为 <urdf>.calibrated.urdf")
    p.add_argument("--root-link", default="base",
                   help="FK 根 link（机器人基座）")
    p.add_argument("--parent-link", default="fr3_link8",
                   help="测量球心 frame 挂载的父 link（FR3 法兰口）")
    p.add_argument("--ball-link", default=FR3_BALL_LINK,
                   help="球心 link 名")
    p.add_argument("--ball-xyz", default="0 0 0.035",
                   help="球心相对 parent-link 的 xyz（米），按你实际工具长度填")
    p.add_argument("--joints", default=",".join(FR3_ARM_JOINTS),
                   help="要拟合的关节名（逗号分隔）")
    p.add_argument("--mode", choices=("reference", "per_placement"), default="reference",
                   help="reference=所有位姿共用 --spacing；per_placement=每个位姿用自己 manifest 的间距")
    p.add_argument("--spacing", type=float, default=None,
                   help="reference 模式下的夹具间距（米）；缺省从各 manifest 的 fixture.spacing_m 取（需一致）")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-4,
                   help="正则化权重")
    p.add_argument("--max-iter", type=int, default=30, help="IPOPT 最大迭代")
    p.add_argument("--noise-sigma", type=float, default=0.0,
                   help="只扰动正则化先验（拟合仍从无噪标称起步），与上游 variance_noise 一致")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    p.add_argument("--self-check", action="store_true",
                   help="不真拟合，跑模块自检 + 装配冒烟")
    return p.parse_args(argv)


def load_all_placements(data_dir: Path, mode: str, spacing_override: float | None) -> list[Placement]:
    """遍历采样根目录下每个子目录，装成一个 Placement。

    目录须含 manifest.yaml + hole_0.csv + hole_1.csv（见 load_placement）。
    """
    if not data_dir.is_dir():
        raise SystemExit(f"[mukca] 数据目录不存在: {data_dir}")
    dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not dirs:
        raise SystemExit(f"[mukca] {data_dir} 下没有插座位姿子目录")
    placements = [load_placement(d) for d in dirs]
    # reference 模式：所有位姿必须同一个间距
    if mode == "reference":
        d0 = spacing_override if spacing_override is not None else placements[0].spacing_m
        for pl in placements:
            if abs(pl.spacing_m - d0) > 1e-9:
                raise SystemExit(
                    f"[mukca] reference 模式要求所有位姿同一间距，但 {pl.key} 是 {pl.spacing_m}m "
                    f"!= {d0}m。用 --spacing 统一，或改 --mode per_placement")
        if spacing_override is not None:
            # 用命令行统一间距覆盖每个位姿（数据仍按各自 spacing 加载，仅拟合时统一）
            placements = [Placement(pl.key, pl.joints, pl.q0, pl.q1, float(spacing_override),
                                    pl.spacing_sigma_m) for pl in placements]
    return placements


def run(argv=None) -> int:
    args = parse_args(argv)

    if args.self_check:
        from arm_control.mukca import _self_check
        _self_check()
        print("[mukca] 模块自检通过（CSV/URDF/FK/objective/admission）。")
        # 装配冒烟：确保 fit() 能 import（依赖就绪）
        try:
            from arm_control.mukca_casadi import FitSpec  # noqa: F401
            print("[mukca] mukca_casadi 可导入（已装 casadi + forwardkinematics）。")
        except Exception as exc:
            print(f"[mukca] 警告: mukca_casadi 导入失败（跑真拟合需要 casadi/forwardkinematics）: {exc}")
        return 0

    if args.data is None:
        raise SystemExit("[mukca] 需要 --data <采样目录>（或加 --self-check 演练）")

    urdf_src = Path(args.urdf)
    if not urdf_src.exists():
        raise SystemExit(f"[mukca] URDF 不存在: {urdf_src}")
    urdf_xml = urdf_src.read_text()

    joints = tuple(j.strip() for j in args.joints.split(",") if j.strip())
    if not joints:
        raise SystemExit("[mukca] --joints 不能为空")

    # 1) 在目标臂下加测量球心 frame（只用于标定，不写进运行描述）
    ball_xyz = [float(v) for v in args.ball_xyz.split()]
    if len(ball_xyz) != 3:
        raise SystemExit("[mukca] --ball-xyz 需 3 个数（x y z 米）")
    urdf_fit = add_ball_joint(urdf_xml, args.parent_link, ball_xyz,
                              joint=FR3_BALL_JOINT, link=args.ball_link)

    # 2) 读全部插座位姿
    print(f"[mukca] 数据目录: {args.data}  mode={args.mode}")
    placements = load_all_placements(args.data, args.mode, args.spacing)
    for pl in placements:
        print(f"  placement {pl.key}: socket0={len(pl.q0)} socket1={len(pl.q1)} d={pl.spacing_m}m")

    # 3) 拟合
    from arm_control.mukca_casadi import FitSpec, fit, origins
    spec = FitSpec(
        urdf_xml=urdf_fit,
        root_link=args.root_link,
        ball_link=args.ball_link,
        selected_joints=list(joints),
        tool_joints=(FR3_BALL_JOINT,),
        lam=args.lam,
        max_iter=args.max_iter,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        mode=args.mode,
    )
    print(f"[mukca] 拟合 {len(joints)} 个关节的 origin（{len(spec.selected_joints)*6} 参数）...")
    result = fit(spec, placements)

    # 4) 校验
    errs = admission_errors(result)
    print(f"  solver: {result['return_status']}  iter={result['iter_count']}  success={result['solver_success']}")
    print(f"  J_before={result['J_before']:.6e}  J_after={result['J_after']:.6e}")
    if errs:
        print("[mukca] 结果不合格:")
        for e in errs:
            print(f"    - {e}")
        print("[mukca] 建议: 检查采样质量/数量、模式与间距、放大 max_iter，再重跑。")
        return 1

    # 5) 打印每个关节拟合前后的 origin，便于人工核对
    nominal = read_origins(urdf_xml, joints)
    fitted = origins(result)
    for j in joints:
        print(f"  {j}:")
        print(f"    标称 xyz={np.round(nominal[j][:3],6)}  rpy={np.round(nominal[j][3:],6)}")
        print(f"    拟合 xyz={np.round(fitted[j][:3],6)}  rpy={np.round(fitted[j][3:],6)}")

    # 6) 写回新 URDF（不覆盖原文件）
    out = Path(args.out) if args.out else Path(str(urdf_src) + ".calibrated.urdf")
    new_xml = rewrite_origins(urdf_xml, fitted)
    out.write_text(new_xml)
    print(f"[mukca] 已写回调校后 URDF -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
