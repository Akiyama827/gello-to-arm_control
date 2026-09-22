#!/usr/bin/env python3
"""宇树 S288 小臂标定 CLI：读数、找零、定方向、标夹爪。

它直接读 ``S288JointChain`` 的原始量，打印每个电机的**多圈角度 q_out**、
**绝对单圈角 ExPos**、以及二者的一致性，帮你把 ``joint_offsets`` / ``joint_signs``
标定出来，并可一键写回 YAML（自动备份 ``.bak``）。

子命令
------
  read     打印每个电机的角度/速度/温度/电压/错误；``--watch`` 持续刷新
  zero     把当前姿态记为各关节零位 -> 输出 joint_offsets
  signs    交互式确定每个臂关节的正方向 -> 输出 joint_signs
  gripper  记录夹爪完全张开/闭合的角度 -> 输出 gripper_open_rad/gripper_close_rad

示例
----
  # 看实时读数，验证接线/方向（Ctrl-C 退出）
  PYTHONPATH=. python -B examples/s288_calibrate.py read --watch

  # 把当前姿态记为零位，并写回配置（自动备份 .bak）
  PYTHONPATH=. python -B examples/s288_calibrate.py zero --apply

  # 逐关节找零：一次只摆一个关节、回车记录，其余关节不动（推荐）
  PYTHONPATH=. python -B examples/s288_calibrate.py zero --per-joint --apply
  #   只重标某几个：--per-joint --only 1,4,7

  # 用绝对单圈编码器 ExPos 当角度源（上电即绝对角，不依赖多圈计数）
  PYTHONPATH=. python -B examples/s288_calibrate.py zero --from-ex --apply

  # 交互式定方向
  PYTHONPATH=. python -B examples/s288_calibrate.py signs --apply

  # 标夹爪
  PYTHONPATH=. python -B examples/s288_calibrate.py gripper --apply

  # 无硬件时用假总线演练（read/zero/--apply 都可跑）
  PYTHONPATH=. python -B examples/s288_calibrate.py read --bus fake
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_control.leader_follower.config import (  # noqa: E402
    build_leader,
    config_from_yaml,
)

DEFAULT_CONFIG = ROOT / "examples/configs/leader_follower_s288_sim.yaml"
TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def _fmt_list(vals, prec: int = 5) -> str:
    return "[" + ", ".join(f"{float(v):.{prec}f}" for v in vals) + "]"


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) + np.pi) % TWO_PI - np.pi


def _load(args):
    cfg = config_from_yaml(args.config)
    if args.ids:
        cfg.leader.motor_ids = [int(x) for x in args.ids.replace(" ", "").split(",") if x]
    if args.port:
        cfg.leader.port = args.port
    if args.bus == "fake":
        cfg.leader.use_fake_bus = True
    elif args.bus == "serial":
        cfg.leader.use_fake_bus = False
        cfg.leader.bus = "serial"
    leader = build_leader(cfg.leader)
    return cfg, leader


def _read_raw_ex(leader):
    """按当前 use_ex_pos 语义返回 ``(角度源, ExPos)``。"""
    raw, ex = leader.chain.read_positions_and_ex_positions()
    raw = np.asarray(raw, dtype=float)
    ex = np.asarray(ex, dtype=float)
    if leader.use_ex_pos:
        src = raw.copy()
        finite = np.isfinite(ex)
        src[finite] = ex[finite]
        return src, ex
    return raw, ex


def _offsets_signs(cfg, n):
    offs = np.asarray(cfg.leader.joint_offsets, dtype=float)
    signs = np.asarray(cfg.leader.joint_signs, dtype=float)
    if offs.shape != (n,):
        offs = np.zeros(n)
    if signs.shape != (n,):
        signs = np.ones(n)
    return offs, signs


def _apply_yaml(path, updates: dict[str, str]) -> None:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    for key, value in updates.items():
        pat = re.compile(rf"^(\s*){re.escape(key)}:([^#\n]*)(#[^\n]*)?$", re.M)

        def _repl(m):
            comment = f"  {m.group(3)}" if m.group(3) else ""
            return f"{m.group(1)}{key}: {value}{comment}"

        text, n = pat.subn(_repl, text)
        if n == 0:
            # 键不存在：插到 leader 段里的锚点之后（默认 joint_signs，其次 start_joints）
            text = _insert_after_anchor(text, key, value)
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy(path, backup)
    path.write_text(text, encoding="utf-8")
    print(f"[标定] 已写回 {path}（备份 {backup.name}）")


def _insert_after_anchor(text: str, key: str, value: str) -> str:
    for anchor in ("joint_signs", "start_joints", "joint_offsets"):
        pat = re.compile(rf"^(\s*){re.escape(anchor)}:.*$", re.M)
        m = pat.search(text)
        if m:
            indent = m.group(1)
            insert_at = m.end()
            return text[:insert_at] + f"\n{indent}{key}: {value}" + text[insert_at:]
    raise SystemExit(f"[标定] YAML 里找不到键 {key!r}，也没有可用的插入锚点")


def _emit(updates: dict[str, str], args) -> None:
    print("\n--- 可粘贴到 YAML 的片段 ---")
    for k, v in updates.items():
        print(f"  {k}: {v}")
    print("---------------------------\n")
    if args.apply:
        _apply_yaml(args.config, updates)
    else:
        print("[标定] 未写回（加 --apply 才会改 YAML）", flush=True)


def _prompt(args, msg: str) -> None:
    if args.yes:
        print(f"[标定] {msg}（--yes 跳过）", flush=True)
        return
    try:
        input(f"[标定] {msg}，然后回车…")
    except EOFError:
        print("\n[标定] 无 stdin，按 --yes 处理")


# --------------------------------------------------------------------------- #
# read
# --------------------------------------------------------------------------- #
def cmd_read(args) -> int:
    cfg, leader = _load(args)
    ids = list(leader.chain.motor_ids)
    n = len(ids)
    offs, signs = _offsets_signs(cfg, n)
    n_arm = int(cfg.leader.n_arm_joints)
    bus_name = "fake" if cfg.leader.use_fake_bus else cfg.leader.bus
    header = (
        f"{'#':>2} {'motor':>5} {'q_out':>10} {'ExPos':>9} {'wrap-mis':>9} "
        f"{'joint':>9} {'dq':>8} {'T(C)':>6} {'V':>6} {'err':>4}"
    )
    print(
        f"[标定] bus={bus_name}  ids={ids}  use_ex_pos={cfg.leader.use_ex_pos}  "
        f"gripper_index={cfg.leader.gripper_index}",
        flush=True,
    )
    print(
        "[标定] wrap-mis = (q_out mod 2π) − ExPos 折到 (-π,π]；≈0 表示多圈与绝对\n"
        "        单圈一致（多圈可信），明显非 0 说明多圈计数可能不可信。",
        flush=True,
    )
    try:
        while True:
            states = leader.chain.bus.read_states()
            raw = np.array(
                [states[mid].q_out if mid in states else np.nan for mid in ids],
                dtype=float,
            )
            ex = np.array(
                [states[mid].ex_pos_rad if mid in states else np.nan for mid in ids],
                dtype=float,
            )
            if cfg.leader.use_ex_pos:
                finite = np.isfinite(ex)
                src = raw.copy()
                src[finite] = ex[finite]
            else:
                src = raw
            if args.watch:
                sys.stdout.write("\033[H\033[J")
            print(f"\nt={time.strftime('%H:%M:%S')}  {header}")
            for i, mid in enumerate(ids):
                st = states.get(mid)
                if st is None:
                    print(f"{i:>2} {mid:>5}   <无反馈>")
                    continue
                q = float(st.q_out)
                exi = float(ex[i])
                mis = (
                    float(_wrap_pi(q - exi)) if np.isfinite(exi) else float("nan")
                )
                joint = float((src[i] - offs[i]) * signs[i])
                tag = " <-夹爪" if i == n_arm else ""
                print(
                    f"{i:>2} {mid:>5} {q:>10.4f} {exi:>9.4f} {mis:>9.4f} "
                    f"{joint:>9.4f} {float(st.dq_out):>8.3f} "
                    f"{float(st.temperature_c):>6.1f} {float(st.voltage_v):>6.2f} "
                    f"{int(st.error):>4}{tag}"
                )
            if not args.watch:
                return 0
            time.sleep(max(args.interval, 0.02))
    except KeyboardInterrupt:
        print("\n[标定] 退出 read", flush=True)
        return 0
    finally:
        leader.close()


# --------------------------------------------------------------------------- #
# zero
# --------------------------------------------------------------------------- #
def _parse_only(spec, n: int) -> list[int]:
    """把 ``--only 1,3,5`` 解析成 0 基下标；空表示全部。"""
    if not spec:
        return list(range(n))
    out: list[int] = []
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        j = int(tok)
        if not (1 <= j <= n):
            raise SystemExit(f"[标定] --only {j} 超出范围 1..{n}")
        out.append(j - 1)
    return out


def cmd_zero(args) -> int:
    cfg, leader = _load(args)
    ids = list(leader.chain.motor_ids)
    n = len(ids)
    prev_offsets, _ = _offsets_signs(cfg, n)
    selected = _parse_only(args.only, n)
    offsets = prev_offsets.copy()  # 未选中的关节保留原零位
    used_ex = False
    try:
        if args.per_joint:
            print(
                f"[标定] 逐关节找零：共 {len(selected)} 个关节，逐个摆到位后回车"
                "（未选中的保留原零位）。",
                flush=True,
            )
            for i in selected:
                tag = "夹爪" if i == cfg.leader.gripper_index else f"J{i + 1}"
                _prompt(args, f"把 {tag}（motor {ids[i]}）摆到零位")
                raw, ex = _read_raw_ex(leader)
                if args.from_ex and np.isfinite(ex[i]):
                    offsets[i] = ex[i]
                    used_ex = True
                    src = "ExPos"
                else:
                    offsets[i] = raw[i]
                    src = "q_out"
                print(f"  {tag}: offset = {offsets[i]:+.5f}  ({src})", flush=True)
        else:
            _prompt(args, "把机械臂摆到**全部关节的零位**")
            raw, ex = _read_raw_ex(leader)
            if args.from_ex and not np.any(np.isfinite(ex)):
                print("[标定] 反馈里没有 ExPos，无法用 --from-ex", file=sys.stderr, flush=True)
                return 2
            for i in selected:
                if args.from_ex and np.isfinite(ex[i]):
                    offsets[i] = ex[i]
                    used_ex = True
                else:
                    offsets[i] = raw[i]
        use_ex = bool(args.from_ex and used_ex)
        mode = "ExPos 绝对零位" if use_ex else "多圈 q_out 零位"
        print(f"[标定] 已记录{mode}（use_ex_pos: {str(use_ex).lower()}）。", flush=True)
        updates = {
            "joint_offsets": _fmt_list(offsets),
            "use_ex_pos": "true" if use_ex else "false",
        }
        _emit(updates, args)
        return 0
    finally:
        leader.close()


# --------------------------------------------------------------------------- #
# signs
# --------------------------------------------------------------------------- #
def cmd_signs(args) -> int:
    cfg, leader = _load(args)
    ids = list(leader.chain.motor_ids)
    n = len(ids)
    n_arm = int(cfg.leader.n_arm_joints)
    _, signs = _offsets_signs(cfg, n)
    new_signs = signs.copy()
    print(
        "[标定] 逐关节：把该关节朝**你希望 joint 值增大的方向**缓慢移动，然后回车。",
        flush=True,
    )
    try:
        for i in range(n_arm):
            before, _ = _read_raw_ex(leader)
            _prompt(args, f"把 J{i + 1}（motor {ids[i]}）朝关节角增大的方向移动")
            time.sleep(args.interval)
            after, _ = _read_raw_ex(leader)
            delta = float(after[i] - before[i])
            if abs(delta) < args.min_delta:
                print(
                    f"  J{i + 1}: 位移太小（{delta:+.4f} rad），保持 {new_signs[i]:+.0f}",
                    flush=True,
                )
            else:
                new_signs[i] = 1.0 if delta > 0 else -1.0
                print(
                    f"  J{i + 1}: Δraw={delta:+.4f} rad -> sign={new_signs[i]:+.0f}",
                    flush=True,
                )
        # 夹爪方向：约定 sign=+1，张开角 > 闭合角（gripper 标定再定绝对值）
        print("[标定] 夹爪方向固定 +1（张开角 > 闭合角，由 gripper 子命令标定）。", flush=True)
        updates = {"joint_signs": _fmt_list(new_signs, prec=0)}
        _emit(updates, args)
        return 0
    finally:
        leader.close()


# --------------------------------------------------------------------------- #
# gripper
# --------------------------------------------------------------------------- #
def _gripper_calibrated(leader, cfg, n) -> float:
    offs, signs = _offsets_signs(cfg, n)
    raw, _ = _read_raw_ex(leader)
    gi = cfg.leader.gripper_index
    if gi is None:
        gi = n - 1
    return float((raw[gi] - offs[gi]) * signs[gi])


def cmd_gripper(args) -> int:
    cfg, leader = _load(args)
    ids = list(leader.chain.motor_ids)
    n = len(ids)
    gi = cfg.leader.gripper_index if cfg.leader.gripper_index is not None else n - 1
    print(f"[标定] 夹爪是第 {gi} 个（motor {ids[gi]}），1=张开 0=闭合。", flush=True)
    try:
        _prompt(args, "把夹爪**完全张开**")
        open_rad = _gripper_calibrated(leader, cfg, n)
        _prompt(args, "把夹爪**完全闭合**")
        close_rad = _gripper_calibrated(leader, cfg, n)
        if open_rad <= close_rad:
            print(
                f"[标定] 警告：张开角 {open_rad:.4f} <= 闭合角 {close_rad:.4f}，"
                "请检查 joint_signs 或夹爪接线方向。",
                flush=True,
            )
        print(f"[标定] 张开={open_rad:.4f} rad  闭合={close_rad:.4f} rad", flush=True)
        updates = {
            "gripper_open_rad": f"{open_rad:.5f}",
            "gripper_close_rad": f"{close_rad:.5f}",
        }
        _emit(updates, args)
        return 0
    finally:
        leader.close()


# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    def default(value):
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--config", default=default(str(DEFAULT_CONFIG)),
                        help="YAML 配置路径")
    parser.add_argument("--bus", choices=("config", "serial", "fake"),
                        default=default("config"),
                        help="总线来源：config=按 YAML（默认）；serial/fake=覆盖")
    parser.add_argument("--port", default=default(None), help="覆盖串口路径")
    parser.add_argument("--ids", default=default(None),
                        help="覆盖电机 ID，如 1,2,3,4,5,6,7,8")
    parser.add_argument("--apply", action="store_true", default=default(False),
                        help="把结果写回 YAML（自动备份 .bak）")
    parser.add_argument("--yes", action="store_true", default=default(False),
                        help="跳过交互提示（自动化用）")
    parser.add_argument("--interval", type=float, default=default(0.4),
                        help="动作间隔/采样间隔（秒）")
    parser.add_argument("--min-delta", type=float, default=default(1e-3),
                        help="判定运动的最小 Δraw（rad）")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="宇树 S288 小臂标定 CLI（读原始角 / 找零 / 定方向 / 标夹爪）"
    )
    _add_common(parser, suppress=False)

    # 让常用选项也能放在子命令之后（子命令里的同名项默认 SUPPRESS，不覆盖主解析器）
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common, suppress=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_read = sub.add_parser("read", parents=[common], help="打印实时读数")
    p_read.add_argument("--watch", action="store_true", help="持续刷新（Ctrl-C 退出）")
    p_read.set_defaults(func=cmd_read)

    p_zero = sub.add_parser("zero", parents=[common], help="记录零位 -> joint_offsets")
    p_zero.add_argument("--from-ex", action="store_true",
                        help="用绝对单圈 ExPos 作零位（同时置 use_ex_pos: true）")
    p_zero.add_argument("--per-joint", action="store_true",
                        help="逐个关节找零：一个摆好回车，其余关节不动（推荐）")
    p_zero.add_argument("--only", default=None,
                        help="只标这些关节（1 基、逗号分隔，如 1,3,5）")
    p_zero.set_defaults(func=cmd_zero)

    p_signs = sub.add_parser("signs", parents=[common], help="交互式确定关节正方向 -> joint_signs")
    p_signs.set_defaults(func=cmd_signs)

    p_grip = sub.add_parser("gripper", parents=[common], help="记录夹爪张开/闭合角")
    p_grip.set_defaults(func=cmd_gripper)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
