#!/usr/bin/env python3
"""S288 单电机探针：这台电机能不能被**主动命令**驱动？能不能被**手动反驱**？

背景（现场）：第 8 个 S288（夹爪，ID8）在全行程扳动时 `q_out`/`ExPos` 几乎不变
（6 秒采样只有 0.0001 rad），且 `err=256` 重上电不消。需要区分：

  (a) 电机 / 编码器 / 驱动硬件坏；
  (b) 电机被固件"抱住"（零刚度命令下也不让反驱）；
  (c) 扳机根本没有耦合到这台电机（"扳机逻辑和别的电机不一样"）。

本探针不改配置、不写文件，分两阶段：

  阶段 1「自由反驱」：只发零刚度零阻尼帧，请**手动**全行程来回扳动扳机，
                      统计 `q_out`/`ExPos`/`dq` 的范围。
                      - 范围明显 -> 电机能被反驱（扳机确实耦合到它）；
                      - 范围 ≈ 0  -> 零刚度下不动（被抱住 / 未耦合 / 坏）。
  阶段 2「主动驱动」：给该电机发小幅慢速**位置正弦**（kp>0），看 `q_out` 是否跟随。
                      - 跟随 -> 电机 + 编码器 + 协议都正常；
                      - 不跟随 -> 电机 / 驱动侧有问题。

两阶段合起来即可判定问题在"输入（反驱/耦合）"还是"输出（电机/驱动）"。

运行（fish）：
    source .venv/bin/activate.fish
    PYTHONPATH=. python -B examples/s288_gripper_probe.py \
        --config examples/configs/leader_follower_s288.local.yaml --id 8

无硬件演练：加 `--bus fake`（读数恒定，仅走流程）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_control.leader_follower.config import config_from_yaml  # noqa: E402
from arm_control.leader_follower.s288 import (  # noqa: E402
    FakeS288Bus,
    S288Codec,
    S288Command,
    S288Spec,
    SerialS288Bus,
)


def _make_bus(args, port: str, mid: int, spec: S288Spec):
    if args.bus == "fake":
        return FakeS288Bus([mid], spec=spec)
    return SerialS288Bus([mid], port=port, spec=spec, codec=S288Codec())


def _state_of(bus, mid: int):
    return bus.read_states()[mid]


def _fmt(st) -> str:
    return (
        f"q_out={st.q_out:+.4f}  ExPos={st.ex_pos_rad:+.4f}  dq={st.dq_out:+.4f}  "
        f"tau={st.tau_out:+.4f}  T={st.temperature_c:.0f}C  V={st.voltage_v:.1f}  "
        f"err={st.error}  warn={st.warn}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S288 单电机探针（反驱 + 主动驱动）")
    parser.add_argument("--config", default=str(ROOT / "examples/configs/leader_follower_s288_sim.yaml"))
    parser.add_argument("--id", type=int, default=8, help="被测电机 ID（默认 8=夹爪）")
    parser.add_argument("--port", default=None, help="覆盖配置里的串口")
    parser.add_argument("--bus", choices=("serial", "fake"), default="serial")
    parser.add_argument("--free-s", type=float, default=5.0, help="阶段1（自由反驱）时长")
    parser.add_argument("--drive-s", type=float, default=8.0, help="阶段2（主动驱动）时长")
    parser.add_argument("--amp", type=float, default=0.30, help="驱动正弦幅度（rad，输出端）")
    parser.add_argument("--period", type=float, default=6.0, help="驱动正弦周期（秒）")
    parser.add_argument("--kp", type=float, default=0.5, help="输出端位置刚度（保守值）")
    parser.add_argument("--kd", type=float, default=0.05, help="输出端阻尼")
    parser.add_argument("--yes", action="store_true", help="不等待回车，直接开始")
    args = parser.parse_args(argv)

    cfg = config_from_yaml(args.config)
    port = args.port or cfg.leader.port
    spec = S288Spec()
    mid = int(args.id)

    print(f"[probe] bus={args.bus}  port={port}  motor_id={mid}", flush=True)
    bus = _make_bus(args, port, mid, spec)
    try:
        st = _state_of(bus, mid)
        print(f"[probe] 初始状态：{_fmt(st)}", flush=True)
        if st.error:
            print(f"[probe] 注意：err=0x{st.error:X}（非 0，可能是历史告警或故障锁存）", flush=True)

        # --- 阶段 1：自由反驱 -------------------------------------------------
        print(
            f"\n[probe] 阶段1：自由反驱（零刚度）。请在 {args.free_s:.0f}s 内"
            "全行程来回扳动扳机/夹爪。",
            flush=True,
        )
        if not args.yes:
            try:
                input("[probe] 准备好后回车开始…")
            except EOFError:
                pass
        lo = hi = None
        ex_lo = ex_hi = None
        dq_abs = 0.0
        t0 = time.time()
        bus.write_commands({mid: S288Command()})  # 确保零刚度
        while time.time() - t0 < args.free_s:
            st = _state_of(bus, mid)
            lo = st.q_out if lo is None else min(lo, st.q_out)
            hi = st.q_out if hi is None else max(hi, st.q_out)
            ex_lo = st.ex_pos_rad if ex_lo is None else min(ex_lo, st.ex_pos_rad)
            ex_hi = st.ex_pos_rad if ex_hi is None else max(ex_hi, st.ex_pos_rad)
            dq_abs = max(dq_abs, abs(st.dq_out))
            time.sleep(0.02)
        span = (hi - lo) if lo is not None else 0.0
        ex_span = (ex_hi - ex_lo) if ex_lo is not None else 0.0
        print(
            f"[probe] 阶段1结果：q_out 范围 [{lo:+.4f}, {hi:+.4f}]（span={span:.4f} rad，"
            f"max|dq|={dq_abs:.4f} rad/s）；ExPos span={ex_span:.4f} rad",
            flush=True,
        )
        if span < 0.02 and ex_span < 0.02:
            print(
                "[probe]  -> 零刚度下几乎不动：电机没被反驱（被抱住 / 未耦合 / 坏）。",
                flush=True,
            )
        else:
            print("[probe]  -> 电机能被反驱：扳机/机械确实耦合到这台电机。", flush=True)

        # --- 阶段 2：主动驱动 -------------------------------------------------
        st = _state_of(bus, mid)
        center = st.q_out
        print(
            f"\n[probe] 阶段2：主动驱动 center={center:+.4f} ± {args.amp:.3f} rad，"
            f"周期 {args.period:.0f}s，kp={args.kp} kd={args.kd}，共 {args.drive_s:.0f}s。",
            flush=True,
        )
        print("[probe] 电机应带动扳机小幅慢速摆动；若被卡住会顶住（时间短、力小）。", flush=True)
        t0 = time.time()
        span_cmd = 2 * args.amp
        q_lo = q_hi = None
        ex_lo = ex_hi = None
        tick = 0
        while time.time() - t0 < args.drive_s:
            t = time.time() - t0
            q_des = center + args.amp * np.sin(2 * np.pi * t / args.period)
            bus.write_commands(
                {mid: S288Command(q_out=float(q_des), kp_out=args.kp, kd_out=args.kd)}
            )
            st = _state_of(bus, mid)
            q_lo = st.q_out if q_lo is None else min(q_lo, st.q_out)
            q_hi = st.q_out if q_hi is None else max(q_hi, st.q_out)
            if np.isfinite(st.ex_pos_rad):
                ex_lo = st.ex_pos_rad if ex_lo is None else min(ex_lo, st.ex_pos_rad)
                ex_hi = st.ex_pos_rad if ex_hi is None else max(ex_hi, st.ex_pos_rad)
            tick += 1
            if tick % 10 == 0:
                print(
                    f"    t={t:4.1f}s  目标={q_des:+.4f}  q_out={st.q_out:+.4f}  "
                    f"ExPos={st.ex_pos_rad:+.4f}  err=0x{st.error:X}",
                    flush=True,
                )
            time.sleep(0.02)
        q_span = (q_hi - q_lo) if q_lo is not None else 0.0
        ex_span = (ex_hi - ex_lo) if ex_lo is not None else float("nan")
        print(
            f"[probe] 阶段2结果：q_out span={q_span:.4f} rad；ExPos span={ex_span:.4f} rad"
            f"（命令 span≈{span_cmd:.3f} rad）",
            flush=True,
        )
        # 用 ExPos 判定：q_out 可能正是坏掉的那一路，不能拿它判断电机能不能转。
        follows = np.isfinite(ex_span) and ex_span > 0.4 * span_cmd
        if follows:
            print(
                "[probe]  -> 电机在主动转（ExPos 跟随命令）：驱动/FOC 正常，"
                "坏的只是转子多圈 q_out 反馈。",
                flush=True,
            )
        elif q_span < 0.05 and (not np.isfinite(ex_span) or ex_span < 0.05):
            print(
                "[probe]  -> 电机不跟随（q_out 与 ExPos 都不动）：电机/驱动侧有问题，"
                "或此 ID 没接电机。",
                flush=True,
            )
        else:
            print(
                "[probe]  -> 部分跟随/受阻：可能有机械阻力/限位/固件告警，结合 err 判断。",
                flush=True,
            )
    finally:
        try:
            bus.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
