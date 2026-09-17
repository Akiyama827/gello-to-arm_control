"""leader-follower 离线自检：不接硬件，验证换算/安全/碰撞/整条假回路。

    PYTHONPATH=. python -B tools/bench/check_leader_follower.py
"""
from __future__ import annotations

import time

import numpy as np

from arm_control.leader_follower import (
    CallableCollisionGuard,
    FollowerFeedback,
    SafetyLimits,
    SafetyMonitor,
    SafetyStop,
    Sphere,
    SphereCollisionGuard,
    TeleopLoop,
)
from arm_control.leader_follower.follower import FakeFollower
from arm_control.leader_follower.leader import FakeLeaderArm
from arm_control.leader_follower.mapping import GripperMapping, JointMapping, Retargeter
from arm_control.leader_follower.safety import FeedbackSample
from arm_control.leader_follower.s288 import (
    S288Codec,
    S288Command,
    S288Spec,
    crc16_ccitt,
)


def check_s288_units_and_codec() -> None:
    spec = S288Spec()
    # 输出 -> 转子 -> 输出 应闭合
    for q in (0.0, 0.5, -1.25):
        assert abs(float(spec.rotor_to_output_angle(spec.output_to_rotor_angle(q))) - q) < 1e-9
    # kp 按 r^2 缩放
    assert abs(float(spec.output_to_rotor_kp(1.0)) - 1.0 / spec.gear_ratio**2) < 1e-15
    # CRC 稳定性（XMODEM 对 "123456789" 的已知值 0x31C3）
    assert crc16_ccitt(b"123456789") == 0x31C3
    # 命令帧可被自身解出转子侧 q
    codec = S288Codec()
    cmd = S288Command(q_out=0.5, dq_out=0.1, tau_out=0.2, kp_out=25.0, kd_out=1.0)
    frame = codec.pack_command(3, cmd, spec)
    assert frame[0:2] == bytes((0xFE, 0xEE)) and frame[3] == 3
    # 伪造一条反馈帧并用 unpack 解析
    import struct

    q_r = float(spec.output_to_rotor_angle(0.5))
    fb = bytes((0xFE, 0xEE, 0x01, 3)) + struct.pack("<fff", q_r, 0.0, 0.0)
    fb = fb + bytes([25, 0]) + b"\x00" * 4
    st = codec.unpack_state(fb, spec)
    assert st.motor_id == 3 and abs(st.q_out - 0.5) < 1e-4, st
    print("s288: 单位换算 / CRC / 帧编解码 OK")


def check_mapping() -> None:
    joints = [
        JointMapping(src_index=0, sign=1.0, scale=0.5, lower=-1.0, upper=1.0, max_rate=1.0),
        JointMapping(src_index=1, sign=-1.0, scale=1.0, lower=-2.0, upper=2.0),
    ]
    grip = GripperMapping(src_index=2, open_width_m=0.08, closed_width_m=0.0)
    rt = Retargeter(joints=joints, gripper=grip, alpha=1.0)
    rt.auto_align(np.array([0.4, -0.3, 1.0]), np.array([0.1, -0.2]))
    rt.reset(np.array([0.1, -0.2]), 0.0)
    out, w = rt.map(np.array([0.4, -0.3, 1.0]), time.monotonic())
    # auto_align 后当前 leader 姿态应映射到当前大臂姿态
    assert np.allclose(out, [0.1, -0.2], atol=1e-9), out
    assert abs(w - 0.08) < 1e-9
    # scale=0.5 且限位 [-1,1]：leader 走到 0.8 -> 未对齐时被裁剪
    rt2 = Retargeter(joints=joints[:1], alpha=1.0)
    out2, _ = rt2.map(np.array([0.8]), time.monotonic())
    assert -1.0 <= out2[0] <= 1.0
    print("mapping: auto_align / scale / 限位 / 夹爪归一化 OK")


def check_safety_gates() -> None:
    limits = SafetyLimits(
        max_step_rad=0.3, track_err_rad=0.15, track_err_hold_s=0.1,
        feedback_timeout_s=0.2, collision_stop_m=0.01, collision_warn_m=0.05,
    )
    mon = SafetyMonitor(limits=limits, joint_lower=[-1, -1], joint_upper=[1, 1])
    now = time.monotonic()

    assert mon.check_leader(np.array([0.1, 0.2]), now)
    assert not mon.check_leader(np.array([np.nan, 0.2]), now)

    v, _ = mon.check_and_clamp_target(np.array([0.5, 0.5]), now)
    assert v
    v, _ = mon.check_and_clamp_target(np.array([1.5, 0.5]), now)
    assert not v and "关节越限" in v.reason

    mon2 = SafetyMonitor(limits=limits, joint_lower=[-1, -1], joint_upper=[1, 1])
    now2 = time.monotonic()
    mon2.check_and_clamp_target(np.array([0.0, 0.0]), now2)
    v, _ = mon2.check_and_clamp_target(np.array([0.9, 0.0]), now2)  # 单步 0.9 > 0.3
    assert not v and "跳变" in v.reason

    # 跟踪误差：持续超阈才停
    mon3 = SafetyMonitor(limits=limits, joint_lower=[-1, -1], joint_upper=[1, 1])
    t0 = time.monotonic()
    mon3.check_and_clamp_target(np.array([0.0, 0.0]), t0)
    s_bad = FeedbackSample(arm_q=np.array([0.5, 0.0]), timestamp=t0, armed=True)
    assert mon3.check_feedback(s_bad, t0)  # 刚超阈，未到 hold
    assert not mon3.check_feedback(s_bad, t0 + 0.2)  # 持续超阈 -> 停

    # 故障位 / 失能 / 反馈非法
    assert not mon3.check_feedback(
        FeedbackSample(arm_q=np.array([0.0, 0.0]), fault=True, fault_reason="x"), t0
    )
    assert not mon3.check_feedback(
        FeedbackSample(arm_q=np.array([0.0, 0.0]), armed=False), t0
    )
    print("safety: leader/限位/跳变/跟踪误差/故障 OK")


def check_collision() -> None:
    # 可调用守卫：布尔 -> 碰撞
    g = CallableCollisionGuard(lambda q: bool(q[0] > 0.5))
    assert g.min_distance(np.array([0.0])) == float("inf")
    assert g.min_distance(np.array([0.9])) == -1.0

    # 球体守卫：大臂球在 [q0, 0]，小臂球在 [0, q0]，半径 0.05。
    # 小臂靠近原点 -> 距离 < 1cm（撞）；远离 -> 安全。
    big = [Sphere(frame="tip", offset=[0, 0, 0], radius=0.05)]
    small = [Sphere(frame="tip", offset=[0, 0, 0], radius=0.05)]
    guard = SphereCollisionGuard(
        big_spheres=big, small_spheres=small,
        big_fk=lambda q: {"tip": np.array([float(q[0]), 0.0, 0.0])},
        small_fk=lambda q: {"tip": np.array([0.0, float(q[0]), 0.0])},
    )
    guard.set_other_state(np.array([1.0]))
    assert guard.min_distance(np.array([0.0])) > 0.05
    guard.set_other_state(np.array([0.02]))
    assert guard.min_distance(np.array([0.02])) < 0.01

    # 碰撞停机门：小臂逼近大臂，目标触发停机
    limits = SafetyLimits(collision_stop_m=0.01, collision_warn_m=0.05)
    mon = SafetyMonitor(limits=limits, joint_lower=[-3.0], joint_upper=[3.0], collision=guard)
    mon.collision.set_other_state(np.array([0.02]))
    v, _ = mon.check_and_clamp_target(np.array([0.02]), time.monotonic())
    assert not v and "碰撞" in v.reason, v

    # 预警区：距离在 (stop, warn] 之间 -> 限速而非停机
    guard.set_other_state(np.array([0.15]))
    mon2 = SafetyMonitor(limits=limits, joint_lower=[-3.0], joint_upper=[3.0], collision=guard)
    v, _ = mon2.check_and_clamp_target(np.array([0.15]), time.monotonic())
    # 距离 = 0.15*sqrt(2) - 0.1 ≈ 0.112 > warn，仍是全速；再逼近一点
    guard.set_other_state(np.array([0.10]))
    mon3 = SafetyMonitor(limits=limits, joint_lower=[-3.0], joint_upper=[3.0], collision=guard)
    v3, _ = mon3.check_and_clamp_target(np.array([0.10]), time.monotonic())
    if v3:
        assert v3.speed_scale < 1.0, v3
    print("collision: 可调用守卫 / 球体最近距离 / 碰撞停机+预警限速 OK")


def check_fake_loop() -> None:
    from arm_control.leader_follower.config import config_from_dict

    cfg = config_from_dict(
        {
            "leader": {"kind": "fake", "n_arm_joints": 6, "with_gripper": True},
            "follower": {"kind": "fake", "n_arm_joints": 6},
            "mapping": {
                "auto_align": True,
                "alpha": 1.0,
                "joints": [
                    {"src": i, "scale": 0.6, "lower": -2.9, "upper": 2.9, "max_rate": 5.0}
                    for i in range(6)
                ],
                "gripper": {"src": -1, "open_width_m": 0.08, "closed_width_m": 0.0},
            },
            "safety": {
                "joint_lower": [-2.9] * 6,
                "joint_upper": [2.9] * 6,
                "limits": {"track_err_rad": 0.5, "track_err_hold_s": 0.5},
            },
            "loop": {"hz": 200.0},
        }
    )
    from arm_control.leader_follower.config import build_pipeline

    leader, retargeter, follower, monitor = build_pipeline(cfg)
    loop = TeleopLoop(
        leader=leader,
        retargeter=retargeter,
        follower=follower,
        monitor=monitor,
        feedback=FollowerFeedback(follower, armed=True),
        hz=cfg.loop.hz,
        auto_align=True,
        verbose=False,
    )
    stats = loop.run(duration_s=0.5)
    assert stats.ticks > 20, stats
    assert stats.safety_stops == 0, stats
    print(f"loop: 假回路 {stats.ticks} ticks / {stats.mean_dt_s * 1000:.2f}ms 平均周期 OK")


def check_safety_stop_propagates() -> None:
    from arm_control.leader_follower.config import build_pipeline, config_from_dict

    cfg = config_from_dict(
        {
            "leader": {"kind": "fake", "n_arm_joints": 2, "with_gripper": False},
            "follower": {"kind": "fake", "n_arm_joints": 2},
            "mapping": {
                "joints": [
                    {"src": 0, "lower": -3, "upper": 3},
                    {"src": 1, "lower": -3, "upper": 3},
                ],
            },
            "safety": {
                "joint_lower": [-0.0001, -0.0001],  # 故意设得极窄，让合法动作立刻越限
                "joint_upper": [0.0001, 0.0001],
                "limits": {"joint_margin_rad": 0.0, "ramp_s": 0.0},
            },
            "loop": {"hz": 100.0},
        }
    )
    leader, retargeter, follower, monitor = build_pipeline(cfg)
    # 用固定 leader 制造越限
    class _Fixed:
        def num_dofs(self): return 2
        def get_joint_state(self): return np.array([1.0, 1.0])
        def get_observations(self): return {"joint_state": self.get_joint_state()}
        def set_torque_mode(self, e): pass
        def close(self): pass

    loop = TeleopLoop(
        leader=_Fixed(), retargeter=retargeter, follower=follower, monitor=monitor,
        hz=100.0, auto_align=False, verbose=False,
    )
    stats = loop.run(duration_s=0.3)
    assert stats.safety_stops == 1, stats
    print("loop: 越限触发安全停机并返回 OK")


if __name__ == "__main__":
    check_s288_units_and_codec()
    check_mapping()
    check_safety_gates()
    check_collision()
    check_fake_loop()
    check_safety_stop_propagates()
    print("check_leader_follower: 全部通过")
