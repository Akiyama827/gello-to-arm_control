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
    crc32_unitree,
)


def check_s288_units_and_codec() -> None:
    spec = S288Spec()
    # 输出 -> 转子 -> 输出 应闭合
    for q in (0.0, 0.5, -1.25):
        assert abs(float(spec.rotor_to_output_angle(spec.output_to_rotor_angle(q))) - q) < 1e-9
    # kp 按 r^2 缩放
    assert abs(float(spec.output_to_rotor_kp(1.0)) - 1.0 / spec.gear_ratio**2) < 1e-15
    # 定点换算往返闭合（官方 digital_servo 公式）
    for q in (0.0, 0.5, -1.25, 2.0):
        raw = int(round(float(spec.output_pos_to_raw(q))))
        assert abs(float(spec.raw_to_output_pos(raw)) - q) < 1e-4, q
    # CRC32 已知向量（由官方 digital_servo/python/servo_demo.py 算得）
    assert crc32_unitree(b"\x00\x00\x00\x00") == 0xC704DD7B
    assert crc32_unitree(bytes(range(16))) == 0x081B46CA
    # 空闲命令帧（id=1, mode=1, timeout=1, 全零）与官方逐字节一致
    codec = S288Codec()
    idle = codec.pack_command(1, S288Command(), spec)
    assert idle.hex() == "feee91000000000000000000000000009ce9c752", idle.hex()
    assert len(idle) == 20 and idle[0:2] == bytes((0xFE, 0xEE)) and idle[3] == 0
    # 命令帧的 mode_byte：低 4 位 id、[6:4] mode=1、最高位 timeout=1
    cmd = S288Command(q_out=0.5, dq_out=0.1, tau_out=0.2, kp_out=25.0, kd_out=1.0)
    frame = codec.pack_command(3, cmd, spec)
    assert frame[0:2] == bytes((0xFE, 0xEE))
    assert (frame[2] & 0x0F) == 3 and ((frame[2] >> 4) & 0x07) == 1 and (frame[2] >> 7) == 1
    # 伪造一条合法反馈帧并用 unpack 解析
    fb = codec.build_feedback_frame(
        3, spec, q_out=0.5, dq_out=0.1, tau_out=0.2, ex_pos_rad=0.3
    )
    st = codec.unpack_state(fb, spec)
    assert st.motor_id == 3 and abs(st.q_out - 0.5) < 1e-4, st
    # 速度原始值较粗（每 raw ≈ 0.0085 rad/s），容差放到 5e-3
    assert abs(st.dq_out - 0.1) < 5e-3 and abs(st.tau_out - 0.2) < 1e-3, st
    # 篡改一个字节 -> CRC 校验必须失败
    bad = bytearray(fb)
    bad[10] ^= 0xFF
    try:
        codec.unpack_state(bytes(bad), spec)
        raise AssertionError("篡改后的反馈帧应因 CRC 失败而报错")
    except ValueError:
        pass
    print("s288: 单位换算 / CRC32 / 官方帧编解码 OK")


def check_s288_leader_8motors() -> None:
    from arm_control.leader_follower.config import build_leader, config_from_dict

    cfg = config_from_dict(
        {
            "leader": {
                "kind": "s288", "n_arm_joints": 7, "with_gripper": True,
                "bus": "fake", "motor_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "gripper_index": 7, "gripper_open_rad": 0.8, "gripper_close_rad": 0.0,
                "joint_signs": [1, -1, -1, -1, 1, 1, 1, 1], "alpha": 1.0,
            },
            "follower": {"kind": "fake", "n_arm_joints": 6},
            "mapping": {
                "joints": [{"src": i} for i in range(6)],
                "gripper": {"src": 7},
            },
        }
    )
    leader = build_leader(cfg.leader)
    assert leader.num_dofs() == 8
    leader.chain.command_positions([0.1] * 7 + [0.4])
    time.sleep(0.3)
    state = leader.get_joint_state()
    assert state.shape == (8,), state
    assert abs(state[7] - 0.5) < 0.02, state  # 夹爪 0.4/0.8 -> 0.5

    # 电机数与布局不符要早报错
    bad = config_from_dict(
        {
            "leader": {
                "kind": "s288", "n_arm_joints": 6, "with_gripper": True,
                "bus": "fake", "motor_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            },
            "follower": {"kind": "fake", "n_arm_joints": 6},
            "mapping": {"joints": [{"src": i} for i in range(6)]},
        }
    )
    try:
        build_leader(bad.leader)
        raise AssertionError("电机数与 n_arm_joints 不符时应报错")
    except ValueError:
        pass
    print("s288: 8 电机(7 臂 + 夹爪)装配 / 夹爪归一化 / 布局校验 OK")


def check_s288_ex_pos() -> None:
    """ExPos 绝对单圈模式：read_arrays 三元组 + 圈内回绕后仍还原角度。"""
    from arm_control.leader_follower.config import build_leader, config_from_dict

    cfg = config_from_dict(
        {
            "leader": {
                "kind": "s288", "n_arm_joints": 7, "with_gripper": True,
                "bus": "fake", "motor_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "gripper_index": 7, "gripper_open_rad": 1.0, "gripper_close_rad": 0.0,
                "joint_offsets": [0.1] * 8, "joint_signs": [1.0] * 8,
                "alpha": 1.0, "use_ex_pos": True,
            },
        }
    )
    leader = build_leader(cfg.leader)
    pos, vel, ex = leader.chain.read_arrays()
    assert pos.shape == (8,) and vel.shape == (8,) and ex.shape == (8,)
    assert np.all(np.isfinite(ex)), ex

    def wrapped_expected(target: float) -> float:
        return (target - 0.1 + np.pi) % (2 * np.pi) - np.pi

    for target in (0.5, 3.0, -3.0):
        leader.chain.command_positions([target] * 7 + [1.0])
        time.sleep(0.35)
        st = leader.get_joint_state()
        assert abs(st[0] - wrapped_expected(target)) < 0.05, (target, st[0])
    print("s288: ExPos 绝对单圈 / read_arrays / 圈内回绕 OK")


def check_dora_follower_messages() -> None:
    """校验 DoraJogFollower 产出 jog/control/gripper 的格式（FR3 主路径）。"""
    from arm_control.leader_follower.follower import DoraJogFollower
    from arm_control.messages import (
        pack_control_update,
        unpack_control_update,
        unpack_jog,
        unpack_motor_command,
    )

    class FakeNode:
        def __init__(self) -> None:
            self.sent: list[tuple[str, object]] = []

        def send_output(self, name, value=None) -> None:
            self.sent.append((name, value))

    assert unpack_control_update(pack_control_update(arm=True)) == {"arm": True}

    node = FakeNode()
    follower = DoraJogFollower(num_arm_joints=7, node=node)
    follower.open()  # 发 control(arm=True) 使能
    follower.send([0.1] * 7, 0.03)  # 夹爪 3cm 单指
    follower.safe_stop()
    follower.close()

    topics = [name for name, _ in node.sent]
    assert topics[0] == "control", topics
    assert "jog" in topics and "gripper" in topics, topics

    jog = next(v for n, v in node.sent if n == "jog")
    assert np.allclose(unpack_jog(jog)["q"], [0.1] * 7)
    assert unpack_jog(jog)["reason"] == "leader-follower"

    gripper = next(v for n, v in node.sent if n == "gripper")
    pos = unpack_motor_command(gripper, 2)["position"]
    assert np.allclose(pos, [0.03, 0.03]), pos  # 单指位移，两指同值

    controls = [unpack_control_update(v) for n, v in node.sent if n == "control"]
    assert {"arm": True} in controls, controls
    assert any(c.get("cancel") for c in controls), controls
    print("dora: jog/control/gripper 消息格式（FR3，单指米）OK")


def check_dora_feedback() -> None:
    """校验 FR3 回读：motor_state/health/event/gripper_state -> 安全层。"""
    from arm_control.leader_follower.dora_feedback import DoraFollowerFeedback
    from arm_control.leader_follower.follower import DoraJogFollower
    from arm_control.messages import (
        pack_controller_event,
        pack_json_message,
        pack_motor_state,
    )

    class FakeNode:
        def __init__(self, events):
            self._events = list(events)
            self.sent = []

        def try_recv(self):
            return self._events.pop(0) if self._events else None

        def next(self, timeout=None):
            return self._events.pop(0) if self._events else None

        def send_output(self, name, value=None):
            self.sent.append((name, value))

    n = 7
    q = np.linspace(-1.0, 1.0, n)
    dq = np.full(n, 0.1)
    pos_cmd = q + 0.2
    zeros = np.zeros(n)
    motor_state = pack_motor_state(q, dq, pos_cmd, zeros, zeros, 100.0, 2.0, zeros)
    health = pack_json_message("motor_health", {"armed": True, "latched_fault": ""})
    grip = pack_json_message("gripper_state", {"width": 0.06, "is_grasped": False})
    event_ok = pack_controller_event(kind="ready", q=q)

    node = FakeNode(
        [
            {"type": "INPUT", "id": "motor_state", "value": motor_state},
            {"type": "INPUT", "id": "motor_health", "value": health},
            {"type": "INPUT", "id": "gripper_state", "value": grip},
            {"type": "INPUT", "id": "controller_event", "value": event_ok},
        ]
    )
    fb = DoraFollowerFeedback(node, n)
    sample = fb.sample()
    assert sample is not None, "有 motor_state 时应给出样本"
    assert np.allclose(sample.arm_q, q), sample.arm_q
    assert np.allclose(sample.arm_dq, dq), sample.arm_dq
    assert sample.armed is True and not sample.fault, sample
    assert abs(sample.gripper_width_m - 0.06) < 1e-9
    # 启动对齐取实测位形，夹爪给单指米
    arm, finger = fb.latest_arm_state()
    assert np.allclose(arm, q) and abs(finger - 0.03) < 1e-9

    # follower.read_state 注入 state_provider 后应返回实测，而非零点
    follower = DoraJogFollower(
        num_arm_joints=n, node=node, state_provider=fb.latest_arm_state
    )
    arm2, grip2 = follower.read_state()
    assert np.allclose(arm2, q), arm2
    assert abs(grip2 - 0.03) < 1e-9

    # 缓存样本不能骗过"反馈陈旧"门：时间前进 -> 用样本自带 timestamp 判停
    mon = SafetyMonitor(
        limits=SafetyLimits(feedback_timeout_s=0.2),
        joint_lower=[-1.5] * n,
        joint_upper=[1.5] * n,
    )
    now = time.monotonic()
    assert mon.check_feedback(sample, now), "刚到的样本应通过"
    assert not mon.check_feedback(sample, now + 0.25), "缓存样本应变陈旧并停机"

    # 控制器 fault 事件与失能都要能被安全层看到
    node_fault = FakeNode(
        [
            {"type": "INPUT", "id": "motor_state", "value": motor_state},
            {
                "type": "INPUT",
                "id": "controller_event",
                "value": pack_controller_event(kind="fault", ok=False, reason="rt fault"),
            },
        ]
    )
    fb_fault = DoraFollowerFeedback(node_fault, n)
    bad = fb_fault.sample()
    assert bad is not None and bad.fault and "rt fault" in bad.fault_reason, bad
    assert not mon.check_feedback(bad, time.monotonic()), "fault 样本应停机"

    node_disarm = FakeNode(
        [
            {"type": "INPUT", "id": "motor_state", "value": motor_state},
            {
                "type": "INPUT",
                "id": "motor_health",
                "value": pack_json_message("motor_health", {"armed": False}),
            },
        ]
    )
    fb_disarm = DoraFollowerFeedback(node_disarm, n)
    dis = fb_disarm.sample()
    assert dis is not None and dis.armed is False, dis
    assert not mon.check_feedback(dis, time.monotonic()), "失能样本应停机"

    # dora STOP -> 故障样本
    fb_stop = DoraFollowerFeedback(FakeNode([{"type": "STOP"}]), n)
    stopped = fb_stop.sample()
    assert stopped is not None and stopped.fault, stopped
    print("dora: 回读(motor_state/health/event/gripper) + 陈旧/故障/失能停机 OK")


def check_mapping() -> None:
    joints = [
        JointMapping(src_index=0, sign=1.0, scale=0.5, lower=-1.0, upper=1.0, max_rate=1.0),
        JointMapping(src_index=1, sign=-1.0, scale=1.0, lower=-2.0, upper=2.0),
    ]
    grip = GripperMapping(src_index=2, open_finger_m=0.08, closed_finger_m=0.0)
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
            "leader": {"kind": "fake", "n_arm_joints": 7, "with_gripper": True},
            "follower": {"kind": "fake", "n_arm_joints": 7},
            "mapping": {
                "auto_align": True,
                "alpha": 1.0,
                "joints": [
                    {"src": i, "scale": 0.6, "lower": -2.9, "upper": 2.9, "max_rate": 5.0}
                    for i in range(7)
                ],
                "gripper": {"src": -1, "open_finger_m": 0.04, "closed_finger_m": 0.0},
            },
            "safety": {
                "joint_lower": [-2.9] * 7,
                "joint_upper": [2.9] * 7,
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
    check_s288_leader_8motors()
    check_s288_ex_pos()
    check_dora_follower_messages()
    check_dora_feedback()
    check_mapping()
    check_safety_gates()
    check_collision()
    check_fake_loop()
    check_safety_stop_propagates()
    print("check_leader_follower: 全部通过")
