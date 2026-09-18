"""大机械臂（follower / 被控设备）抽象与三种下发后端。

arm_control 里所有运动都走 Dora（`plan` / `jog` / `control` / `gripper`），
所以默认后端是 `DoraJogFollower`：每 tick 发一条 `jog`（单帧关节目标，
0.2s 不过期即停，天然安全）+ `gripper`。另提供：

    DryRunFollower   不接硬件，只记录/打印（默认，用来先验证映射）
    DoraJogFollower  通过 Dora 节点把 jog + gripper 发给 arm_controller /
                     franka_gripper（FR3 就是这个接法）
    RtFollower       绕开 Dora，直连远程 RT 服务器（RtBackend.apply_command）；
                     只驱动 FR3 的 7 个臂关节，夹爪不走这条通道
    FakeFollower     仿真大臂，用于回路自检

统一接口：`send(arm_q_rad, gripper_finger_m)`；`read_state()` 返回大臂当前
(arm_q, gripper_finger_m)，供启动 auto_align 使用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import numpy as np


class FollowerArm(Protocol):
    num_arm_joints: int

    def open(self) -> None:
        ...

    def read_state(self) -> tuple[np.ndarray, float]:
        ...

    def send(self, arm_q: Sequence[float], gripper_finger_m: float) -> None:
        ...

    def safe_stop(self) -> None:
        ...

    def close(self) -> None:
        ...


# --------------------------------------------------------------------------- #
# 空跑
# --------------------------------------------------------------------------- #
class DryRunFollower:
    """只记录目标、不碰硬件；打印限频。默认后端。"""

    def __init__(
        self,
        num_arm_joints: int = 7,
        log_period_s: float = 1.0,
        verbose: bool = True,
        initial_q: Optional[Sequence[float]] = None,
        initial_finger_m: float = 0.0,
    ) -> None:
        self.num_arm_joints = int(num_arm_joints)
        self._log_period_s = float(log_period_s)
        self._verbose = bool(verbose)
        self._last_log = 0.0
        self._last_cmd: Optional[tuple[np.ndarray, float]] = None
        self._sent = 0
        self._q0 = (
            np.zeros(self.num_arm_joints)
            if initial_q is None
            else np.asarray(initial_q, dtype=float)
        )
        self._grip0 = float(initial_finger_m)

    def open(self) -> None:
        if self._verbose:
            print("[follower:dry-run] 已就绪（不驱动任何硬件）")

    def read_state(self) -> tuple[np.ndarray, float]:
        return self._q0.copy(), self._grip0

    def send(self, arm_q: Sequence[float], gripper_finger_m: float) -> None:
        q = np.asarray(arm_q, dtype=float)
        if q.shape != (self.num_arm_joints,):
            raise ValueError(f"大臂目标长度应为 {self.num_arm_joints}，得到 {q.shape}")
        self._last_cmd = (q, float(gripper_finger_m))
        self._sent += 1
        now = time.monotonic()
        if self._verbose and now - self._last_log >= self._log_period_s:
            self._last_log = now
            qs = ", ".join(f"{v:+.3f}" for v in q)
            print(
                f"[follower:dry-run] #{self._sent:6d}  q=[{qs}]  "
                f"gripper={gripper_finger_m * 1000:.1f}mm"
            )

    def safe_stop(self) -> None:
        self._last_cmd = None

    def close(self) -> None:
        self.safe_stop()


# --------------------------------------------------------------------------- #
# 仿真
# --------------------------------------------------------------------------- #
class FakeFollower:
    """一阶跟踪的假大臂，读回自身位形，用于端到端自检。"""

    def __init__(
        self,
        num_arm_joints: int = 7,
        gripper_open_finger_m: float = 0.04,
        time_constant_s: float = 0.08,
        initial_q: Optional[Sequence[float]] = None,
        initial_finger_m: float = 0.0,
    ) -> None:
        self.num_arm_joints = int(num_arm_joints)
        self.gripper_open_finger_m = float(gripper_open_finger_m)
        self._q = (
            np.zeros(self.num_arm_joints)
            if initial_q is None
            else np.asarray(initial_q, dtype=float).copy()
        )
        self._q_cmd = self._q.copy()
        self._grip = float(initial_finger_m)
        self._grip_cmd = float(initial_finger_m)
        self._tau = max(float(time_constant_s), 1e-3)
        self._last_t = time.monotonic()

    def open(self) -> None:
        pass

    def _step(self) -> None:
        now = time.monotonic()
        dt = max(now - self._last_t, 0.0)
        self._last_t = now
        a = min(dt / self._tau, 1.0) if dt > 0 else 0.0
        self._q = self._q + (self._q_cmd - self._q) * a
        self._grip = self._grip + (self._grip_cmd - self._grip) * a

    def read_state(self) -> tuple[np.ndarray, float]:
        self._step()
        return self._q.copy(), float(self._grip)

    def send(self, arm_q: Sequence[float], gripper_finger_m: float) -> None:
        self._step()
        q = np.asarray(arm_q, dtype=float)
        if q.shape != (self.num_arm_joints,):
            raise ValueError(f"大臂目标长度应为 {self.num_arm_joints}，得到 {q.shape}")
        self._q_cmd = q
        self._grip_cmd = float(gripper_finger_m)

    def safe_stop(self) -> None:
        self._q_cmd = self._q.copy()

    def close(self) -> None:
        self.safe_stop()


# --------------------------------------------------------------------------- #
# Dora（arm_control 的原生通道）
# --------------------------------------------------------------------------- #
@dataclass
class DoraJogFollower:
    """通过 Dora 节点下发：`jog`（臂）+ `gripper`（夹爪）+ `control`（使能）。

    对应 `dataflows/real_motion.yml` 里 arm_console 输出的三个 topic。`jog` 是
    单帧目标，`jog_timeout_s`（默认 0.2s）内未刷新即自动停，所以 leader 断流
    时大臂会保持而不是失控。默认在 `open()` 时发 `control(arm=True)` 使能。
    """

    num_arm_joints: int = 7
    kp: Optional[Sequence[float]] = None
    kd: Optional[Sequence[float]] = None
    reason: str = "leader-follower"
    auto_arm: bool = True
    node: object = field(default=None, repr=False)
    # 启动对齐用的实测位形来源：返回 (arm_q[n], gripper_finger_m)。接上
    # `motor_state` 回读后由节点注入；不接则回退到零点（部署侧自行对齐）。
    state_provider: Optional[Callable[[], tuple[np.ndarray, float]]] = field(
        default=None, repr=False
    )

    _sent: int = field(default=0, init=False)
    _opened: bool = field(default=False, init=False)

    def _ensure_node(self):
        if self.node is None:
            from dora import Node

            self.node = Node()
        return self.node

    def open(self) -> None:
        node = self._ensure_node()
        if self.auto_arm:
            from arm_control.messages import pack_control_update

            node.send_output("control", pack_control_update(arm=True))
            print("[follower:dora] 已发送 control(arm=True)")
        self._opened = True

    def read_state(self) -> tuple[np.ndarray, float]:
        # `jog` 通道本身不回读；若节点注入了 `state_provider`（订阅
        # `plant_interface/motor_state` + `franka_gripper/gripper_state`），
        # 就用实测位形做启动对齐基准；否则回退零点，由部署侧自行对齐。
        if self.state_provider is not None:
            arm, finger = self.state_provider()
            return np.asarray(arm, dtype=float).ravel(), float(finger)
        return np.zeros(self.num_arm_joints), 0.0

    def send(self, arm_q: Sequence[float], gripper_finger_m: float) -> None:
        from arm_control.messages import pack_jog, pack_motor_command

        node = self._ensure_node()
        q = np.asarray(arm_q, dtype=float)
        if q.shape != (self.num_arm_joints,):
            raise ValueError(f"大臂目标长度应为 {self.num_arm_joints}，得到 {q.shape}")
        node.send_output("jog", pack_jog(q=q, reason=self.reason))
        # 夹爪走 gripper topic，格式为 2 指 motor_command：`franka_gripper` 取
        # position[0] 作为**单指位移（米）**并令 width = 2*finger；DM 臂则由
        # joint_mimics 把单指位移换算成夹爪电机角。两指槽填同一个值。
        zeros = np.zeros(2)
        node.send_output(
            "gripper",
            pack_motor_command(
                [float(gripper_finger_m), float(gripper_finger_m)],
                zeros,
                zeros,
                zeros,
                zeros,
            ),
        )
        self._sent += 1

    def safe_stop(self) -> None:
        # jog 自然过期即停；显式 cancel 取消当前 leg 并保持 ARMED。
        if not self._opened:
            return
        try:
            from arm_control.messages import pack_control_update

            self._ensure_node().send_output(
                "control", pack_control_update(cancel=True, reason="leader-stale")
            )
        except Exception:
            pass

    def close(self) -> None:
        if not self._opened:
            return
        try:
            from arm_control.messages import pack_control_update

            self._ensure_node().send_output(
                "control", pack_control_update(cancel=True, arm=False, reason="leader-follower-exit")
            )
            print("[follower:dora] 已发送 control(cancel, arm=False)")
        except Exception:
            pass
        self._opened = False


# --------------------------------------------------------------------------- #
# 远程 RT（低延迟备选）
# --------------------------------------------------------------------------- #
@dataclass
class RtFollower:
    """直连远程 RT 服务器（`RtBackend`），100Hz 流式发伺服词。

    **FR3 注意**：这条通道只驱动 7 个臂关节（`gripper_slot=False`），
    Franka Hand 是独立设备、不走 RT（走 `franka_gripper` 节点）。也就是说
    用 RT 后端遥操作时夹爪不动；要连夹爪一起遥操作请用 `DoraJogFollower`。

    对 DM 臂这类"夹爪和臂在同一总线"的机械臂，可把 `gripper_slot=True`，
    把夹爪作为最后一个电机槽一起下发（此时 `gripper_finger_m` 需已换算成该
    槽的电机语义）。

    `joint_names` 必须与部署 `arm.joints` 完全一致。RT 服务器自带
    staleness->hold、fault latch、力矩限幅，因此这里只需持续发帧；停发超过
    hold_ms 会 HOLD、超过 fault_ms 会 LATCH（需 DISARM->ARM 恢复）。
    """

    joint_names: Sequence[str]
    host: str = "127.0.0.1"
    udp_port: int = 47800
    tcp_port: int = 47801
    kp: Optional[Sequence[float]] = None
    kd: Optional[Sequence[float]] = None
    gripper_slot: bool = False
    pose_hold: Optional[dict] = None

    num_arm_joints: int = field(init=False)

    _backend: object = field(default=None, init=False, repr=False)

    # FR3 的关节刚度/阻尼（examples/profiles/motion_franka.yaml）。
    _FR3_KP = (1200.0, 1200.0, 1200.0, 1200.0, 800.0, 600.0, 400.0)
    _FR3_KD = (30.0, 30.0, 30.0, 30.0, 20.0, 15.0, 10.0)

    def __post_init__(self) -> None:
        self.joint_names = list(self.joint_names)
        self.num_arm_joints = len(self.joint_names) - (1 if self.gripper_slot else 0)

    def open(self) -> None:
        from arm_control.plants.remote_rt.client import RtBackend, RtConfig

        self._backend = RtBackend(
            RtConfig(host=self.host, udp_port=self.udp_port, tcp_port=self.tcp_port),
            list(self.joint_names),
        )
        self._backend.open()
        self._backend.enable_all()
        print(f"[follower:rt] 已连接 {self.host} 并使能 {len(self.joint_names)} 关节")

    def _gains(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.joint_names)
        is_fr3 = n == 7 and not self.gripper_slot
        kp = (
            np.asarray(self.kp, dtype=float)
            if self.kp is not None
            else (np.asarray(self._FR3_KP) if is_fr3 else np.full(n, 20.0))
        )
        kd = (
            np.asarray(self.kd, dtype=float)
            if self.kd is not None
            else (np.asarray(self._FR3_KD) if is_fr3 else np.full(n, 0.5))
        )
        return kp, kd

    def read_state(self) -> tuple[np.ndarray, float]:
        if self._backend is None:
            return np.zeros(self.num_arm_joints), 0.0
        st = self._backend.motor_state()["position"]
        arm = np.asarray(st, dtype=float)[: self.num_arm_joints]
        grip = float(st[self.num_arm_joints]) if self.gripper_slot else 0.0
        return arm, grip

    def send(self, arm_q: Sequence[float], gripper_finger_m: float) -> None:
        assert self._backend is not None, "RtFollower.open() 未调用"
        kp, kd = self._gains()
        q = np.asarray(arm_q, dtype=float)
        if self.gripper_slot:
            q = np.concatenate([q, [float(gripper_finger_m)]])
        command = {
            "position": q,
            "velocity": np.zeros(len(q)),
            "torque": np.zeros(len(q)),
            "kp": kp,
            "kd": kd,
        }
        if self.pose_hold is not None:
            command["pose_hold"] = self.pose_hold
        self._backend.apply_command(command)

    def safe_stop(self) -> None:
        if self._backend is not None:
            self._backend.safe_stop()

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
