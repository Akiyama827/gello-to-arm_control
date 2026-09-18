"""大臂回读：把 Dora 图里的反馈 topic 收敛成 `FeedbackSample`。

小臂→大臂的 `jog` 是**单向下发**通道，默认不回读。安全层要判"大臂没按预期
运行"，就必须另接一组反馈：

* ``plant_interface/motor_state``      实测关节位置/速度（跟踪误差、速度异常）
* ``plant_interface/motor_health``      `armed` / `latched_fault`（失能/故障位）
* ``arm_controller/controller_event``   控制器 `fault`（控制器停了）
* ``franka_gripper/gripper_state``      夹爪开度（仅记录，安全层暂不判）

同一个 Dora `Node` 既发 `jog`/`control`/`gripper` 又收上面这些 topic，所以本类
提供非阻塞 ``drain()``：主循环每 tick 调一次 ``sample()``，把自上次以来到达的
事件全部吸收，只保留最新一条。**新鲜度用样本自带的到达时刻 `timestamp`，
不是"这一 tick 调过 `sample()`"**，这样即使调用方每 tick 拿到同一个缓存样本，
`SafetyMonitor` 的反馈陈旧门仍然会触发。
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np

from .safety import FeedbackSample

# motor_state 每电机 8 个 float（见 contracts/motor.py 的 _MS）。
_MOTOR_STATE_STRIDE = 8


class DoraFollowerFeedback:
    """从 Dora `Node` 抽大臂反馈；可作 `TeleopLoop(feedback=fb.sample)`。"""

    def __init__(
        self,
        node,
        num_arm_joints: int,
        *,
        arm_state_id: str = "motor_state",
        gripper_state_id: str = "gripper_state",
        health_ids: Sequence[str] = ("motor_health", "controller_arm"),
        event_id: str = "controller_event",
    ) -> None:
        self._node = node
        self._n = int(num_arm_joints)
        self._arm_state_id = str(arm_state_id)
        self._gripper_state_id = str(gripper_state_id)
        self._health_ids = tuple(str(x) for x in health_ids)
        self._event_id = str(event_id)

        self._q: Optional[np.ndarray] = None
        self._dq: Optional[np.ndarray] = None
        self._width_m: Optional[float] = None
        self._t = 0.0

        self._armed: Optional[bool] = None
        self._health_fault = ""
        self._event_fault = ""
        self._stopped = False

    # ------------------------------------------------------------------ #
    # 收包
    # ------------------------------------------------------------------ #
    def _try_recv(self):
        """非阻塞取一个事件；空/不可用都返回 None。"""
        fn = getattr(self._node, "try_recv", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
        try:
            return self._node.next(timeout=0.0)
        except Exception:
            return None

    def drain(self) -> int:
        """吸收当前所有待处理事件，返回处理条数。"""
        handled = 0
        while True:
            event = self._try_recv()
            if event is None:
                break
            etype = event.get("type") if isinstance(event, dict) else None
            if etype == "STOP":
                # dora 要求停机：当作一次故障反馈，交给安全层停机。
                self._stopped = True
                break
            if etype != "INPUT":
                continue
            self.handle(event.get("id"), event.get("value"))
            handled += 1
        return handled

    def handle(self, input_id, value) -> None:
        """处理单条输入；也供离线测试直接调用。"""
        if value is None:
            return
        if input_id == self._arm_state_id:
            self._parse_motor_state(value)
        elif input_id == self._gripper_state_id:
            self._parse_gripper_state(value)
        elif input_id in self._health_ids:
            self._parse_health(value)
        elif input_id == self._event_id:
            self._parse_controller_event(value)

    def _parse_motor_state(self, value) -> None:
        from arm_control.messages import unpack_motor_state

        n = self._n
        try:
            state = unpack_motor_state(value, n)
        except Exception:
            n = max(len(value) // _MOTOR_STATE_STRIDE, 1)
            state = unpack_motor_state(value, n)
        pos = np.asarray(state["position"], dtype=float).ravel()[: self._n]
        vel = np.asarray(state["velocity"], dtype=float).ravel()[: self._n]
        self._q = pos
        self._dq = vel
        self._t = time.monotonic()

    def _parse_gripper_state(self, value) -> None:
        from arm_control.messages import unpack_json_message

        msg = unpack_json_message(value, expected_schema="gripper_state")
        width = msg.get("width")
        # 早期样本可能 width=None（未测到），保留 None 而不是当 0。
        self._width_m = None if width is None else float(width)

    def _parse_health(self, value) -> None:
        from arm_control.messages import unpack_json_message

        msg = unpack_json_message(value)
        if "armed" in msg:
            self._armed = bool(msg["armed"])
        latched = msg.get("latched_fault") or ""
        self._health_fault = str(latched) if latched else ""

    def _parse_controller_event(self, value) -> None:
        from arm_control.messages import unpack_controller_event

        result = unpack_controller_event(value)
        if result.get("kind") == "fault":
            # 控制器故障是终态，粘住不自动清除：恢复要有一次重启。
            self._event_fault = str(result.get("reason") or "controller fault")

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def _fault_reason(self) -> str:
        parts = [p for p in (self._event_fault, self._health_fault) if p]
        return "; ".join(parts)

    @property
    def stopped(self) -> bool:
        return self._stopped

    def latest_arm_state(self) -> tuple[np.ndarray, float]:
        """启动对齐用：返回 (arm_q[n], gripper_finger_m)。无回读则零点。"""
        self.drain()
        if self._q is None:
            return np.zeros(self._n), 0.0
        finger = 0.0 if self._width_m is None else self._width_m / 2.0
        return self._q.copy(), float(finger)

    def sample(self) -> Optional[FeedbackSample]:
        """`TeleopLoop` 的 feedback 回调：吸收本 tick 的反馈并给出一帧。"""
        self.drain()
        if self._stopped:
            return FeedbackSample(
                arm_q=np.zeros(self._n),
                timestamp=time.monotonic(),
                fault=True,
                fault_reason="收到 dora STOP",
            )
        if self._q is None:
            # 还没收到过 motor_state：返回 None，让 feedback_required 计时。
            return None
        return FeedbackSample(
            arm_q=self._q.copy(),
            arm_dq=None if self._dq is None else self._dq.copy(),
            gripper_width_m=self._width_m,
            timestamp=self._t,
            fault=bool(self._fault_reason()),
            fault_reason=self._fault_reason(),
            armed=self._armed,
        )

    def prime(self, timeout_s: float = 10.0) -> None:
        """阻塞等待首帧 `motor_state`（启动对齐需要实测位形）。

        用带超时的 `next()`，只在节点启动阶段调用；收到即返回。超时抛错，
        宁可拒绝启动也不要对零点做对齐。
        """
        deadline = time.monotonic() + float(timeout_s)
        got_stop = False
        while self._q is None and time.monotonic() < deadline:
            try:
                event = self._node.next(timeout=0.1)
            except Exception:
                event = None
            if event is None:
                continue
            etype = event.get("type") if isinstance(event, dict) else None
            if etype == "STOP":
                got_stop = True
                self._stopped = True
                break
            if etype == "INPUT":
                self.handle(event.get("id"), event.get("value"))
        if self._q is None and not got_stop:
            raise RuntimeError(
                f"启动超时：{timeout_s:.1f}s 内未收到大臂 motor_state；"
                "确认 plant_interface 已启动、motor_state 已接线。"
            )
