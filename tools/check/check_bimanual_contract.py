"""Device-free pair lifecycle and admission checks against real ArmControllers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from arm_control.control.arm_controller import ArmController
from arm_control.control.doubles import _FakeExecutor
from arm_control.contracts.motion import pack_plan, pack_control_update, unpack_controller_event
from arm_control.contracts.motor import pack_motor_state
from arm_control.multi_arm import (
    ArmPlan,
    BimanualPlan,
)
from arm_control.multi_arm.coordinator import BimanualCoordinator
from arm_control.control.inprocess import QueueControllerNode
from arm_control.multi_arm.types import ArmControlState, PairLifecycle as P


class LocalArmHandle:
    """An ``ArmHandle`` over an in-process ``ArmController`` (on a ``QueueControllerNode``): the coordinator's contract
    without supervisors. Production handles are ``SdkArmHandle`` (one per supervisor); this one lives here only."""

    def __init__(self, controller) -> None:
        self._controller = controller
        self._node = controller.node

    @property
    def arm_id(self) -> str:
        return self._controller.arm_id

    @property
    def controller(self):
        return self._controller

    def poll_events(self) -> list[dict]:
        """New controller events since the last poll (staging ACKs, results)."""
        return [
            unpack_controller_event(arrow)
            for arrow in self._node.drain_controller_events()
        ]

    def state(self) -> ArmControlState:
        c = self._controller
        # Zeros from a controller that has never measured anything are NOT a
        # measured pose; only report joints once the plant has streamed and
        # the controller went ready at least once.
        measured = c.last_state if c._last_state_at is not None else None
        return ArmControlState(
            arm_id=c.arm_id,
            position=None if measured is None else measured.position,
            velocity=None if measured is None else measured.velocity,
            armed=bool(c.armed),
            ready=bool(c.ready_sent and c.bridge_armed),
            executing=bool(c._running),
            running_plan_id=c.current_plan_id or None,
            mode_law=c.mode_law,
            stopped=bool(c.stopped),
            frozen=bool(c.frozen),
            observed_at=c._last_state_at,
            session_id=c.session_id,
            model_token=c.model_token,
        )

    def stage(self, plan: ArmPlan) -> None:
        self._controller.on_plan(plan.payload)

    def execute(self, plan_id: str) -> None:
        self._controller.on_control(pack_control_update(execute=plan_id))

    def cancel(self, reason: str = "", *, plan_id: str | None = None) -> None:
        self._controller.on_control(
            pack_control_update(cancel=True, reason=reason, cancel_plan_id=plan_id)
        )

    def set_armed(self, armed: bool) -> None:
        # Arming is a LIFECYCLE operation through the controller's own
        # control path (the same message an operator console sends) -- never
        # a flag write, and never implied by a plan arriving.
        self._controller.on_control(pack_control_update(arm=bool(armed)))

    def set_mode(self, law: str, spec: Mapping | None = None) -> None:
        if law == "soft":
            if spec is None:
                raise ValueError("set_mode('soft') requires a pose-hold spec")
            self._controller.on_control(pack_control_update(pose_hold=dict(spec)))
        elif law == "joint":
            # Leaving Soft goes through the controller's own cancel path
            # ("leaving Soft"); a controller already in joint mode is left
            # alone -- cancelling it here would abort a leg for nothing.
            if self._controller.mode_law == "soft":
                self._controller.on_control(
                    pack_control_update(cancel=True, reason="select joint mode")
                )
        else:
            raise ValueError(f"unknown control law {law!r} (joint or soft)")

    def set_gains(self, kp=None, kd=None) -> None:
        fields: dict[str, Any] = {}
        if kp is not None:
            fields["kp"] = kp
        if kd is not None:
            fields["kd"] = kd
        if fields:
            self._controller.on_control(pack_control_update(gains=fields))

    def command_gripper(self, finger_m: float) -> None:
        raise ValueError("an arm controller drives no gripper: the Hand is its own device (hand_bridge / the sim Hand)")


def measured(n):
    z = np.zeros(n)
    return pack_motor_state(z, z, z, z, z, z, z, z)


class Harness:
    def __init__(self, *, staging_timeout_s=2, state_timeout_s=1, controller_kwargs=None):
        self.t = 10.0
        self.controllers, self.handles = {}, {}
        for a, n in [("fr3_1", 7), ("other_arm", 6)]:
            c = ArmController(
                QueueControllerNode(),
                _FakeExecutor(n),
                arm_id=a,
                clock=lambda: self.t,
                plant_reports_health=False,
                model_token=f"{a}:model:r2",
                **(dict(state_period_s=0.01) | (controller_kwargs or {})),
            )
            c.on_control(pack_control_update(arm=True))
            c.on_motor_state(measured(n))
            c.node.sent.clear()
            self.controllers[a] = c
            self.handles[a] = LocalArmHandle(c)
        self.coord = BimanualCoordinator(
            self.handles,
            clock=lambda: self.t,
            staging_timeout_s=staging_timeout_s,
            state_timeout_s=state_timeout_s,
        )

    def pair(self, name="pair", *, bad_model=False, bad_gains=False, distance=0.):
        members = []
        for i, (a, c) in enumerate(self.controllers.items()):
            q = np.zeros((2, c.n_arm))
            q[-1] = distance
            payload = pack_plan(
                plan_id=f"{name}:{a}",
                phase="test",
                gated=True,
                times=[0.0, 1.0],
                positions=q,
                velocities=np.zeros_like(q),
                kp=np.full(c.n_arm, -1.0 if bad_gains and i else 600.0),
                kd=np.full(c.n_arm, 20.0),
                model_token="wrong" if bad_model and i else c.model_token,
            )
            members.append(ArmPlan(a, f"{name}:{a}", payload))
        return BimanualPlan(name, *members)

    def stage(self, **kwargs):
        p = self.pair(**kwargs)
        self.coord.stage_pair(p)
        self.coord.pump()
        return p

    def settle(self):
        for _ in range(300):
            self.t += 0.01
            for c in self.controllers.values():
                c.on_motor_state(measured(c.n_arm))
                # A real graph consumes commands; the fake plant doesn't need them.
                c.node.sent = [
                    (t, v) for t, v in c.node.sent if t == "controller_event"
                ]
            self.coord.pump()
            if self.coord.lifecycle.terminal:
                return
        raise AssertionError("pair did not finish")


def refused(fn, text):
    try:
        fn()
    except (ValueError, KeyError) as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError(f"accepted {text}")


def check_barrier_and_completion():
    h = Harness()
    p = h.pair()
    h.coord.stage_pair(p)
    refused(lambda: h.coord.execute_pair(p.pair_id), "not ready")
    a, b = h.handles
    for e in h.handles[a].poll_events():
        h.coord.on_member_event(a, e)
    assert h.coord.lifecycle is P.STAGING
    h.coord.on_member_event(b, dict(kind="plan_staged", plan_id="old", ok=True))
    h.coord.on_member_event(
        b, dict(kind="plan_staged", plan_id=p.members()[b].plan_id, ok=True)
    )
    assert h.coord.lifecycle is P.STAGING
    h.coord.pump()
    assert h.coord.lifecycle is P.READY
    assert not any(c._running for c in h.controllers.values())
    h.coord.execute_pair(p.pair_id)
    assert all(c._running for c in h.controllers.values())
    h.coord.execute_pair(p.pair_id)  # duplicate release is harmless
    h.settle()
    assert h.coord.lifecycle is P.COMPLETE and h.coord.pair_state().ok() is True
    refused(lambda: h.coord.stage_pair(p), "consumed")
    assert tuple(h.coord.arms) == (a, b)
    assert h.coord.state().arms()[b].position.shape == (6,)


def check_rejections():
    h = Harness()
    refused(lambda: h.coord.stage_pair(h.pair(bad_model=True)), "model token")
    assert all(not c.current_plan_id for c in h.controllers.values())
    h.stage(bad_gains=True)
    assert h.coord.lifecycle is P.REJECTED
    h.coord.pump()
    assert not h.coord.pair_state().cancellation_pending
    assert all(not c._running and not c.current_plan_id for c in h.controllers.values())
    for change in ["stale", "unknown", "disarmed", "restart", "model"]:
        h = Harness()
        h.stage()
        c = h.controllers["other_arm"]
        if change == "stale":
            h.t += 1.1
        if change == "unknown":
            c._last_state_at = None
        if change == "disarmed":
            c.on_control(pack_control_update(arm=False))
        if change == "restart":
            c.session_id = "new-process"
        if change == "model":
            c.model_token = "new-model"
        h.coord.execute_pair("pair")
        assert h.coord.lifecycle in (P.REJECTED, P.FAULTED), change
        assert not any(c._running for c in h.controllers.values()), change


def check_partial_release_and_cancellation():
    h = Harness()
    h.stage()
    right = h.handles["other_arm"]

    def failed_send(_):
        raise OSError("link unavailable")

    right.execute = failed_send
    h.coord.execute_pair("pair")
    assert h.coord.lifecycle is P.FAULTED
    assert not any(c._running for c in h.controllers.values())
    assert h.coord.pair_state().cancellation_pending  # sends are not ACKs
    refused(lambda: h.coord.stage_pair(h.pair("next")), "unconfirmed")
    h.coord.pump()
    assert not h.coord.pair_state().cancellation_pending
    h.stage(name="next")
    h.coord.cancel_pair("operator")
    h.coord.pump()
    assert (
        h.coord.lifecycle is P.CANCELLED
        and not h.coord.pair_state().cancellation_pending
    )
    h.stage(name="third")
    # A delayed cancellation for an older plan cannot clear a new member plan.
    left = h.controllers["fr3_1"]
    left.on_control(pack_control_update(cancel=True, cancel_plan_id="pair:fr3_1"))
    assert left.current_plan_id == "third:fr3_1"
    h.coord.pump()
    assert h.coord.lifecycle is P.READY


def check_lost_ack_and_fault():
    h = Harness(staging_timeout_s=0.1)
    h.coord.stage_pair(h.pair())
    h.handles["other_arm"].poll_events = lambda: []
    h.t += 0.2
    h.coord.pump()
    assert h.coord.lifecycle is P.REJECTED and "timeout" in h.coord.pair_state().reason
    h = Harness()
    h.stage()
    h.coord.execute_pair("pair")
    h.controllers["other_arm"].on_control(pack_control_update(arm=False))
    h.coord.pump()
    h.coord.pump()
    assert h.coord.lifecycle is P.FAULTED
    assert not h.controllers["fr3_1"]._running


def main():
    check_barrier_and_completion()
    check_rejections()
    check_partial_release_and_cancellation()
    check_lost_ack_and_fault()
    print(
        "bimanual contract: PASS (7+6 DOF, ACK barrier, model/session/freshness, "
        "gain rejection, partial release, observed cancel, stale IDs, fault)"
    )


if __name__ == "__main__":
    main()
