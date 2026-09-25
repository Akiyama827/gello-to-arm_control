"""Device-free pair lifecycle and admission checks against real ArmControllers."""

from __future__ import annotations

import numpy as np

from arm_control.control.arm_controller import ArmController
from arm_control.control.doubles import _FakeExecutor
from arm_control.messages import pack_motor_state, pack_plan, pack_control_update
from arm_control.multi_arm import (
    ArmPlan,
    BimanualPlan,
    BimanualInterface,
    LocalArmHandle,
)
from arm_control.multi_arm.coordinator import BimanualCoordinator
from arm_control.control.inprocess import QueueControllerNode
from arm_control.multi_arm.types import PairLifecycle as P


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
                gripper=dict(
                    n_motors=n, mimic=None, open_finger_m=0.04, gains=(100.0, 1.0)
                ),
                clock=lambda: self.t,
                plant_reports_health=False,
                model_token=f"{a}:model:r2",
                **(dict(state_period_s=0.01) | (controller_kwargs or {})),
            )
            c.on_control(pack_control_update(arm=True))
            c.on_motor_state(measured(n))
            c.node.sent.clear()
            self.controllers[a] = c
            self.handles[a] = LocalArmHandle(c, clock=lambda: self.t)
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
    robot = BimanualInterface(h.coord)
    assert robot.left.arm_id == a and robot.right.arm_id == b
    assert robot.state().arms()[b].position.shape == (6,)


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


def check_topic_adapter():
    from collections import deque
    from arm_control.contracts.motion import (
        pack_controller_event,
        unpack_control_update,
        unpack_controller_event,
    )
    from arm_control.contracts.multi_arm import pack_pair_plan, pack_pair_command
    from arm_control.multi_arm.adapter import TopicArmHandle, run

    h = Harness()
    pair = h.pair()

    class Node:
        def __init__(self):
            self.inputs, self.sent = deque(), []

        def add(self, topic, value):
            self.inputs.append(dict(type="INPUT", id=topic, value=value))

        def next(self, timeout):
            h.t += 0.001
            return self.inputs.popleft() if self.inputs else dict(type="STOP")

        def send_output(self, topic, value):
            self.sent.append((topic, value))

    node = Node()
    for a, c in h.controllers.items():
        node.add(
            f"controller_event_{a}",
            pack_controller_event(
                kind="mode", reason="joint", runtime_state=c.runtime_state
            ),
        )
        node.add(f"motor_state_{a}", measured(c.n_arm))
    node.add(
        "bimanual_plan",
        pack_pair_plan(
            pair_id=pair.pair_id,
            plans={a: p.payload for a, p in pair.members().items()},
        ),
    )
    for a, c in h.controllers.items():
        plan_id = pair.members()[a].plan_id
        node.add(
            f"controller_event_{a}",
            pack_controller_event(
                kind="plan_staged",
                plan_id=plan_id,
                runtime_state=dict(c.runtime_state, plan_id=plan_id),
            ),
        )
    node.add("bimanual_command", pack_pair_command(pair_id="pair", action="execute"))
    run(node, list(h.controllers), clock=lambda: h.t)
    for a, p in pair.members().items():
        assert sum(t == f"plan_{a}" for t, _ in node.sent) == 1
        controls = [
            unpack_control_update(v) for t, v in node.sent if t == f"control_{a}"
        ]
        assert any(c.get("execute") == p.plan_id for c in controls)
        assert any(c.get("cancel_plan_id") == p.plan_id for c in controls)

    # Runtime heartbeats cannot freshen stale motor data; restart drops old q.
    c = h.controllers["fr3_1"]
    handle = TopicArmHandle(node, c.arm_id, clock=lambda: h.t)
    event = unpack_controller_event(
        pack_controller_event(
            kind="mode", reason="joint", runtime_state=c.runtime_state
        )
    )
    handle.observe_event(event)
    handle.observe_motor_state(dict(position=np.zeros(9), velocity=np.zeros(9)))
    assert len(handle.state().position) == 7
    observed = handle.state().observed_at
    h.t += 2
    handle.observe_event(event)
    assert handle.state().observed_at == observed
    handle.observe_event(
        dict(event, runtime_state=dict(c.runtime_state, session_id="new"))
    )
    assert handle.state().position is None
    try:
        pack_controller_event(
            kind="mode", runtime_state=dict(c.runtime_state, armed="false")
        )
    except ValueError:
        pass
    else:
        raise AssertionError("malformed runtime heartbeat admitted")


def main():
    check_barrier_and_completion()
    check_rejections()
    check_partial_release_and_cancellation()
    check_lost_ack_and_fault()
    check_topic_adapter()
    print(
        "bimanual contract: PASS (7+6 DOF, ACK barrier, model/session/freshness, "
        "gain rejection, partial release, observed cancel, stale IDs, fault)"
    )


if __name__ == "__main__":
    main()
