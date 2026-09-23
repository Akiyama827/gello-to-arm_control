"""Dora transport for the reusable pair coordinator.

Inputs: bimanual_plan, bimanual_command, arm_command, controller_event_<id>,
motor_state_<id>. Outputs: plan_<id>, control_<id>, gripper_<id>,
bimanual_state, pair_result. Only member controllers produce motor commands.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping

from dora import Node

from arm_control.contracts.motor import _MS, unpack_motor_state
from arm_control.contracts.motion import (
    pack_control_update,
    unpack_controller_event,
    unpack_plan,
)
from arm_control.contracts.multi_arm import (
    pack_bimanual_state,
    pack_pair_result,
    plan_payload_from_json,
    unpack_arm_command,
    unpack_pair_command,
    unpack_pair_plan,
)
from arm_control.messages import pack_motor_command
from arm_control.multi_arm.coordinator import BimanualCoordinator
from arm_control.multi_arm.types import ArmControlState, ArmPlan, BimanualPlan


def arm_ids_from_env():
    arms = [p.strip() for p in os.environ.get("BIMANUAL_ARMS", "left,right").split(",")]
    if len(arms) != 2 or len(set(arms)) != 2 or not all(arms):
        raise ValueError("BIMANUAL_ARMS must name exactly two distinct arm ids")
    return arms


class TopicArmHandle:
    def __init__(self, node, arm_id: str, *, clock: Callable = time.monotonic):
        self.node, self.arm_id, self._clock = node, arm_id, clock
        self._runtime = {}
        self._runtime_at = None
        self._motor_at = None
        self._q = self._qd = None
        self._mode_law = None
        self._fault_reason = None

    def observe_event(self, event: Mapping):
        runtime = event.get("runtime_state")
        if runtime is not None:
            if runtime.get("session_id") != self._runtime.get("session_id"):
                self._q = self._qd = self._motor_at = None
            self._runtime = dict(runtime)
            self._runtime_at = self._clock()
        if event.get("kind") == "mode" and event.get("ok") is True:
            self._mode_law = (event.get("mode_state") or {}).get(
                "law", event.get("reason")
            )
        if event.get("kind") == "fault":
            self._fault_reason = event.get("reason", "controller fault")
        elif event.get("kind") == "ready":
            self._fault_reason = None

    def observe_motor_state(self, state: Mapping):
        n = self._runtime.get("n_arm")
        if not isinstance(n, int) or n <= 0 or len(state["position"]) < n:
            return  # wait for the controller's joint-count/identity heartbeat
        self._q, self._qd = state["position"][:n].copy(), state["velocity"][:n].copy()
        self._motor_at = self._clock()

    def motor_age_sec(self):
        return None if self._motor_at is None else self._clock() - self._motor_at

    def state(self):
        r = self._runtime
        stamps = (self._motor_at, self._runtime_at)
        observed = min(stamps) if all(t is not None for t in stamps) else None
        return ArmControlState(
            arm_id=self.arm_id,
            position=self._q,
            velocity=self._qd,
            armed=r.get("armed"),
            ready=r.get("ready"),
            executing=r.get("executing"),
            running_plan_id=r.get("plan_id") or None,
            mode_law=self._mode_law,
            stopped=r.get("stopped"),
            frozen=r.get("frozen"),
            fault_reason=self._fault_reason,
            observed_at=observed,
            session_id=r.get("session_id"),
            model_token=r.get("model_token"),
        )

    def stage(self, plan):
        self.node.send_output(f"plan_{self.arm_id}", plan.payload)

    def execute(self, plan_id):
        self.node.send_output(
            f"control_{self.arm_id}", pack_control_update(execute=plan_id)
        )

    def cancel(self, reason="", *, plan_id=None):
        self.node.send_output(
            f"control_{self.arm_id}",
            pack_control_update(cancel=True, reason=reason, cancel_plan_id=plan_id),
        )

    def set_armed(self, armed):
        if not isinstance(armed, bool):
            raise ValueError("arm value must be boolean")
        self.node.send_output(f"control_{self.arm_id}", pack_control_update(arm=armed))

    def set_mode(self, law, spec=None):
        if law == "soft" and spec is not None:
            body = dict(pose_hold=dict(spec))
        elif law == "joint":
            body = dict(cancel=True, reason="select joint mode")
        else:
            raise ValueError("unknown control law or missing Soft spec")
        self.node.send_output(f"control_{self.arm_id}", pack_control_update(**body))

    def set_gains(self, kp=None, kd=None):
        body = {k: v for k, v in dict(kp=kp, kd=kd).items() if v is not None}
        self.node.send_output(f"control_{self.arm_id}", pack_control_update(gains=body))

    def command_gripper(self, finger_m):
        import math

        width = float(finger_m)
        if not math.isfinite(width) or width < 0:
            raise ValueError("finger width must be finite and nonnegative")
        self.node.send_output(
            f"gripper_{self.arm_id}",
            pack_motor_command([width] * 2, [0.0] * 2, [0.0] * 2, [0.0] * 2, [0.0] * 2),
        )


def decode_pair_plan(payload):
    body = unpack_pair_plan(payload)
    members = []
    for arm_id, raw in body["plans"].items():
        wire = plan_payload_from_json(raw)
        plan = unpack_plan(wire)
        members.append(
            ArmPlan(arm_id, plan["plan_id"], wire, gated=plan.get("gated") is True)
        )
    return BimanualPlan(body["pair_id"], *members)


def run(node, arms, *, clock=time.monotonic):
    handles = {a: TopicArmHandle(node, a, clock=clock) for a in arms}
    coordinator = BimanualCoordinator(
        handles,
        clock=clock,
        staging_timeout_s=float(os.environ.get("BIMANUAL_STAGE_TIMEOUT_SEC", "2")),
        state_timeout_s=float(os.environ.get("BIMANUAL_STATE_STALE_SEC", "1")),
    )
    coordinator.on_pair_result(
        lambda s: node.send_output("pair_result", pack_pair_result(s.to_payload()))
    )
    last_publish = float("-inf")
    try:
        while True:
            event = node.next(timeout=0.02)
            if event and event["type"] == "STOP":
                break
            if event and event["type"] == "INPUT":
                eid, value = event["id"], event["value"]
                try:
                    if eid == "bimanual_plan":
                        coordinator.stage_pair(decode_pair_plan(value))
                    elif eid == "bimanual_command":
                        c = unpack_pair_command(value)
                        if c["pair_id"] != coordinator.active_pair_id:
                            raise ValueError("pair command names a stale pair")
                        if c["action"] == "execute":
                            coordinator.execute_pair(c["pair_id"])
                        else:
                            coordinator.cancel_pair(
                                c.get("reason") or "operator cancel"
                            )
                    elif eid == "arm_command":
                        c = unpack_arm_command(value)
                        h = handles[c["arm_id"]]
                        action, v = c["action"], c.get("value")
                        if (
                            coordinator.lifecycle.active
                            or coordinator.pair_state().cancellation_pending
                        ):
                            coordinator.cancel_pair("per-arm action invalidated pair")
                            # A disarm is always forwarded; other mutations wait
                            # for explicit observed cancellation, then a new request.
                            if action != "arm" or v is not False:
                                raise ValueError(
                                    "pair cancellation requested; retry after cleared"
                                )
                        if action == "arm":
                            h.set_armed(v)
                        elif action == "gripper":
                            h.command_gripper(v)
                        elif action == "mode":
                            h.set_mode(v["law"], v.get("spec"))
                        elif action == "gains":
                            h.set_gains(**v)
                    elif eid.startswith("controller_event_"):
                        a = eid.removeprefix("controller_event_")
                        decoded = unpack_controller_event(value)
                        handles[a].observe_event(decoded)
                        coordinator.on_member_event(a, decoded)
                    elif eid.startswith("motor_state_"):
                        a = eid.removeprefix("motor_state_")
                        handles[a].observe_motor_state(
                            unpack_motor_state(value, len(value) // _MS)
                        )
                except (ValueError, TypeError, KeyError) as exc:
                    node.send_output(
                        "pair_result",
                        pack_pair_result(
                            dict(
                                pair_id=coordinator.active_pair_id,
                                lifecycle="request_rejected",
                                ok=False,
                                reason=str(exc),
                            )
                        ),
                    )
            coordinator.check_timeouts()
            if clock() - last_publish >= 0.1:
                node.send_output(
                    "bimanual_state",
                    pack_bimanual_state(coordinator.state().to_payload()),
                )
                last_publish = clock()
    finally:
        coordinator.cancel_pair("coordinator shutdown")


def main():
    run(Node(), arm_ids_from_env())
