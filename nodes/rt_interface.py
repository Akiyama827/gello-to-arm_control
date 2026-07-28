"""Dora node: plant bridge to an arm behind the RT machine (arm_rt_server).

The naming peer of ``hardware_interface`` (direct DM CAN) and
``franka_interface`` (direct FCI, read-only): same inputs
(``motor_command`` / ``arm``), same outputs (``motor_state`` /
``motor_state_viz`` / ``motor_health`` / ``model_revision``), so a dataflow
moves an arm onto the RT machine by pointing ``plant_interface`` at this file
— the orchestrator, executor, and teleop never learn the difference.

Deliberately NOT here:
- A deadman. The SERVER owns staleness->hold and fault latching; a dead PC
  leaves the arm holding. This node only mirrors server flags into
  ``motor_health`` so the operator and orchestrator see them.
- Grasp handling. The FR3's Hand is its own TCP device driven from the PC
  (see ``franka_interface``); a DM gripper is a bus motor slot that rides
  ``motor_command`` through the server like any other joint, with the grasp
  GATE staying PC-side policy.
"""
from __future__ import annotations

import time

import numpy as np
from dora import Node

from arm_control.config import arm_joints, ee_frame, load_robot_config
from arm_control.hardware.rt_backend import RtBackend, RtLinkError
from arm_control.messages import (
    pack_json_message,
    pack_motor_state,
    unpack_json_message,
    unpack_motor_command,
)
from arm_control.node_utils import ShutdownFlag, install_signal_handlers


def _pack(state: dict[str, np.ndarray]):
    return pack_motor_state(
        state["position"], state["velocity"], state["position_cmd"],
        state["velocity_cmd"], state["torque_cmd"], state["kp"],
        state["kd"], state["torque"],
    )


def main() -> None:
    cfg = load_robot_config()
    rate_hz = float(cfg.get("hardware_update_rate_hz", cfg.update_rate_hz))
    period = 1.0 / rate_hz
    viz_period = 1.0 / float(cfg.get("viz_publish_rate_hz", 60.0))
    joints = arm_joints(cfg)

    backend = RtBackend.from_config(cfg)
    node = Node()
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)

    last_step = last_viz = 0.0
    step_count = 0
    health_prev: tuple | None = None
    model_revision_sent = False
    armed_wanted = False
    last_drop_warn = 0.0

    try:
        backend.open()
        print(
            f"[rt_interface] bridge up: '{backend.backend_name}' via "
            f"{backend.config.host}, {backend.num_motors} joints, DISARMED",
            flush=True,
        )
        while not shutdown.stop_requested:
            event = node.next(0)
            if event is not None:
                etype, eid = event["type"], event.get("id", "")
                if etype == "STOP":
                    break
                if etype == "INPUT" and eid == "motor_command":
                    if armed_wanted:
                        backend.apply_command(
                            unpack_motor_command(event["value"], backend.num_motors)
                        )
                    elif time.perf_counter() - last_drop_warn > 2.0:
                        # Never swallow silently: an executor upstream is
                        # streaming into a disarmed bridge, and from the
                        # operator's seat that looks like "nothing happened".
                        last_drop_warn = time.perf_counter()
                        print(
                            "[rt_interface] dropping motor_command — DISARMED "
                            "(arm via the operator gate)",
                            flush=True,
                        )
                elif etype == "INPUT" and eid == "arm":
                    if bool(unpack_json_message(event["value"]).get("armed", False)):
                        try:
                            backend.enable_all()
                            armed_wanted = True
                            print("[rt_interface] ARMED (server holds until commands flow)", flush=True)
                        except RtLinkError as exc:
                            print(f"[rt_interface] arm refused: {exc}", flush=True)
                    else:
                        armed_wanted = False
                        backend.safe_stop()
                        print("[rt_interface] DISARMED", flush=True)
            else:
                time.sleep(period * 0.2)

            now = time.perf_counter()
            if now - last_step < period:
                continue
            last_step = now
            step_count += 1

            health = backend.motor_health()
            # Publish state ONLY while the RT stream is fresh. Republishing a
            # frozen snapshot at graph rate would defeat the executor's own
            # staleness deadman (it keys on Dora arrival time) and let it
            # keep commanding against a dead picture of the arm.
            if health["state_fresh"]:
                state = backend.motor_state()
                node.send_output("motor_state", _pack(state))
                if now - last_viz >= viz_period:
                    last_viz = now
                    node.send_output("motor_state_viz", _pack(state))
            edge = (health["armed"], health["latched_fault"], health["any_fault"])
            if edge != health_prev or step_count % max(1, int(rate_hz / 2.0)) == 0:
                node.send_output("motor_health", pack_json_message("motor_health", health))
            health_prev = edge

            if not model_revision_sent:
                node.send_output(
                    "model_revision",
                    pack_json_message(
                        "model_revision",
                        {
                            "revision": 0,
                            "backend": f"rt:{backend.backend_name}",
                            "joint_names": joints,
                            "ee_frame": ee_frame(cfg),
                            "num_motors": backend.num_motors,
                        },
                    ),
                )
                model_revision_sent = True
    except RtLinkError as exc:
        print(f"[rt_interface] {exc}", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
