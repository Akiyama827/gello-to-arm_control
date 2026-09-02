"""Dora node: Franka Research 3 bridge — READ-ONLY state + gripper + payload.

NO MOTION FROM THIS PROCESS (user decision 2026-07-26): the FR3 runs torque
control only, and the torque servo lives in a C++ 1 kHz loop on the realtime
machine. This node is the arm's Python-side presence for everything else —
state streaming, health/fault reporting, the Franka Hand, payload declaration
— with the same output topics as ``nodes/hardware_interface.py`` so
visualizers and orchestrators read both arms identically. Any
``motor_command`` that arrives is dropped with a (once) warning, never
forwarded.

Notes that stay true in this shape:

* ``float`` mode has no analogue here: hand-guide the FR3 for calibration with
  the enabling device on the arm (Franka's Guiding mode, physical button);
  libfranka only reports it as ``RobotMode.Guiding``.
* No ``SafetyController`` — that class is DM-motor shaped. The FR3 enforces
  its own limits and reports through ``robot_mode`` + ``current_errors``.
* No ``GraspGate`` — the Franka Hand detects objects itself; ``grasp()``
  returns the verdict and ``GripperState.is_grasped`` reports drops.
* Payload is declared via ``Robot.set_load`` on a successful grasp
  (``module_grasps`` gives mass and CoM), never through tau_ff.
* No ``can_bus_status`` (no CAN bus); packet health rides
  ``motor_health.success_rate``.
"""
from __future__ import annotations

# ruff: noqa: E402

import time

from dora import Node


from arm_control.config import arm_joints, ee_frame, load_robot_config
from arm_control.messages import (
    pack_grasp_result,
    unpack_grasp_request,
)
from arm_control.hardware.franka_backend import (
    FrankaBackendUnavailableError,
    FrankaHardwareBackend,
)
from arm_control.messages import (
    pack_json_message,
    pack_motor_state_dict as _pack,
    unpack_json_message,
    unpack_motor_command,
)
from arm_control.node_utils import ShutdownFlag, install_signal_handlers


class ArmGate:
    """Operator arm/disarm latch. No deadman: this process cannot move the arm
    (the RT machine owns motion and runs its own watchdog), so there is nothing
    for a host-side timer to stop — arming here gates gripper actions only."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.armed = False
        self.latched_fault: str | None = None
        self._cmd_warned = False

    def arm(self) -> None:
        if self.latched_fault is not None:
            print(f"[franka] arm refused — latched fault: {self.latched_fault}", flush=True)
            return
        self.backend.enable_all()
        self.armed = True
        print("[franka] ARMED — monitor + gripper (no motion from this host)", flush=True)

    def disarm(self, reason: str | None = None) -> None:
        self.backend.safe_stop()
        self.armed = False
        if reason is not None:
            self.latched_fault = reason
            print(f"[franka] DISARM latched: {reason}", flush=True)
        else:
            print("[franka] DISARMED (operator)", flush=True)

    def on_command(self, _command: dict) -> None:
        if not self._cmd_warned:
            self._cmd_warned = True
            print(
                "[franka] motor_command DROPPED — the FR3 position path was "
                "removed (torque control runs on the RT machine); this bridge "
                "is read-only + gripper",
                flush=True,
            )


def main() -> None:
    cfg = load_robot_config()
    rate_hz = float(cfg.get("hardware_update_rate_hz", cfg.update_rate_hz))
    period = 1.0 / rate_hz
    viz_period = 1.0 / float(cfg.get("viz_publish_rate_hz", 60.0))
    grasps = dict(cfg.get("module_grasps") or {})
    joints = arm_joints(cfg)

    backend = FrankaHardwareBackend.from_config(cfg)
    node = Node()
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    gate = ArmGate(backend)

    last_step = last_viz = last_grip = 0.0
    step_count = 0
    health_prev: tuple | None = None
    model_revision_sent = False

    try:
        backend.open()
        print(
            f"[franka] bridge up: {backend.num_motors} joints, DISARMED, "
            "read-only + gripper",
            flush=True,
        )

        while not shutdown.stop_requested:
            event = node.next(0)
            if event is not None:
                etype, eid = event["type"], event.get("id", "")
                if etype == "STOP":
                    break
                if etype == "INPUT" and eid == "motor_command":
                    gate.on_command(
                        unpack_motor_command(event["value"], backend.num_motors)
                    )
                elif etype == "INPUT" and eid == "grasp_request":
                    payload = unpack_grasp_request(event["value"])
                    module_id = str(payload.get("module_id", ""))
                    immediate = backend.request_grasp(
                        str(payload.get("request_id", "")),
                        module_id,
                        str(payload.get("mode", "close")),
                    )
                    if immediate is not None:
                        node.send_output("grasp_result", pack_grasp_result(**immediate))
                    elif str(payload.get("mode", "close")) == "release":
                        backend.set_payload(0.0)
                elif etype == "INPUT" and eid == "arm":
                    if bool(unpack_json_message(event["value"]).get("armed", False)):
                        gate.arm()
                    else:
                        gate.disarm()
            else:
                time.sleep(period * 0.2)

            now = time.perf_counter()
            if now - last_step < period:
                continue
            last_step = now
            step_count += 1

            # A completed grasp: report it, and declare the new load to
            # libfranka so its controller carries the module (the FR3 analogue
            # of the DM path's payload tau_ff).
            result = backend.poll_grasp_result()
            if result is not None:
                node.send_output("grasp_result", pack_grasp_result(**result))
                grasp_cfg = dict(grasps.get(result["module_id"]) or {})
                if result["ok"] and grasp_cfg:
                    backend.set_payload(
                        float(grasp_cfg.get("mass_kg", 0.0)),
                        grasp_cfg.get("com_offset_ee", (0.0, 0.0, 0.0)),
                    )

            state = backend.motor_state()
            node.send_output("motor_state", _pack(state))
            if now - last_viz >= viz_period:
                last_viz = now
                node.send_output("motor_state_viz", _pack(state))

            # Health at 2 Hz plus immediately on any CHANGE (edge, not level, so
            # a latched fault doesn't flood the topic). The armed edge doubles as
            # the orchestrator's arm ACK.
            health = backend.motor_health()
            if gate.latched_fault:
                health["latched_fault"] = gate.latched_fault
                health["any_fault"] = True
            elif health["latched_fault"] and not gate.latched_fault:
                # The robot faulted on its own (reflex / user stop): latch on our
                # side too so the arm gate refuses to re-arm silently.
                gate.latched_fault = health["latched_fault"]
                gate.armed = False
            health["armed"] = gate.armed and health["armed"]
            edge = (health["armed"], health["latched_fault"], health["any_fault"])
            if edge != health_prev or step_count % max(1, int(rate_hz / 2.0)) == 0:
                node.send_output("motor_health", pack_json_message("motor_health", health))
            health_prev = edge

            if now - last_grip > 0.05:
                last_grip = now
                grip = backend.gripper_state()
                if grip is not None:
                    node.send_output(
                        "gripper_state", pack_json_message("gripper_state", grip)
                    )

            if not model_revision_sent:
                node.send_output(
                    "model_revision",
                    pack_json_message(
                        "model_revision",
                        {
                            "revision": 0,
                            "backend": "pylibfranka",
                            "joint_names": joints,
                            "ee_frame": ee_frame(cfg),
                            "num_motors": backend.num_motors,
                        },
                    ),
                )
                model_revision_sent = True
    except FrankaBackendUnavailableError as exc:
        print(f"[franka_interface] {exc}", flush=True)
    finally:
        backend.close()
        if backend.last_close_errors:
            print(
                f"[franka_interface] close completed with "
                f"{len(backend.last_close_errors)} cleanup error(s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
