"""Dora node for the real DM motor backend (SocketCAN or DM SDK)."""
from __future__ import annotations

# ruff: noqa: E402

import time

import numpy as np
from dora import Node


from arm_control.end_effectors.torque_gripper import GraspController
from arm_control.end_effectors.grasp import GraspGate
from arm_control.control.safety import SafetyController
from arm_control.config import load_robot_config
from arm_control.messages import (
    pack_grasp_result,
    unpack_grasp_request,
)
from arm_control.plants.dm.backend import DmBackendUnavailableError, DmHardwareBackend
from arm_control.messages import (
    pack_json_message,
    pack_motor_state,
    pack_motor_state_dict as _pack_visual_motor_state,
    unpack_json_message,
    unpack_motor_command,
)
from arm_control.node_utils import (
    ShutdownFlag,
    _load_mode_config,
    _zero_command,
    install_signal_handlers,
)

# Per-DM-type default feed-forward torque clamp (N·m), used when the scenario
# YAML omits `safety.torque_limits`.  Conservative, under the DM MIT encode peak
# (4340/4340P ±28, 4310 ±10); bench-tunable via config.
_DEFAULT_TORQUE_LIMIT_BY_TYPE = {"4310": 3.0, "4310p": 3.0, "4340": 9.0, "4340p": 9.0}


def _gripper_mimic_cfg(cfg) -> dict | None:
    for mimic in (cfg.get("joint_mimics") or {}).values():
        if isinstance(mimic, dict):
            return mimic
    return None


def _gripper_motor_index(backend, cfg) -> int | None:
    sources = {
        str(m.get("source"))
        for m in (cfg.get("joint_mimics") or {}).values()
        if isinstance(m, dict)
    }
    for i, motor in enumerate(backend.motors):
        if motor.name in sources or motor.joint in sources:
            return i
    return None


def _make_grasp(cfg, backend) -> GraspController | None:
    mimic = _gripper_mimic_cfg(cfg)
    index = _gripper_motor_index(backend, cfg)
    if mimic is None or index is None:
        return None
    gate = GraspGate.from_config(cfg.get("grasp") or {}, mimic)
    return GraspController(gate, index)


def _safety_torque_limits(cfg, backend) -> np.ndarray:
    safety = cfg.get("safety") or {}
    raw = safety.get("torque_limits")
    if raw and len(raw) == backend.num_motors:
        return np.array([float(v) for v in raw], dtype=np.float64)
    return np.array(
        [_DEFAULT_TORQUE_LIMIT_BY_TYPE.get(m.motor_type, 3.0) for m in backend.motors],
        dtype=np.float64,
    )


def _make_safety(cfg, backend) -> SafetyController:
    safety = cfg.get("safety") or {}
    return SafetyController(
        backend,
        deadman_timeout_s=float(safety.get("deadman_timeout_s", 0.1)),
        temp_limit_c=float(safety.get("temp_limit_c", 85.0)),
        torque_limits=_safety_torque_limits(cfg, backend),
        arm_ramp_sec=float(safety.get("arm_ramp_sec", 1.0)),
    )


def _effective_listen_mode(cfg) -> bool:
    mode_cfg = _load_mode_config()
    plant_cfg = mode_cfg.get("plant") or {}
    if "listen_mode" in plant_cfg:
        return bool(plant_cfg["listen_mode"])
    return bool(cfg.get("listen_mode", False))


_PLANT_MODES = ("listen", "float", "command")


def _plant_mode(cfg) -> str:
    """Resolve the bridge plant mode: ``listen`` | ``float`` | ``command``.

    An explicit ``plant.mode`` in the mode config wins; otherwise derive from
    ``listen_mode`` for back-compat (true -> listen, false -> command).  ``float``
    is distinct from ``command``: it enables motors on open and forwards
    impedance setpoints straight through, with no arm-gate / grasp / deadman
    layer.  Sharing the ``listen_mode: false`` flag once conflated the two, which
    left float's motors disabled (safety layer, disarmed).
    """
    mode_cfg = _load_mode_config()
    requested = (mode_cfg.get("plant") or {}).get("mode")
    if requested is not None:
        if requested not in _PLANT_MODES:
            raise ValueError(
                f"plant.mode must be one of {_PLANT_MODES}, got {requested!r}"
            )
        return str(requested)
    return "listen" if _effective_listen_mode(cfg) else "command"


def main() -> None:
    cfg = load_robot_config()
    rate_hz = float(cfg.get("hardware_update_rate_hz", cfg.update_rate_hz))
    period = 1.0 / rate_hz
    viz_rate_hz = float(cfg.get("viz_publish_rate_hz", 60.0))
    viz_period = 1.0 / viz_rate_hz
    log_rates = bool(cfg.get("viz_log_rates", False))
    mode = _plant_mode(cfg)
    backend = DmHardwareBackend.from_config(cfg)
    n = backend.num_motors
    command = _zero_command(n)
    node = Node()
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    last_step = 0.0
    last_viz = 0.0
    last_rate_log = 0.0
    step_count = 0
    rate_step_count = 0
    viz_count = 0
    model_revision_sent = False
    safety: SafetyController | None = None
    grasp: GraspController | None = None
    health_fault_prev: tuple | None = None  # (armed, latched_fault, any_fault) edge state

    try:
        backend.open()
        real_backend = cfg.get("real_backend") or cfg.raw.get("real_backend", "")
        if not real_backend.startswith("dm_"):
            adapter = f"SocketCAN ({real_backend})"
        else:
            adapter = f"{backend.bus.adapter} ch{backend.bus.channel}"
        print(
            f"[hardware_interface] backend open: {n} motors on {adapter}",
            flush=True,
        )
        if mode == "listen":
            backend.enable_all()
            print("[hardware_interface] listen mode — motors enabled", flush=True)
        elif mode == "float":
            # Float mode: enable on open and forward impedance setpoints straight
            # through. No arm-gate/grasp/deadman — the arm is meant to be live and
            # backdrivable (gravity comp). enable_all() also solicits the first
            # feedback reply that bootstraps the impedance loop.
            backend.enable_all()
            print(
                "[hardware_interface] float mode — motors enabled, "
                "direct impedance forwarding (no safety/grasp gate)",
                flush=True,
            )
        else:
            # Command mode: DISARMED by default. Do NOT enable on open — the
            # safety layer sends DM_ENABLE on `arm`, and forwards real setpoints
            # only while armed.
            safety = _make_safety(cfg, backend)
            grasp = _make_grasp(cfg, backend)
            print(
                f"[hardware_interface] command mode — DISARMED "
                f"(deadman {safety.deadman_timeout_s:g}s, "
                f"grasp gate {'on' if grasp else 'off'})",
                flush=True,
            )

        while not shutdown.stop_requested:
            event = node.next(0)
            if event is not None:
                etype = event["type"]
                eid = event.get("id", "")
                if etype == "INPUT" and eid == "motor_command":
                    command = unpack_motor_command(event["value"], n)
                    if grasp is not None and grasp.active:
                        pass  # grasp owns the send this tick (merged gripper slot)
                    elif safety is not None:
                        safety.on_command(command, time.monotonic())
                    else:
                        backend.apply_command(command)
                elif etype == "INPUT" and eid == "grasp_request":
                    if grasp is not None:
                        result = grasp.request(unpack_grasp_request(event["value"]))
                        if result is not None:
                            node.send_output(
                                "grasp_result", pack_grasp_result(**result)
                            )
                elif etype == "INPUT" and eid == "arm":
                    if safety is not None:
                        payload = unpack_json_message(event["value"])
                        if payload.get("armed", False):
                            safety.arm(time.monotonic())
                        else:
                            safety.disarm()
                elif etype == "STOP":
                    break
            else:
                time.sleep(period * 0.2)

            now = time.perf_counter()
            if now - last_step < period:
                continue
            last_step = now
            step_count += 1
            rate_step_count += 1

            # Deadman watchdog: evaluated every tick from the wall clock, so it
            # fires even when no motor_command event arrives (upstream crash).
            if safety is not None:
                safety.tick(time.monotonic())

            # Send MIT keepalive and capture fresh state
            if mode == "listen":
                backend.listen_step()
            elif safety is not None and safety.awaiting_first_command:
                # Armed but the orchestrator hasn't commanded yet: send a zero-torque
                # MIT keepalive so the motors report their TRUE pose (solicit a reply)
                # while the arm stays a zero-torque hold. The orchestrator defers
                # coordinator.start() until this fresh state arrives, then commands —
                # ending the window well inside the deadman timeout.
                backend.listen_step()
            state = backend.motor_state()

            # Grasp gate: while a grasp is active it owns the gripper motor slot,
            # driving the merged command through the safety layer each tick (which
            # also feeds the deadman) and emitting grasp_result on grasped/missed.
            if grasp is not None and grasp.active and safety is not None:
                cmd, result = grasp.step(
                    state,
                    command,
                    safety.armed,
                    safety.latched_fault is not None,
                )
                if cmd is not None:
                    safety.on_command(cmd, time.monotonic())
                if result is not None:
                    node.send_output("grasp_result", pack_grasp_result(**result))

            node.send_output(
                "motor_state",
                pack_motor_state(
                    state["position"],
                    state["velocity"],
                    state["position_cmd"],
                    state["velocity_cmd"],
                    state["torque_cmd"],
                    state["kp"],
                    state["kd"],
                    state["torque"],
                ),
            )

            if now - last_viz >= viz_period:
                last_viz = now
                viz_count += 1
                node.send_output("motor_state_viz", _pack_visual_motor_state(state))

            # Fault reflex runs every tick for a fast safe-stop; health is
            # published at 2 Hz, plus immediately on any CHANGE of the
            # (armed, latched_fault, any_fault) tuple (edge, not level, so a
            # latched fault doesn't flood the topic). The armed edge doubles as
            # the arm ACK: the orchestrator refuses to start a sequence from a
            # motor_state until it has seen armed:true from the bridge, closing
            # the queued-pre-arm-state race.
            if safety is not None:
                health = safety.check_health(time.monotonic())
                periodic = step_count % max(1, int(rate_hz / 2.0)) == 0
                health_edge = (health["armed"], health["latched_fault"], health["any_fault"])
                if health_edge != health_fault_prev or periodic:
                    node.send_output(
                        "motor_health", pack_json_message("motor_health", health)
                    )
                health_fault_prev = health_edge

            if step_count % max(1, int(rate_hz / 2.0)) == 0:
                node.send_output(
                    "can_bus_status",
                    pack_json_message(
                        "can_bus_status",
                        {"bus_load_pct": round(backend.bus_load_pct, 1)},
                    ),
                )

            if not model_revision_sent:
                node.send_output(
                    "model_revision",
                    pack_json_message(
                        "model_revision",
                        {
                            "revision": 0,
                            "backend": "socketcan" if not real_backend.startswith("dm_") else "dm_usb2fdcan",
                            "joint_names": list(cfg.joint_names),
                            "motor_names": [motor.name for motor in backend.motors],
                            "motor_types": [motor.motor_type for motor in backend.motors],
                            "num_motors": n,
                        },
                    ),
                )
                model_revision_sent = True

            if log_rates and now - last_rate_log >= 1.0:
                if last_rate_log > 0:
                    dt = now - last_rate_log
                    print(
                        f"[hardware_interface] rates: loop={rate_step_count / dt:.1f}Hz "
                        f"viz={viz_count / dt:.1f}Hz",
                        flush=True,
                    )
                last_rate_log = now
                rate_step_count = 0
                viz_count = 0
    except DmBackendUnavailableError as exc:
        print(f"[hardware_interface] {exc}", flush=True)
    finally:
        backend.close()
        if backend.last_close_errors:
            print(
                f"[hardware_interface] close completed with "
                f"{len(backend.last_close_errors)} cleanup error(s)",
                flush=True,
            )


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
