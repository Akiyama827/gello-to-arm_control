"""Dora node: Tier-1 SIM bridge for the shadow run (pick-and-dock).

Lets the HARDWARE orchestrator run unmodified against the MuJoCo composed
scene: it speaks the real bridge's 7-motor contract (motor_state/motor_command,
arm -> motor_health ACK) while the plant speaks per-arm slices of the composed
scene (assembler: 6, base: 2, plus the finger servos). This node HOSTS the
grasp policy — the same arm_control.bridge GraspGate/GraspController the bench
bridge runs: grasp_request comes in here, the gate ramps the close and
thresholds the grip through the plant's gripper_state effort stream, and
grasp_result goes back out (GRASPED/MISSED, LOST on a drop).

Translation:
- motor_command (7: 6 arm + gripper motor) -> motor_command_assembler (6):
  arm slots pass through (armed only); the gripper MOTOR slot is mapped onto
  the scene's two finger servos via the joint_mimics calibration and sent as
  motor_command_gripper — the fingers physically close (and stop on the
  module via the pad<->module contact pair).
- motor_command_base (2): gentle PD hold at the configured socket pose so the
  dock base stays put.
- motor_state_assembler (6) -> motor_state (7): arm slots pass through; the
  gripper slot echoes the sim's TRUE finger position (gripper_state input,
  finger metres -> motor via the mimic) so a contact-jammed finger reaches
  the operator mirror; synthesized open only until the first echo arrives.
- arm {armed} -> motor_health {armed, ...} published on every change plus
  periodically — the same armed-edge ACK the real bridge publishes.
- PARK: while the orchestrator is not streaming (disarmed, or pre-first
  command), the shim holds the assembler at its current pose — the sim
  stand-in for a real arm staying put by friction; a real disarmed bridge
  sends nothing. No deadman emulation (SafetyController owns that on
  hardware).
"""
from __future__ import annotations

# ruff: noqa: E402


import numpy as np
from dora import Node


from arm_control.bridge import GraspController, GraspGate
from arm_control.node_utils import _zero_command
from arm_control.config import load_robot_config
from arm_control.messages import (
    pack_grasp_result,
    unpack_grasp_request,
)
from arm_control.messages import (
    pack_json_message,
    pack_motor_command,
    pack_motor_state,
    unpack_json_message,
    unpack_motor_command,
    unpack_motor_state,
)

# ponytail: fixed gentle holds (base scenery + disarmed park); config knobs if
# a scenario ever needs different scene dynamics.
_BASE_KP, _BASE_KD = 60.0, 2.0
_PARK_KP, _PARK_KD = 60.0, 2.0
# Engage park almost immediately: the arm spawns at a clean, gravity-holdable
# q=0, but 0.2 s of free-fall is enough to drop the weak wrist into a
# self-contact tangle the planner then rightly refuses to start from.
_PARK_GAP_STATES = 20
_STREAM_EVERY = 10  # shim streams holds at ~100 Hz from the 1 kHz state feed
# Republish rate: the plant feeds 1 kHz but the Python orchestrator (tick +
# RNEA per state) lags behind that over dora — queues grow and the whole loop
# runs in slow motion. The real bridge publishes ~400 Hz; ~200 Hz is plenty.
_PUBLISH_EVERY = 5

N_ARM = 6
# Neutral 7-slot template for GraspController._merge — only the gripper slot
# of the merged command is consumed (the arm slots go to the plant separately).
_CMD7_TEMPLATE = _zero_command(7)


def command_7_to_6(cmd: dict) -> tuple:
    """7-motor bridge command -> 6-motor assembler slice (gripper dropped)."""
    return tuple(
        np.asarray(cmd[key], dtype=float)[:N_ARM]
        for key in ("position", "velocity", "torque", "kp", "kd")
    )


def gripper_motor_to_fingers(motor: float, mimic: dict) -> np.ndarray:
    """Gripper MOTOR position -> the two finger prismatic targets (clamped).

    Wraps the CANONICAL hardware mimic (motor_open <-> lower = fingers OPEN,
    motor_closed <-> upper = fingers CLOSED; measured on the composed scene:
    q=0 is a 72 mm pad gap, q=0.0439 is full close). Do not hand-roll this
    map — an inverted copy once spawned the scene with the fingers closed
    inside the grasp volume.
    """
    from arm_control.joint_motor_map import gripper_motor_to_finger

    lower = float(mimic.get("lower", 0.0))
    upper = float(mimic["upper"])  # required: a guessed travel closes wrong
    lo, hi = min(lower, upper), max(lower, upper)
    value = float(np.clip(gripper_motor_to_finger(float(motor), mimic), lo, hi))
    return np.full(2, value)


def base_hold_command(q_hold: np.ndarray | None = None) -> tuple:
    """Gentle PD hold for the 2-DOF dock base (default zeros).

    ``sim_base_hold_q`` in the scenario config poses the socket — e.g. pitching
    the dock port to face UP so the upright-carried module docks top-down.
    """
    q = np.zeros(2) if q_hold is None else np.asarray(q_hold, dtype=float)[:2]
    zeros = np.zeros(2)
    return q, zeros, zeros, np.full(2, _BASE_KP), np.full(2, _BASE_KD)


def park_command(q_arm) -> tuple:
    """Hold the assembler at the captured pose while nobody is streaming."""
    zeros = np.zeros(N_ARM)
    return (
        np.asarray(q_arm, dtype=float)[:N_ARM],
        zeros,
        zeros,
        np.full(N_ARM, _PARK_KP),
        np.full(N_ARM, _PARK_KD),
    )


def state_6_to_7(state: dict, gripper_open_motor: float) -> tuple:
    """6 assembler rows -> 7: arm passthrough, gripper slot synthesized open."""
    out = []
    for key in ("position", "velocity", "torque"):
        v = np.zeros(7)
        v[:N_ARM] = np.asarray(state[key], dtype=float)[:N_ARM]
        if key == "position":
            v[6] = gripper_open_motor
        out.append(v)
    return tuple(out)


def main() -> None:
    cfg = load_robot_config()
    mimic = next(
        (m for m in (cfg.get("joint_mimics") or {}).values() if isinstance(m, dict)),
        {},
    )
    gripper_open = float(mimic.get("motor_open", 0.0))
    base_hold_q = (
        np.asarray(cfg.get("sim_base_hold_q"), dtype=float)
        if cfg.get("sim_base_hold_q") is not None
        else None
    )
    # THE grasp gate — the same policy machine the bench runs
    # (arm_control/bridge): thresholds from the scenario `grasp:` block,
    # endpoints from the gripper mimic. The plant only reports finger
    # position + servo effort; close sequencing, GRASPED/MISSED latching and
    # LOST drop events all live here at bridge altitude.
    gate = GraspGate.from_config(dict(cfg.get("grasp") or {}), mimic)
    grasp = GraspController(gate, gripper_index=6)
    # Finger-N -> gripper-motor-N.m estimate: the mimic jacobian
    # |d finger / d motor| maps the summed pad forces onto the motor axis.
    mimic_jac = abs(
        (float(mimic["upper"]) - float(mimic.get("lower", 0.0)))
        / (float(mimic["motor_closed"]) - float(mimic.get("motor_open", 0.0)))
    )
    tau_est = 0.0
    gs_ticks = 0
    state7_template = {"position": np.zeros(7), "torque": np.zeros(7)}
    armed = False
    published_health = -1  # force an immediate first publish
    states_seen = 0
    last_cmd_state = -(10**9)
    park_q: np.ndarray | None = None
    zeros = np.zeros(7)
    # TRUE gripper motor echo (from the sim's finger joints); None until the
    # first gripper_state arrives, then the synthesized-open fallback retires.
    gripper_motor_echo: float | None = None

    node = Node()
    print("[sim_bridge] shadow-run bridge shim up (7<->6+base translation)", flush=True)
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        topic = event["id"]
        if topic == "arm":
            armed = bool(unpack_json_message(event["value"]).get("armed", False))
            node.send_output("motor_health", _health(armed))
            published_health = states_seen
        elif topic == "motor_command":
            cmd = unpack_motor_command(event["value"], 7)
            if armed:
                node.send_output(
                    "motor_command_assembler",
                    pack_motor_command(*command_7_to_6(cmd)),
                )
                if not grasp.active:
                    # Gripper slot passthrough only while NO grasp owns the
                    # fingers (close->hold->release) — same ownership rule as
                    # the hardware bridge's GraspController.
                    fingers = gripper_motor_to_fingers(
                        float(np.asarray(cmd["position"], dtype=float)[6]), mimic
                    )
                    zeros2 = np.zeros(2)
                    node.send_output(
                        "motor_command_gripper",
                        pack_motor_command(fingers, zeros2, zeros2, zeros2, zeros2),
                    )
                last_cmd_state = states_seen
                park_q = None
        elif topic == "grasp_request":
            req = unpack_grasp_request(event["value"])
            print(
                f"[sim_bridge] grasp_request mode={req.get('mode')} "
                f"id={str(req.get('request_id'))[:8]}",
                flush=True,
            )
            immediate = grasp.request(req)
            if immediate is not None:
                node.send_output("grasp_result", pack_grasp_result(**immediate))
        elif topic == "gripper_state":
            gs_ticks += 1
            if gs_ticks % 500 == 0:  # ~10 s heartbeat: prove the sensor flows
                print(
                    f"[sim_bridge] gripper sensing alive: tick {gs_ticks}, "
                    f"tau_est={tau_est:.3f} N.m, grasp_active={grasp.active}",
                    flush=True,
                )
            body = unpack_json_message(event["value"])
            fingers = body.get("positions") or []
            efforts = body.get("efforts") or []
            if fingers:
                from arm_control.joint_motor_map import (
                    gripper_finger_to_motor,
                )

                gripper_motor_echo = float(
                    gripper_finger_to_motor(float(fingers[0]), mimic)
                )
                tau_est = mimic_jac * float(sum(abs(float(e)) for e in efforts))
                # Step the gate on the 50 Hz sensor tick: it ramps the close,
                # latches GRASPED on torque, reports MISSED on empty-closed
                # and LOST on a drop. Its command owns the fingers while a
                # grasp is active.
                state7_template["position"][6] = gripper_motor_echo
                state7_template["torque"][6] = tau_est
                cmd7, result = grasp.step(
                    state7_template, _CMD7_TEMPLATE, armed, False
                )
                if cmd7 is not None:
                    fingers_cmd = gripper_motor_to_fingers(
                        float(cmd7["position"][6]), mimic
                    )
                    z2 = np.zeros(2)
                    node.send_output(
                        "motor_command_gripper",
                        pack_motor_command(fingers_cmd, z2, z2, z2, z2),
                    )
                if result is not None:
                    print(
                        f"[sim_bridge] grasp gate: ok={result['ok']} "
                        f"({result['reason']}) tau_est={tau_est:.3f} N.m",
                        flush=True,
                    )
                    node.send_output("grasp_result", pack_grasp_result(**result))
        elif topic == "motor_state_assembler":
            state = unpack_motor_state(event["value"], N_ARM)
            states_seen += 1
            if states_seen % _PUBLISH_EVERY == 0:
                pos, vel, tau = state_6_to_7(
                    state,
                    gripper_open if gripper_motor_echo is None else gripper_motor_echo,
                )
                node.send_output(
                    "motor_state",
                    pack_motor_state(pos, vel, tau, zeros, zeros, zeros, zeros, zeros),
                )
            if states_seen % _STREAM_EVERY == 0:
                node.send_output(
                    "motor_command_base",
                    pack_motor_command(*base_hold_command(base_hold_q)),
                )
                if states_seen - last_cmd_state > _PARK_GAP_STATES:
                    if park_q is None:
                        park_q = np.asarray(
                            state["position"], dtype=float
                        )[:N_ARM].copy()
                    node.send_output(
                        "motor_command_assembler",
                        pack_motor_command(*park_command(park_q)),
                    )
            if states_seen - published_health >= 500:  # ~2 Hz at 1 kHz plant
                node.send_output("motor_health", _health(armed))
                published_health = states_seen


def _health(armed: bool):
    return pack_json_message(
        "motor_health",
        {"armed": armed, "latched_fault": "", "any_fault": False},
    )


if __name__ == "__main__":
    main()
