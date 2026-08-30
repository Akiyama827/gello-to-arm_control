"""Dora node: Tier-1 SIM bridge for the shadow run (pick-and-dock).

One configured actor bridge adapts the real bridge contract to one actor slice
of a MuJoCo scene. ``SIM_ACTOR_ID`` selects the slice (legacy default:
``assembler``). This node hosts only optional gripper behavior; scene policy
belongs to the caller.

Translation:
- motor_command (7: 6 arm + gripper motor) -> motor_command_<actor> (6):
  actor slots pass through (armed only); the gripper MOTOR slot is mapped onto
  the scene's two finger servos via the joint_mimics calibration and sent as
  motor_command_gripper — the fingers physically close (and stop on the
  module via the pad<->module contact pair).
- motor_state_<actor> (6) -> motor_state (7): actor slots pass through; the
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


import time
import os

import numpy as np
from dora import Node


from arm_control.bridge import GraspController, GraspGate, HandGraspFsm
from arm_control.node_utils import _zero_command
from arm_control.config import arm_joints, load_robot_config
from arm_control.messages import (
    pack_grasp_result,
    unpack_grasp_request,
)
from arm_control.messages import (
    pack_cartesian_block,
    pack_json_message,
    pack_motor_command,
    pack_motor_state,
    unpack_json_message,
    unpack_motor_command,
    unpack_motor_state,
)

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

def command_arm_slice(cmd: dict, n_arm: int) -> tuple:
    """Bridge command -> arm slice (drops a trailing gripper motor slot, if any).

    The optional Cartesian block rides along verbatim: it is an EE-level
    quantity with no per-motor structure, so slicing does not apply to it.
    Trailing element so it lands on ``pack_motor_command``'s ``cartesian``.
    """
    sliced = tuple(
        np.asarray(cmd[key], dtype=float)[:n_arm]
        for key in ("position", "velocity", "torque", "kp", "kd")
    )
    cart = cmd.get("cartesian")
    if cart is None:
        return sliced
    return sliced + (
        pack_cartesian_block(cart["pose"], cart["task_R"], cart["kc"], cart["dc"]),
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


def park_command(q_arm, kp, kd) -> tuple:
    """Hold the assembler at the captured pose while nobody is streaming.

    Uses the arm's OWN configured gains: the old fixed gentle hold (60/2)
    sagged a heavy arm ~1.5 rad before the operator armed (FR3 gravity vs
    kp 60), and the drooped wrist then started inside the cloud-voxel plan
    obstacles — planning refused before anything moved.
    """
    n_arm = len(kp)
    zeros = np.zeros(n_arm)
    return (
        np.asarray(q_arm, dtype=float)[:n_arm],
        zeros,
        zeros,
        np.asarray(kp, dtype=float),
        np.asarray(kd, dtype=float),
    )


def state_to_bridge(state: dict, n_arm: int, gripper_motor: float | None) -> tuple:
    """Arm rows -> bridge rows: passthrough, plus a gripper motor slot when the
    bridge contract has one (DM: 7 = 6 arm + gripper; FR3: arm-only)."""
    n_bridge = n_arm if gripper_motor is None else n_arm + 1
    out = []
    for key in ("position", "velocity", "torque"):
        v = np.zeros(n_bridge)
        v[:n_arm] = np.asarray(state[key], dtype=float)[:n_arm]
        if key == "position" and gripper_motor is not None:
            v[n_arm] = gripper_motor
        out.append(v)
    return tuple(out)


class SimHand:
    """Franka-Hand grasp semantics over the plant's position-servoed fingers.

    Real-Hand parity in three moves. (1) The width command RAMPS at the
    configured speed — a raw servo target slams the fingers in milliseconds
    and the pads cam off the module's curved surface (measured: 8.4 mm
    settle on a 57 mm module). (2) Verdict is width-only, like grasp():
    fingers BLOCKED short of the commanded width = object present (an empty
    close reaches the target exactly), plus the epsilon band. (3) On HELD,
    a force phase drives the targets to full close against the jam so the
    plant's per-finger force cap becomes the sustained grip (a position
    servo AT the settle width holds ~zero force — the module would ratchet
    out during the carry). Feeds the same HandGraspFsm the real node runs.
    """

    SETTLE_WINDOW_S = 0.2
    SETTLE_TOL_M = 0.0005
    CONTACT_TOL_M = 0.003  # blocked this far short of the command = contact
    FORCE_CLOSE_EXTRA_M = 0.010  # hold-phase squeeze past the jam width
    EFFORT_FLOOR_N = 2.0  # held module keeps the servos loaded (5-12 N meas.)

    def __init__(self, fsm: HandGraspFsm) -> None:
        self.fsm = fsm
        self.mode = "idle"  # idle | closing | holding | opening
        self.gdone_count = 0
        self.gdone_ok = False
        self.state: dict | None = None
        self.finger_cmd: np.ndarray | None = None  # per-finger targets to send
        self._cmd_w: float | None = None    # ramped width command
        self._target_w = 0.0
        self._last_t: float | None = None
        self._hist: list[tuple[float, float]] = []  # (t, width)

    def command(self, action: str, now: float) -> None:
        self._target_w = (
            self.fsm.grasp_width_m if action == "grasp" else self.fsm.open_width_m
        )
        self.mode = "closing" if action == "grasp" else "opening"
        self._last_t = now
        self._hist.clear()
        if action != "grasp":
            # Open is INSTANT: the plant's release detection keys on the
            # commanded target reaching the open rest — a ramped open lags
            # ~0.7 s, long enough for the released module to slide off the
            # pads before the dock weld check runs. (The ramp matters for
            # CLOSING, where a slam cams the pads off the curved hull.)
            self._cmd_w = self._target_w
            self.finger_cmd = np.full(2, self._cmd_w / 2.0)
        elif self._cmd_w is None:
            # First action: ramp from the measured width (or fully open).
            self._cmd_w = (
                float(self.state["width"]) if self.state else 2 * 0.04
            )

    def on_gripper_state(self, positions, efforts, now: float) -> None:
        width = float(sum(float(p) for p in positions))
        in_band = (
            self.fsm.grasp_width_m - self.fsm.epsilon_inner_m
            <= width
            <= self.fsm.grasp_width_m + self.fsm.epsilon_outer_m
        )
        if self.mode in ("closing", "opening") and self._cmd_w is not None:
            dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
            self._last_t = now
            step = self.fsm.speed_mps * dt
            if self._cmd_w > self._target_w:
                self._cmd_w = max(self._target_w, self._cmd_w - step)
            else:
                self._cmd_w = min(self._target_w, self._cmd_w + step)
            self.finger_cmd = np.full(2, self._cmd_w / 2.0)
        if self.mode == "opening" and self._cmd_w == self._target_w:
            self.mode = "idle"
        if self.mode == "closing" and self._cmd_w == self._target_w:
            self._hist.append((now, width))
            self._hist = [(t, w) for t, w in self._hist
                          if now - t <= self.SETTLE_WINDOW_S]
            settled = (
                len(self._hist) >= 2
                and self._hist[-1][0] - self._hist[0][0] >= 0.5 * self.SETTLE_WINDOW_S
                and max(w for _, w in self._hist) - min(w for _, w in self._hist)
                < self.SETTLE_TOL_M
            )
            if settled:
                contact = width > self._target_w + self.CONTACT_TOL_M
                self.gdone_ok = in_band and contact
                self.gdone_count += 1
                print(
                    f"[sim_bridge] hand verdict: {'HELD' if self.gdone_ok else 'not held'} "
                    f"(width {width * 1000:.1f} mm, cmd {self._target_w * 1000:.1f} mm)",
                    flush=True,
                )
                if self.gdone_ok:
                    # BOUNDED force phase: close a further 10 mm against the
                    # jam (~6 N/finger at the plant's servo kp). Full close
                    # was tried and squeezes the curved hull out of the flat
                    # pinch (width collapsed 48 -> 19 mm, module escaped);
                    # the noslip post-pass carries the hold from there.
                    self.mode = "holding"
                    self._cmd_w = max(0.0, width - self.FORCE_CLOSE_EXTRA_M)
                    self.finger_cmd = np.full(2, self._cmd_w / 2.0)
                else:
                    self.mode = "idle"
        # Holding truth: width band + servo LOAD. Width-blocked alone is
        # fragile — soft contacts let the pads sink to the hold command
        # during a carry (measured: width -> cmd exactly, module still
        # riding). A dropped module parks the servos AT target with ~zero
        # effort; a held one keeps them loaded (5-12 N measured).
        effort = float(sum(abs(float(e)) for e in efforts))
        is_grasped = (
            self.mode == "holding" and in_band and effort > self.EFFORT_FLOOR_N
        )
        self.state = {"width": width, "is_grasped": is_grasped}


def main() -> None:
    cfg = load_robot_config()
    actor_id = os.environ.get("SIM_ACTOR_ID", "assembler")
    n_arm = len(arm_joints(cfg))
    mimic = next(
        (m for m in (cfg.get("joint_mimics") or {}).values() if isinstance(m, dict)),
        {},
    )
    # An absent joint_mimics block is the arm's signal that its gripper is not
    # a motor slot on the bus (the FR3's Franka Hand): bridge contract is
    # arm-only and grasps run through the SimHand + HandGraspFsm pair — the
    # same verdict semantics the real hand path produces.
    hand: SimHand | None = None
    grasp = None
    if mimic:
        gripper_open = float(mimic.get("motor_open", 0.0))
        n_bridge = n_arm + 1
        # THE grasp gate — the same policy machine the bench runs
        # (arm_control/bridge): thresholds from the scenario `grasp:` block,
        # endpoints from the gripper mimic. The plant only reports finger
        # position + servo effort; close sequencing, GRASPED/MISSED latching
        # and LOST drop events all live here at bridge altitude.
        gate = GraspGate.from_config(dict(cfg.get("grasp") or {}), mimic)
        grasp = GraspController(gate, gripper_index=n_arm)
        # Finger-N -> gripper-motor-N.m estimate: the mimic jacobian
        # |d finger / d motor| maps the summed pad forces onto the motor axis.
        mimic_jac = abs(
            (float(mimic["upper"]) - float(mimic.get("lower", 0.0)))
            / (float(mimic["motor_closed"]) - float(mimic.get("motor_open", 0.0)))
        )
        cmd_template = _zero_command(n_bridge)
    else:
        gripper_open = None
        n_bridge = n_arm
        gcfg = dict((cfg.get("franka") or {}).get("gripper") or {})
        hand = SimHand(HandGraspFsm(gcfg))
    arm_cfg = dict(cfg.get("arm") or {})
    park_kp = np.asarray(arm_cfg.get("kp", [_PARK_KP] * n_arm), float)[:n_arm]
    park_kd = np.asarray(arm_cfg.get("kd", [_PARK_KD] * n_arm), float)[:n_arm]
    tau_est = 0.0
    gs_ticks = 0
    state_template = {"position": np.zeros(n_bridge), "torque": np.zeros(n_bridge)}
    armed = False
    published_health = -1  # force an immediate first publish
    states_seen = 0
    last_cmd_state = -(10**9)
    park_q: np.ndarray | None = None
    zeros = np.zeros(n_bridge)
    # TRUE gripper motor echo (from the sim's finger joints); None until the
    # first gripper_state arrives, then the synthesized-open fallback retires.
    gripper_motor_echo: float | None = None

    node = Node()
    mode = "hand-grasp" if hand is not None else "gate-grasp"
    print(
        f"[sim_bridge] shadow-run bridge shim up ({n_bridge}<->{n_arm}+base, "
        f"{mode})",
        flush=True,
    )
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
            cmd = unpack_motor_command(event["value"], n_bridge)
            if armed:
                node.send_output(
                    f"motor_command_{actor_id}",
                    pack_motor_command(*command_arm_slice(cmd, n_arm)),
                )
                if grasp is not None and not grasp.active:
                    # Gripper slot passthrough only while NO grasp owns the
                    # fingers (close->hold->release) — same ownership rule as
                    # the hardware bridge's GraspController.
                    fingers = gripper_motor_to_fingers(
                        float(np.asarray(cmd["position"], dtype=float)[n_arm]), mimic
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
            if hand is not None:
                action, immediate = hand.fsm.on_request(
                    req, hand.gdone_count, time.monotonic()
                )
                hand.command(action, time.monotonic())
            else:
                immediate = grasp.request(req)
            if immediate is not None:
                node.send_output("grasp_result", pack_grasp_result(**immediate))
        elif topic == "gripper_state":
            gs_ticks += 1
            if gs_ticks % 500 == 0:  # ~10 s heartbeat: prove the sensor flows
                active = (hand.mode if hand is not None
                          else f"grasp_active={grasp.active}")
                print(
                    f"[sim_bridge] gripper sensing alive: tick {gs_ticks}, "
                    f"tau_est={tau_est:.3f} N.m, {active}",
                    flush=True,
                )
            body = unpack_json_message(event["value"])
            fingers = body.get("positions") or []
            efforts = body.get("efforts") or []
            if fingers and hand is not None:
                now = time.monotonic()
                hand.on_gripper_state(fingers, efforts, now)
                if hand.finger_cmd is not None:
                    z2 = np.zeros(2)
                    node.send_output(
                        "motor_command_gripper",
                        pack_motor_command(hand.finger_cmd, z2, z2, z2, z2),
                    )
                result = hand.fsm.poll(
                    hand.state, hand.gdone_count, hand.gdone_ok, now
                )
                if result is not None:
                    print(
                        f"[sim_bridge] hand grasp: ok={result['ok']} "
                        f"({result['reason']})",
                        flush=True,
                    )
                    node.send_output("grasp_result", pack_grasp_result(**result))
            elif fingers:
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
                state_template["position"][n_arm] = gripper_motor_echo
                state_template["torque"][n_arm] = tau_est
                cmd_g, result = grasp.step(
                    state_template, cmd_template, armed, False
                )
                if cmd_g is not None:
                    fingers_cmd = gripper_motor_to_fingers(
                        float(cmd_g["position"][n_arm]), mimic
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
        elif topic == f"motor_state_{actor_id}":
            state = unpack_motor_state(event["value"], n_arm)
            states_seen += 1
            if states_seen % _PUBLISH_EVERY == 0:
                pos, vel, tau = state_to_bridge(
                    state,
                    n_arm,
                    None if gripper_open is None
                    else (gripper_open if gripper_motor_echo is None
                          else gripper_motor_echo),
                )
                node.send_output(
                    "motor_state",
                    pack_motor_state(pos, vel, tau, zeros, zeros, zeros, zeros, zeros),
                )
            if states_seen % _STREAM_EVERY == 0:
                if states_seen - last_cmd_state > _PARK_GAP_STATES:
                    if park_q is None:
                        park_q = np.asarray(
                            state["position"], dtype=float
                        )[:n_arm].copy()
                    node.send_output(
                        f"motor_command_{actor_id}",
                        pack_motor_command(*park_command(park_q, park_kp, park_kd)),
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
