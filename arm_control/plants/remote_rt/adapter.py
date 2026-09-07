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

from dora import Node

from arm_control.config import arm_joints, ee_frame, load_robot_config
from arm_control.hardware.rt_backend import RtBackend, RtLinkError
from arm_control.messages import (
    pack_json_message,
    pack_motor_state_dict as _pack,
    unpack_json_message,
    unpack_motor_command,
)
from arm_control.node_utils import ShutdownFlag, install_signal_handlers


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
    drop_warned = False
    last_fresh = True
    last_cmd_sent = 0.0
    last_ack_seq = -1
    last_ack_t = 0.0
    ack_warned = False

    try:
        backend.open()
        print(
            f"[rt_interface] bridge up: '{backend.backend_name}' via "
            f"{backend.config.host}, {backend.num_motors} joints, DISARMED",
            flush=True,
        )
        while not shutdown.stop_requested:
            # Block IN dora (GIL released) instead of poll-then-sleep: the old
            # `next(0)` + `time.sleep(2ms)` pattern added a measured mean
            # ~1 ms / worst ~2 ms to EVERY command's forwarding latency —
            # the single largest avoidable term in the whole command path.
            event = node.next(timeout=period * 0.2)
            if event is not None:
                etype, eid = event["type"], event.get("id", "")
                if etype == "STOP":
                    break
                if etype == "INPUT" and eid == "motor_command":
                    if armed_wanted:
                        t_cmd = time.perf_counter()
                        if last_cmd_sent and t_cmd - last_cmd_sent > 0.3:
                            # The other half of the CMD_LOST forensics: a gap
                            # HERE means commands stopped ARRIVING from the
                            # executor (delivery/backpressure), not sending.
                            print(
                                f"[rt_interface] command gap "
                                f"{t_cmd - last_cmd_sent:.2f}s (upstream paused?)",
                                flush=True,
                            )
                        last_cmd_sent = t_cmd
                        backend.apply_command(
                            unpack_motor_command(event["value"], backend.num_motors)
                        )
                    elif not drop_warned:
                        # Never swallow silently — but say it ONCE per disarm
                        # episode: the executor streams its hold continuously,
                        # and a repeating line is noise that buries signal.
                        drop_warned = True
                        print(
                            "[rt_interface] dropping motor_command — DISARMED "
                            "(ARM on the teleop page grants authority)",
                            flush=True,
                        )
                elif etype == "INPUT" and eid == "arm":
                    if bool(unpack_json_message(event["value"]).get("armed", False)):
                        try:
                            backend.enable_all()
                            armed_wanted = True
                            drop_warned = False
                            last_cmd_sent = 0.0  # don't count the disarmed era
                            print("[rt_interface] ARMED (server holds until commands flow)", flush=True)
                        except RtLinkError as exc:
                            print(f"[rt_interface] arm refused: {exc}", flush=True)
                    else:
                        armed_wanted = False
                        drop_warned = False  # next disarm episode warns once again
                        # safe_stop VERIFIES the drop (ack + state stream);
                        # printing DISARMED unconditionally taught operators
                        # to trust a line that could be false while the server
                        # held at full stiffness (audit 2026-07-29).
                        if backend.safe_stop() or backend.safe_stop():
                            print("[rt_interface] DISARMED (confirmed)", flush=True)
                        else:
                            print(
                                "[rt_interface] DISARM NOT CONFIRMED — server may "
                                "still be armed; do not approach the arm",
                                flush=True,
                            )
                elif etype == "INPUT" and eid == "active_mask":
                    try:
                        backend.set_active_mask(
                            int(unpack_json_message(event["value"])["mask"])
                        )
                        print(
                            f"[rt_interface] active slots 0x"
                            f"{backend.motor_health()['active_mask']:04X}",
                            flush=True,
                        )
                    except (KeyError, TypeError, ValueError, RuntimeError, RtLinkError) as exc:
                        print(f"[rt_interface] active-mask refused: {exc}", flush=True)

            now = time.perf_counter()
            if now - last_step < period:
                continue
            last_step = now
            step_count += 1

            health = backend.motor_health()
            # A server that receives our stream but REJECTS every packet
            # (joint-count mismatch, flow pinned to another port) holds
            # silently with no fault: the ONE observable is the acked seq
            # freezing while our sent seq advances. Say it.
            if armed_wanted and health["armed"] and health["state_fresh"]:
                if health["last_cmd_seq"] != last_ack_seq:
                    last_ack_seq = health["last_cmd_seq"]
                    last_ack_t = now
                    ack_warned = False
                elif (
                    not ack_warned
                    and last_cmd_sent
                    and health["sent_cmd_seq"] - last_ack_seq > rate_hz
                    and now - last_ack_t > 1.0
                ):
                    ack_warned = True
                    print(
                        f"[rt_interface] server is IGNORING our commands "
                        f"(acked seq stuck at {last_ack_seq}, sent "
                        f"{health['sent_cmd_seq']}) — joint count or command "
                        "flow pin mismatch?",
                        flush=True,
                    )
            if health["state_fresh"] != last_fresh:
                # Transitions are logged, not the steady state: a silent gate
                # here made a server CMD_LOST latch undiagnosable on rung 3
                # (commands stop when state stops — the cause needs a line).
                last_fresh = health["state_fresh"]
                print(
                    "[rt_interface] RT state stream "
                    + ("recovered" if last_fresh
                       else f"STALE ({health['state_age_s']:.2f}s) — gating "
                            "motor_state; executor will pause commands"),
                    flush=True,
                )
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


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
