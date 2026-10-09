"""Dora MuJoCo backend for simulated plant state and command topics.

Drop-in successor to the removed ``drake_interface``; identical Dora contract:

- Composed scene (config ``scene`` block): one ``motor_command_<arm>`` input
  per ``scene.arm_slices`` entry, one ``motor_state_<arm>`` output per arm,
  ``topology_state`` + ``model_revision`` on topology change. NO grasp
  protocol: close/success/drop policy lives above the plant; this plant
  only senses (gripper_state positions+efforts) and reacts to its own world
  (fixture yields to grip force, keyed dock mate engages only on a release
  actually AT the seat — no magnets; the orchestrator's position-sensed
  verify gates the release).
- Per-arm idle decay (no command within ``idle_timeout_sec``) zeros that
  arm's gains; arms that never commanded keep the startup spawn hold.

Engine differences live in ``arm_control.plants.mujoco.backend``: welds
are physical (module rides the gripper, snaps into the dock basin) and the
optional viewer is MuJoCo's own (``sim_launch_viewer``, desk sessions).
"""
from __future__ import annotations

# ruff: noqa: E402

import time
import os

import numpy as np
from dora import Node


from arm_control.config import load_robot_config
from arm_control.messages import (
    pack_json_message,
    pack_motor_state,
    pack_scene_result,
    pack_scene_state,
    scene_state_payload,
    unpack_motor_command,
    unpack_json_message,
    unpack_scene_command,
)
from arm_control.node_utils import (
    ShutdownFlag,
    install_signal_handlers,
)
from arm_control.plants.mujoco.scene_core import SceneCore, build_world


def _without_schema(payload: dict) -> dict:
    body = dict(payload)
    body.pop("schema", None)
    return body


_undamped_warned = [False]


def _warn_undamped(command: dict) -> None:
    """Warn once when a command carries stiffness but no damping.

    The sim plant closes an explicit PD law, so kp > 0 with kd == 0 is an
    UNDAMPED second-order system: it rings at constant amplitude forever and
    never satisfies the executor's done() gate. That is a legitimate config on
    a real arm whose firmware or controller owns damping (the FR3 — libfranka
    derives it from the stiffness), which is exactly why it reaches the sim
    unnoticed. Cheap to say out loud, expensive to diagnose from a plot.
    """
    if _undamped_warned[0]:
        return
    kp = np.asarray(command.get("kp", ()), dtype=float)
    kd = np.asarray(command.get("kd", ()), dtype=float)
    if kp.size and float(np.max(np.abs(kp))) > 0.0 and not float(np.max(np.abs(kd))) > 0.0:
        _undamped_warned[0] = True
        print(
            "[mujoco_interface] WARNING: kp is non-zero but kd is ALL ZERO — the "
            "sim PD law has no damping and will oscillate without settling. Set "
            "arm.kd (or controller.kd) for the SIM config; a real arm whose "
            "controller owns damping still needs a damped value here.",
            flush=True,
        )


def main() -> None:
    cfg = load_robot_config()
    rate_hz = float(cfg.get("sim_update_rate_hz", cfg.update_rate_hz))
    period = 1.0 / rate_hz
    idle_timeout = float(cfg.get("idle_timeout_sec", 0.1))
    debug_forces = os.environ.get('MUJOCO_DEBUG_FORCES', '0') == '1'

    scene_cfg = cfg.get("scene")
    # The world and its per-tick rules (slices, deadmans, gripper, scene changes, stepping) live in plants/mujoco/scene_core,
    # shared with the supervisor's scene link. This node is its Dora front end: messages in, messages out, publication timers.
    backend, arm_slices, joint_names, n = build_world(cfg, period=period)
    core = SceneCore(backend, arm_slices, n, period=period, idle_timeout=idle_timeout)

    # Mirror the twin's ground truth (module/base/fixtures) into Rerun so ONE
    # viewer shows sim truth + plan previews. Fully thread-isolated: with no
    # viewer attached the gRPC sink blocks, and that must only ever park the
    # mirror thread — never this step loop.
    if cfg.get("sim_scene_rerun"):
        from arm_control.simulation.rerun_scene import start_mirror_thread

        # A GETTER, not a snapshot: a graft rebinds backend.model/.data, and a
        # mirror holding the originals would freeze the scene at the dock.
        start_mirror_thread(
            lambda: (backend.model, backend.data),
            exclude_prefixes=tuple(dict(scene_cfg or {}).get("arm_prefixes") or ()),
        )
        print("[mujoco_interface] twin ground truth mirrored to Rerun (sim/…)", flush=True)

    node = Node()
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)

    last_gain_sig: dict[str, tuple] = {}
    _plant_err_at = [0.0]
    last_step = 0.0
    last_qpos_pub = 0.0
    last_grip_pub = 0.0
    last_inhand_pub = 0.0
    last_wrench_pub = 0.0
    last_eqf_pub = 0.0
    last_scene_pub = 0.0
    model_revision_sent = False


    try:
        while not shutdown.stop_requested:
            event = node.next(timeout=period)
            if event is not None:
                etype = event["type"]
                eid = event.get("id", "")

                if etype == "INPUT" and eid == "arm" and not arm_slices:
                    core.set_single_armed(bool(unpack_json_message(event["value"]).get("armed", False)))
                elif etype == "INPUT" and eid == "motor_command_gripper":
                    # Finger servo targets (not an arm slice): drive the
                    # scene's gripper joints so closing is physical. A nonzero
                    # TORQUE slot means grasp(force) instead of move(width):
                    # close under that force cap and let contact stop the
                    # fingers. Reuses the motor_command torque field rather
                    # than adding a second gripper message.
                    gripper_cmd = unpack_motor_command(event["value"], 2)
                    core.apply_gripper(gripper_cmd["position"], force_n=max(abs(t) for t in gripper_cmd["torque"]))
                elif etype == "INPUT" and eid == "scene_command" and backend.scene_state:
                    request = unpack_scene_command(event["value"])
                    try:
                        payload = core.apply_scene(request)
                    except (KeyError, TypeError, ValueError) as exc:
                        node.send_output(
                            "scene_result",
                            pack_scene_result(
                                request_id=request["request_id"], revision=request["revision"],
                                ok=False, reason=str(exc),
                            ),
                        )
                    else:
                        node.send_output(
                            "scene_result",
                            pack_scene_result(
                                request_id=request["request_id"], revision=payload["revision"], ok=True,
                            ),
                        )
                        node.send_output("scene_state", pack_scene_state(**payload))
                elif etype == "INPUT" and eid.startswith("motor_command_"):
                    arm = eid.removeprefix("motor_command_")
                    if arm in arm_slices:
                        sub = unpack_motor_command(event["value"], arm_slices[arm]["n"])
                        core.set_slice(arm, sub, time.monotonic())
                        # What the plant is actually GIVEN, whenever it
                        # changes. Every servo diagnosis upstream of here is
                        # guesswork without it: the sender's gains and the
                        # applied gains are two different numbers separated by
                        # a bridge, a slice and a pack/unpack.
                        _sig = (
                            tuple(np.round(sub["kp"], 3)),
                            tuple(np.round(sub["kd"], 3)),
                            bool(np.any(np.abs(sub["torque"]) > 1e-9)),
                        )
                        if _sig != last_gain_sig.get(arm):
                            last_gain_sig[arm] = _sig
                            print(
                                f"[mujoco_interface] {arm} gains applied: "
                                f"kp={np.round(sub['kp'], 1).tolist()} "
                                f"kd={np.round(sub['kd'], 2).tolist()} "
                                f"|tau_ff|max={float(np.max(np.abs(sub['torque']))):.3f}",
                                flush=True,
                            )
                elif etype == "INPUT" and not arm_slices and eid == "motor_command":
                    single = unpack_motor_command(event["value"], n)
                    if not core.set_single(single, time.monotonic()):
                        continue  # disarmed: as before the move, this iteration neither steps nor publishes
                    _warn_undamped(single)
                elif etype == "STOP":
                    break

            now = time.monotonic()
            if now - last_step < period:
                continue
            # Advance the deadline by exactly one period rather than resetting it
            # to `now`: `node.next(timeout=period)` returns EARLY whenever an
            # input arrives, so the loop then notices the deadline a few ms late
            # and `last_step = now` folded that lateness into the next deadline.
            # Under steady command traffic that slipped a few percent every
            # cycle and the "sim time behind wall clock" debt grew without bound
            # even though the plant costs 0.14 ms of a 10 ms budget (measured,
            # ~70x realtime). Re-sync only after a REAL stall, so a genuinely
            # slow step never makes the loop spin trying to catch up.
            last_step += period
            if now - last_step > period:
                last_step = now

            # Deadmans (each slice's idle timeout, crediting time this node spent blocked; the single-model Soft expiry),
            # one step, and the sim-time-behind-wall accounting: SceneCore.tick.
            state = core.tick(now)
            command = core.command
            # The PLANT's own view of its steady error, straight from the
            # servo's inputs: no bridge, no decimation, no orchestrator. If
            # this reads ~0 while the orchestrator reports a residual, the
            # miss is in the state path, not the servo.
            if debug_forces and arm_slices and now - _plant_err_at[0] > 2.0:
                _plant_err_at[0] = now
                for _arm, _info in arm_slices.items():
                    _s, _m = _info["start"], _info["n"]
                    _v = np.max(np.abs(state["velocity"][_s : _s + _m]))
                    if _v > 0.01 or command["kp"][_s] <= 0.0:
                        continue  # moving, or the slice is limp: not a steady error
                    _e = command["position"][_s : _s + _m] - state["position"][_s : _s + _m]
                    # Decompose what BALANCES the servo at steady state.
                    # kp*err says how hard the servo pushes; these say what
                    # pushes back, by name, instead of by inference.
                    _d = backend.data
                    _v6 = backend._vadr[_s : _s + _m]
                    _bits = {
                        "applied": _d.qfrc_actuator[_v6],
                        "constraint": _d.qfrc_constraint[_v6],
                        "passive": _d.qfrc_passive[_v6],
                        "bias": _d.qfrc_bias[_v6],
                        "gravcomp": _d.qfrc_gravcomp[_v6],
                    }
                    print(
                        f"[mujoco_interface] {_arm} steady err (plant's own): "
                        f"max={np.max(np.abs(_e)):.5f} rad "
                        f"{np.round(_e, 4).tolist()}\n"
                        + "\n".join(
                            f"[mujoco_interface]     qfrc_{k:11s}"
                            f"{np.round(v, 2).tolist()}"
                            for k, v in _bits.items()
                        )
                        + "\n[mujoco_interface]     active equalities: "
                        + str({
                            _k: [round(float(x), 2) for x in _val[:3]]
                            for _k, _val in backend.active_constraint_reactions().items()
                        })
                        + f"\n[mujoco_interface]     ncon={_d.ncon} pairs="
                        + str(sorted({
                            tuple(sorted((
                                backend.model.body(
                                    backend.model.geom_bodyid[_d.contact[_c].geom1]).name,
                                backend.model.body(
                                    backend.model.geom_bodyid[_d.contact[_c].geom2]).name,
                            )))
                            for _c in range(_d.ncon)
                        }))
                        + " geoms=" + str(sorted({
                            tuple(sorted((backend.model.geom(c.geom1).name or f'geom{c.geom1}',
                                          backend.model.geom(c.geom2).name or f'geom{c.geom2}')))
                            for c in _d.contact
                        })),
                        flush=True,
                    )
            pos_cmd = np.zeros(n, dtype=np.float64)
            echoed = np.asarray(state.get("position_cmd", []), dtype=np.float64)
            pos_cmd[: min(n, echoed.size)] = echoed[: min(n, echoed.size)]

            if arm_slices:
                for arm, info in arm_slices.items():
                    s, m = info["start"], info["n"]
                    node.send_output(
                        f"motor_state_{arm}",
                        pack_motor_state(
                            state["position"][s : s + m],
                            state["velocity"][s : s + m],
                            pos_cmd[s : s + m],
                            command["velocity"][s : s + m],
                            command["torque"][s : s + m],
                            command["kp"][s : s + m],
                            command["kd"][s : s + m],
                            np.zeros(m),  # torque_fb (sim: n/a)
                        ),
                    )
            else:
                node.send_output(
                    "motor_state",
                    pack_motor_state(
                        state["position"],
                        state["velocity"],
                        pos_cmd,
                        command["velocity"],
                        command["torque"],
                        command["kp"],
                        command["kd"],
                        np.zeros(n),
                    ),
                )

            # Live scene feedback for the synthetic camera: sim_perception
            # mirrors this qpos into its own copy of the composed model, so
            # the sampled cloud TRACKS the physics (a static spawn snapshot
            # kept "seeing" the module at the fixture after pickup).
            # The scene state as a CURRENT-VALUE channel, not just an event.
            # Consumers start at revision 0 and can only learn the real one
            # from an echo; echoes used to follow a successful command, which
            # made the first command any consumer sent unsendable (it was
            # built on 0 and rejected as stale). The startup publish below
            # patches a cold start, but a node that comes up LATE -- or comes
            # back -- has already missed it and is stuck the same way. This is
            # the current value, always available, cheap: a handful of names
            # and ints every 2 s.
            if backend.scene_state is not None and now - last_scene_pub > 2.0:
                last_scene_pub = now
                try:
                    node.send_output(
                        "scene_state",
                        pack_scene_state(**scene_state_payload(backend.scene_state)),
                    )
                except Exception:
                    pass  # graphs without the output declared still run
            if backend.data is not None and now - last_qpos_pub > 0.2:
                last_qpos_pub = now
                try:
                    node.send_output(
                        "sim_qpos",
                        pack_json_message(
                            "sim_qpos", {"qpos": backend.data.qpos.tolist()}
                        ),
                    )
                except Exception:
                    pass  # graphs without the output declared still run
            # In-hand truth at 10 Hz: module pose in the EE frame, the sim
            # source for the orchestrator's slip monitor (bench source = the
            # wrist camera's end-cap tag re-read, same message).
            if scene_cfg and now - last_inhand_pub > 0.1:
                last_inhand_pub = now
                pose = backend.inhand_pose()
                if pose is not None:
                    try:
                        node.send_output(
                            "inhand_pose",
                            pack_json_message(
                                "inhand_pose", {"pose_xyzquat": pose}
                            ),
                        )
                    except Exception:
                        pass  # graphs without the output declared still run
            # Named equality reactions at 20 Hz. The plant reports WHICH
            # constraints are carrying load and how much; it never decides
            # what that means. A holder that lets go when the arm pulls hard
            # enough is scene knowledge, so the threshold and the release
            # both live with the consumer that owns the scene.
            if scene_cfg and now - last_eqf_pub > 0.05:
                last_eqf_pub = now
                try:
                    node.send_output(
                        "constraint_force",
                        pack_json_message(
                            "constraint_force",
                            {"reactions": backend.active_constraint_reactions()},
                        ),
                    )
                except Exception:
                    pass  # graphs without the output declared still run
            # Measured EE wrench at 20 Hz — the sim's stand-in for the FR3's
            # O_F_ext_hat_K (same frame, same sign). Published so contact
            # detection has a signal to grow into; nothing acts on it yet.
            if scene_cfg and now - last_wrench_pub > 0.05:
                last_wrench_pub = now
                try:
                    node.send_output(
                        "ee_wrench",
                        pack_json_message(
                            "ee_wrench",
                            {
                                "frame": "world",
                                "wrench": backend.ee_wrench().tolist(),
                            },
                        ),
                    )
                except Exception:
                    pass  # graphs without the output declared still run
            # Gripper SENSING at 50 Hz — its own cadence: the bridge's grasp
            # gate both ramps the close and thresholds the grip through this
            # stream (5 Hz would stretch the gate's close ramp to ~20 s).
            if backend.data is not None and now - last_grip_pub > 0.02:
                last_grip_pub = now
                finger_q = backend.gripper_positions()
                if finger_q.size:
                    try:
                        node.send_output(
                            "gripper_state",
                            pack_json_message(
                                "gripper_state",
                                {
                                    "positions": finger_q.tolist(),
                                    # Raw per-finger servo forces (N): the
                                    # grasp policy above the plant reads these —
                                    # the plant only senses.
                                    "efforts": backend.gripper_efforts().tolist(),
                                },
                            ),
                        )
                    except Exception:
                        pass  # graphs without the output declared still run

            if not model_revision_sent:
                try:
                    node.send_output(
                        "topology_state",
                        pack_json_message(
                            "topology_state", _without_schema(backend.topology_state())
                        ),
                    )
                except Exception:
                    pass  # graphs without the output declared still run
                node.send_output(
                    "model_revision",
                    pack_json_message(
                        "model_revision",
                        {
                            "revision": backend.model_revision,
                            "model_path": str(backend.model_path),
                            "joint_names": list(backend.joint_names),
                            "num_motors": n,
                            "supports_pose_hold": backend.supports_pose_hold,
                        },
                    ),
                )
                # The plant's scene revision, once, at startup. Every scene
                # consumer starts at revision 0 and learns the real one from
                # an echo -- but echoes were only sent AFTER a successful
                # command, so the first command any consumer sent was built
                # on 0 and the plant rejected it as stale. Chicken and egg:
                # you needed a successful scene_command to learn how to make
                # a successful scene_command.
                if backend.scene_state is not None:
                    try:
                        node.send_output(
                            "scene_state",
                            pack_scene_state(**scene_state_payload(backend.scene_state)),
                        )
                    except Exception:
                        pass  # graphs without the output declared still run
                model_revision_sent = True
    finally:
        backend.close()


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
