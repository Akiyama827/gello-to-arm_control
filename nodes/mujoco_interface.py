"""Dora MuJoCo backend for simulated plant state and command topics.

Drop-in successor to the removed ``drake_interface``; identical Dora contract:

- Composed scene (config ``scene`` block): one ``motor_command_<arm>`` input
  per ``scene.arm_slices`` entry, one ``motor_state_<arm>`` output per arm,
  ``topology_state`` + ``model_revision`` on topology change. NO grasp
  protocol: the bridge's GraspGate owns close/success/drop policy; this plant
  only senses (gripper_state positions+efforts) and reacts to its own world
  (fixture yields to grip force, keyed dock mate engages only on a release
  actually AT the seat — no magnets; the orchestrator's position-sensed
  verify gates the release).
- Per-arm idle decay (no command within ``idle_timeout_sec``) zeros that
  arm's gains; arms that never commanded keep the startup spawn hold.

Engine differences live in ``arm_control.simulation.mujoco_backend``: welds
are physical (module rides the gripper, snaps into the dock basin) and the
optional viewer is MuJoCo's own (``sim_launch_viewer``, desk sessions).
"""
from __future__ import annotations

# ruff: noqa: E402

import time
import os
import importlib

import numpy as np
from dora import Node


from arm_control.config import gripper_joints, load_robot_config
from arm_control.messages import (
    pack_json_message,
    pack_motor_state,
    pack_scene_result,
    pack_scene_state,
    unpack_motor_command,
    unpack_scene_command,
)
from arm_control.node_utils import ShutdownFlag, _zeros, install_signal_handlers
from arm_control.simulation.mujoco_backend import MuJoCoBackend
from arm_control.simulation.scene_backend import (
    _resolve,
    build_scene_backend,
    build_workcell_backend,
)
from arm_control.scene import Attachment, SceneState


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


def _scene_state_from_payload(
    current: SceneState, payload: dict, revision: int | None = None
) -> SceneState:
    """``revision`` is a SIBLING of ``state`` on a scene_command
    (``pack_scene_command`` puts it there), not a member of it, so a command's
    state dict carries none and the caller must pass it. Only a scene_STATE
    echo, where revision does sit inside, may leave it out."""
    attachments = {
        name: Attachment(**value) for name, value in dict(payload.get("attachments", {})).items()
    }
    return SceneState(
        actor_q={name: list(values) for name, values in dict(payload.get("actor_q", current.actor_q)).items()},
        attachments=attachments,
        constraints={name: bool(value) for name, value in dict(payload.get("constraints", current.constraints)).items()},
        revision=int(payload["revision"] if revision is None else revision),
    )


def _scene_state_payload(state: SceneState) -> dict:
    return {
        "revision": state.revision,
        "actor_q": state.actor_q,
        "attachments": {
            name: {
                "object_name": item.object_name, "body": item.body,
                "parent_frame": item.parent_frame, "child_frame": item.child_frame,
                "mate_pose": item.mate_pose,
            }
            for name, item in state.attachments.items()
        },
        "constraints": state.constraints,
    }


def main() -> None:
    cfg = load_robot_config()
    rate_hz = float(cfg.get("sim_update_rate_hz", cfg.update_rate_hz))
    period = 1.0 / rate_hz
    idle_timeout = float(cfg.get("idle_timeout_sec", 0.1))

    scene_path = os.environ.get("WORKCELL_SCENE")
    scene_cfg = cfg.get("scene")
    if scene_path:
        loader = None
        loader_name = os.environ.get("WORKCELL_LOADER")
        if loader_name:
            module_name, function_name = loader_name.split(":", 1)
            loader = getattr(importlib.import_module(module_name), function_name)
        backend, arm_slices, joint_names, n = build_workcell_backend(
            scene_path,
            control_period=period,
            launch_viewer=bool(cfg.get("sim_launch_viewer", False)),
            enable_self_collision=bool(cfg.get("sim_self_collision", False)),
            loader=loader,
        )
        print(
            f"[mujoco_interface] generic workcell: {n} actuators, slices={arm_slices}",
            flush=True,
        )
    elif scene_cfg:
        backend, arm_slices, joint_names, n = build_scene_backend(
            scene_cfg,
            control_period=period,
            launch_viewer=bool(cfg.get("sim_launch_viewer", False)),
            enable_self_collision=bool(cfg.get("sim_self_collision", False)),
        )
        print(
            f"[mujoco_interface] composed scene: {n} actuators, slices={arm_slices}",
            flush=True,
        )
    else:
        # Single-model mode (view/motion graphs): one MJCF/URDF, plain
        # ``motor_command`` in / ``motor_state`` out, no welds, no ground.
        joint_names = list(cfg.joint_names)
        n = len(joint_names) if joint_names else cfg.num_motors
        arm_slices = {}
        backend = MuJoCoBackend(
            joint_names=joint_names,
            single_model_path=_resolve(str(cfg.get("sim_model_path", ""))),
            timestep=float(cfg.get("sim_timestep", 0.001)),
            control_period=period,
            ground_z=None,
            launch_viewer=bool(cfg.get("sim_launch_viewer", False)),
            # Single-model joints carry no scene prefix; servo the fingers so
            # they hold instead of free-sliding (and take gripper commands).
            # From the arm's OWN config (`arm.gripper_joints`) — a new robot is a
            # new YAML, never an edit here.
            gripper_joints=tuple(gripper_joints(cfg)),
        )
        print(f"[mujoco_interface] single model: {backend.model_path}", flush=True)
    backend.load()

    # THE PLANT DECLARES THE STARTING STATE. On the bench the arm is wherever
    # it is and the host reads it; in the twin the plant spawns it somewhere
    # defined. MuJoCo's implicit zero is not a declaration, it is an accident --
    # and for the FR3's Hand zero means SHUT, so every graph started with the
    # jaws closed around whatever they were parked over. Measured: at the bench
    # grasp pose the shut jaws bury 29.85 mm into the module, the contact
    # solver pushes back, and the approach settles ~0.039 rad short of target
    # forever (the planner never sees it -- it checks collisions at the grasp
    # profile's finger width, not the plant's actual one). Open to the arm's
    # own configured width, exactly as run_workcell_add_demo.py does.
    # NB the count comes from the PLANT, not gripper_joints(cfg): that list is
    # deliberately EMPTY for an arm whose gripper is its own device (the FR3's
    # Hand), while the composed scene still has the finger joints.
    _fingers = int(backend.gripper_positions().size)
    # Same accessor sim_bridge uses for this block (the gripper config sits
    # under the robot key, not the arm block).
    _gcfg = dict((cfg.get("franka") or {}).get("gripper") or {})
    _open_w = float(_gcfg.get("open_width_m", 0.0))
    if _fingers and _open_w > 0:
        backend.apply_gripper_command([_open_w / _fingers] * _fingers)
        print(
            f"[mujoco_interface] jaws spawned OPEN at {_open_w * 1000:.0f} mm",
            flush=True,
        )

    # Mirror the twin's ground truth (module/base/fixtures) into Rerun so ONE
    # viewer shows sim truth + plan previews. Fully thread-isolated: with no
    # viewer attached the gRPC sink blocks, and that must only ever park the
    # mirror thread — never this step loop.
    if cfg.get("sim_scene_rerun"):
        from arm_control.simulation.rerun_scene import start_mirror_thread

        start_mirror_thread(
            backend.model,
            backend.data,
            exclude_prefixes=tuple(dict(scene_cfg or {}).get("arm_prefixes") or ()),
        )
        print("[mujoco_interface] twin ground truth mirrored to Rerun (sim/…)", flush=True)

    node = Node()
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)

    command = {
        # The SPAWN POSE, read back from the plant — not zeros. The comment
        # below has always said "hold at the spawn pose"; the code held at
        # q=0, which for any arm whose spawn pose is not the origin is not a
        # hold at all but a full-authority move to the origin. Measured on the
        # FR3 scene: the arm slams off its ready pose at up to 30 rad/s during
        # the pre-arm window, and whether it has wrecked the scene by the time
        # the operator arms is a race against sim_bridge's park taking over.
        # That is the "start pose is in collision per the plan world" the
        # planner intermittently refused on. Pre-existing, law-independent
        # (the old Python PD does it too) — found while A/B-ing the servo law.
        "position": backend.motor_state()["position"].copy(),
        "velocity": _zeros(n),
        "torque": _zeros(n),
        # Gentle hold at the spawn pose until real commands arrive: with zero
        # gains the arm free-falls during graph startup and settles into a
        # tangle downstream planners then refuse to start from.
        "kp": _zeros(n) + 60.0,
        "kd": _zeros(n) + 2.0,
        # Optional Cartesian-impedance target + task-frame K_c/D_c. None until
        # a command carries one; None = joint-space PD only, i.e. today.
        "cartesian": None,
    }
    last_cmd_time: dict[str, float] = {arm: 0.0 for arm in arm_slices}
    idle_warned: dict[str, bool] = {arm: False for arm in arm_slices}
    last_step = 0.0
    last_qpos_pub = 0.0
    last_grip_pub = 0.0
    last_inhand_pub = 0.0
    last_wrench_pub = 0.0
    last_eqf_pub = 0.0
    _slow_warned_at = [0.0]
    _slow_debt = [0.0]   # debt at last warning (warn only on NEW drift)
    _sim_steps = [0]
    _wall_t0 = [0.0]
    model_revision_sent = False

    def _zero_slice(arm: str) -> None:
        info = arm_slices[arm]
        s, m = info["start"], info["n"]
        command["torque"][s : s + m] = 0.0
        command["kp"][s : s + m] = 0.0
        command["kd"][s : s + m] = 0.0
        # The Cartesian spring decays WITH the joint gains. Leaving it live
        # while kp/kd go to zero is the worst of both: a limp arm still being
        # pulled toward a target nobody is refreshing.
        command["cartesian"] = None

    try:
        while not shutdown.stop_requested:
            event = node.next(timeout=period)
            if event is not None:
                etype = event["type"]
                eid = event.get("id", "")

                if etype == "INPUT" and eid == "motor_command_gripper":
                    # Finger servo targets (not an arm slice): drive the
                    # scene's gripper joints so closing is physical. A nonzero
                    # TORQUE slot means grasp(force) instead of move(width):
                    # close under that force cap and let contact stop the
                    # fingers. Reuses the motor_command torque field rather
                    # than adding a second gripper message.
                    gripper_cmd = unpack_motor_command(event["value"], 2)
                    backend.apply_gripper_command(
                        gripper_cmd["position"],
                        force_n=float(max(abs(t) for t in gripper_cmd["torque"])),
                    )
                elif etype == "INPUT" and eid == "scene_command" and backend.scene_state:
                    request = unpack_scene_command(event["value"])
                    try:
                        if request["revision"] <= backend.scene_state.revision:
                            raise ValueError("stale scene revision")
                        next_state = _scene_state_from_payload(
                            backend.scene_state, request["state"], request["revision"]
                        )
                        backend.apply_scene_state(next_state)
                    except (KeyError, TypeError, ValueError) as exc:
                        node.send_output(
                            "scene_result",
                            pack_scene_result(
                                request_id=request["request_id"], revision=request["revision"],
                                ok=False, reason=str(exc),
                            ),
                        )
                    else:
                        payload = _scene_state_payload(backend.scene_state)
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
                        info = arm_slices[arm]
                        s, m = info["start"], info["n"]
                        sub = unpack_motor_command(event["value"], m)
                        last_cmd_time[arm] = time.monotonic()
                        for key in ("position", "velocity", "torque", "kp", "kd"):
                            command[key][s : s + m] = sub[key]
                        # Cartesian impedance is an EE-level term, so it has no
                        # slice — the arm that owns the configured EE body owns
                        # it. Last writer wins; in every scene we run, exactly
                        # one arm ever sends one.
                        command["cartesian"] = sub["cartesian"]
                elif etype == "INPUT" and not arm_slices and eid == "motor_command":
                    command = unpack_motor_command(event["value"], n)
                    _warn_undamped(command)
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

            for arm in arm_slices:
                if last_cmd_time[arm] > 0.0 and now - last_cmd_time[arm] > idle_timeout:
                    # Say it out loud. A silent slice going limp looks
                    # downstream like a servo that cannot track: the arm sags
                    # off its target and whoever measures next reads the sag
                    # as tracking error. Once per gap, not per tick.
                    if not idle_warned[arm]:
                        idle_warned[arm] = True
                        print(
                            f"[mujoco_interface] {arm}: no command for "
                            f"{now - last_cmd_time[arm]:.2f} s — zeroing gains "
                            f"(idle_timeout_sec={idle_timeout})",
                            flush=True,
                        )
                    _zero_slice(arm)
                elif last_cmd_time[arm] > 0.0:
                    idle_warned[arm] = False

            state = backend.step(command)
            # Warn on CUMULATIVE drift, not single-step spikes: viewer sync
            # steps legitimately take ~14 ms 15x/s while the physics has
            # ample headroom (0.04 ms/step measured) and queued timer ticks
            # drain back-to-back afterwards — per-step warnings cried wolf
            # every second. True lag = wall elapsed vs sim time advanced.
            _sim_steps[0] += 1
            if _wall_t0[0] == 0.0:
                _wall_t0[0] = now
            debt = (now - _wall_t0[0]) - _sim_steps[0] * period
            if (
                debt > 0.5
                and debt - _slow_debt[0] > 0.25
                and now - _slow_warned_at[0] > 5.0
            ):
                _slow_warned_at[0] = now
                _slow_debt[0] = debt
                print(
                    f"[mujoco_interface] sim time {debt:.1f} s behind wall "
                    "clock and falling further (plant-clock pacing keeps "
                    "execution safe; motion plays in slow motion)",
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
                                    # bridge's grasp gate thresholds these —
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
                            pack_scene_state(**_scene_state_payload(backend.scene_state)),
                        )
                    except Exception:
                        pass  # graphs without the output declared still run
                model_revision_sent = True
    finally:
        backend.close()


if __name__ == "__main__":
    main()
