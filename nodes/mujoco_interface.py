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

import numpy as np
from dora import Node


from arm_control.config import gripper_joints, load_robot_config
from arm_control.messages import (
    pack_json_message,
    pack_motor_state,
    unpack_motor_command,
)
from arm_control.node_utils import ShutdownFlag, _zeros, install_signal_handlers
from arm_control.simulation.mujoco_backend import MuJoCoBackend
from arm_control.simulation.scene_backend import build_scene_backend, _resolve


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

    scene_cfg = cfg.get("scene")
    if scene_cfg:
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
        "position": _zeros(n),
        "velocity": _zeros(n),
        "torque": _zeros(n),
        # Gentle hold at the spawn pose until real commands arrive: with zero
        # gains the arm free-falls during graph startup and settles into a
        # tangle downstream planners then refuse to start from.
        "kp": _zeros(n) + 60.0,
        "kd": _zeros(n) + 2.0,
    }
    last_cmd_time: dict[str, float] = {arm: 0.0 for arm in arm_slices}
    last_step = 0.0
    last_qpos_pub = 0.0
    last_grip_pub = 0.0
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

    try:
        while not shutdown.stop_requested:
            event = node.next(timeout=period)
            if event is not None:
                etype = event["type"]
                eid = event.get("id", "")

                if etype == "INPUT" and eid == "motor_command_gripper":
                    # Finger servo targets (not an arm slice): drive the
                    # scene's gripper joints so closing is physical.
                    backend.apply_gripper_command(
                        unpack_motor_command(event["value"], 2)["position"]
                    )
                elif etype == "INPUT" and eid.startswith("motor_command_"):
                    arm = eid.removeprefix("motor_command_")
                    if arm in arm_slices:
                        info = arm_slices[arm]
                        s, m = info["start"], info["n"]
                        sub = unpack_motor_command(event["value"], m)
                        last_cmd_time[arm] = time.monotonic()
                        for key in ("position", "velocity", "torque", "kp", "kd"):
                            command[key][s : s + m] = sub[key]
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
                    _zero_slice(arm)

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
                model_revision_sent = True
    finally:
        backend.close()


if __name__ == "__main__":
    main()
