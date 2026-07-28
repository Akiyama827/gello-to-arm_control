"""Trajectory executor node: streams motor commands tracking a planned trajectory.

Thin Dora wrapper around ``JointTrajectoryExecutor`` (RNEA feedforward + PD
targets, the same engine the M1 orchestrator uses). Drake-free — runs on the
Jetson unchanged.

States: idle (hold a LATCHED pose — gravity ff + PD toward where the arm was
when idle began, re-latched if the arm is moved externally past
``hold_relatch_rad`` so a disarmed/repositioned arm never stores up a jump)
-> active (track the trajectory samples) -> past the end the executor keeps
stepping the clamped final sample, i.e. a stiff hold at the goal. An empty
trajectory (n_samples == 0) is the stop message: back to the latched hold.

The trajectory's joint columns map onto the FIRST n motor slots in order; the
remaining slots (Gripper) hold their measured position — unless a ``gripper``
input (2-finger ``motor_command_gripper`` format, finger metres) has set a
target, which is then held in the gripper motor slot via the joint_mimics
calibration.
"""
from __future__ import annotations

# ruff: noqa: E402

import time
from dataclasses import replace

import numpy as np
from dora import Node


from arm_control.config import _arm_block, arm_joints, load_robot_config
from arm_control.dynamics import PinocchioDynamics
from arm_control.execution.trajectory_executor import JointState, JointTrajectoryExecutor
from arm_control.joint_motor_map import gripper_finger_to_motor
from arm_control.messages import (
    pack_motor_command,
    unpack_motor_command,
    unpack_motor_state,
    unpack_trajectory,
)
from arm_control.planning.trajectory import JointTrajectory
from arm_control.node_utils import _load_mode_config, expand_named_values


def accept_trajectory(traj: dict, measured: np.ndarray, n_arm: int, tol: float) -> str | None:
    """Reason the trajectory must be refused, or None when it is safe to load."""
    nj = traj["positions"].shape[1]
    if nj != n_arm:
        return f"trajectory has {nj} joints, executor plans {n_arm}"
    err = float(np.max(np.abs(traj["positions"][0] - measured[:n_arm])))
    if err > tol:
        return (
            f"start point {err:.3f} rad from measured pose (tol {tol:g}) "
            "— re-plan from the current pose"
        )
    return None


def update_hold_pose(
    hold_q: np.ndarray | None, q_arm: np.ndarray, relatch_rad: float
) -> np.ndarray:
    """Latched idle hold: keep the anchor unless the arm was moved externally.

    Servoing to a latched pose (instead of chasing the live measured pose)
    is what makes the idle arm actually STAY PUT — chasing measured lets any
    imbalance integrate into drift (visible in frictionless sim). Re-latching
    past ``relatch_rad`` keeps the no-jump property: a disarmed arm that got
    hand-moved re-anchors instead of snapping back on arm.
    """
    if hold_q is None or float(np.max(np.abs(q_arm - hold_q))) > relatch_rad:
        return q_arm.copy()
    return hold_q


def tracking_error_exceeded(arm_cmd, measured: np.ndarray, limit: float) -> bool:
    """Runaway guard: true when the servo target has left the real arm behind.

    Catches an obstructed/stalled arm AND the execute-while-disarmed case
    (bridge swallows commands, targets advance anyway) — abort back to the
    compliant measured-pose hold instead of storing up a jump.
    """
    n_arm = len(arm_cmd.q_des)
    return float(np.max(np.abs(arm_cmd.q_des - measured[:n_arm]))) > limit


def merge_command(arm_cmd, measured: np.ndarray, n: int, kp: np.ndarray, kd: np.ndarray):
    """Compose the full n-motor command: arm servo + hold for the remaining slots."""
    n_arm = len(arm_cmd.q_des)
    q_des = measured.copy()
    q_des[:n_arm] = arm_cmd.q_des
    qd_des = np.zeros(n)
    qd_des[:n_arm] = arm_cmd.qd_des
    tau = np.zeros(n)
    tau[:n_arm] = arm_cmd.tau_ff
    kp_out = kp.copy()
    kp_out[:n_arm] = arm_cmd.kp
    kd_out = kd.copy()
    kd_out[:n_arm] = arm_cmd.kd
    return q_des, qd_des, tau, kp_out, kd_out


def main() -> None:
    cfg = load_robot_config()
    mode_cfg = _load_mode_config()
    controller_cfg = dict(mode_cfg.get("controller") or {})
    planner_cfg = dict(mode_cfg.get("planner") or {})
    names = list(cfg.joint_names or cfg.motor_names)
    n = cfg.num_motors
    # Default to the arm's own joint list rather than "all motors but the last":
    # that guess only holds for arms whose gripper is a trailing motor slot.
    arm_names = [str(j) for j in (planner_cfg.get("joints") or arm_joints(cfg))]
    if arm_names != names[: len(arm_names)]:
        raise ValueError(f"planner.joints {arm_names} must be a prefix of {names}")
    n_arm = len(arm_names)

    rate_hz = float(controller_cfg.get("command_rate_hz", cfg.update_rate_hz))
    period = 1.0 / rate_hz
    state_timeout = float(controller_cfg.get("state_timeout_sec", cfg.state_timeout_sec))
    start_tol = float(controller_cfg.get("start_pos_tol_rad", 0.1))
    abort_tol = float(controller_cfg.get("abort_pos_err_rad", 0.35))
    relatch_tol = float(controller_cfg.get("hold_relatch_rad", 0.3))
    # Gain CEILINGS are per-arm hardware facts, not policy: 500/5 are the DM MIT
    # wire-format limits (pack_mit_control_frame clips above them), while the
    # FR3 takes joint stiffness in N·m/rad and needs ~1200. Defaults keep the DM
    # path byte-identical; an arm that needs other bounds states them.
    kp_max = float(controller_cfg.get("kp_max", 500.0))
    kd_max = float(controller_cfg.get("kd_max", 5.0))
    tau_max = float(controller_cfg.get("torque_limit_max", 100.0))
    def _gains(key: str, arm_key: str, clamp_max: float) -> np.ndarray:
        """Per-motor gain vector from the mode config, else the arm's own table.

        Without the fallback a mode config keyed by ANOTHER arm's joint names
        expands to all-zeros — a limp arm on the bench, silently. The arm table
        (``arm.<arm_key>``) is arm-length (it says nothing about non-arm motor
        slots such as the DM gripper), so it is zero-padded up to the full motor
        list; those trailing slots are only ever set by an explicit mode-config
        entry.
        """
        spec = controller_cfg.get(key)
        if spec is None:
            spec = _arm_block(cfg).get(arm_key)
            if isinstance(spec, list) and len(spec) == n_arm < len(names):
                spec = list(spec) + [0.0] * (len(names) - n_arm)
        return expand_named_values(
            spec, names=names, default=0.0, clamp_min=0.0, clamp_max=clamp_max
        )

    kp = _gains("kp", "kp", kp_max)
    kd = _gains("kd", "kd", kd_max)
    tau_lim = _gains("torque_limits", "max_tau", tau_max)
    if not float(np.max(np.abs(kp[:n_arm]))) > 0.0:
        # All-zero arm stiffness is never intentional — it is a limp arm that
        # holds nothing. The usual cause is a mode config whose per-joint gain
        # keys name a DIFFERENT arm (expand_named_values falls back to 0.0 per
        # missing name), which a dataflow's per-node ARM_CONTROL_MODE_CONFIG can
        # pin behind your back. Fail here, not on the bench.
        raise ValueError(
            f"resolved kp is all zeros for joints {arm_names}. Check that "
            "controller.kp in the mode config is keyed by THESE joint names "
            "(ARM_CONTROL_MODE_CONFIG, including any per-node env: override in "
            "the dataflow), or drop it to inherit arm.kp from the arm config."
        )

    # Plants that compensate gravity themselves (FR3 control box) get RNEA
    # MINUS gravity — full RNEA would double-count it and push the arm up.
    plant_gc = bool(_arm_block(cfg).get("plant_gravity_comp", False))
    dynamics = PinocchioDynamics(cfg.urdf_path, arm_names)
    executor = JointTrajectoryExecutor(
        "arm",
        arm_names,
        dynamics,
        kp_default=kp[:n_arm],
        kd_default=kd[:n_arm],
        max_torque=tau_lim[:n_arm],
    )

    # First mimic entry = the gripper's motor slot, whatever the joint is named
    # on this arm (the DM arm calls it Gripper_1; another arm won't).
    mimic = next(
        (dict(m) for m in (cfg.get("joint_mimics") or {}).values() if isinstance(m, dict)),
        {},
    )
    state: dict[str, np.ndarray] | None = None
    hold_q: np.ndarray | None = None
    grip_q: float | None = None  # gripper motor target (rad); None = hold measured
    last_state_t = 0.0
    last_step = time.monotonic()
    node = Node()
    print(f"[trajectory_executor] ready — {n_arm} planned joints @ {rate_hz:g}Hz", flush=True)

    while True:
        event = node.next(timeout=period)
        now = time.monotonic()
        if event is not None:
            if event["type"] == "INPUT" and event["id"] == "motor_state":
                state = unpack_motor_state(event["value"], n)
                last_state_t = now
            elif event["type"] == "INPUT" and event["id"] in ("trajectory", "trajectory_replay"):
                traj = unpack_trajectory(event["value"])
                if len(traj["times"]) == 0:
                    executor.clear_trajectory()
                    hold_q = None  # latch wherever the arm is right now
                    print("[trajectory_executor] stop — holding at the stop pose", flush=True)
                elif state is None:
                    print("[trajectory_executor] REFUSED: no motor state yet", flush=True)
                else:
                    reason = accept_trajectory(traj, state["position"], n_arm, start_tol)
                    if reason is not None:
                        print(f"[trajectory_executor] REFUSED: {reason}", flush=True)
                    else:
                        hold_q = None
                        executor.load_trajectory(
                            JointTrajectory(
                                times=traj["times"],
                                positions=traj["positions"],
                                velocities=traj["velocities"],
                            ),
                            t_start=now,
                        )
                        print(
                            f"[trajectory_executor] tracking {len(traj['times'])} samples "
                            f"over {traj['times'][-1]:.2f}s",
                            flush=True,
                        )
            elif event["type"] == "INPUT" and event["id"] == "gripper":
                if n > n_arm and mimic:
                    finger_m = float(unpack_motor_command(event["value"], 2)["position"][0])
                    lo, hi = sorted(
                        (float(mimic["motor_open"]), float(mimic["motor_closed"]))
                    )
                    grip_q = float(
                        np.clip(gripper_finger_to_motor(finger_m, mimic), lo, hi)
                    )
                else:
                    print(
                        "[trajectory_executor] gripper input ignored: no gripper "
                        "slot / joint_mimics in config",
                        flush=True,
                    )
            elif event["type"] == "STOP":
                break

        if now - last_step < period:
            continue
        last_step = now
        if state is None or now - last_state_t > state_timeout:
            continue

        q = state["position"]
        arm_state = JointState(position=q[:n_arm], velocity=state["velocity"][:n_arm])
        arm_cmd = None
        if executor.has_trajectory:
            arm_cmd = executor.step(now, arm_state)
            if tracking_error_exceeded(arm_cmd, q, abort_tol):
                executor.clear_trajectory()
                arm_cmd = None
                hold_q = None  # re-anchor at the measured pose
                print(
                    f"[trajectory_executor] ABORT: tracking error > {abort_tol:g} rad "
                    "(obstructed or disarmed?) — compliant hold at measured pose",
                    flush=True,
                )
        if arm_cmd is None:
            # Idle: the executor's hold primitive at the latched anchor
            # (gravity FF + torque clamp — the same anchored hold every other
            # tier uses; this node used to hand-roll it inline).
            hold_q = update_hold_pose(hold_q, arm_state.position, relatch_tol)
            arm_cmd = executor.hold_command(
                JointState(position=hold_q, velocity=np.zeros(n_arm))
            )
        if plant_gc:
            arm_cmd = replace(
                arm_cmd, tau_ff=arm_cmd.tau_ff - dynamics.gravity(arm_state.position)
            )
        q_des, qd_des, tau, kp_out, kd_out = merge_command(arm_cmd, q, n, kp, kd)
        if grip_q is not None:
            q_des[n_arm] = grip_q  # gripper motor slot: teleop target, not measured
        node.send_output(
            "motor_command", pack_motor_command(q_des, qd_des, tau, kp_out, kd_out)
        )


if __name__ == "__main__":
    main()
