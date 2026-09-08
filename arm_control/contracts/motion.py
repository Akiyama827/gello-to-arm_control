"""Trajectory, planner/controller, settings, and jog wire contracts."""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from ._arrow import _pack, _unpack, _check_length, pack_json_message, unpack_json_message


def pack_trajectory(times, positions, velocities) -> pa.Array:
    """Pack a sampled joint trajectory: [n_samples, n_joints, t..., q..., qd...].

    The n_joints columns map onto the FIRST n_joints motor slots in order
    (Joint1..Joint6); remaining slots (Gripper) are held by the executor.
    n_samples == 0 is the stop/hold message.
    """
    times = np.asarray(times, dtype=np.float64).ravel()
    q = np.asarray(positions, dtype=np.float64).reshape(len(times), -1) if len(times) else np.zeros((0, 0))
    qd = np.asarray(velocities, dtype=np.float64).reshape(q.shape) if len(times) else q
    header = np.array([q.shape[0], q.shape[1]], dtype=np.float64)
    return _pack(np.concatenate([header, times, q.ravel(), qd.ravel()]))


def unpack_trajectory(arrow: pa.Array) -> dict:
    flat = _unpack(arrow)
    if flat.size < 2:
        raise ValueError("unpack_trajectory: missing header")
    n, j = int(flat[0]), int(flat[1])
    _check_length("unpack_trajectory", flat, 2 + n + 2 * n * j)
    return {
        "times": flat[2 : 2 + n],
        "positions": flat[2 + n : 2 + n + n * j].reshape(n, j),
        "velocities": flat[2 + n + n * j :].reshape(n, j),
    }


def _completion_body(completion) -> dict | None:
    """Validate the leg-completion rule. Unknown keys are a caller bug.

    Silently dropping a misspelled tolerance would leave the leg judged by a
    default the caller never chose -- and the failure mode is a leg that
    completes while still moving.
    """
    if completion is None:
        return None
    unknown = set(completion) - {"vel_tol", "goal_q", "residual_max"}
    if unknown:
        raise ValueError(f"pack_plan: unknown completion key(s) {sorted(unknown)}")
    goal = completion.get("goal_q")
    if goal is not None and completion.get("residual_max") is None:
        raise ValueError("pack_plan: completion goal_q needs a residual_max")
    return {
        "vel_tol": float(completion["vel_tol"]),
        "goal_q": None if goal is None
        else np.asarray(goal, dtype=float).ravel().tolist(),
        "residual_max": (
            None if completion.get("residual_max") is None
            else float(completion["residual_max"])
        ),
    }


def pack_plan(
    *,
    plan_id: str,
    phase: str,
    gated: bool,
    times,
    positions,
    velocities,
    kp,
    kd,
    cartesian_poses=None,
    cartesian=None,
    completion=None,
) -> pa.Array:
    """One planned leg, planner -> controller.

    JSON, not a packed float layout, and deliberately so: this crosses the wire
    ONCE PER LEG (a long 7-DOF leg is a few thousand floats, ~50 kB), never per
    tick, so the 1 kHz-stream argument for a binary layout does not apply and a
    self-describing message is worth far more here -- every field below is one
    the controller must not have to guess.

    ``gated`` is the operator contract: a gated plan is loaded for REVIEW and
    must not run until an ``execute`` naming this exact ``plan_id`` arrives.

    ``completion`` carries the leg-completion RULE with the leg, so the
    controller never has to know which leg is special -- or what makes it
    special. Absent (the default) means the executor's own done(): the plan
    played out and the arm is at the target. Present, it is the SETTLE rule --
    the plan played out, the arm has stopped to ``vel_tol``, and (if
    ``goal_q`` is given) it stopped within ``residual_max`` of it. The caller
    picks those numbers because the caller is the one that knows why this leg
    cannot be judged on joint residual.
    """
    times = np.asarray(times, dtype=float).ravel()
    q = np.asarray(positions, dtype=float).reshape(len(times), -1)
    qd = np.asarray(velocities, dtype=float).reshape(q.shape)
    body = {
        "plan_id": str(plan_id),
        "phase": str(phase),
        "gated": bool(gated),
        "times": times.tolist(),
        "positions": q.tolist(),
        "velocities": qd.tolist(),
        "kp": np.asarray(kp, dtype=float).ravel().tolist(),
        "kd": np.asarray(kd, dtype=float).ravel().tolist(),
        "completion": _completion_body(completion),
        # FK of every sample, precomputed in WORLD. The controller owns no
        # kinematics by design, and the Cartesian impedance target is exactly
        # FK(q_des) -- so it ships with the samples rather than being re-derived
        # downstream from a model the controller is not allowed to load.
        "cartesian_poses": (
            None if cartesian_poses is None
            else np.asarray(cartesian_poses, dtype=float).reshape(len(times), 7).tolist()
        ),
        "cartesian": (
            None if cartesian is None
            else {
                "task_R": np.asarray(cartesian["task_R"], dtype=float).reshape(3, 3).tolist(),
                "kc": np.asarray(cartesian["kc"], dtype=float).ravel().tolist(),
                "dc": np.asarray(cartesian["dc"], dtype=float).ravel().tolist(),
            }
        ),
    }
    return pack_json_message("plan", body)


def unpack_plan(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema="plan")
    n = len(body["times"])
    for key in ("positions", "velocities"):
        if len(body[key]) != n:
            raise ValueError(f"unpack_plan: {key} has {len(body[key])} rows, want {n}")
    for key in ("times", "positions", "velocities", "kp", "kd"):
        body[key] = np.asarray(body[key], dtype=float)
    if body.get("cartesian_poses") is not None:
        body["cartesian_poses"] = np.asarray(body["cartesian_poses"], dtype=float)
    if body.get("completion") is not None:
        spec = body["completion"]
        if spec.get("goal_q") is not None:
            spec["goal_q"] = np.asarray(spec["goal_q"], dtype=float)
    if body.get("cartesian") is not None:
        body["cartesian"] = {
            k: np.asarray(v, dtype=float) for k, v in body["cartesian"].items()
        }
    return body


def pack_control_update(**fields) -> pa.Array:
    """Control-plane updates that are not a leg: arm, payload, execute, cancel,
    gains, pose_hold, hold, stop.

    NOT ``rt_protocol.pack_control``, which is the RT server's binary link
    frame (ctl_type / seq / t_mono_ns) and has nothing to do with this. Same
    package, unrelated jobs -- hence the longer name here.

    One topic rather than five, because they are all the same thing -- the
    planner telling the controller something about how to behave that is not a
    trajectory -- and because they must arrive in ORDER relative to each other
    (a payload declaration that overtook the execute it belongs to would servo
    one leg with the wrong feedforward). Absent keys mean "unchanged".
    """
    # `cancel`, `hold` and `stop` are three DIFFERENT things and the names are
    # worth keeping apart: cancel aborts a leg and stays armed (an operator's
    # Stop button), hold freezes at a milestone forever, stop is terminal.
    known = {"arm", "payload", "execute", "cancel", "gains", "hold", "stop", "reason", "pose_hold"}
    unknown = set(fields) - known
    if unknown:
        raise ValueError(f"pack_control_update: unknown field(s) {sorted(unknown)}")
    return pack_json_message("control", dict(fields))


def unpack_control_update(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema="control")
    body.pop("schema", None)
    return body


def _mode_state_body(state) -> dict:
    if not isinstance(state, dict) or state.get("law") not in ("joint", "soft"):
        raise ValueError("mode_state requires law joint or soft")
    kp = np.asarray(state.get("kp"), dtype=float)
    kd = np.asarray(state.get("kd"), dtype=float)
    if (kp.ndim != 1 or not kp.size or kd.shape != kp.shape
            or not np.isfinite(kp).all() or not np.isfinite(kd).all()):
        raise ValueError("mode_state requires matching finite gain arrays")
    return {"law": state["law"], "kp": kp.tolist(), "kd": kd.tolist()}


def pack_controller_event(
    *, kind: str, plan_id: str = "", ok: bool = True, reason: str = "", q=None,
    mode_state=None,
) -> pa.Array:
    """Controller -> planner. ``kind`` is one of:

    ``ready``      the plant is armed and reporting; ``q`` is the first fresh
                   measured pose, which is where the sequence must start from.
    ``leg_result`` ``plan_id`` finished (``ok``) or gave up (``reason``).
    ``fault``      the controller stopped driving; nothing else will move.
    ``mode``       accepted ``soft``/``joint`` in reason, or ok=False refusal.

    Every result carries its ``plan_id`` so a late reply from a superseded plan
    is dropped rather than credited to the current one.

    Optional ``mode_state`` reports the actual law and executor joint gains;
    old senders and receivers may omit it. Mode heartbeats use the same event.
    """
    if kind not in {"ready", "leg_result", "fault", "mode"}:
        raise ValueError(f"pack_controller_event: unknown kind {kind!r}")
    return pack_json_message(
        "controller_event",
        {
            "kind": kind,
            "plan_id": str(plan_id),
            "ok": bool(ok),
            "reason": str(reason),
            "q": None if q is None else np.asarray(q, dtype=float).ravel().tolist(),
            **({} if mode_state is None else {"mode_state": _mode_state_body(mode_state)}),
        },
    )


def unpack_controller_event(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema="controller_event")
    body.pop("schema", None)
    if body.get("q") is not None:
        body["q"] = np.asarray(body["q"], dtype=float)
    if "mode_state" in body:
        body["mode_state"] = _mode_state_body(body["mode_state"])
    return body


def pack_controller_settings(payload: dict) -> pa.Array:
    return pack_json_message("controller_settings", payload)


def unpack_controller_settings(arrow: pa.Array) -> dict:
    body = unpack_json_message(arrow, expected_schema="controller_settings")
    body.pop("schema", None)
    return body




# --------------------------------------------------------------------------- #
# jog — a live joint setpoint from an operator holding a direction, at whatever
# rate the console ticks. Deliberately NOT a trajectory and NOT a plan: it has
# no duration, no review, and no completion. It EXPIRES, which is the whole
# safety property (see ArmController.on_jog), so nothing here carries a
# timestamp -- the receiver stamps arrival, because a clock the sender controls
# is a clock a wedged sender can lie about.
# --------------------------------------------------------------------------- #


def pack_jog(*, q, reason: str = "") -> pa.Array:
    """One joint-space setpoint for the arm joints, in order."""
    return pack_json_message(
        "jog",
        {"q": [float(v) for v in np.asarray(q, dtype=float).ravel()],
         "reason": str(reason)},
    )


def unpack_jog(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="jog")
