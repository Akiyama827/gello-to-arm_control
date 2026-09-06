"""Arrow packing and unpacking for Dora inter-node communication.

All messages are flat float64 Arrow arrays. The layouts intentionally match
the caller's historical ``nodes/schemas.py`` functions.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

_MS = 8  # pos, vel, pos_cmd, vel_cmd, tor_cmd, kp, kd, tor_fb
_MC = 5
# Optional Cartesian-impedance tail on motor_command: 7 target pose
# [x,y,z,qw,qx,qy,qz] + 9 task-frame rotation (3x3 ROW-MAJOR) + 6 K_c + 6 D_c.
# Appended AFTER the n*_MC interleaved motor block, so a command without it is
# byte-identical to what this module has always packed.
_CART = 28


def _pack(arr: np.ndarray) -> pa.Array:
    return pa.array(np.asarray(arr, dtype=np.float64).ravel())


def _unpack(arrow: pa.Array) -> np.ndarray:
    return np.asarray(arrow, dtype=np.float64)


def pack_points(points: np.ndarray) -> pa.Array:
    """Nx3 point cloud as a flat float64 array (low-rate preview payloads)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be Nx3")
    return _pack(pts)


def unpack_points(arrow: pa.Array) -> np.ndarray:
    flat = _unpack(arrow)
    if flat.size % 3:
        raise ValueError("point payload length not divisible by 3")
    return flat.reshape(-1, 3)


def pack_json_message(schema: str, payload: dict) -> pa.Array:
    """Pack a low-rate structured message as a one-element Arrow string array."""
    if not schema:
        raise ValueError("schema must be non-empty")
    body = {"schema": schema}
    body.update(payload)
    return pa.array([json.dumps(body, sort_keys=True, separators=(",", ":"))], type=pa.string())


def unpack_json_message(arrow: pa.Array, *, expected_schema: str | None = None) -> dict:
    if len(arrow) != 1:
        raise ValueError(f"unpack_json_message: expected 1 string value, got {len(arrow)}")
    raw = arrow[0].as_py()
    if not isinstance(raw, str):
        raise ValueError("unpack_json_message: expected Arrow string payload")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError("unpack_json_message: expected JSON object")
    schema = body.get("schema")
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(f"unpack_json_message: expected schema {expected_schema}, got {schema}")
    return body


def pack_motor_state(pos, vel, pos_cmd, vel_cmd, tor_cmd, kp, kd, torque) -> pa.Array:
    n = len(pos)
    buf = np.empty(n * _MS, dtype=np.float64)
    buf[0::_MS] = pos
    buf[1::_MS] = vel
    buf[2::_MS] = pos_cmd
    buf[3::_MS] = vel_cmd
    buf[4::_MS] = tor_cmd
    buf[5::_MS] = kp
    buf[6::_MS] = kd
    buf[7::_MS] = torque
    return _pack(buf)


def pack_motor_state_dict(state: dict[str, np.ndarray]) -> pa.Array:
    """Round-trip partner of :func:`unpack_motor_state` — pack its own dict.

    Every backend node held a private copy of this splat; they are one
    function, so it lives here beside the field order it depends on.
    """
    return pack_motor_state(
        state["position"], state["velocity"], state["position_cmd"],
        state["velocity_cmd"], state["torque_cmd"], state["kp"],
        state["kd"], state["torque"],
    )


def unpack_motor_state(arrow: pa.Array, n: int) -> dict:
    flat_raw = _unpack(arrow)
    _check_length("unpack_motor_state", flat_raw, n * _MS)
    flat = flat_raw.reshape(n, _MS)
    return {
        "position": flat[:, 0],
        "velocity": flat[:, 1],
        "position_cmd": flat[:, 2],
        "velocity_cmd": flat[:, 3],
        "torque_cmd": flat[:, 4],
        "kp": flat[:, 5],
        "kd": flat[:, 6],
        "torque": flat[:, 7],
    }


def pack_cartesian_block(pose, task_R, kc, dc) -> np.ndarray:
    """The optional Cartesian-impedance tail as a flat 28-float array.

    ``pose``   target EE pose, ``[x,y,z,qw,qx,qy,qz]``, WORLD frame (the frame
               the plant's Jacobian and EE pose are in).
    ``task_R`` 3x3 ROW-MAJOR rotation whose COLUMNS are the task frame's axes
               in world coordinates. K_c/D_c are diagonals IN THAT FRAME, so a
               dock's stiffness is stated once (stiff along the insertion axis)
               and never re-derived per dock orientation.
    ``kc``/``dc`` 6 each: ``[kx,ky,kz,krx,kry,krz]`` task-frame diagonals,
               N/m and N·m/rad (damping N·s/m and N·m·s/rad).

    Sent as a matrix, not a quaternion, because the servo law consumes a
    row-major 3x3 — one less convention to get wrong at either end.
    """
    out = np.concatenate(
        [
            np.asarray(pose, dtype=float).ravel(),
            np.asarray(task_R, dtype=float).ravel(),
            np.asarray(kc, dtype=float).ravel(),
            np.asarray(dc, dtype=float).ravel(),
        ]
    )
    if out.size != _CART:
        raise ValueError(f"cartesian block must be 7+9+6+6={_CART} floats, got {out.size}")
    return out


def pack_motor_command(pos, vel, tor, kp, kd, cartesian=None) -> pa.Array:
    """Interleaved 5-float-per-motor command, plus an OPTIONAL Cartesian tail.

    ``cartesian`` is the 28-float block from :func:`pack_cartesian_block` (or
    None). Omitting it produces exactly the array this function has always
    produced — that is what makes the block optional for graphs that never
    send one.
    """
    n = len(pos)
    buf = np.empty(n * _MC, dtype=np.float64)
    buf[0::_MC] = pos
    buf[1::_MC] = vel
    buf[2::_MC] = tor
    buf[3::_MC] = kp
    buf[4::_MC] = kd
    if cartesian is None:
        return _pack(buf)
    tail = np.asarray(cartesian, dtype=float).ravel()
    if tail.size != _CART:
        raise ValueError(f"cartesian tail must be {_CART} floats, got {tail.size}")
    return _pack(np.concatenate([buf, tail]))


def unpack_motor_command(arrow: pa.Array, n: int) -> dict:
    """Inverse of :func:`pack_motor_command`; ``cartesian`` is None when absent.

    The LENGTH is the presence flag: n*5 = no Cartesian block, n*5+28 = one.
    Anything else is a layout bug and raises, same as before.
    """
    flat_raw = _unpack(arrow)
    if flat_raw.size not in (n * _MC, n * _MC + _CART):
        _check_length("unpack_motor_command", flat_raw, n * _MC)
    flat = flat_raw[: n * _MC].reshape(n, _MC)
    tail = flat_raw[n * _MC :]
    return {
        "position": flat[:, 0],
        "velocity": flat[:, 1],
        "torque": flat[:, 2],
        "kp": flat[:, 3],
        "kd": flat[:, 4],
        "cartesian": None
        if tail.size == 0
        else {
            "pose": tail[:7],
            "task_R": tail[7:16].reshape(3, 3),
            "kc": tail[16:22],
            "dc": tail[22:28],
        },
    }


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
    gains, hold, stop.

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
    known = {"arm", "payload", "execute", "cancel", "gains", "hold", "stop", "reason"}
    unknown = set(fields) - known
    if unknown:
        raise ValueError(f"pack_control_update: unknown field(s) {sorted(unknown)}")
    return pack_json_message("control", dict(fields))


def unpack_control_update(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema="control")
    body.pop("schema", None)
    return body


def pack_controller_event(
    *, kind: str, plan_id: str = "", ok: bool = True, reason: str = "", q=None
) -> pa.Array:
    """Controller -> planner. ``kind`` is one of:

    ``ready``      the plant is armed and reporting; ``q`` is the first fresh
                   measured pose, which is where the sequence must start from.
    ``leg_result`` ``plan_id`` finished (``ok``) or gave up (``reason``).
    ``fault``      the controller stopped driving; nothing else will move.

    Every result carries its ``plan_id`` so a late reply from a superseded plan
    is dropped rather than credited to the current one.
    """
    if kind not in {"ready", "leg_result", "fault"}:
        raise ValueError(f"pack_controller_event: unknown kind {kind!r}")
    return pack_json_message(
        "controller_event",
        {
            "kind": kind,
            "plan_id": str(plan_id),
            "ok": bool(ok),
            "reason": str(reason),
            "q": None if q is None else np.asarray(q, dtype=float).ravel().tolist(),
        },
    )


def unpack_controller_event(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema="controller_event")
    body.pop("schema", None)
    if body.get("q") is not None:
        body["q"] = np.asarray(body["q"], dtype=float)
    return body


def pack_controller_settings(payload: dict) -> pa.Array:
    return pack_json_message("controller_settings", payload)


def unpack_controller_settings(arrow: pa.Array) -> dict:
    body = unpack_json_message(arrow, expected_schema="controller_settings")
    body.pop("schema", None)
    return body


def _scene_message(schema: str, body: dict) -> pa.Array:
    request_id = body.get("request_id")
    if schema != "scene_state" and (not isinstance(request_id, str) or not request_id):
        raise ValueError(f"{schema}.request_id must be a non-empty string")
    revision = body.get("revision")
    if not isinstance(revision, int) or revision < 0:
        raise ValueError(f"{schema}.revision must be a non-negative integer")
    return pack_json_message(schema, body)


def pack_scene_command(*, request_id: str, revision: int, state: dict) -> pa.Array:
    if not isinstance(state, dict):
        raise ValueError("scene_command.state must be a mapping")
    return _scene_message("scene_command", {"request_id": request_id, "revision": revision, "state": state})


def unpack_scene_command(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema="scene_command")
    _scene_message("scene_command", body)
    if not isinstance(body.get("state"), dict):
        raise ValueError("scene_command.state must be a mapping")
    return body


def pack_scene_result(*, request_id: str, revision: int, ok: bool, reason: str = "") -> pa.Array:
    return _scene_message("scene_result", {"request_id": request_id, "revision": revision, "ok": bool(ok), "reason": str(reason)})


def pack_scene_state(*, revision: int, actor_q: dict, attachments: dict, constraints: dict) -> pa.Array:
    return _scene_message("scene_state", {"revision": revision, "actor_q": actor_q, "attachments": attachments, "constraints": constraints})


# --------------------------------------------------------------------------- #
# SceneState <-> wire. These lived twice -- once in the caller's scene grafter,
# once in its MuJoCo interface -- written independently and genuinely divergent
# by the time anyone noticed: one took `revision` as an argument and one
# demanded it inside the payload, one omitted `actor_q` when empty and one
# always sent it. That divergence produced the KeyError('revision') that made
# every scene command fail. One copy, here, next to the messages they encode.
#
# There are TWO encoders because there are two messages, and they legitimately
# differ. Naming both is the point: a single encoder with a flag is how the
# rules drifted back into the callers last time.


def scene_command_state(state) -> dict:
    """The ``state`` field of a ``scene_command``: what the sender WANTS.

    No ``revision``: on a command that is a SIBLING of this dict, not a member
    (see ``pack_scene_command``), and a copy inside would be a second source of
    truth for the value the whole optimistic-concurrency check turns on.

    ``actor_q`` is OMITTED while empty, never sent as ``{}``. The plant reads a
    present-but-empty actor_q as "set every actor to nothing" and would drop the
    base tilt on the next recompile; absent means "no opinion, keep yours". A
    node learns actor_q only from a scene_state echo, and the plant publishes
    one only after a command -- so the FIRST command any node sends is always
    the empty case, and this is not a rare path.
    """
    payload: dict = {
        "attachments": {
            name: {
                "object_name": a.object_name,
                "body": a.body,
                "parent_frame": a.parent_frame,
                "child_frame": a.child_frame,
                "mate_pose": list(a.mate_pose),
            }
            for name, a in state.attachments.items()
        },
        "constraints": dict(state.constraints),
    }
    if state.actor_q:
        payload["actor_q"] = dict(state.actor_q)
    return payload


def scene_state_payload(state) -> dict:
    """A full ``scene_state`` echo: what the plant HAS. All four keys, always.

    This one is authoritative rather than a request, so nothing may be elided:
    a consumer bootstrapping from an echo needs every field, and ``revision``
    sits inside because there is no command to carry it alongside.
    """
    return {"revision": int(state.revision), "actor_q": dict(state.actor_q),
            **{k: v for k, v in scene_command_state(state).items()
               if k != "actor_q"}}


def scene_state_from_payload(current, payload: dict, revision: int | None = None):
    """Decode either message back onto ``current`` (absent keys keep its values).

    ``revision`` is passed in for a scene_COMMAND, whose state dict carries
    none by the rule above; a scene_STATE echo has it inside and may leave the
    argument out.
    """
    from arm_control.scene import Attachment, SceneState

    return SceneState(
        actor_q={
            name: list(values)
            for name, values in dict(
                payload.get("actor_q", current.actor_q)
            ).items()
        },
        attachments={
            name: Attachment(**value)
            for name, value in dict(payload.get("attachments", {})).items()
        },
        constraints={
            name: bool(value)
            for name, value in dict(
                payload.get("constraints", current.constraints)
            ).items()
        },
        revision=int(payload["revision"] if revision is None else revision),
    )


def _check_length(name: str, values: np.ndarray, expected: int) -> None:
    if values.size != expected:
        raise ValueError(f"{name}: expected {expected} float64 values, got {values.size}")


# --------------------------------------------------------------------------- #
# Grasp request/result — the orchestrator <-> bridge wire contract. Lives here
# with the other codecs (NOT with docking policy): both real bridges, the sim
# bridge, and the orchestrator must agree on it, and this module is the one
# thing all of them already import.
# --------------------------------------------------------------------------- #


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


def pack_grasp_request(
    *,
    request_id: str,
    target_id: str,
    mode: str = "close",
    gripper_body: str = "gripper",
) -> pa.Array:
    return pack_json_message(
        "grasp_request",
        {
            "request_id": request_id,
            "target_id": target_id,
            "mode": mode,
            "gripper_body": gripper_body,
        },
    )


def unpack_grasp_request(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="grasp_request")


def pack_grasp_result(*, request_id: str, target_id: str, ok: bool, reason: str = "") -> pa.Array:
    return pack_json_message(
        "grasp_result",
        {
            "request_id": request_id,
            "target_id": target_id,
            "ok": bool(ok),
            "reason": reason,
        },
    )


def unpack_grasp_result(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="grasp_result")


# --------------------------------------------------------------------------- #
# object_poses — the pose-estimator -> consumer contract. Hand-rolled on both
# ends until 2026-07-26; once perception lives in its own repo this codec is
# the ONLY place the schema exists, so drift between publisher and consumer
# becomes an import error instead of a silent field mismatch.
#
# ``object_id`` is OPAQUE here: this package never interprets it. What the
# tracked bodies ARE -- modules, fixtures, bricks -- is the caller's domain,
# and naming them in the wire format is what made this codec look like an
# assembly-task message rather than the pose contract it is.
# --------------------------------------------------------------------------- #


def pack_object_poses(
    *,
    camera: str,
    frame: str,
    stamp: float,
    objects: list[dict],
    tags: list[dict] | None = None,
) -> pa.Array:
    """``objects``: dicts with ``object_id`` and ``pose_xyzquat`` (7 floats,
    [x y z qw qx qy qz]) plus free-form extras (``fitness``…). ``frame`` names
    the frame the poses are expressed in — consumers assert on it."""
    for m in objects:
        pose = m.get("pose_xyzquat")
        if pose is None or len(pose) != 7:
            raise ValueError(f"object entry needs a 7-float pose_xyzquat: {m}")
    return pack_json_message(
        "object_poses",
        {
            "camera": camera,
            "frame": frame,
            "stamp": float(stamp),
            "objects": objects,
            "tags": list(tags or []),
        },
    )


def unpack_object_poses(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="object_poses")


__all__ = [
    "pack_motor_state",
    "pack_motor_state_dict",
    "unpack_motor_state",
    "pack_motor_command",
    "unpack_motor_command",
    "pack_cartesian_block",
    "pack_trajectory",
    "unpack_trajectory",
    "pack_plan",
    "unpack_plan",
    "pack_control_update",
    "unpack_control_update",
    "pack_controller_event",
    "unpack_controller_event",
    "pack_controller_settings",
    "unpack_controller_settings",
    "scene_command_state",
    "scene_state_payload",
    "scene_state_from_payload",
    "pack_json_message",
    "unpack_json_message",
    "pack_jog",
    "unpack_jog",
    "pack_grasp_request",
    "unpack_grasp_request",
    "pack_grasp_result",
    "unpack_grasp_result",
    "pack_object_poses",
    "unpack_object_poses",
]


def _self_check() -> None:
    """Round-trip the planner/controller wire pair (arm_control/messages)."""
    n, samples = 7, 5
    times = np.linspace(0.0, 1.0, samples)
    q = np.tile(np.arange(n, dtype=float), (samples, 1))
    plan = unpack_plan(
        pack_plan(
            plan_id="p1", phase="final_approach", gated=True,
            times=times, positions=q, velocities=q * 0.0,
            kp=np.full(n, 1200.0), kd=np.full(n, 30.0),
            cartesian_poses=np.tile([0, 0, 0, 1, 0, 0, 0], (samples, 1)),
            cartesian={"task_R": np.eye(3), "kc": np.ones(6), "dc": np.ones(6)},
            completion={"vel_tol": 0.002, "goal_q": np.arange(n, dtype=float),
                        "residual_max": 0.01},
        )
    )
    assert plan["plan_id"] == "p1" and plan["gated"] is True
    assert plan["positions"].shape == (samples, n), plan["positions"].shape
    assert plan["cartesian_poses"].shape == (samples, 7)
    assert plan["cartesian"]["task_R"].shape == (3, 3)
    assert plan["completion"]["goal_q"].shape == (n,), plan["completion"]
    assert plan["completion"]["vel_tol"] == 0.002

    # A misspelled tolerance must not silently leave the leg on the default
    # rule: that is a leg judged done while it is still moving.
    for bad in ({"vel_tol": 0.1, "residual": 0.01}, {"vel_tol": 0.1, "goal_q": q[0]}):
        try:
            pack_plan(plan_id="p", phase="x", gated=False, times=times,
                      positions=q, velocities=q, kp=np.ones(n), kd=np.ones(n),
                      completion=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"completion={bad} must be refused")

    # A leg with no Cartesian block and no completion rule stays None, not
    # zeros -- the controller branches on exactly this.
    bare = unpack_plan(
        pack_plan(plan_id="p2", phase="lift", gated=False, times=times,
                  positions=q, velocities=q, kp=np.ones(n), kd=np.ones(n))
    )
    assert bare["cartesian"] is None and bare["completion"] is None
    assert bare["cartesian_poses"] is None

    # control: absent keys mean "unchanged", so an empty dict must survive and
    # an unknown key must be refused at PACK time, not silently ignored.
    assert unpack_control_update(pack_control_update()) == {}
    assert unpack_control_update(pack_control_update(arm=True, execute="p1")) == {
        "arm": True, "execute": "p1"
    }
    try:
        pack_control_update(payload={"mass_kg": 1.0}, bogus=1)
    except ValueError as exc:
        assert "bogus" in str(exc), exc
    else:
        raise AssertionError("pack_control_update accepted an unknown field")

    ev = unpack_controller_event(
        pack_controller_event(kind="ready", q=np.zeros(n))
    )
    assert ev["kind"] == "ready" and ev["q"].shape == (n,)
    ev = unpack_controller_event(
        pack_controller_event(kind="leg_result", plan_id="p1", ok=False, reason="x")
    )
    _check_scene_codec()
    assert ev["plan_id"] == "p1" and ev["ok"] is False and ev["q"] is None
    try:
        pack_controller_event(kind="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("pack_controller_event accepted an unknown kind")
    print("messages self-check ok")


def _check_scene_codec() -> None:
    """The two encoders differ ON PURPOSE; assert exactly how, in one place."""
    from arm_control.scene import Attachment, SceneState

    empty = SceneState(actor_q={}, attachments={}, constraints={}, revision=3)
    # A command with nothing to say about actors must not say "no actors".
    assert "actor_q" not in scene_command_state(empty), scene_command_state(empty)
    assert "revision" not in scene_command_state(empty)
    # The echo is authoritative: every key, every time.
    echo = scene_state_payload(empty)
    assert set(echo) == {"revision", "actor_q", "attachments", "constraints"}, echo
    assert echo["revision"] == 3 and echo["actor_q"] == {}

    full = SceneState(
        actor_q={"base": [0.1, 0.0]},
        attachments={"m": Attachment("part_a", "body", "p", "c", (0, 0, 0, 1, 0, 0, 0))},
        constraints={"hold": False},
        revision=7,
    )
    assert scene_command_state(full)["actor_q"] == {"base": [0.1, 0.0]}
    # Round trip both ways. A command's revision arrives as an ARGUMENT (it is
    # a sibling on the wire); demanding it inside the payload is the
    # KeyError('revision') this consolidation exists to prevent.
    back = scene_state_from_payload(empty, scene_command_state(full), revision=7)
    assert back.revision == 7 and back.actor_q == {"base": [0.1, 0.0]}
    assert back.attachments["m"].object_name == "part_a"
    assert back.constraints == {"hold": False}
    echoed = scene_state_from_payload(empty, scene_state_payload(full))
    assert echoed.revision == 7 and echoed.actor_q == {"base": [0.1, 0.0]}
    # Absent keys keep the CURRENT value rather than clearing it -- which is
    # what makes the omit-when-empty rule safe on the receiving side.
    kept = scene_state_from_payload(full, {"revision": 9, "attachments": {}})
    assert kept.actor_q == {"base": [0.1, 0.0]} and kept.constraints == {"hold": False}


if __name__ == "__main__":
    _self_check()
