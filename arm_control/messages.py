"""Arrow packing and unpacking for Dora inter-node communication.

All messages are flat float64 Arrow arrays. The layouts intentionally match
the historical ``Control/nodes/schemas.py`` functions.
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


def _check_length(name: str, values: np.ndarray, expected: int) -> None:
    if values.size != expected:
        raise ValueError(f"{name}: expected {expected} float64 values, got {values.size}")


# --------------------------------------------------------------------------- #
# Grasp request/result — the orchestrator <-> bridge wire contract. Lives here
# with the other codecs (NOT with docking policy): both real bridges, the sim
# bridge, and the orchestrator must agree on it, and this module is the one
# thing all of them already import.
# --------------------------------------------------------------------------- #


def pack_grasp_request(
    *,
    request_id: str,
    module_id: str,
    mode: str = "close",
    gripper_body: str = "gripper",
) -> pa.Array:
    return pack_json_message(
        "grasp_request",
        {
            "request_id": request_id,
            "module_id": module_id,
            "mode": mode,
            "gripper_body": gripper_body,
        },
    )


def unpack_grasp_request(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="grasp_request")


def pack_grasp_result(*, request_id: str, module_id: str, ok: bool, reason: str = "") -> pa.Array:
    return pack_json_message(
        "grasp_result",
        {
            "request_id": request_id,
            "module_id": module_id,
            "ok": bool(ok),
            "reason": reason,
        },
    )


def unpack_grasp_result(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="grasp_result")


# --------------------------------------------------------------------------- #
# module_poses — the perception -> orchestrator contract. Hand-rolled on both
# ends until 2026-07-26; once perception lives in its own repo this codec is
# the ONLY place the schema exists, so drift between publisher and consumer
# becomes an import error instead of a silent field mismatch.
# --------------------------------------------------------------------------- #


def pack_module_poses(
    *,
    camera: str,
    frame: str,
    stamp: float,
    modules: list[dict],
    tags: list[dict] | None = None,
) -> pa.Array:
    """``modules``: dicts with ``module_id`` and ``pose_xyzquat`` (7 floats,
    [x y z qw qx qy qz]) plus free-form extras (``fitness``…). ``frame`` names
    the frame the poses are expressed in — consumers assert on it."""
    for m in modules:
        pose = m.get("pose_xyzquat")
        if pose is None or len(pose) != 7:
            raise ValueError(f"module entry needs a 7-float pose_xyzquat: {m}")
    return pack_json_message(
        "module_poses",
        {
            "camera": camera,
            "frame": frame,
            "stamp": float(stamp),
            "modules": modules,
            "tags": list(tags or []),
        },
    )


def unpack_module_poses(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="module_poses")


__all__ = [
    "pack_motor_state",
    "unpack_motor_state",
    "pack_motor_command",
    "unpack_motor_command",
    "pack_cartesian_block",
    "pack_trajectory",
    "unpack_trajectory",
    "pack_controller_settings",
    "unpack_controller_settings",
    "pack_json_message",
    "unpack_json_message",
    "pack_grasp_request",
    "unpack_grasp_request",
    "pack_grasp_result",
    "unpack_grasp_result",
    "pack_module_poses",
    "unpack_module_poses",
]
