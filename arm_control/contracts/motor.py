"""Motor state and command wire layouts, including the Cartesian tail."""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from ._arrow import _pack, _unpack, _check_length
from .impedance import CARTESIAN_SIZE, POSE_HOLD_SIZE, cartesian_values, pose_hold_values, unpack_pose_hold_values

_MS = 8  # pos, vel, pos_cmd, vel_cmd, tor_cmd, kp, kd, tor_fb
_MC = 5
# Optional Cartesian-impedance tail on motor_command: the RT v8 tail (``impedance.cartesian_values``): 9 task-frame rotation
# (3x3 ROW-MAJOR, robot BASE frame) + 6 K_c + 6 D_c. No target pose: every plant computes it as FK(q_des) with its own model
# (operator 2026-10-09). Appended AFTER the n*_MC interleaved motor block, so a command without it is byte-identical to what
# this module has always packed.
_CART = CARTESIAN_SIZE


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


def pack_cartesian_block(spec: dict) -> np.ndarray:
    """The optional Cartesian-impedance tail of a validated base-frame spec ``{frame: "base", task_R, kc, dc}``.

    ``task_R`` is a 3x3 rotation whose COLUMNS are the task frame's axes in the robot BASE frame; ``kc``/``dc`` are task-frame
    diagonals ``[x, y, z, rx, ry, rz]`` (N/m, N.m/rad; N.s/m, N.m.s/rad), so a dock's stiffness is stated once (stiff along the
    insertion axis). A matrix, not a quaternion, because the servo law consumes a row-major 3x3.
    """
    return np.asarray(cartesian_values(spec), dtype=float)


def pack_motor_command(pos, vel, tor, kp, kd, cartesian=None, *, pose_hold=None) -> pa.Array:
    """Interleaved 5-float-per-motor command, plus an OPTIONAL Cartesian tail.

    ``cartesian`` is the 21-float block from :func:`pack_cartesian_block` (or
    None). Omitting it produces exactly the array this function has always
    produced — that is what makes the block optional for graphs that never
    send one. Alternatively, ``pose_hold`` is a validated 15-float spec
    (id, kc[6], dc[6], nullspace kp/kd) whose target the plant captures.
    Both tails together are invalid.
    """
    n = len(pos)
    buf = np.empty(n * _MC, dtype=np.float64)
    buf[0::_MC] = pos
    buf[1::_MC] = vel
    buf[2::_MC] = tor
    buf[3::_MC] = kp
    buf[4::_MC] = kd
    if pose_hold is not None:
        if cartesian is not None:
            raise ValueError("pose hold and trajectory Cartesian impedance are exclusive")
        return _pack(np.concatenate([buf, pose_hold_values(pose_hold)]))
    if cartesian is None:
        return _pack(buf)
    tail = np.asarray(cartesian, dtype=float).ravel()
    if tail.size != _CART:
        raise ValueError(f"cartesian tail must be {_CART} floats, got {tail.size}")
    return _pack(np.concatenate([buf, tail]))


def unpack_motor_command(arrow: pa.Array, n: int) -> dict:
    """Inverse of :func:`pack_motor_command`; ``cartesian`` is None when absent.

    The LENGTH is the presence flag: n*5 = no tail, n*5+21 = a base-frame Cartesian spec, n*5+15 = pose hold.
    Anything else is a layout bug and raises, same as before.
    """
    flat_raw = _unpack(arrow)
    if flat_raw.size not in (n * _MC, n * _MC + _CART, n * _MC + POSE_HOLD_SIZE):
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
        if tail.size != _CART
        else {
            "frame": "base",
            "task_R": tail[0:9].reshape(3, 3),
            "kc": tail[9:15],
            "dc": tail[15:21],
        },
        "pose_hold": unpack_pose_hold_values(tail) if tail.size == POSE_HOLD_SIZE else None,
    }
