"""Motor state and command wire layouts, including the Cartesian tail."""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from ._arrow import _pack, _unpack, _check_length

_MS = 8  # pos, vel, pos_cmd, vel_cmd, tor_cmd, kp, kd, tor_fb
_MC = 5
# Optional Cartesian-impedance tail on motor_command: 7 target pose
# [x,y,z,qw,qx,qy,qz] + 9 task-frame rotation (3x3 ROW-MAJOR) + 6 K_c + 6 D_c.
# Appended AFTER the n*_MC interleaved motor block, so a command without it is
# byte-identical to what this module has always packed.
_CART = 28


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
