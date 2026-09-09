#!/usr/bin/env python3
"""Offline cap admission and protocol checks. Only starts local fake servers.

From the library root:
    PYTHONPATH=. python tools/bench/rt/torque_cap.py /path/to/arm_rt_server
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from arm_control.plants.dm.backend import (
    DEFAULT_LIMITS_BY_TYPE,
    decode_mit_reply,
    pack_mit_control_frame,
)
from arm_control.plants.remote_rt.protocol import golden_lines


def main(binary: Path) -> None:
    binary = binary.resolve(strict=True)
    local = ["--bind", "127.0.0.1", "--udp-port", "49810", "--tcp-port", "49811"]
    invalid = ("", "junk", "27junk", "nan", "inf", "-inf", "0", "-1", "1e999", "1e-999")
    for value in invalid:
        result = subprocess.run(
            [str(binary), "--backend", "fake", *local, "--tau-max", value],
            capture_output=True, text=True, timeout=3,
        )
        assert result.returncode == 2 and "--tau-max must be" in result.stderr, result
        assert "arm_rt_server:" not in result.stdout, result  # before startup
    missing = subprocess.run(
        [str(binary), "--tau-max"], capture_output=True, text=True, timeout=3,
    )
    assert missing.returncode == 2 and "needs a value" in missing.stderr, missing

    # This value is valid as a float, but cannot encode even a near-zero DM
    # torque. Factory admission must reject it BEFORE opening a CAN socket.
    tiny = subprocess.run(
        [str(binary), "--backend", "dm", "--dm-spec", "cap_check_none;1:4340",
         *local, "--tau-max", "0.000001"],
        capture_output=True, text=True, timeout=3,
    )
    assert tiny.returncode != 0 and "representable" in tiny.stderr, tiny
    assert "no CAN interface" not in tiny.stderr and "socket(PF_CAN)" not in tiny.stderr

    for args, expected in (([], "50 50"), (["--tau-max", "27"], "27 27"),
                           (["--tau-max", "100"], "50 50")):
        process = subprocess.Popen(
            [str(binary), "--backend", "fake", "--n", "2", *local, *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            try:
                stdout, stderr = process.communicate(timeout=.4)
                raise AssertionError(("fake server exited before the check", stdout, stderr))
            except subprocess.TimeoutExpired:
                process.terminate()  # only the fake process spawned above
                stdout, stderr = process.communicate(timeout=3)
            assert process.returncode == 0, (stdout, stderr)
            assert f"effective torque limits (N.m): {expected}\n" in stdout, stdout
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    print("PASS CLI cap validation, DM pre-bus rejection, effective startup limits")

    subprocess.run([str(binary.parent / "torque_cap_selfcheck")], check=True)
    protocol = subprocess.check_output(
        [str(binary.parent / "protocol_selfcheck")], text=True,
    ).splitlines()
    assert protocol == golden_lines(), "C++/Python RT protocol drift"

    # Exercise the existing unmodified Python codecs against every C++ golden.
    for line in subprocess.check_output([str(binary), "--mit-check"], text=True).splitlines():
        tag, motor_type, *fields = line.split()
        limits = DEFAULT_LIMITS_BY_TYPE[motor_type]
        if tag == "PACK":
            p, v, kp, kd, tau = map(float, fields[:5])
            packed = pack_mit_control_frame(
                position=p, velocity=v, kp=kp, kd=kd, torque=tau, limits=limits,
            )
            assert packed.hex() == fields[5], line
        else:
            assert tag == "DEC", line
            state = decode_mit_reply(bytes.fromhex(fields[0]), limits)
            assert state is not None
            values = (state.motor_id, state.error, state.position, state.velocity,
                      state.torque, state.t_mos, state.t_rotor)
            assert [f"{v:.9g}" for v in values] == fields[1:], line
    print("PASS unchanged Python/C++ RT and MIT protocol parity")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
