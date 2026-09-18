"""Run with PYTHONPATH=. python tools/bench/check_pose_hold_contract.py."""
import math

import numpy as np

from arm_control.contracts.impedance import pose_hold_values, unpack_pose_hold_values
from arm_control.contracts.motor import pack_motor_command, unpack_motor_command


def main():
    spec = dict(id=7, kc=[300, 300, 300, 30, 30, 30],
                dc=[35, 35, 35, 8, 8, 8], nullspace_kp=0.5, nullspace_kd=2)
    assert unpack_pose_hold_values(pose_hold_values(spec)) == spec
    z = np.zeros(7)
    plain = pack_motor_command(z, z, z, z, z)
    soft = pack_motor_command(z, z, z, z, z, pose_hold=spec)
    assert len(plain) == 35 and len(soft) == 50
    assert unpack_motor_command(plain, 7)["pose_hold"] is None
    assert unpack_motor_command(soft, 7)["pose_hold"] == spec
    for key, value in (("id", 0), ("id", 1.5), ("id", 2**32),
                       ("kc", [math.nan]*6), ("kc", [1001]*6),
                       ("dc", [-1]*6), ("dc", [0]*5),
                       ("nullspace_kp", -1), ("nullspace_kd", math.inf)):
        try:
            pose_hold_values(dict(spec, **{key: value}))
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError((key, value))
    try:
        pack_motor_command(z, z, z, z, z, cartesian=np.zeros(28), pose_hold=spec)
    except ValueError:
        pass
    else:
        raise AssertionError("two Cartesian modes accepted")
    print("pose hold contract: plain compatibility, roundtrip and validation OK")


if __name__ == "__main__":
    main()
