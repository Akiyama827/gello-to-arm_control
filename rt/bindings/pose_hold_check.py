"""Offline binding + encoder assertions; no robot connection or pytest.

PYTHONPATH=. python rt/bindings/pose_hold_check.py /path/to/arm_rt_servo.cpython-310-x86_64-linux-gnu.so
"""
import importlib.util
import struct
import sys

import numpy as np
if len(sys.argv)>1:
    module_spec=importlib.util.spec_from_file_location("arm_rt_servo",sys.argv[1])
    arm_rt_servo=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(arm_rt_servo)
else:
    import arm_rt_servo

from arm_control.contracts.impedance import pose_hold_values
from arm_control.plants.remote_rt import protocol
from arm_control.plants.remote_rt.client import RtBackend, RtConfig, RtLinkError


def rejected(call):
    try:
        call()
    except (ValueError, RtLinkError):
        return
    raise AssertionError("invalid input was accepted")


spec = dict(id=3, kc=[100.]*3+[10.]*3, dc=[20.]*3+[2.]*3,
            nullspace_kp=1., nullspace_kd=3.)
q = np.zeros(7)
J = np.eye(6, 7)
pose = np.array([0., 0., 0., 1., 0., 0., 0.])
args = dict(q=q, dq=q, J=J, pose=pose, bias=q, spec=pose_hold_values(spec), dt=.5)
hold = arm_rt_servo.PoseHold()
assert np.allclose(hold.torque(**args), 0)
pose[0] = .1
assert np.isclose(hold.torque(**args)[0], -10)
for key, value in [("J", J.ravel()), ("q", q.reshape(1, 7)), ("pose", pose[:6]),
                   ("dt", 0.), ("dq", np.full(7, np.nan))]:
    rejected(lambda key=key, value=value: hold.torque(**(args | {key: value})))
for key, value in [("id", 0), ("id", 1.5), ("id", 2**32),
                   ("nullspace_kp", -1), ("nullspace_kd", 11),
                   ("kc", [1001.]*6), ("dc", [float("nan")]*6)]:
    bad = spec | {key: value}
    rejected(lambda bad=bad: pose_hold_values(bad))
    native = [float(bad["id"]), *bad["kc"], *bad["dc"], bad["nullspace_kp"], bad["nullspace_kd"]]
    rejected(lambda native=native: hold.torque(**(args | {"spec": native})))
command = dict(n=7, seq=42, t_mono_ns=protocol._GOLDEN_T,
               q_des=[.1*j for j in range(7)], qd_des=[.01*j for j in range(7)],
               tau_ff=list(range(7)), kp=[100.+j for j in range(7)], kd=[.5*j for j in range(7)])
joint = protocol.pack_command(**command)
soft = protocol.pack_command(**command, pose_hold=spec)
assert len(joint)==664 and len(soft)==784
assert soft.hex()==protocol.golden_lines()[4].split()[1]
assert struct.unpack_from("<H",joint,4)[0]==protocol.VERSION
assert struct.unpack_from("<H",soft,4)[0]==protocol.POSE_HOLD_VERSION
for key, value in [("q_des", [0.]*6), ("kd", [-1.]*7), ("tau_ff", [float("inf")]*7)]:
    rejected(lambda key=key,value=value: protocol.pack_command(**(command | {key:value})))
client = RtBackend(RtConfig(host="127.0.0.1"), [str(i) for i in range(7)])
assert not client.supports_pose_hold and not client.motor_health()["supports_pose_hold"]
rejected(lambda: client.apply_command(dict(position=q, pose_hold=spec)))
rejected(lambda: client.apply_command(dict(position=q, cartesian={})))
assert client._cmd_seq==0  # rejection occurs before any send
print("pose_hold_check: binding, joint/pose encoder parity, validation, capability refusal passed")
