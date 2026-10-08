"""Native/Python pure sampler parity; pass the built reference_probe.so path."""

import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from arm_control.motion import JointTrajectory

binary = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[3]
lib = C.CDLL(str(binary))
P = C.POINTER(C.c_double)
lib.probe_create.argtypes = [C.c_int, C.c_int, P, P, P]
lib.probe_create.restype = C.c_void_p
lib.probe_sample.argtypes = [C.c_void_p, C.c_double, P]
lib.probe_sample.restype = C.c_bool
lib.probe_destroy.argtypes = [C.c_void_p]
lib.probe_bytes.restype = C.c_size_t
lib.probe_mailbox_bytes.restype = C.c_size_t
lib.probe_benchmark.argtypes = [C.c_void_p, C.c_int, P]
lib.probe_benchmark.restype = C.c_double


def pointer(a):
    return a.ctypes.data_as(P) if a is not None else P()


def create(t, q, v):
    return lib.probe_create(q.shape[1], len(t), pointer(t), pointer(q), pointer(v))


def sample(handle, t, n):
    out = np.empty(1 + 3 * n)
    assert lib.probe_sample(handle, t, pointer(out))
    return out


def oracle(traj, t):
    point = traj.sample_at(t)
    return np.r_[point.time, point.position, point.velocity, point.acceleration]


rng = np.random.default_rng(20260926)
max_errors = np.zeros(3)
count = 0
cases = 0
for n in (1, 7, 16):
    for k in (2, 3, 17, 128):
        for hermite in (False, True):
            times = np.r_[-0.7, -0.7 + np.cumsum(rng.uniform(0.01, 0.2, k - 1))]
            q = rng.normal(size=(k, n))
            v = rng.normal(size=(k, n)) if hermite else None
            traj = JointTrajectory(times, q, v)
            handle = create(times, q, v)
            assert handle
            queries = np.r_[
                times[0] - 1,
                times[-1] + 1,
                times,
                np.nextafter(times, -np.inf),
                np.nextafter(times, np.inf),
                rng.uniform(times[0], times[-1], 257),
                -np.inf,
                np.inf,
            ]
            try:
                initial = [sample(handle, t, n) for t in queries]
                for t, got in zip(queries, initial):
                    want = oracle(traj, t)
                    assert got[0] == want[0]
                    np.testing.assert_allclose(
                        got[1:], want[1:], rtol=2e-13, atol=2e-10
                    )
                    max_errors = np.maximum(
                        max_errors, abs(got[1:] - want[1:]).reshape(3, n).max(axis=1)
                    )
                # Caller mutation after staging cannot change the stored immutable curve.
                times[:] = 0
                q[:] = 100
                if v is not None:
                    v[:] = 100
                for index in rng.permutation(len(queries)):
                    assert (
                        sample(handle, queries[index], n).tobytes()
                        == initial[index].tobytes()
                    )
                bad = np.empty(1 + 3 * n)
                assert not lib.probe_sample(handle, np.nan, pointer(bad))
                count += len(queries)
                cases += 1
            finally:
                lib.probe_destroy(handle)

# Independent analytic cubic, including every derivative away from clamped endpoints.
t = np.array([-0.4, 0.1, 0.7, 1.5])
a = rng.normal(size=(4, 7))
q = a[0] + t[:, None] * a[1] + t[:, None] ** 2 * a[2] + t[:, None] ** 3 * a[3]
v = a[1] + 2 * t[:, None] * a[2] + 3 * t[:, None] ** 2 * a[3]
h = create(t, q, v)
assert h
try:
    for x in np.linspace(t[0], t[-1], 259)[1:-1]:
        want = np.r_[
            a[0] + x * a[1] + x * x * a[2] + x * x * x * a[3],
            a[1] + 2 * x * a[2] + 3 * x * x * a[3],
            2 * a[2] + 6 * x * a[3],
        ]
        np.testing.assert_allclose(sample(h, x, 7)[1:], want, rtol=1e-12, atol=1e-12)
finally:
    lib.probe_destroy(h)

# A joint held still ON a limit is sampled exactly at it, natively too (offset form; the weighted sum went one ulp past).
hi = -0.1518
t = np.array([0.0, 2.0])
for v in (np.zeros((2, 2)), None):
    q = np.array([[0.0, hi], [0.3, hi]])
    h = create(t, q, v)
    assert h
    try:
        held = {sample(h, x, 2)[2] for x in np.linspace(0, 2, 2001)}
        assert held == {hi}, sorted(held)[-3:]
    finally:
        lib.probe_destroy(h)

invalid = []
for name, t, q, v in [
    ("one_knot", np.zeros(1), np.zeros((1, 7)), None),
    ("capacity", np.arange(129, dtype=float), np.zeros((129, 7)), None),
    ("zero_joints", np.arange(2, dtype=float), np.zeros((2, 0)), None),
    ("joint_capacity", np.arange(2, dtype=float), np.zeros((2, 17)), None),
    ("duplicate_time", np.zeros(2), np.zeros((2, 7)), None),
    ("reversed_time", np.array([1.0, 0.0]), np.zeros((2, 7)), None),
    ("nan_time", np.array([0.0, np.nan]), np.zeros((2, 7)), None),
    ("infinite_time", np.array([0.0, np.inf]), np.zeros((2, 7)), None),
    ("nan_position", np.array([0.0, 1.0]), np.full((2, 7), np.nan), None),
    (
        "infinite_velocity",
        np.array([0.0, 1.0]),
        np.zeros((2, 7)),
        np.full((2, 7), np.inf),
    ),
]:
    assert not create(t, q, v), name
    invalid.append(name)

t = np.linspace(0, 1, 128)
q = rng.normal(size=(128, 16))
v = rng.normal(size=(128, 16))
h = create(t, q, v)
checksum = C.c_double()
try:
    timing = [lib.probe_benchmark(h, 100000, C.byref(checksum)) for _ in range(5)]
finally:
    lib.probe_destroy(h)
result = dict(
    status="PASS",
    cases=cases,
    samples=count,
    analytic_samples=257,
    max_abs_error_q_qd_qdd=max_errors.tolist(),
    repeat_and_shuffle="byte-identical",
    caller_mutation="no change",
    rejected=invalid,
    capacity=dict(
        knots=128,
        joints=16,
        storage_bytes=lib.probe_bytes(),
        mailbox_bytes=lib.probe_mailbox_bytes(),
    ),
    host_only_mean_ns_per_sample=timing,
    hardware_qualified=False,
    source_sha256=hashlib.sha256(
        (root / "rt/include/arm_rt/reference.hpp").read_bytes()
    ).hexdigest(),
    binary_sha256=hashlib.sha256(
        binary.read_bytes()
    ).hexdigest(),
    compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
    python=sys.version.split()[0],
)
print(json.dumps(result, indent=2))
