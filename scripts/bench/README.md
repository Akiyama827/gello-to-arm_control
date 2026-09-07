# Bench tools

Hardware bring-up and measurement, run **by hand against real hardware**. None
of them is imported by a node, launched by a dataflow, or run in any check.
They live apart from `scripts/` so that directory holds only what you run as
part of normal work (`run_console.py`, `setup_fr3_assets.py`, `check_dm_spec.py`).

| Tool | What it measures |
|---|---|
| `bench_ramp.py` | The DM gain ladder — climb from the kp 20 / kd 0.5 anchor a rung at a time |
| `dm_read_params.py` | DM motor parameter registers over SocketCAN. READ-ONLY: 0x7FF reads only, never a write, save, enable or MIT frame |
| `gripper_bench.py` | Franka Hand grasp force and slip. The Hand has no force sensor, so this infers it |
| `rt_handguide.py` | Gravity float / impedance hold under the RT server. Sends no motion commands — it tests server reflexes |
| `rt_timing_bench.py` | The RT servo thread's per-tick wakeup jitter, passively, off a disarmed server |

Two probes were deleted rather than moved here (2026-09-07): `test_dmcan_raw.py`
and `test_motor_01.py`, one-off DM bring-up checks from 2026-07-27 that
`dm_read_params.py` supersedes — and it is read-only where they enabled motors.
Their `test_` prefix also claimed a name this repo does not use: there is no
pytest here, and a collector would have tried to run them against hardware.
Git has them if a DM bring-up ever needs them again.
