# Reusable bench tools

Run hardware diagnostics only with an operator and the appropriate deployment
procedure. CLI `--help` is safe; a read-only diagnostic may still transmit
protocol packets. These tools are not launched by runtime dataflows.

| Tool | What it measures |
|---|---|
| `dm/read_params.py` | DM register reads; never writes, saves, enables, or sends MIT commands |
| `rt/timing.py` | RT servo wakeup jitter from a disarmed server |

Concrete gain-ramp, Hand-grasp, and hand-guiding procedures belong to the
consuming project's `Control/tools/bench/`. The standalone console application
is `examples/run_console.py`; FR3 asset staging is `tools/assets/setup_fr3.py`.

Two probes were deleted rather than moved here (2026-09-07): `test_dmcan_raw.py`
and `test_motor_01.py`, one-off DM bring-up checks from 2026-07-27 that
`dm_read_params.py` supersedes — and it is read-only where they enabled motors.
Their `test_` prefix also claimed a name this repo does not use: there is no
pytest here, and a collector would have tried to run them against hardware.
Git has them if a DM bring-up ever needs them again.
