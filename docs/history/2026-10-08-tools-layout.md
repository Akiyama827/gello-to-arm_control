# tools/ by what a tool may touch (tools cleanup, library slice L2)

L1 made every device-free check run; this slice moves files only, so the folder says what a tool may touch (`tools/README.md`):

* `tools/check/`: every device-free check (from `tools/bench/` and `tools/inspect/`), and the suite, renamed
  `rt_offline_suite.py` → `check/suite.py` (it has not been RT-only for a while).
* `tools/check/rt/`: the fake-RT regressions (from `tools/bench/rt/`); the three helper-only modules are now `_bench_target.py`,
  `_envelope_age.py`, `_first_envelope_trace.py`.
* `tools/diag/`: `constrained_cpu.py`, `tick_jitter.py` (opt-in measurements on fake servers; `constrained_cpu` now finds the
  regressions in `check/rt/`).
* `tools/hw/`: `rt_timing.py` (was `bench/rt/timing.py`, a passive observer of a real server), `remote_fake_rt.py` (was
  `bench/check_remote_fake_rt.py`; it talks to the RT host), `dm_read_params.py` (was `bench/dm/read_params.py`, CAN).
* `tools/bench/README.md` (stale: it named a file that no longer existed) is replaced by `tools/README.md`. `tools/inspect/` and
  `tools/bench/` no longer exist.

Every current reference is rewritten (README, RUNBOOK, ARCHITECTURE, clients, rt/README, rt/CMakeLists.txt CTest paths, code
comments and docstrings, pyproject). Dated records (docs/history, the project's dated status, plans and evidence) keep the paths
they were written with.

Checked: the suite passes with the same entries as before the move, and the moved `hw/` and `diag/` tools import and parse
arguments (`--help`).

## Follow-up: no library module imports the repo's `tools/`

`franka_adapter --demo` imported `tools.check.check_hand_grasp` and re-ran three of its checks. That made a library module
depend on the repo's tool tree and broke under a project path, where the project's own `tools` package shadows this one.
`check_hand_grasp` is its own suite entry and runs all of them, so the demo now checks only the FSM it owns. `kinematic_fit`'s
docstring no longer names a project file. `grep -rn "from tools\." arm_control` is empty. Related checks run:
`franka_adapter --demo`, `check_hand_grasp`.
