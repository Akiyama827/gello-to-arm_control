# tools/

Where a tool lives says what it may touch.

| Folder | What | Rule |
|---|---|---|
| `check/` | Device-free, assert-based checks (`check_<subject>.py`) and the suite, `check/suite.py` | The suite runs every one; a device-free check no suite runs is deleted |
| `check/rt/` | Fake-RT protocol regressions; `_<name>.py` are helpers they import | Run by the suite (two through CTest) |
| `diag/` | Opt-in measurements and investigations on fake servers | Not pass gates; never run by a suite |
| `hw/` | Touches a device, the RT host or a CAN bus | Operator only; never run by a suite; read the options and get hardware authorization first |
| `assets/` | Stages third-party descriptions (`setup_fr3.py`) | — |

Run the suite from anywhere: `python tools/check/suite.py --build-dir <dir>`. It builds `rt/`, runs every check and module
self-check, and writes `offline-summary.json`. Checks that simulate the FR3 need it staged (`tools/assets/setup_fr3.py`);
unstaged, they do not run and one `fr3_assets` FAIL names them.
