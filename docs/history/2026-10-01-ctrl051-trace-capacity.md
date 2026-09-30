# CTRL-051 remainder: trace overflow, atomic write, capacity policy, operator export (2026-10-01)

Card acceptance items covered: dropped-record detection, disk-full and blocked-writer behaviour, an operator export path, capacity
policy; all offline (no robot, no transport).

- `PairFaultTrace` counts every recorded sample; a persisted trace states `samples_kept` and `dropped_samples` (samples evicted from the
  bounded 500-tick history), so overflow is explicit, never silent.
- Atomic write: the trace is encoded once, written to `<name>.json.partial`, fsync'ed, its SHA-256 sidecar written, then renamed. A full disk or
  crash leaves no file that looks like a trace (the check injects ENOSPC at fsync and compares the directory byte for byte before and after). The
  existing behaviour is unchanged: the writer runs on its own thread after stop was requested, and a write failure still preserves the stop.
- Capacity policy (`RETAIN_FILES` 20, `RETAIN_BYTES` 256 MiB, adjustable defaults, not safety constants): **nothing is deleted automatically**
  (a trace can be the only record of a real fault). Every save that finds the directory over the policy prints a visible warning; pruning is an
  explicit operator action: `python -m arm_control.multi_arm.trace_export --prune [--keep-files N] [--yes]` (dry run without `--yes`).
- `arm_control/multi_arm/trace_export.py`: `--verify` (sidecar hash, schema, `samples_kept` equals the samples present) and `--dest DIR` (copies
  verified traces and sidecars plus `EXPORT_MANIFEST.json` into an empty directory, exclusive create, refuses a destination inside the source, skips
  and reports tampered traces). It imports no plant, SDK or supervisor module (checked in a subprocess).
- Check: `tools/bench/check_trace_retention.py`; `check_dual_hardware.py` still passes (real console fault trace).
- Not done (still open in CTRL-051): SITL/RT-side trace export, supervisor correlation (the pair console does not run the supervisor), and
  continuous (non-fault) recording with rotation. A session of the console on the PC continues to write fault traces into
  `Control/.cache/dual_arm/faults`; this slice changes their format additively (`dropped_samples`, `samples_kept`, `retention` keys and a sidecar).
