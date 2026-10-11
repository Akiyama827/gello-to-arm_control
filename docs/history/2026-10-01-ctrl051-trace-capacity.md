# CTRL-051 remainder: trace overflow, atomic write, capacity policy, operator export (2026-10-01)

Covers, offline (no robot, no transport): explicit overflow metadata for the bounded history, atomic and durable trace writes, a capacity
policy with a visible warning, and an operator verify/export/prune path. It does NOT close the card's acceptance: "dropped-record
detection" here is ring-buffer eviction (`dropped_samples` counts every tick older than the 500-tick window, so it grows by the tick rate
for the whole session: truthful metadata, not detection of records lost by a slow consumer or fanout); blocked-writer behaviour predates this
slice (`check_dual_hardware`); consumer-loss is not tested; "replay cannot contact robot addresses" is only an import check of three module
prefixes. CTRL-051 stays IN_PROGRESS.

- `PairFaultTrace` counts every recorded sample; a persisted trace states `samples_kept` and `dropped_samples` (samples evicted from the
  bounded 500-tick history), so overflow is explicit, never silent.
- Atomic write: the trace is encoded once, written to `<name>.json.partial`, fsync'ed, its SHA-256 sidecar created exclusively and fsync'ed, then renamed and the directory fsync'ed; only files this call created are cleaned up on failure, and a name collision (same session, same microsecond) refuses instead of replacing (round-1 review found both the cleanup and the replace could destroy another trace). A full disk or
  crash leaves no file that looks like a trace (the check injects ENOSPC at fsync and compares the directory byte for byte before and after). The
  existing behaviour is unchanged: the writer runs on its own thread after stop was requested, and a write failure still preserves the stop.
- Capacity policy (`RETAIN_FILES` 20, `RETAIN_BYTES` 256 MiB, adjustable defaults, not safety constants): **nothing is deleted automatically**
  (a trace can be the only record of a real fault). Every save that finds the directory over the policy prints a visible warning; pruning is an
  explicit operator action that can only delete traces with a byte-identical exported copy: `python -m arm_control.multi_arm.trace_export --dest DIR`, then `--prune [--keep-files N] --yes --confirm-exported DIR` (dry run without `--yes`; `--keep-files` must be >= 0).
- `arm_control/multi_arm/trace_export.py`: `--verify` (sidecar hash, schema, `samples_kept` equals the samples present) and `--dest DIR` (copies
  verified traces and sidecars plus `EXPORT_MANIFEST.json` into an empty directory, exclusive create, refuses a destination inside the source, skips
  and reports tampered traces). It imports no plant, SDK or supervisor module (checked in a subprocess).
- Check: `tools/bench/check_trace_retention.py`; `check_dual_hardware.py` still passes (real console fault trace).
- Not done (still open in CTRL-051): SITL/RT-side trace export, supervisor correlation (the pair console does not run the supervisor), and
  continuous (non-fault) recording with rotation. A session of the console on the PC continues to write fault traces into
  `Control/.cache/dual_arm/faults`; this slice changes their format additively (`dropped_samples`, `samples_kept`, `retention` keys and a sidecar).

Round-1 review (ACCEPT; the findings were fixed): traces written before sidecars existed (the real fault records in `Control/.cache/dual_arm/faults`)
are reported as `legacy`, exported and never treated as a failed verification; a directory used as a root (an export) no longer treats its
`EXPORT_MANIFEST.json` as a trace (timestamp glob `2*.json`), symlinks are skipped, `*.partial` crash leftovers are listed as incomplete and count toward
capacity; `verify` no longer raises on JSON that is not a trace; prune reports a removed trace even if its sidecar unlink fails. The check gained crash,
sidecar-ENOSPC, collision, legacy, non-trace JSON, `samples_kept` mismatch, exclusive-export and prune-confirmation cases (14 mutations were run by the
reviewer against the first version: 4 survived, each now has a case). `_save_fault_trace` passes `recorded` although the writer runs after the tick loop has
exited (no race in production); no check asserts `dropped_samples` on a console trace.
