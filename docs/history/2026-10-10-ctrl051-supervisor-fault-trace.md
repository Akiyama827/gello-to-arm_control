# CTRL-051/010: each supervisor writes its own fault trace; traces keep the newest 20; one export root

Operator decisions: the fault trace holds the last 10 s before a fault plus 2 s after, at most 50 MB per file, the newest
20 kept (2026-10-09); pruning to 20 applies to both the supervisor's and the console's traces (2026-10-10, replacing
"nothing is deleted automatically"); exports go to `/mnt/data/experiment_data/modular_arm/control_exports/<date>/`, never
inside a repository (2026-10-09).

New:
* `multi_arm.diagnostics.SupervisorFaultTrace`, owned by every `RobotRuntime` (`runtime.fault_trace`): a ring of the
  supervisor's trace/1 rows sized for `FAULT_PRE_S` + `FAULT_POST_S` (10 + 2 s). A fault opens a window: a FAULTED receipt
  (`_receipt`), a new or changed latched fault in the link health (the RT's, or the supervisor's own authority fault), or a
  controller stop. Faults inside the window join it (`faults`, at most 50 listed, the rest counted). 2 s after the first
  fault the window's rows go to ONE writer thread; the loop never encodes or touches the disk. A window that closes while the
  writer is still busy is SKIPPED and counted (`skipped`, and `skipped_traces_before` in the next trace), never queued
  without bound; a failed write (disk full) is printed and counted (`errors`). `close()` writes an open window with what it
  has (`post_window_complete: false`). Schema `arm_control.supervisor_fault/v1`, under
  `$ARM_CONTROL_ROOT/.cache/supervisor/<instance>/faults/`. It needs no trace/1 reader: a console that died still leaves
  the supervisor's record.
* `diagnostics.write_trace` (the console's atomic writer, factored out and shared): temporary file + fsync + rename, SHA-256
  sidecar, never replaces a trace; then keeps the newest `RETAIN_FILES` (20) of its directory (the new trace protected).
  At most `MAX_TRACE_BYTES` (50 MB): the oldest samples go first, counted in `truncated_samples`. Samples are encoded one at
  a time with a 1 ms yield every 50, so a writer thread sharing the servo's interpreter does not hold the GIL for a whole
  trace. Measured (stub link, 1,202 rows, 1.2 MB): without the yield one servo tick came 21 ms after the previous one
  during the write; with it the largest gap in two writes was 10.8 ms (none over 15 ms).
* `trace_export` verifies both schemas; `--root` reads a supervisor's directory; a bare `--dest` exports to
  `$ARM_CONTROL_EXPORT_ROOT/<date>/faults-<source>-<time>/` (CTRL-010; refused when the variable is unset).
  `tools/hw/rt_timing.py --save` writes the RT timing trial's raw samples and summary under the same root (`rt_timing-<time>/`).
  The library names only the variable; the project's RUNBOOK sets it to the operator's path.
* `SupervisedHardware.diagnostic_sample` adds `trace_age_s` per arm: a supervisor that answers trace/1 but whose loop stopped
  ticking shows as an ageing row (the supervisors share this PC's monotonic clock); one that stops answering was already
  `trace_stale`. With the supervisor-written trace this closes CTRL-051's consumer-loss item: a lost consumer loses no fault
  record, a stalled producer is visible to the consumer, and a late consumer counts `trace_dropped`.

Changed: `RETAIN_BYTES` is now 20 x 50 MB (a bound, not a warning threshold); `capacity_warning`'s text says the next save
prunes. check_supervisor_ledger's stub link gained `config.host`: without it every trace/1 row failed to build (counted as
`errors`, so the stub's trace was always empty, unnoticed).

Checks: check_supervisor_ledger (new `check_fault_trace`: window bounds [-pre, +post] against the fault time, two faults in
one trace, sha256-verified, a busy writer skips while the loop keeps ticking, a failed write counted, an open window written
on close), check_trace_retention (save prunes to RETAIN_FILES with sidecars, the 50 MB cap drops oldest and verifies, a bare
--dest needs the export root), check_supervised_hardware (`trace_age_s`), rt_timing --save against a loopback fake server;
supervisor/pair checks (sim_supervisor, sdk_supervisor, dual_supervisor, state_history, dora_arm, stream_supervisor,
pair_planning, cooperative, dual_console, bimanual_sync, scene_cohost); project check_dual_hardware on the staged bench pair,
check_supervised_sims, sim-add e2e (passed, 74.4 s; no fault trace written).

Not done: copying the RT box's own gap journal off the box (an operator step at the bench; the box is never polled from
here); a trace/1 or status field reporting the writer's skipped/errors counts (printed and carried in the next trace only).
