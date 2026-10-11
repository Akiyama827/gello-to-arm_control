# CTRL-051/010: each supervisor writes its own fault trace; traces keep the newest 20; one export root

Operator decisions: the fault trace holds the last 10 s before a fault plus 2 s after, at most 50 MB per file, the newest
20 kept (2026-10-09); pruning to 20 applies to both the supervisor's and the console's traces (2026-10-10, replacing
"nothing is deleted automatically"); exports go to `/mnt/data/experiment_data/modular_arm/control_exports/<date>/`, never
inside a repository (2026-10-09).

New:
* `multi_arm.diagnostics.SupervisorFaultTrace`, owned by every `RobotRuntime` (`runtime.fault_trace`): a ring of the
  supervisor's trace/1 rows sized for `FAULT_PRE_S` + `FAULT_POST_S` (10 + 2 s). A fault opens a window: a FAULTED receipt
  (`_receipt`), a KIND of latched fault not yet seen while the latch has stayed set (the RT's, or the supervisor's own
  authority fault; the kind is the text without its parenthesised age, so "rt state stream stale (0.42s)" is one fault
  however long it lasts, and a link flapping between "stale" and "frozen" is one episode; the set clears with the latch),
  or a controller stop (with its reason). Faults inside the window join it (`faults`, at most 50 listed, the rest counted). 2 s after the first
  fault the window's rows go to ONE writer thread; the loop never encodes or touches the disk. A window that closes while the
  writer is still busy is SKIPPED and counted (`skipped`, and `skipped_traces_before` in the next trace), never queued
  without bound; a failed write (disk full) is printed and counted (`errors`). `close()` writes an open window with what it
  has (`post_window_complete: false`). Schema `arm_control.supervisor_fault/v1`, under
  `$ARM_CONTROL_ROOT/.cache/supervisor/<instance>/faults/` (a simulated plant's: `.../sim_faults/`). It needs no trace/1 reader: a console that died still leaves
  the supervisor's record.
* `diagnostics.write_trace` (the console's atomic writer, factored out and shared): temporary file + fsync + rename, SHA-256
  sidecar, never replaces a trace; then keeps the newest `RETAIN_FILES` (20) of its directory (the new trace protected).
  At most `MAX_TRACE_BYTES` (50 MB): the oldest samples go first, counted in `truncated_samples`. Samples are encoded one at
  a time with a 1 ms yield after every ~2 ms of encoding, so a writer thread sharing the servo's interpreter does not hold
  the GIL for a whole trace. Measured (stub link, 1,202 rows, 1.2 MB): without the yield one servo tick came 21 ms after
  the previous one during the write; with it the largest gap in two writes was 11.9 ms. A 10 ms Python loop beside
  synthetic traces: 3.8 MB 11.4 ms, 7.4 MB 12.6 ms, 19.4 MB 15.1 ms; the reviewer measured 17.5-22.2 ms (one gap per write)
  for 19.4 MB inside a supervisor, realistic sizes (3.6-7 MB) within the 12.3-12.5 ms baseline (the final join and write are
  not sliced).
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

Review round 1 (Opus xhigh, BLOCK: 2 MUST) and what changed:
* MUST: the trigger compared the latched fault TEXT with the last one, and `RtBackend`/`SimLink` put the age in that text
  ("rt state stream stale (0.42s)"), so one stale link opened a new window every tick: a trace every ~2 s (reproduced on
  the C++ fake server under SIGSTOP), and newest-20 pruning would have deleted the onset trace after ~42 s -> only the
  rising edge of a latched fault opens a window. check_fault_trace: a latched text changing every 5 ms for 1.2 s gives
  exactly one trace (the text-compare rule fails it, verified).
* MUST: `close()` wrote the open window BEFORE the DISARM (a slow write delayed the stop up to `timeout_s`; a raise skipped
  the DISARM and the remaining runtimes' close) -> the fault trace closes after the link's DISARM and never raises; the
  check times close() -> DISARM < 0.3 s with a 1 s write (the old order fails it, verified).
* SHOULD: `trigger`/`poll` could raise into the loop (a refused executor submit stopped the controller and disarmed the
  arm) -> both are guarded, counted in `errors`; the check forces a failing flush and the arm stays armed.
* SHOULD: checks wrote into, and would prune, real arms' directories (the staged pair and the real-add/calibrate stagings
  use the real instance ids) -> `ARM_CONTROL_FAULT_ROOT` redirects every fault directory (console and supervisor
  subprocesses); both suites set a scratch one, and check_dual_hardware, check_supervised_real and check_base_supervisor
  default to one.
* SHOULD: the project RUNBOOK's commands lacked the root/PYTHONPATH prefix and a bare `--dest` read the library's console
  directory -> prefixed, `--root` explicit.
* NITs taken: close() waits for a busy writer, then writes the open window; the yield is time-based; the controller-stop
  trigger carries the stop reason (`ArmController.stop_reason`); `export_dir` names to the microsecond and refuses a
  relative export root; trace_export reports a refused export instead of a traceback; rt_timing creates its destination
  before the collection; `capacity_warning` (no production caller) and its helper deleted; joined faults get what remains
  of the first fault's 2 s (documented). Left: an authority fault set and cleared within one drain never triggers; a single
  sample larger than the 50 MB budget leaves an empty trace; crash-leftover `.partial` files are listed by trace_export but
  never pruned.

Review round 2 (Opus xhigh, PASS-WITH-FIXES; both MUSTs and all SHOULDs confirmed by running code: the SIGSTOP stale-link
reproduction now writes one trace and keeps the onset; close -> DISARM 9-10 ms with a 3 s write):
* SHOULD: the simulated bench pair runs its supervisors under the real instance ids (`fr3_1`/`fr3_2`), and the console's
  directory was shared by sim and hardware, so simulation faults would prune unexported bench traces -> simulations write
  to `sim_faults/` (supervisor: a `SimLink` plant; console: a non-`remote_rt` pair); check_scene_cohost asserts a simulated
  supervisor's trace is marked sim, check_trace_retention that the directories differ.
* NITs taken: a latched fault of another kind replacing one still latched gets its own trace (check: "rt servo frozen (...)"
  then "rt fault 3: ..." without a clear tick -> two traces; a refused ARM against a still-faulted RT may trace each
  attempt, operator-paced); `ARM_CONTROL_FAULT_ROOT` must be absolute and a supervisor prints where its traces go when it
  is set; check_dual_hardware's dead `CONTROL_ROOT` patches replaced by the variable, no temporary directory when it is
  already set; the export-root message no longer claims to check "outside any repository"; a failed fault-trace close is
  counted; a raising native-reference close no longer skips the rest of `RobotRuntime.close` (including the DISARM: that
  gap predated this slice). Left: the suites' scratch fault roots are not removed.

Review round 3 (Opus xhigh, PASS-WITH-FIXES, no MUST; sim/hardware split verified live: the sim bench pair wrote to
`fr3_1/sim_faults`, the RT staging to `faults`; every latched_fault producer checked against the kind regex):
* SHOULD (conditional): a degraded link flips between "rt servo frozen (...)" and "rt state stream stale (...)" when
  re-sends are lost, and each flip after a window closed reopened one (9 traces in 4 s on the stub) -> the kinds seen in
  one latched episode are remembered until the latch clears; check: 1.5 s of flapping = one trace (a last-kind-only
  rule fails it, verified), frozen -> "rt fault 3" still two.
* NITs taken: both trace headers carry `sim`; a bare `--dest` names the export `<faults|sim_faults>-<source>-<time>`;
  trace_export refuses a relative fault root cleanly; both RUNBOOKs describe `sim_faults/` and the checks' scratch root.
  Left: an authority fault with an empty message never triggers (pre-existing); two SimLink plant failures of one exception
  type share a trace.
