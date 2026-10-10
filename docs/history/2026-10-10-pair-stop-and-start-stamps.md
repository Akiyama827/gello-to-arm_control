# The pair's supervisors are stopped together; releases are stamped

Bench, step 16 (dual console, 2026-10-10): after the console exited, FR3 #2's supervisor was still running (re-parented to
systemd, DISARMED, holding FR3 #2's RT session) while FR3 #1's had stopped. `multi_arm.launch.running_pair` (and
`start_pair`'s failure path) stopped the supervisors one after another. The bench ran `run.py real dual 2>&1 | tee ...`:
Ctrl-C reached the whole pipeline, `tee` died and the console's stdout became a dead pipe; after FR3 #1's supervisor stopped,
`stop_supervisor`'s "log kept" print raised BrokenPipeError and FR3 #2's was never signalled (its log shows `[supervisor]
stopped` only at the manual stop). The orphan was stopped by hand (SIGTERM: clean). (Corrected in review: not a second
Ctrl-C, which dual_main's flag-only handlers absorb.)

* `supervisor.launch.stop_supervisors(procs)`: SIGTERM to every supervisor first (each DISARMs and exits on its own), then
  each wait in a nested `finally`, so an interrupt during one still stops and reaps the rest. Used by both pair paths.
  `python -m arm_control.supervisor.launch` (in the suite's module checks): an interrupt injected into the first wait leaves
  no survivor.
* Step 16's acceptance ("a pair leg's references start < 2 ms apart") had nothing to read on success: trace rows persist only
  on a fault and the controller's `executing` line had no time. The `executing` line now prints `start=` (the trajectory's
  anchor: a scheduled pair release gives both arms the same stamp by construction, so it shows the anchoring, not a skew), and
  each controller prints `first sample <plan> at <t> (phase ...)` on its first commanding tick of the leg. On hardware both
  supervisors read the PC's monotonic clock, so the two arms' first-sample lines differ by the PC-side start skew (each
  supervisor's 100 Hz tick phase included, so up to ~10 ms by design; the trajectory's phase is time-based, so a late first
  tick enters it at its elapsed phase). The physical skew at the robots stays unmeasured (both RT servers' state stamps are on
  the RT box's clock, at 250 Hz: an instrument for later).

Review round 1 (Opus xhigh, PASS-WITH-FIXES): MUST: the first version's `start=`/`taken=` claim was false (`start=` equal by
construction, 1 us apart in check_dual_console) -> the first-sample line above. SHOULDs: the root-cause narrative corrected
(dead stdout, not a second Ctrl-C); project `run.py real dual` maps SIGTERM/SIGHUP to an interrupt around the pair (a closed
terminal killed the console with no cleanup and orphaned both supervisors); bench logs use `tee -i`. NITs: the self-check now
proves every supervisor is signalled before the first wait and removes its temp dirs; check_dual_hardware uses
stop_supervisors.
