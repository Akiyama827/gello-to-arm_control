# The pair's supervisors are stopped together; releases are stamped

Bench, step 16 (dual console, 2026-10-10): after the console exited, FR3 #2's supervisor was still running (re-parented to
systemd, DISARMED, holding FR3 #2's RT session) while FR3 #1's had stopped. `multi_arm.launch.running_pair` (and
`start_pair`'s failure path) stopped the supervisors one after another, each with a bounded wait, so an interrupt during the
first wait (a second Ctrl-C) skipped the second entirely. The orphan was stopped by hand (SIGTERM: clean `[supervisor]
stopped`).

* `supervisor.launch.stop_supervisors(procs)`: SIGTERM to every supervisor first (each DISARMs and exits on its own), then
  each wait in a nested `finally`, so an interrupt during one still stops and reaps the rest. Used by both pair paths.
  `python -m arm_control.supervisor.launch` (in the suite's module checks): an interrupt injected into the first wait leaves
  no survivor.
* Step 16's acceptance ("a pair leg's references start < 2 ms apart") had nothing to read on success: trace rows persist only
  on a fault and the controller's `executing` line had no time. It now prints `start=` (the trajectory's start, on the
  reference clock) and `taken=` (when this controller took the release). A pair's two supervisors share the PC's monotonic
  clock, so the difference of their lines is the PC-side start skew; the physical skew at the robots stays unmeasured.
