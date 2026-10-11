# SupervisedHardware: the dual console's runtime over two supervisors (dual-arm slice 4)

**Console decoupled from controllers.** `DualArmConsole` now reads each arm only through its `ArmHandle` (`state().position`,
`armed`, `ready`) and the model's joint limits (`PinocchioDynamics(...).joint_limits`, the same source the controller reads,
computed once per arm in the console). It no longer touches `runtime.controllers`. Differences from before, all in the stricter
direction: the Hand command needs `state().ready` (controller ready and bridge armed, plus the runtime's own conditions) rather
than `ready_sent` alone; a plan from an arm that has never reported a measured pose is refused ("no measured pose") instead of
planning from zeros; after a rejected sample the console keeps the last good pose, as the display always did. Checked with
`check_dual_hardware` (16 PASS, plus a case that a Hand command needs the handle's readiness) and `check_dual_console` (plus
cases that the displayed limits equal the controller's, a rejected sample keeps the last good pose and a pose is refused before
any measurement); mutants of each of those four console changes are killed.

**`multi_arm/hands.py`.** The Hand methods (`finger_positions`, `set_fingers`, `hand_busy`, `cancel_hands`) moved out of
`RemoteHardware` into the `HandControl` mixin, verbatim, so both runtimes share them.

**`multi_arm/supervised_hardware.SupervisedHardware`.** The runtime surface the console uses, with each arm's authority in
its supervisor: an `ArmClient` and a `SdkArmHandle` per arm, no controller, no RT link. `connect()` claims both arms (never
arms or moves), starts the Hands and waits for fresh measured feedback. ARM/DISARM run on a per-arm worker (the console holds
its lock while calling): not-ready while pending, a failure is the arm's fault until the next request succeeds, arming while
one is pending is refused, a DISARM voids queued ARMs. `stop()` queues a DISARM on both; `close()` queues a DISARM and waits for it
behind any request in flight (bounded by the SDK timeouts), then closes the clients (the supervisor cancels live executions; the
robot holds). An invalid measurement from the supervisor is not fresh; the last good pose stays for display. If the supervisor channel is lost, the DISARM cannot be delivered: the supervisor then abandons the session and leaves the arm armed and
holding (its design), so close() on a dead channel is not a stop. Independent mode only: `set_shared_grasps` is refused (shared
coupling needs the PC-side closure and stays on `RemoteHardware`, retirement condition: the closure moves behind
`state_history/1`, slice 3, and the group path). `diagnostic_sample` reports only what this runtime has (measured joints from
the supervisor), no RT packets or controller internals. Gains through `set_gains` need named presets, which this runtime does
not pass yet.

Selected by passing the class as the console's `runtime_factory`; there is no default and no launch or unit file change (graph
migration is the operator's separate stop).

Known limits (not blocking, from review): the SDK channel serialises calls, so a console tick's reads queue behind the worker's
`await_authority` slices while an ARM/DISARM is being confirmed (measured 0.5 ms per tick on the fake; on hardware a confirm
could delay a tick and trip the console's 0.5 s action gate); a refused ARM makes the arm not fresh, so the operator DISARMs
(which clears the fault) before arming again; the console's `close()` and its tick thread can touch the pending list together
when a tick is blocked on IPC (the same pattern exists on `RemoteHardware`); `check_dual_modes` and `check_cooperative` fail
identically on the base with `ARM_CONTROL_ROOT` at the Control root (a camera-mount collision with the shared-object plate,
unrelated) and pass with it at the package root.

Not claimed: console-level parity with `RemoteHardware` through supervisors. That needs an FR3 supervisor deployment with the
pair's model tokens (an operator/hardware configuration); the fake-RT fixture here is a 3-joint arm and a stub scene.
Bench: `tools/bench/check_supervised_hardware.py` (offline suite). Mutation survivors: `close()` cancelling the Hands is
redundant with the DISARM path, and the generation guard / cancelling queued ARMs is redundant with ARM being refused while any
request is pending (equivalent mutants); a duplicate DISARM not being suppressed is harmless.
