# Every device-free check runs in the suite (tools cleanup, library slice L1)

Operator request: the tools folders had grown past anyone's overview. A read-only audit found that the offline suite ran 9 of the 36
top-level `tools/bench/check_*.py`; the rest were listed in the RUNBOOK and run by hand, or by nothing. A check no one runs rots
unseen (the project's `tools/demo/workcell_add.py` had been broken since 09-09). Rule from here: **a device-free check that no suite
runs is deleted.** Folder moves and renames are the next slice (L2); this one changes only what runs.

* `rt_offline_suite.py` now also runs, after the build-dependent checks and whether or not the build passed:
  * 24 device-free checks (`CHECKS`): arm_controller, bimanual_contract, bimanual_sync, closure_record, console_authority,
    console_grasp, cooperative, dual_console, dual_modes, execution_policy, hand_grasp, motion_profiles, node_shutdown,
    pair_planning, pose_hold, refinement, retiming, rt_telemetry, stream, supervisor_ledger, and inspect/ console_controls,
    console_page, controller_mode_feedback, ui_scene_injection;
  * 10 module self-checks (`MODULE_CHECKS`, `python -m`) and `examples/run_console.py --self-check`.
* Merged: `check_pose_hold_contract` + `check_pose_hold_controller` → `check_pose_hold.py`; `check_stream_contract` +
  `check_stream_target` → `check_stream.py` (the codec part runs first, because it asserts the SDK imported no supervisor or plant;
  the target part imports the supervisor lazily after it).
* Deleted, with no current reference: `check_integral_state.py` and `check_payload_replacement.py` (capture/baseline tools of the
  finished 09-26/27 refactors), `rt/first_envelope_probe.py` (the 09-28 investigation, recorded in
  2026-09-30-ctrl033-selection-unresolved.md) and `rt/reference_mode_boundary.py` (its own README row said "historical").
* Fixed: `check_supervisor_ledger.py` had failed since `4598952` added `hold_ms` to the link contract (`AttributeError: 'StubLink'
  object has no attribute 'hold_ms'`); its stub now states `hold_ms = None`, an RT HELLO without a hold deadline.

Not in this suite, by design:
* `check_dual_hardware.py` (the only check of `RemoteHardware`, which the live `real dual` console builds) needs the project's staged
  pair config (`stage_bench_pair(hardware=True)`); the project suite runs it. Run by hand on 10-08 against this tree's
  `arm_rt_server`: 16 PASS.
* `python -m arm_control.kinematic_fit --self-check` and `-m arm_control.model_bundle` read the project's FT description through
  `CONTROL_ROOT`. Without it the first fails and the second prints SKIPPED and exits 0; the project suite runs both. With the
  project root, kinematic_fit passes.
* `check_pose_hold_sim.py` needs the historical `legacy-fr3-soft-20260907` URDF fixture, which no repo holds (copies exist only
  on this PC). It passes against that fixture. Whether to commit the fixture to project evidence or keep it a by-hand check is an
  operator decision.
* `nodes/arm_console.py --self-check` is the same self-check as `console_graphs`. `examples/run_console.py --check` and
  `tools/assets/setup_fr3.py --check` exit 0 whether or not assets are staged: they are status reports, not gates.
* Hardware or network tools (`check_remote_fake_rt.py`, `bench/dm/read_params.py`, `rt/timing.py`) and opt-in measurements
  (`rt/constrained_cpu.py`, `rt/tick_jitter.py`) move to `hw/` and `diag/` in L2.
