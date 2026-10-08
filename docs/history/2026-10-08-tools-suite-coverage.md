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
* `check_pose_hold_sim.py` needed a historical URDF fixture no repo holds; superseded by the follow-up below (it now runs).
* `nodes/arm_console.py --self-check` is the same self-check as `console_graphs`. `examples/run_console.py --check` and
  `tools/assets/setup_fr3.py --check` exit 0 whether or not assets are staged: they are status reports, not gates.
* Hardware or network tools (`check_remote_fake_rt.py`, `bench/dm/read_params.py`, `rt/timing.py`) and opt-in measurements
  (`rt/constrained_cpu.py`, `rt/tick_jitter.py`) move to `hw/` and `diag/` in L2.

## Soft simulation check re-targeted (follow-up, operator: "the proposal seems correct")

`check_pose_hold_sim.py` pinned a historical URDF (`legacy-fr3-soft-20260907`, in no repo) by hash. Run against the selected
descriptions to drop that pin, it diverged on the library's staged generic FR3 (EE 6.2 cm off, 18 rad/s still moving at the end)
while passing on FR3 #1's deployed description (5.2 mm, 0.002 rad/s). Root cause: the check built its own plant without the robot's
`sim_armature` (0.1, Franka's reflected rotor inertia, `robots/fr3/sim.yaml`), which the sim applies
(`plants/mujoco/single.py`). The generic franka_description's `fr3_link7` inertia is ~4e-7 kg*m^2 (the deployed description's is
~2e-4), so without armature the wrist is near-massless under Soft's rotational damping. Not a defect of Soft or of the sim path.

The check now builds the plant with `build_single_model_backend` from an entry config (`--config`, default the library example),
with no fixture and no hash; the limits are unchanged. Measured with a freshly built `arm_rt_servo` (the compiled RT law the sim's
Soft always uses; Soft refuses without it): generic FR3 6.9 mm peak / 0.026 rad/s at the end, FR3 #1 description 5.3 mm /
0.002 rad/s, both PASS. It joins the suite; the project suite runs it on each deployed arm's description.

## Review round 1 (BLOCK) and fixes

* A fresh checkout was red (PASS=71 FAIL=5): five added checks simulate the staged FR3 (`franka/`, gitignored, staged by
  `tools/assets/setup_fr3.py`), and the author's checkout had it. Now the suite names them (`NEEDS_FR3`, which also holds
  `check_pose_hold_sim` and the `--refine` variant): unstaged, they do not run and one `fr3_assets` FAIL lists them, so a fresh
  checkout is red until the assets are staged, never silently smaller. Confirmed in a fresh worktree (below).
* "Every device-free check" was not yet true: 37 module self-checks, `rt/bindings/pose_hold_check.py`, `setup_fr3.py
  --self-check`, `check_dual_modes --refine` and the `remote_rt.client` loopback (it SKIPped, rc 0, looking for `rt/build`) were
  unrun. All are in now: every module with a `__main__` self-check (`MODULE_CHECKS`, 43 entries with flags), `SCRIPT_CHECKS`, and
  the loopback rung against the suite's built server (`ARM_RT_SERVER_BIN`). Not run, by design: the node CLIs (they start a node),
  `kinematic_fit` and `model_bundle` (project description), `simulation.mujoco_backend` (a shim re-running a subset of
  `plants.mujoco.backend --self-check`), `multi_arm.trace_export` and `supervisor.service` (tools/servers, not checks).
* The project has no suite runner yet; `check_dual_hardware`, `kinematic_fit`, `model_bundle` and the Soft check on each deployed
  description are run by the project suite the next slice (P1) creates. Until then they are run by nothing.
* Outside the tree: `inspect/check_console_page.py` needs `node` on PATH; the Soft checks need `arm_rt_servo` installed (Soft
  refuses without it).
