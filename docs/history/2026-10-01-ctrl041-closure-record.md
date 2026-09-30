# CTRL-041 slice 2: the five closure quantities and the fault-trace schema version (2026-10-01)

Design: `Control/docs/ctrl041-design.md` (approved with amendments 2026-10-01: extend the existing diagnostics output
with a schema version; no new record file; offline/shadow, no threshold change).

- `arm_control/multi_arm/closure.py`: `closure_record(...)` builds one `arm_control.closure/v1` record per tick with the five
  quantities in separate slots: `planned_reference` (the planner's geometric closure/path maxima), `rt_selected_reference`
  (absent unless a native reference is selected: the pair runs streamed today, so it says so), `tracking` (per-arm joint error
  of the measured state against the RT command and the PC command, each arm's own latest published state), `common_time_inferred`
  (TCP agreement at a common time, with the alignment provenance: common time, clock domain, latest skew, both brackets'
  sequence numbers, times and fractions), `physical` (present only when independently measured; absent otherwise). Every present
  value carries its frame/model/time basis; every absent one carries the reason (never a zero or stale value); non-finite values
  are absent. Motion and stationary windows are labelled, with `stationary_for_s` so an analysis can split settle from hold without
  this module inventing a threshold.
- `PairFaultTrace.save` writes `arm_control.dual_fault/v2` (`schema_version` 2, `closure_schema`); the console adds `closure_v1`
  beside the older `closure` tuple in every recorded sample (v1 readers, including `reconstruct_lineage`, are unaffected: additive).
  The timestamp note now names the attested clock domain (CTRL-041 slice 1), not the host string.
- Checks: `tools/bench/check_closure_record.py` (pure), `check_dual_hardware.py` extended (a real fault trace from the two fake RT servers
  is v2 and its `closure_v1` values equal the recorded `closure` and the aligned brackets), plus `check_dual_console`,
  `check_bimanual_contract`, `check_bimanual_sync`, `check_cooperative` re-run.
- Not done: `rt_selected_reference` needs a native-reference pair (CTRL-053 composition); `physical` needs an independent measurement
  (mocap/camera, an operator session); the replayed-diagnostic-agreement acceptance case (a replay of the recorded values reproducing the
  recorded closure) is covered only for `common_time_inferred` by the extended hardware check; no plots. This record changes no
  threshold and gates nothing.
