# CTRL-053 graph 3: `dual_arm_sim.yml` deleted (operator decision 2026-10-09)

The graph ran two in-process Dora `arm_controller`s and two `mujoco_interface` plants in two independent worlds, a
process-isolation diagnostic the RUNBOOK already called superseded by the shared-world `run_console --dual`. Migrating it would
have needed a two-supervisor launcher for no new capability: with supervisors each arm is already its own process. Operator:
delete it.

Deleted: `dataflows/dual_arm_sim.yml`, `examples/configs/sim_arm_b.yaml` (its second instance, used by nothing else), the
run_console self-check's port comparison against it; RUNBOOK says so. Project: `configs/entries/sim_console_demo_b.yaml` (the same
second instance, for that graph only) and its allow-list entries. Checks: `check_dataflows` (8 library graphs), run_console
`--self-check`; project `check_config_duplicates`, `check_dataflows`.

Review: ACCEPT. Taken: the RUNBOOK's reason is scoped (supervisors isolate arms on hardware; the sim `--dual` console is one process, so sim process isolation is no longer exercised). Named: nothing asserts distinct console ports across the real arm configs (run_console's port probe catches a collision at launch only).
