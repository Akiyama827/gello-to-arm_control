# The composed plant honours `sim_gravcomp` (the FR3 twin holds itself up)

Operator decision 2026-10-08 (the ScenePlant review's F1): plant gravity compensation for the twin, as the FR3's control box
does on the bench. Every FR3 simulation already modelled it (`robots/fr3/sim.yaml`: `sim_gravcomp: true`, with
`arm.plant_gravity_comp: true` so the executor ships RNEA minus gravity) except the composed bench (`sim_franka_bench`): its
composed plant path ignored `sim_gravcomp` and its config said neither. Under the supervisor a disarmed twin therefore fell (the
link's release and hold words carry no gravity), and the Dora node's spawn hold sagged.

* `build_world`, composed branch: with `sim_gravcomp`, the arm's bodies (`scene.arm.prefix`, required) are gravity compensated
  before the first compile (gravcomp is baked into the spec).
* Project (parent): `configs/simulation/franka_bench.yaml` includes `robots/fr3/sim.yaml` (its own update rate and
  self-collision still win) and states `arm.plant_gravity_comp: true`.

Measured on the composed bench world, before -> after: the Dora spawn hold over 2 s sagged 0.38 -> 0.0 rad; a kp 200 hold with no
feedforward 0.12 -> 0.0 rad; the supervised twin disarmed for 2 s 3.72 -> 0.0 rad. Gravcomp lands on 30 bodies, all `asm_`
(none on the base or the module). Sim time advances (1.004 s per s) and an armed move with no gravity feedforward tracks to
0.8 mrad. Dora `sim bench`, 120 s startup smoke: all six nodes ready, perception registers the module (fit 1.00), no traceback;
its "sim time behind wall clock" warnings are older than this change (23 both ways, A/B against the old config).

Review round 1 (BLOCK on the record, behaviour verified): gravcomp on 30 `asm_` bodies only (19.0 kg; grafted modules keep
`mod_`, so a carried payload stays the executor's `_payload_tau`); executor and plant agree (spawn hold 0.000 mrad; a
double-count mutant drifts 69.8 mrad, plant gravcomp off 70.7 mrad). Corrected: my record and code comment claimed the composed
arm also gets `ARM_ARMATURE` 0.1. It does not (that loop is workcell-only; measured `dof_armature` 0 on every `asm_` joint), so
the bench is NOT the other FR3 twins' 0.5 ms / armature 0.1 plant; a 2 s step settles either way (4e-5 rad/s). Applying armature
on the composed path would be its own decision. Taken in the parent: the include comment names the keys this path ignores;
a guard check builds the bench world and enforces `sim_gravcomp` == `arm.plant_gravity_comp` for every simulation entry (the
two-file invariant, else torque is double- or zero-counted, ~70 mrad). The parent's config commit is followed directly by its
pin (amending the pushed commit would need a force-push).
