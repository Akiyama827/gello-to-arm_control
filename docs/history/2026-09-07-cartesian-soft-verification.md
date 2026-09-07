# Cartesian Soft verification — 2026-09-07

## Scope and provenance

Baseline parent `d65978189101dd65b0a8ad176ea91191832b8133`, library
`a94f766c0e484fc74331c5fe3fabd812ecc133a1`, matching gitlink; both on
`feat/operator-gated-add-green`. Preserved earlier uncommitted, user-approved
console policy changes: no general deadman hold button; independently expiring
hold-to-jog. Perception work was not touched.

Implementation stays in contracts, control, UI, plants and reusable RT core.
No planner/FK/IK/collision work was added to the controller process; the native
plant computes Cartesian torque. No node-source imports, package reorganization,
project CAD/calibration moves or reverse dependencies were introduced.

Soft holds measured position AND orientation with compliant nullspace. The
row-space projector uses fixed-size, dimension-bounded QR, not an iterative
SVD in the RT tick. It follows Franka's kinematic impedance/nullspace structure,
not a dynamically consistent operational-space controller or safety standard.
Existing torque/slew limits remain the final command boundary.

## Commands and observed results

From the library root in the control Python environment (3.13.13, MuJoCo
3.12.0, Dora CLI/client 1.0.1), unless indicated otherwise:

```sh
PYTHONPATH=. python tools/bench/check_pose_hold_contract.py
PYTHONPATH=. python tools/bench/check_pose_hold_controller.py
PYTHONPATH=. python tools/bench/check_console_authority.py
PYTHONPATH=. python -m arm_control.control.arm_controller
PYTHONPATH=. python nodes/arm_console.py --self-check
PYTHONPATH=. python rt/bindings/pose_hold_check.py
PYTHONPATH=. python tools/bench/check_pose_hold_sim.py
node --input-type=module --check < arm_control/ui/static/console.js
cmake --build /tmp/control-soft-rt-build -j2
cmake --build /tmp/control-soft-franka-build -j2
/tmp/control-soft-rt-build/pose_hold_selfcheck
/tmp/control-soft-rt-build/arm_rt_server --mit-check
ARM_RT_SERVER_BIN=/tmp/control-soft-rt-build/arm_rt_server PYTHONPATH=. python -m arm_control.plants.remote_rt.client
```

All exited zero. `--mit-check` printed golden encode/decode records, not a
live CAN check. Native self-check asserts include no heap allocation, rank
deficiency, quaternion sign invariance, target-id capture and ramp/limiting.
Python encoder assertions check v1/v2 sizes, gain bounds and capability refusal.
Controller checks include stale/nonfinite/moving state refusal, Float-to-Soft
Track-gain fallback, plan/jog exclusion, stop/disarm and measured-pose exit.
Fake RT tracked to 0.1 mrad, held on stale commands, latched stream loss and
recovered through DISARM/ARM. Invalid/oversized/replayed packets cannot refresh
the watchdog in that check. No physical backend was launched.

Exact protocol comparison also passed:

```python
import subprocess
cpp = subprocess.check_output(['/tmp/control-soft-rt-build/protocol_selfcheck'])
py = subprocess.check_output(['python', '-m', 'arm_control.plants.remote_rt.protocol', '--hex'])
assert cpp == py
```

`ruff check` passed for changed Python implementation, contracts and all four
new bench checks plus `rt/bindings/pose_hold_check.py`. AST parsing of every
changed/untracked Python source and `git diff --check` passed. No pytest used.

The freshly compiled binding was installed with
`python -m pip install --no-deps -e rt/bindings` in the console environment.
Earlier checks used Python 3.10.19 in `rob` with an explicit 3.10 binding path.
The first no-build-isolation install failed because that environment lacked
scikit-build-core; normal isolated installation succeeded. A 3.10 binding
cannot be imported by Python 3.13. The `rob` environment lacks Dora metadata,
so full graph checks use the actual control environment, not `rob`.

## Physics and runtime evidence

Headless standalone FR3: 1 N·m external elbow-joint torque applied for three
seconds after two seconds of settling; release followed for five seconds.

| Quantity | Observed |
| --- | ---: |
| Maximum joint displacement, Soft | 0.111795 rad |
| Maximum EE position error | 0.005209 m |
| Maximum EE orientation error | 0.019537 rad |
| Final EE position error | 0.000407 m |
| Final EE orientation error | 0.000994 rad |
| Final maximum joint speed | 0.002027 rad/s |
| Same elbow torque under Track, joint displacement | 0.000898 rad |

The harness checks finite state, compliance, bounded pose deflection, recovery,
settling and a measured-joint Track exit. These are model-specific observations,
not hardware limits. The TCP body's fixed transform is retained in single-model
MuJoCo loading; composed-scene/recompile logic is unchanged.

Launched `PYTHONPATH=. python examples/run_console.py`, then used the real
browser UI: ARM; Soft; refused Plan in Soft; Track; a small joint target;
Plan+preview (3 samples, 0.72 seconds); Execute; successful leg result; Soft
again; DISARM. The mode text and disarmed indication updated, and the layout
was visually inspected. The launcher was stopped by Ctrl-C; all five nodes
reported successful exit. The separately launched Rerun viewer is persistent.

Warnings retained: duplicate visual geom names from staged FR3 URDF; some
simulation wall-clock lag; repeated Dora register-message size warnings during
the concurrent graph session. The latter were not diagnosed or suppressed.
The browser rendered correctly, but a listening Rerun viewer alone is not
proof of uninterrupted visualization transport.

From the consuming project's Control root:

```sh
PYTHONPATH=.:libs/arm_control /home/ubu/miniconda3/envs/rob/bin/python tools/demo/workcell_add.py --output /tmp/control-soft-headless-add.json
PYTHONPATH=.:libs/arm_control python /tmp/control-soft-gated-3VvSwb/run.py
python tools/run.py real franka motion --dry-run
```

Headless add passed all gates, model revision 1, terminal commit. Existing gated
sim harness passed all twelve ordered gates and terminal commit in 51.56 s,
after verifying initial disarm; launcher terminated with its expected 143.
Compact hash-verified evidence is retained in the consumer's
`Control/evidence/cartesian_soft/`. This does not erase earlier intermittent
workcell issues documented in that project's runbook. Real launcher dry-run
resolved the intended real Franka graph/config; it did not launch hardware.

Independent read-only review found stale/nonfinite Soft admission and stale
console health bypass; both were repaired and rechecked. Subsequent native RT,
transport and transition review found no further high-impact issue.

## Explicitly not validated or performed

- No RT service deployment, restart, live robot command or hardware compliance
  experiment. New server capability `pose_hold=2` is required for real Soft.
- No hardware loop-time certification or new joint-limit/speed policy. Existing
  robot protections remain; example gains require deliberate physical bring-up.
- No claim of exact EE invariance, dynamic nullspace decoupling, or safety
  certification. Physical payload and configured EE frame must be verified.
- No Cartesian Soft for composed workcell plants or DM hardware; no assembly
  behavior, gripper semantics or perception ownership was moved.
