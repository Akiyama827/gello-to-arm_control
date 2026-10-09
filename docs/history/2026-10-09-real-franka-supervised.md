# CTRL-053 graphs 5-8: the FR3's hardware graphs run on its supervisor (refactored and checked offline; NOT run on hardware)

Operator 2026-10-09: graphs 1-8 approved in one go; real graphs refactored only, never run or qualified here (the track list holds
the bench rungs).

Library:
* `dataflows/real_franka_motion.yml` (graph 5): `arm_controller` + `rt_interface` -> `supervised_arm` (the supervisor runs the
  controller over the RT link); console, replay, visualizer read it; jog = setpoint/1 (saturated); the console gets
  `plant_capabilities`. The motor CSV comes from `trace_logger`.
* `viz/motor_state_logger.py` + `nodes/trace_logger.py`: the same CSV (same file naming, same columns: measured
  pos/vel/torque, `qdes` = the PC's command, `pos_cmd/vel_cmd/tor_cmd` = the RT's servoed word) from the supervisor's trace/1, one
  row per controller tick; ticks that fell off the 0.64 s ring between reads are counted and reported, never invented; a missing
  field is nan, never 0. Self-check (`python -m arm_control.viz.motor_state_logger --self-check`).
* `supervisor/dora_arm.py`: Soft is offered exactly when the supervisor's plant runs pose hold (`capabilities.pose_hold`), in
  `plant_capabilities` and `motor_health` (it was always False: graph 1's open item).
* `tools/check/check_console_grasp.py`: skips Dora timer inputs.

Project:
* `dataflows/real_franka_workcell_motion.yml` (graph 6): node for node the library graph (project console/visualizer).
* `dataflows/calibration_workcell.yml` (graph 7) and `dataflows/real_workcell_add.yml` (graph 8): `arm_controller` + the FR3's
  `rt_interface` -> `supervised_arm`; the planner reads `motor_health` (a latched supervisor fault stops the sequence, as in
  sim_bench); the Hand, the base driver and the base's RT node take `arm` from `supervised_arm` (the base ARMs with the FR3 until
  it has its own supervisor, graph 4); the motor CSV from `trace_logger`.
* `tools/run.py`: `Scenario.plant`; `real franka motion` (FR3 #1 and #2) is supervised over the RT link; `calibrate` and
  `real-add` too (`WORKCELL_SUPERVISED`); one start path `start_graph_supervisor(config, mode, env, plant)` and one run path
  `run_supervised_graph` (interrupt-safe start, graph, stop) for scenarios and workcell modes; a hardware supervisor runs over the
  STAGED config the graph gets. Dry runs say so.
* `tools/check/check_supervised_real.py` (new, in the suite with the fake server): each of the four hardware launches is staged
  exactly as run.py stages it, then ONLY rt.host/ports point at a local fake arm_rt_server with that instance id and joint order:
  no in-process controller and no FR3 RT node in the graph, the supervisor starts over the RT plant with the gains supervised_arm
  resolves, setpoint/1 and trace/1 offered, trace rows carry the CSV's columns. (The fake reports q = 0, outside the FR3's joint-4
  range: the policy refuses its hold and ARM, as it must; ARM and jog are the sim checks'.)
* `tools/check/check_calibration_sim.py`: the twin IS the calibration graph (Hand included, the sim profile serves a sim Hand);
  only the supervisor's plant differs (sim). End to end: ARM, plan, gated Execute, leg_result, DISARM, the trace CSV's final
  error < 5 mm.
* `tools/check/check_add_contracts.py`: the add graphs' command producers (real-add: none in the graph, the supervisor commands);
  the FR3's end frame checked from the staged config, the RT node's ee_frame gate on the base's node; the manual-execution checks
  build the controller by the same factory the supervisor uses.

Retired in these graphs: the Dora `arm_controller` and the FR3's `rt_interface` node; `motor_state_logger` (Dora topics) where a
supervisor commands. Not retired: `rt_interface` (the base, until graph 4; the DM arm's graphs), `motor_state_logger` (in-process
graphs). Not run on hardware: every behaviour change for the operator is a track-list row.

Review round 1: ACCEPT (no wiring error, nothing producing unsafe motion; RT server down at launch: the supervisor exits in
0.2 s, run.py returns 1 with the log kept). Taken:
* (SHOULD) A bench fault left no evidence: the supervisor's own lines went only to a temp log deleted on a clean stop. Every
  supervisor line is now ECHOED to the launcher's terminal and written to the log (`launch.py`), and a hardware run (`plant=rt`)
  keeps the log and prints its path.
* (SHOULD) The trace CSV repeated the last RT packet as a measurement while the stream was stale (a 0.4 s SIGSTOP of the fake:
  41 ticks, one state_seq): a tick without a fresh packet now writes nan in the RT columns (the PC's own `qdes` stays).
* NITs: the logger reads once more after STOP (the DISARM ticks) and survives a transient SDK error (counted); `qdes` is the
  command computed on THIS tick (the Dora logger paired state k with command k-1: `qdes - pos` differs by about v*dt, ~4 mrad
  at 0.4 rad/s); check_supervised_real no longer claims a Soft assertion it does not make; run.py's real-add mode path comes
  from WORKCELL_SUPERVISED; "final error < 5 mm" is 0.005 rad; the calibration twin also swaps the mode config (motion_franka).
Operator-visible changes on the bench (track list): console and visualizer get motor_state at 100 Hz (was the 60 Hz
motor_state_viz); the CSV's qdes is nan after DISARM (it held the last value); Soft is offered on fr3_1 when the RT server runs
pose hold, but its controller refuses it there (pose_provenance: calibrated), as before.
Base coupling in real-add (read, not run): the base now ARMs only once the FR3's RT state confirms ARMED (both armed on the
operator's request before), and DISARMs once the FR3 reports DISARMED. OPEN, OPERATOR DECISION: when the FR3's supervisor is
LOST the base stays ARMED (as on the retired path: rt_interface ignored a closed input) -- publishing DISARM to it would leave a
tilted base limp under the FR3; keeping it armed leaves it powered with no FR3 authority. Unchanged here (the adapter rule
"unknown is never published as DISARMED" stands).
Named: no per-instance lock across calibrate/real-add/real franka motion (as before); ARM over the RT link untested offline (a
`--q0` for the fake backend would allow it: a C++ change, not taken here). FAIL (not this slice; owner: the machine's load):
check_sim_supervisor's observe-rate assertion (0.84-0.88x real time) while 30+ external `run_expert.py` jobs held load ~94 on
32 cores; the check starts its supervisor directly (not through the changed launcher); re-run when the load drops.
