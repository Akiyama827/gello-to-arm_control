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
