# D1, gripper: the hosted world serves the real Hand protocol (SimHandBridge)

Operator decisions 2026-10-08: the supervisor-hosted twin serves the FR3 Hand through the REAL `hand_bridge` line protocol
(not a new `scene_action/1` op), and D1 is scoped to the gripper now (the base writer and scene mutations come with
`sim_workcell_add`, step 4, designed with D2's multi-device hosting).

`arm_control/end_effectors/sim_hand_bridge.py`:
* `parse_command` and `_Admission` port `rt/src/hand_admission.hpp` rule for rule (one command, never a queue; the client echoes
  the epoch; REJECT `offline`/`stopping`/`stale_epoch`/`busy`/`stale_state`/`invalid_command`; the epoch bumps on admission,
  cancellation and client change), minus the device-session and failed-stop paths, which cannot happen in the sim.
* The server ports `hand_bridge.cpp`'s serve loop: one client at a time; every 0.1 s `STATE`, `CAPS active_stop` and
  `META 2 <epoch> <available> <busy> <measured> <seq>`; `ACCEPT/REJECT`, `DONE <token> <ok>`, `GDONE <ok>` (GRASP only),
  `STOPPING <epoch>`; a new action needs a measured sample under 0.6 s old taken after the last action; a disconnect stops
  committed work and otherwise only cancels (a completed grasp keeps holding).
* The finger physics is the project's measured SimHand model, moved: the width command ramps at the commanded speed; a GRASP
  is HELD when it settles inside the epsilon band and blocked >= 3 mm short of the command (an empty close is not); a held
  grasp drives the fingers shut in force mode at the commanded force; `grasped` needs the band and > 2 N of servo load; a MOVE
  succeeds when it settles within 1 mm of its goal (blocked short = franka's move failure); HOME opens to 80 mm; GSTOP ends
  the action where the fingers are and lets a held grasp go.
* `ScenePlant` hosts it with `sim_hand: {port, bind?}` (port required): started with the plant, closed with it, and driven
  from the plant thread once per tick (a sample after the step, a finger command for the next).

One difference from the old SimHand: an OPEN now ramps at the commanded speed, as the real Hand does; SimHand opened instantly
because the plant's dock-release detection keys on the open target (a ramped open lagged ~0.7 s and let the module slide off
before the weld check). Step 2c must make that detection key on the measured jaws, not lean on an unrealistic Hand.
Retires: the project's SimHand/SimHandMove (`assembly/runtime/sim_bridge.py`) when `sim_bench` runs on this server (2c).

Checks: `tools/check/check_sim_hand.py` (new) drives the server with a fake two-finger plant at 250 Hz feeding `on_sample` as
ScenePlant does: parser bounds; stale-epoch, invalid, busy and stale-state rejects; MOVE -> DONE 1 and no GDONE; GRASP on an
object -> GDONE 1 / DONE 1, held under force, `STATE w 1`; the grasp survives a disconnect; GSTOP releases it; an empty GRASP ->
GDONE 0; GSTOP mid-move stops silently; a disconnect stops a committed move; the project's own `BridgeClient` + `HandGraspFsm`
grasp end to end. Mutants caught: no busy check, no freshness, GDONE for MOVE, stop on every disconnect, no epoch check. Survives
(equivalent): reporting DONE regardless of the epoch -- a stop or disconnect always clears the running action first; kept
because the C++ server has it. On the real composed bench world (scratch): the server starts with the plant, `MOVE 0.04` ->
ACCEPT, DONE 1, the twin's fingers 0.08 -> 0.04 m.
