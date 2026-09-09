# rt/ — the RT machine's servo server

One static C++ binary (`arm_rt_server`) that owns the arm's 1 kHz torque loop
on a realtime Linux box, speaking a small UDP/TCP protocol to the PC's Dora
graph. The PC-side counterpart is `arm_control/plants/remote_rt/client.py` behind
`nodes/rt_interface.py` — the same plant-node contract every other bridge
speaks, so graphs move an arm onto the RT machine by swapping one node path.

**Design rule:** this process is a servo with reflexes, never a brain. Torque
ceilings, staleness thresholds, and slew are launch flags; everything
task-shaped (gain schedules, grasp policy, phases, planning) stays on the PC
and arrives per tick in `CommandPacket`s. Dora ends at this machine's door —
no daemon, no Python, no graph lifecycle owning the arm.

## Layout

| Path | What |
|------|------|
| `include/arm_rt/protocol.hpp` | Wire structs — hand-mirrored in `arm_control/plants/remote_rt/protocol.py`, parity enforced (below) |
| `include/arm_rt/servo_law.hpp` | THE torque law, header-only: PD + ff, clamp, slew vs the robot's own `tau_J_d` echo. Shared by the RT loop and the sim binding |
| `include/arm_rt/seqlock.hpp` | The non-RT↔RT seam (single-writer seqlock, latest-wins) |
| `src/rt_loop.cpp` | The servo thread: backend read → staleness policy → law → write. Constructs the backend ON this thread (libfranka requirement) |
| `src/udp_link.cpp` | Commands in (latest-wins), state out (paced) |
| `src/control_session.cpp` | TCP: arm/disarm/ping + STATUS/FAULT events |
| `src/backend_fake.cpp` | Damped-integrator loopback plant — the PERMANENT test double (see below) |
| `src/backend_franka.cpp` | FR3 over libfranka ActiveControl (`-DWITH_FRANKA=ON`), **unvalidated on hardware** |
| `src/backend_dm.cpp` | DM FDCAN backend with exact reply-ID filters and MIT wire-format self-check |
| `src/hand_bridge.cpp` | Generic Franka Hand TCP bridge; built with libfranka |
| `bindings/` | Optional pybind module `arm_rt_servo` so the MuJoCo twin closes the same compiled law |

Concrete service units and device-specific bridges belong to the consuming
project's deployment tree, not this reusable core. Moving those files does
not alter an installed RT host. The bring-up notes below record existing
engineering checks; they are not certification of a particular deployment.

## Build

```bash
cmake -B build -G Ninja rt && cmake --build build          # fake + selfcheck
cmake -B build -G Ninja -DWITH_FRANKA=ON rt && cmake --build build   # + FR3
pip install -e rt/bindings                                  # optional: sim same-law
PYTHONPATH=. python tools/bench/rt/torque_cap.py build/arm_rt_server  # offline caps + parity
```

## Protocol in one paragraph

UDP fast path, both directions latest-wins with sequence numbers:
`CommandPacket` (q_des, qd_des, tau_ff, kp, kd — the full bridge contract
word, 100 Hz from the executor) and `StatePacket` (q, dq, tau, the servo's
own post-clamp `tau_cmd`, its current target `q_cmd`, flags, plus reserved
FT-sensor fields), streamed at `--state-hz` to the source address of the last
command. TCP control channel, fixed 128-byte frames: HELLO (n + backend
name), ARM, DISARM, PING/PONG, STATUS, FAULT. Clocks are NOT assumed synced —
freshness is arrival-time based, timestamps are diagnostics.

Parity between `protocol.hpp` and the Python protocol is enforced, not hoped
for: `./build/protocol_selfcheck` and `python -m arm_control.plants.remote_rt.protocol
--hex` must print identical golden lines, and `python -m
arm_control.plants.remote_rt.client` diffs them automatically before its loopback
test. Any layout change edits both files in one commit and bumps `VERSION`.

Cartesian Soft adds a separate 784-byte command version 2: the unchanged
664-byte joint-command prefix, followed by 15 doubles
`[id, kc[6], dc[6], nullspace_kp, nullspace_kd]`. State/control and ordinary
joint commands remain version 1. Only Franka advertises `pose_hold=2` in HELLO;
the client rejects Soft without that capability and rejects legacy Cartesian
tails. Unsupported versions, invalid gains and replayed sequences do not renew
the command deadman.

`PoseHold` captures measured EE pose and nullspace joint posture on each new
nonzero id. The shared compiled law uses local pose/Jacobian/Coriolis, ramps
gains over 0.5 s and projects nullspace spring/damping before the existing
torque clamp/slew. Franka adds gravity itself. Staleness or fault resets Soft
and enters measured joint hold using the last accepted joint fallback gains.
The gain ceilings are validation bounds, not hardware safety certification.
Offline checks: `pose_hold_selfcheck` includes Eigen's no-allocation guard;
`rt/bindings/pose_hold_check.py` checks the compiled Python API and encoder.

## Safety semantics (the part to re-read before bench day)

`--tau-max NM` optionally lowers every joint's command ceiling to the smaller
of this positive finite value and its backend limit. Omit it to preserve the
backend defaults. The resolved per-joint limits are printed at startup and
apply to joint/Cartesian commands and initial, stale, and latched-fault holds
through the same clamp/slew law. The final clamp wins even when the robot's
torque echo lies outside the cap. The option cannot be changed by a command
packet; deployment values belong in the consuming project's service unit.

DM keeps its firmware-compatible MIT encoding scale. With an explicit cap,
the torque field uses only codes whose decoded values lie inside that cap;
unrepresentably small caps are rejected before opening the bus. This limits
requested torque, not measured torque, thermal duty, or contact force. It
does not provide a brake or make disarming a gravity-loaded joint safe.

Authority ladder while ARMED, most-alive first:

1. **Fresh command** (< `--hold-ms` old): track it with its own kp/kd/tau_ff.
2. **Stale** (> `--hold-ms`): HOLD the pose captured at staleness, last
   command's gains (or `--hold-kp/--hold-kd` if none ever arrived — the
   state right after arming, which *is* "hold where you are"). tau_ff is
   slewed out, never stepped.
3. **Stale past `--fault-ms`**, control session lost, or plant error: still
   holding, but LATCHED — commands are ignored and ARM is refused until an
   explicit DISARM→ARM cycle. Nothing auto-re-arms. Losing the PC never
   drops the arm: it parks it.

DISARM is the only authority drop (backend `stop()`: for the FR3 that is the
last accepted torque + `motion_finished` — a controlled stop, never zero
torque, which would drop a loaded arm). The staleness clock starts AT ARM,
matching the bench bridges — and so does the **command epoch**: whatever the
command seqlock held before ARM (the client's zero-authority address-teach
packet, a pre-fault leftover setpoint) is never tracked. Found live on the
FR3 impedance rung: the prime packet's zero gains were adopted as "the
task's gains" and the hold spring never engaged. The servo's slew limiter doubles as the startup
ramp: first tick slews from the robot's measured `tau_J_d`, so there is no
separate soft-start path.

The ARM ack rule for clients: `FAULTED` in the STATUS flags is the refusal —
`ARMED` alone is not consent, because a fault-holding server keeps its ARMED
flag on purpose. Symmetrically, a DISARM ack is not proof authority dropped
(the RT loop applies it at its next sample, seconds away inside a blocking
plant call) — `safe_stop()` confirms against the state stream and returns
the verdict.

Hardening from the post-rung-2 adversarial review (all bench-triggered or
review-caught, all in the stale-authority class):

- **Gains are snapshotted at command acceptance** — the hold branch never
  dereferences the live command buffer, so a stray zero-gain datagram (every
  tool's address-teach prime) cannot un-spring a parked arm.
- **ARM is a generation counter, not a level** — a DISARM→ARM pair faster
  than one servo tick still triggers every per-epoch reset (gains, hold
  pose, plant retry, backend stop).
- **While armed, UDP commands are accepted only from the control client's
  IP** — nothing else can steal the state stream or inject authority.
- **The control session has a real deadman** — the server PINGs at 4 Hz and
  treats ~1.2 s of silence as session loss (a half-open socket from a dead
  PC otherwise takes the kernel's 2-hour keepalive to notice).
- NaN/Inf commands are dropped at the UDP boundary; misconfigured safety
  flags (`hold-ms >= fault-ms`, `slew <= 0`) refuse to start; the seqlock
  read is bounded (a preempted writer can't spin the FIFO reader).

## Bring-up ladder

0. **Loopback on the PC** (no hardware): `python -m
   arm_control.plants.remote_rt.client` — protocol parity, tracking through the
   compiled law, staleness→hold, fault latch, refused ARM, DISARM/ARM
   recovery. GREEN 2026-07-27; keep it green.
1. **Fake on the RT box**: same demo with `rt.host` pointed at the box.
   Proves the link, the kernel, and the service unit. Quantify with
   `python tools/bench/rt/timing.py --host <box>` — it listens to the
   disarmed state stream (states only, sends one zero-gain packet) and
   reports the servo's per-tick wakeup jitter from the tick stamps.
2. **FR3 gravity-float, then impedance-hold** — first hardware validation of
   `backend_franka`, zero PC-side commands, driven by
   a deployment-owned hand-guidance tool (explicitly arms, logs, and disarms).
   Run the server manually in the foreground for this rung,
   `--fault-ms 3600000` (armed-with-no-commander is the test's steady
   state), operator on the stop:
   - `--hold-kp 0 --hold-kd 0` → zero torque on top of the robot's own
     gravity compensation; push the arm around by hand.
   - `--hold-kp 30 --hold-kd 2` → the arm springs back around the pose
     captured at ARM.

   **Declare anything bolted past the flange** (`--ee-mass KG [--ee-com
   X,Y,Z]`, the CoM in the flange frame). The value is ADDED to Desk's
   end-effector config — pass the camera+mount or module mass alone, not
   hand+camera. Float has nothing but the robot's own gravity model holding
   it up: 0.2 kg undeclared is ~1 N·m of permanent elbow torque, and the arm
   creeps into its joint-4 limit and trips `joint_velocity_violation`
   (2026-08-06, twice, before the flag existed).
   A hard shove trips the collision reflex — the safe outcome; recover
   DISARM→ARM (the backend runs `automaticErrorRecovery` on re-arm).
3. **FR3 tracking**: slow sine from the PC graph via `rt_interface`;
   compare `q_cmd` vs `q` in the state stream against the sim twin.
4. **Graph integration**: the real motion graph with `plant_interface` →
   `nodes/rt_interface.py`; then the project's task-specific validation.

## RT host checklist (rung 1 prerequisite)

- PREEMPT_RT kernel (`uname -v` says `PREEMPT_RT`); `/etc/security/limits.d/`
  grants `rtprio 99` + `memlock unlimited` — needed even with the systemd
  unit (which sets both) the moment you run the server manually in a
  foreground bring-up session. **99, not 95**: libfranka's `kEnforce`
  requests the *highest* FIFO priority and refuses to start below it
  (confirmed live: 95 ok, 99 EPERM = "unable to set realtime scheduling").
  Re-login after editing — PAM applies limits at session start.
- Isolate the servo cores: `isolcpus=2,3 nohz_full=2,3 irqaffinity=0-1` on
  the kernel cmdline; the unit pins to 2–3.
- Cap idle states: `intel_idle.max_cstate=1 processor.max_cstate=1` on the
  cmdline. PREEMPT_RT does NOT do this for you, and it dominates everything
  else: measured on rung 1, deep core+package C-states cost 90–140 µs of
  wakeup latency (the idle box was *worse* than a desktop PC); with the
  package held awake the same box's servo tick measured p99 3.4 µs /
  max 5.1 µs — under full housekeeping-core load.
- Pin the arm-link NIC's IRQs off the servo cores; disable interrupt
  coalescing on that NIC (`ethtool -C <if> rx-usecs 0`).
- Direct cable to the PC (no switch), static IPs, and for the FR3 a second
  NIC to the control box — the servo's FCI traffic never shares a wire with
  the PC link.

## backend_fake is permanent

Most of this server is not the servo law — it is threads, sockets, seqlocks,
staleness policy, and latching, all of which `backend_fake` exercises on any
machine in seconds. It is also the only safe place to rehearse the failure
paths (packet loss, session death, latched re-arm). It stays dumb by rule: a
damped integrator, never a second simulator — closed-loop realism is the
MuJoCo twin's job on the PC.
