# rt/ — the RT machine's servo server

One C++ binary (`arm_rt_server`) that owns the arm's RT servo loop
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
| `include/arm_rt/mailbox.hpp` | The non-RT↔RT seam (SPSC triple-buffer mailbox, latest-wins, ownership by atomic exchange) |
| `src/rt_loop.cpp` | The servo thread: backend read → staleness policy → law → write. Constructs the backend ON this thread (libfranka requirement) |
| `src/udp_link.cpp` | Commands in (latest-wins), state out (paced) |
| `src/control_session.cpp` | TCP: arm/disarm/ping + STATUS/FAULT events |
| `src/backend_fake.cpp` | Damped-integrator loopback plant — the PERMANENT test double (see below) |
| `src/backend_franka.cpp` | FR3 over libfranka ActiveControl (`-DWITH_FRANKA=ON`); hardware-run at `c23a0c7` ([timing report](https://github.com/Westlake-Work/modular_robotic_arm/blob/f042a681841393a5a0ad7b82469f8ef0997290c8/Control/docs/status/2026-09-24-dual-arm-rt-timing.md)), later changes offline-checked only |
| `src/backend_dm.cpp` | DM FDCAN backend with exact reply-ID filters and MIT wire-format self-check |
| `src/hand_bridge.cpp` | Generic Franka Hand TCP bridge; built with libfranka |
| `bindings/` | Optional pybind module `arm_rt_servo` so the MuJoCo twin closes the same compiled law |

Concrete service units and device-specific bridges belong to the consuming
project's deployment tree, not this reusable core. Moving those files does
not alter an installed RT host. The bring-up notes below record existing
engineering checks; they are not certification of a particular deployment.

## Hand active STOP

`hand_bridge` keeps one libfranka Gripper session. Its action thread owns reads,
move, grasp and homing; a separate worker invokes the same session's thread-safe
`stop()` for `GSTOP`. This uses the vendor's
[move/grasp interruption API](https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Gripper.html).
It neither joins nor blocks the arm RT servo loop.

- `CAPS active_stop` advertises the updated bridge; legacy META 2 framing stays.
- `STOPPING <epoch>` acknowledges receipt, not physical completion. Admission
  stays busy while stopping. Canceled actions cannot report successful DONE.
- Stop is retried while the canceled action is active, then once after its exit
  to cover cancellation just before SDK entry. New commands require a successful
  final stop and a subsequent fresh measured sample. A false/exceptional final
  stop inhibits commands (META `available 0`). The latch clears only after a
  retried stop succeeds and a measured sample read after it arrives. A new
  Hand device session retries the stop once, discarding all old actions; an
  explicit GSTOP also retries. The recovery never opens, homes or re-sends an
  action. A PC client reconnect alone does not clear the latch.
- Client disconnect interrupts an active action and discards queued commands.
  It does not stop an already completed grasp, including when an unsent open was
  queued. Explicit GSTOP does request stop even when idle; do not use it as a
  promise to retain grasp force or a suspended object.
- The gated position-move client requires this capability. Updating Python alone
  does not update the daemon installed on the RT host. Standalone slider/grasp
  interfaces keep their existing admission policy.

This is a software stop, not a safety-rated E-stop. The 20 ms retry interval is
not a physical latency guarantee. Homing interruption, loss of the robot link,
bridge process death and physical stopping latency are not certified by the
offline checks. Deployment and an authorized empty-Hand stopping test remain
separate acceptance steps.

## Bench inventory

Every `tools/bench/rt/*.py` file has one role. **Regression** entries belong
to the device-free C3 suite (`95ea1d9`); `native_reference.py` and
`reference_upload.py` run there through their CTest self-check wrappers.
**Diagnostic** entries are opt-in investigations or support modules, not
routine pass gates. The current `controller_extension.py` regression exposes an
intermittent pending-COMMIT cancellation race (10/15 runs fail at `3c82090`);
C3 reports it as FAIL, and **CTRL-034** owns the behavior correction (branch
`task/control/CTRL-034-pending-commit-cancel`). Reviewer bisection
(failures/runs): 0/21 at `425a943`–`5b06d28`; 1/6 at `55188d3`; 1/7 at
`30037e1`; 0/6 at `5325bb8`; 3/6 at `9184f43`; 10/15 at `3c82090`.

| File | Class | Scope |
|---|---|---|
| `bench_target.py` | diagnostic | Local/remote bench target helper |
| `check_two_instances.py` | regression | Two fake instance isolation |
| `constrained_cpu.py` | diagnostic | Loaded timing matrix runner |
| `controller_extension.py` | regression | Controller preset/payload transitions |
| `envelope_age.py` | diagnostic | Timing collection helper |
| `first_envelope_probe.py` | diagnostic | First-envelope collision investigation |
| `first_envelope_trace.py` | diagnostic | Opt-in client timeline helper |
| `mode_select.py` | regression | Mode wire parity and malformed packets |
| `mode_select_supervisor.py` | regression | Mode handoff through fake RT |
| `native_reference.py` | regression | Native reference and expiry through CTest |
| `pc_stall.py` | regression | PC stall and local cancel |
| `reference_horizon.py` | regression | Bounded horizon execution |
| `reference_mode_boundary.py` | diagnostic | Historical v5/v7 arbitration probe |
| `reference_numerics.py` | regression | Native/Python Hermite parity |
| `reference_protocol.py` | regression | v7 wire parity and malformed packets |
| `reference_supervisor.py` | regression | Supervisor/native execution |
| `reference_upload.py` | regression | Upload/session through CTest |
| `stale_envelope_scope.py` | regression | Prior-session envelope collision |
| `stream_races.py` | regression | Stream admission races |
| `stream_supervisor.py` | regression | Stream SDK/supervisor flow |
| `streamed_hold.py` | regression | Streamed-command stall guard |
| `timing.py` | diagnostic | Passive server timing observation |
| `torque_cap.py` | regression | Torque admission and protocol parity |

No tool is classified as retired: the historical probes remain useful for
diagnosis or legacy-compatibility checks, and `first_envelope_trace.py` is
still imported by the active `pc_stall.py` trace path. There is therefore no
retired-tool deletion commit in C4.

## Build

```bash
cmake -B build -G Ninja rt && cmake --build build          # fake + selfcheck
cmake -B build -G Ninja -DWITH_FRANKA=ON rt && cmake --build build   # + FR3
pip install -e rt/bindings                                  # optional: sim same-law
PYTHONPATH=. python tools/bench/rt/torque_cap.py build/arm_rt_server  # offline caps + parity
build/dm_command_selfcheck                                # native command dispatch, no devices
build/dm_backend_selfcheck                                # DM lifecycle via intercepted CAN syscalls
build/franka_backend_selfcheck                            # FR3 exception containment, scripted franka::
build/fault_journal_selfcheck                             # first-cause fault journal ring
build/ipc_selfcheck                                       # mailbox, command admission; also build with -fsanitize=thread
build/stale_envelope_scope_selfcheck                      # prior ARM envelope cannot own a new reference
PYTHONPATH=. python tools/bench/rt/stale_envelope_scope.py build/arm_rt_server  # 50 fake-RT collisions
```

The consolidated device-free check is:

```bash
python tools/bench/rt_offline_suite.py
```

It builds with `WITH_FRANKA=OFF`, runs all 19 available C++ self-checks through
CTest, checks the SDK in streamed and native modes, then runs the fast local
fake-RT regression benches. `reference_session_selfcheck` and
`reference_runtime_selfcheck` need socket/port harnesses; CTest runs them
through the existing `reference_upload.py` and `native_reference.py` probes.
The command prints a revision-labeled PASS/FAIL line for every check and writes
`rt/build-offline/offline-summary.json` plus individual logs. This does not
connect to an RT host or qualify physical control. With `WITH_FRANKA=ON`, CTest
also registers `hand_bridge_selfcheck`.

## DM wire mappings

`--dm-spec 'IF;ID:TYPE[:MST[:PMAX:VMAX:TMAX]],...'` configures each motor's
host MIT encoding and feedback decoding. `ID` is 1..15; the optional reply
Master ID `MST` is 0..0x7ff and defaults to 0. Decimal and `0x` hexadecimal
IDs are supported. `TYPE` is `4310`, `4310p`, `4340`, or `4340p`.

Omitting the three ranges preserves the generic type defaults, including
4340/4340p velocity ±10 rad/s. An override supplies `MST` and all three
positive symmetric half-ranges: position in rad, velocity in rad/s, torque
in N.m. For motors configured in firmware with PMAX 12.5, VMAX 20, TMAX 28:

```text
--dm-spec 'can0;1:4340:0x11:12.5:20:28,2:4340p:0x12:12.5:20:28'
```

These are host wire maps and must match the motor firmware registers; the
server does not probe or change registers. An identical velocity code 2457
decodes to 4 rad/s with VMAX 20 and 2 rad/s with VMAX 10. The operating
`--tau-max 27` remains a separate ±27 N.m command cap within the TMAX 28 map.
Malformed entries, incomplete triples, duplicate IDs, and nonpositive,
nonfinite, underflowed, or overflowing ranges fail before opening CAN.
The doubled range and its encoder product (65535 position levels, 4095
velocity/torque levels) must also be finite. `arm_rt_server --mit-check`
checks mappings, malformed specs, and caps offline, including Release builds.

### Native MIT actuation (500 Hz)

Python retains trajectory sampling, q/qd/qdd and model feedforward computation.
RT forwards `q_des`, `qd_des`, `kp`, `kd`, `tau_ff` to the motor's native MIT
controller. The motor applies its own PD; RT does not encode externally
computed PD as an additional feedforward torque. No model loading or reference
interpolation occurs in RT, and qdd is not added to the network protocol.
Slower PC updates leave the latest reference in place between packets, while
RT continues reading feedback and applying authority/freshness checks at 500 Hz.

All active motors' five fields are checked for finiteness and wire limits
before any motor is enabled or receives a command. Invalid gains are rejected,
never silently clipped. Normal frames match the existing Python MIT codec.
Native payloads that overlap reserved MIT lifecycle control words are rejected.

The safety limiter is the one exception to forwarding feedforward unchanged:
RT predicts total `PD + ff` from the **encoded** fields and latest feedback.
If necessary it changes only the ff code to keep that prediction inside both
the torque cap and the per-tick slew interval; q/qd/kp/kd codes stay unchanged.
If no representable ff can satisfy those bounds, it faults and disables rather
than sending an unsafe command. Invalid batches, failed writes and failed
feedback reads also trigger a best-effort zero-gain/disable sequence. A broken
bus cannot guarantee delivery of that sequence.

DM `tau_cmd` and its slew reference are the last encoded command's **predicted
total torque at sampled q/dq**, not a motor-accepted or measured torque. Actual
torque remains the feedback `tau` field. Onboard PD reacts between samples, so
neither the sampled cap nor slew check guarantees actual motor torque/current
or its rate. A physical motor-side limit needs independent verification.
Franka and fake retain their existing external torque law unchanged.

Deploy the updated binary and generated specification together: older RT
parsers can silently ignore the range suffix and retain their type defaults.

## Protocol in one paragraph

UDP fast path, both directions latest-wins with sequence numbers:
`CommandPacket` (q_des, qd_des, tau_ff, kp, kd — the full bridge contract
word, typically 100 Hz from the executor) and `StatePacket` (q, dq, tau, the servo's
post-limiter `tau_cmd` (DM: sampled prediction), selected targets `q_cmd` and
`qd_cmd`, flags, plus reserved
FT-sensor fields), streamed at `--state-hz` to the claimed session's address
and to observer leases. TCP control channel, fixed 128-byte frames: HELLO
(n + backend name + launch flags), IDENTITY, CLAIM, ARM, DISARM, PING/PONG,
STATUS, FAULT. Clocks are NOT assumed synced — freshness is arrival-time
based, timestamps are diagnostics.

## Identity, session binding and observers (protocol v5)

One `arm_rt_server` process owns one plant. A second FR3 is a second process
with its own ports, `--rt-cpu`, backend session, command epoch, watchdog and
fault latch; nothing is shared between instances except the host.

- `--instance-id` and `--joint-names` are required for `franka` and `dm`.
  They are provisioned deployment data, printed at startup and hashed into
  every state packet. They are a configured label, not proof of which robot
  is cabled to the FCI port.
- After HELLO the server sends `IDENTITY` (instance id, task model id, a fresh
  random 64-bit session id, joint-order hash, capabilities). The client checks
  them against its own config and sends `CLAIM`; the server compares instance,
  joint count and order, session, requested capabilities and (when named)
  task model. ARM is refused without an accepted claim on this session.
- Every command carries `session_id` and `arm_epoch` (the ARM ack's `seq`).
  A datagram from another/expired session or another epoch is dropped
  before it can refresh the deadman, teach a state address or reach the
  servo buffer; the IP pin, first-sender port pin, replay window and
  finite-value checks remain. While disarmed a claimed session's datagram
  only teaches its state-stream address. This is flow binding, not security.
- `ObserveRequest` (16 bytes) leases a read-only copy of the state stream for
  100–5000 ms, up to 4 observers. It grants nothing, refreshes no deadman,
  never changes the commander's address and needs no TCP session; closing an
  observer sends nothing. `RtObserver` checks `identity_hash` and drops
  states from any other instance.
- DISARM's ack `seq` is its generation. `StatePacket.stop_epoch` reaches it
  only after the servo sampled the disarm and `backend->stop()` returned;
  `safe_stop()` requires a state received after the ack with that epoch.
  That is process-level confirmation, not a physical standstill measurement.
- `CAP_CALIBRATED_TASK_SPACE` is never advertised: pose hold uses the
  backend's own model (`libfranka-vendor` on the FR3), not a PC calibration.

Parity between `protocol.hpp` and the Python protocol is enforced, not hoped
for: `./build/protocol_selfcheck` and `python -m arm_control.plants.remote_rt.protocol
--hex` must print identical golden lines, and `python -m
arm_control.plants.remote_rt.client` diffs them automatically before its loopback
test. Any layout change edits both files in one commit and bumps `VERSION`.

State/control and ordinary joint commands use version 5; CommandPacket is
672 bytes and StatePacket 888. `qd_cmd` echoes the RT-selected desired velocity, including zero in
initial/stale/fault holds, Cartesian pose hold and disarm. It is not measured
joint velocity or proof that a plant accepted/tracked the target. CTL_STATUS
requests report current flags without changing authority or clearing a fault.

Cartesian Soft uses a separate 792-byte command version 6: the unchanged
672-byte joint-command prefix, followed by 15 doubles
`[id, kc[6], dc[6], nullspace_kp, nullspace_kd]`. Only Franka advertises
`CAP_POSE_HOLD` in IDENTITY;
the client rejects Soft without that capability and rejects legacy Cartesian
tails. Unsupported versions, invalid gains and replayed sequences do not renew
the command deadman.
Legacy v1–v4 peers are rejected. Deploy PC clients and every arm/base RT
instance together; the base uses the same wire contract as Franka.

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
apply to joint/Cartesian commands and initial, stale, and latched-fault holds.
Torque backends use the shared clamp/slew law; DM uses the native sampled
limiter described above. The final cap wins even when the previous torque
reference lies outside the cap. The option cannot be changed by a command
packet; deployment values belong in the consuming project's service unit.

DM keeps its firmware-compatible MIT encoding scale. The cap applies to the
predicted total including onboard PD, not just the ff field. Unrepresentably
small caps are rejected before opening the bus; an unrepresentable runtime
limiter correction fails closed. This does not limit measured torque, thermal
duty, or contact force. It
does not provide a brake or make disarming a gravity-loaded joint safe.

Authority ladder while ARMED, most-alive first:

1. **Fresh command** (< `--hold-ms` old): track it with its own kp/kd/tau_ff.
2. **Stale** (> `--hold-ms`): HOLD the pose captured at staleness, last
   command's gains (or `--hold-kp/--hold-kd` if none ever arrived — the
   state right after arming, which *is* "hold where you are"). tau_ff is
   slewed out, never stepped.
3. **Stale past `--fault-ms`** or control session lost, with a healthy plant: still
   holding, but LATCHED — commands are ignored and ARM is refused until an
   explicit DISARM→ARM cycle. Nothing auto-re-arms. Losing the PC never
   requests zero authority: it requests a local hold.
4. **Plant error**: torque writes are suppressed. DM also attempts zero-gain
   commands and disable frames; holding is no longer guaranteed.

DISARM explicitly drops authority (backend `stop()`: for the FR3 that is the
last accepted torque + `motion_finished` — a controlled stop, never zero
torque, which would drop a loaded arm). The staleness clock starts AT ARM,
matching the bench bridges — and so does the **command epoch**: each accepted
command is published as one envelope (payload, its own receive time, the ARM
generation it was validated under), and only the current generation is ever
tracked, so a pre-ARM leftover (the client's zero-authority address-teach
packet, a pre-fault setpoint) never is. Every admission check (joint count,
session, host, ARM epoch, flow pin, sequence) runs before any side effect: a
rejected packet cannot pin the flow or teach the peer.

The RT loop owns tracking authority and the CMD_LOST deadline
(`CommandAuthority`). Both derive from the current generation only: its
ARM-start record, which the ARM handler publishes before the generation, and
its newest accepted envelope. A receive time is stamped when `recvfrom`
returns, and a handler delayed after that point does not re-date it. So a
handler of a retired generation that publishes late is neither tracked nor
able to move the new generation's deadline. The state-stream peer is scoped
by generation under its lock, so such a handler cannot redirect it either.
A command is tracked only while its session is the current claimed one. Found live on the
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

Every latch is also journaled on the box, not only sent as `CTL_FAULT` to a
connected client. The server's stderr (journald) gets
`[rt] fault journal: first cause ...` with code, time, ARM epoch, armed flag
and session. It also gets one `secondary cause` line per different code while
latched, and a `stop failed` line when a backend stop fails. DISARM and client
disconnects do not erase these lines. Text past the 104-byte wire field ends in
`...`; text past the 240-byte record is marked `(truncated)`. A full 16-record
ring drops new records (`dropped=`); a producer that loses the slot claim
4 times drops its record (`contended=`). Each enqueue does bounded work: at
most 4 claim attempts and one scan/copy of at most 240 bytes. That is a step
bound, not a measured RT latency.

The wire latch has one state word: FREE → WRITING → SET, CLEARING while
DISARM resets it, and ARMING while ARM prepares a generation.
- DISARM never clears a latch that is still being written. It waits for the
  writer, which normally takes microseconds, until a 100 ms monotonic
  deadline. If the writer has not finished by then, DISARM still drops
  authority but leaves the latch and says so in its reply.
- ARM is admitted only by moving the word FREE → ARMING. A latched fault, a
  latch still being written (even one whose clear timed out before `fault`
  was published) or a reset in progress refuses it ("DISARM again").
  Authority is raised only after the word is FREE again.
- Every latch and journal record carries the ARM generation it arose in. A
  latch from an older generation that lands while ARM prepares the next one
  is journaled as a secondary cause with its own generation.
- The servo never waits. A cause that arrives while the latch is being
  written, cleared or armed is journaled as a secondary cause, and it
  re-latches on a later tick if it persists.
- `CTL_FAULT` goes out once per latch serial, and only for a SET latch. A
  latch that DISARM cleared before delivery is not sent (the journal keeps
  it). An acknowledgement never discards a newer latch's notice.

Export is best effort. `state_tx` drains the ring while running. After every
thread is joined, `main` drops its inherited SCHED_FIFO priority and drains
once more. That delivers records from the final backend stop or a last control
fault, and it prints a lost-record count if any records were lost. SIGKILL,
power loss or a failed log sink can still lose the tail.

Hardening from the post-rung-2 adversarial review (all bench-triggered or
review-caught, all in the stale-authority class):

- **Gains are snapshotted at command acceptance** — the hold branch never
  dereferences the live command buffer, so a stray zero-gain datagram (every
  tool's address-teach prime) cannot un-spring a parked arm.
- **ARM is a generation counter, not a level** — a DISARM→ARM pair faster
  than one servo tick still triggers every per-epoch reset (gains, hold
  pose, plant retry, backend stop).
- **While armed, UDP commands are accepted only from the control client's
  IP, its pinned port, its session id and the current ARM epoch** — nothing
  else can steal the state stream or inject authority (v5; see above).
- **The control session has a real deadman** — the server PINGs at 4 Hz and
  treats ~1.2 s of silence as session loss (a half-open socket from a dead
  PC otherwise takes the kernel's 2-hour keepalive to notice).
- NaN/Inf commands are dropped at the UDP boundary; misconfigured safety
  flags (`hold-ms >= fault-ms`, `slew <= 0`) refuse to start; the mailbox
  read never waits or retries (a preempted writer can't spin the FIFO reader).

## Bring-up ladder

0. **Loopback on the PC** (no hardware): `python -m
   arm_control.plants.remote_rt.client` — protocol parity, tracking through the
   compiled law, staleness→hold, fault latch, refused ARM, DISARM/ARM
   recovery, server-side claim refusal, foreign-session/stale-epoch/NaN
   rejection, passive observers, stop-epoch confirmation and session restart
   isolation. The original subset was GREEN 2026-07-27; the v5 additions are
   offline-verified against the fake backend only. Keep it green.
   `tools/bench/rt/check_two_instances.py` runs two fake instances at once.
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

## Process-local RT setup gates ARM

At startup the servo thread applies `mlockall`, `SCHED_FIFO` at `--rt-priority`
and, when `--rt-cpu` is given, its CPU pin. It reads back the policy, priority
and CPU mask. This finishes before the backend is published as ready. Until
then ARM is refused (`refused: backend not ready (RT setup pending)`), and
HELLO waits for readiness, so it never reports a setup that has not run. For
`franka` and `dm`, any failed part refuses ARM for the life of the process:
`refused: RT process setup failed (<parts>); fix limits, restart`. The state
stream and observers keep working. The fake backend runs degraded unless
`--require-rt-setup` is given. HELLO reports `rt=ok` or the failed parts.

This checks only what the process sets for itself. The host facts (kernel,
IRQ routing, `ksoftirqd`, isolation, deployed artifacts) belong to the
deployment package's `install.py --check-host`. Whether the whole stack meets
FCI timing is established by hardware qualification.

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

Most of this server is not the servo law — it is threads, sockets, mailboxes,
staleness policy, and latching, all of which `backend_fake` exercises on any
machine in seconds. It is also the only safe place to rehearse the failure
paths (packet loss, session death, latched re-arm). It stays dumb by rule: a
damped integrator, never a second simulator — closed-loop realism is the
MuJoCo twin's job on the PC.

## Native reference extension (v7; software-only qualification)

The base protocol remains **v5**. Its command/state/control bytes and v6 pose
hold are unchanged. `CAP_REFERENCE_HORIZON = 1 << 3` is a separate capability;
The server advertises support; each client must explicitly negotiate it.
A client without it stays on streamed commands. A new server with an old client
stays on v5 state. Passive observer leases continue to receive v5 state.

After v5 HELLO/IDENTITY/CLAIM, a capable client explicitly requests v5
`REFERENCE_MODE` (opcode 11) while disarmed, binding the session to SHA-256 of
its existing opaque execution model token and the existing joint-order hash.
The server must acknowledge in v5 STATUS before either peer uses v7. The
binding lasts until disconnect; a repeated identical request is idempotent,
changing it requires reconnect. This is a PC model binding, not RT verification
of calibration or equality with the vendor task model. No reserved legacy bytes
acquire a new meaning. No v7 command is also a pose-hold command.

All new control frames retain the 24-byte little-endian header and 104-byte
body (128 bytes total), with `arg=0`. v7 operations are BEGIN=32, CHUNK=33,
SEAL=34, COMMIT=35, ABORT=36, CANCEL_REFERENCE=37 and STATUS=38. STATUS echoes
request `seq` and operation and reports a typed result; receipt/admission is
not selection or backend application. Each packed C++ payload declares byte
offsets in `protocol.hpp`; `_REFERENCE_LAYOUTS` is the checked Python mirror.
Reserved bytes, unknown flags/opcodes and wrong versions refuse. Counts do not
change packet sizes. Upload ids are positive monotonically increasing u64
within a session; revisions are positive u32, exactly previous+1, no wrapping.

The upload body is **knot-major little-endian float64**: for each knot, time
(seconds relative to release), N joint positions (rad), then N velocities
(rad/s) for Hermite. Linear references omit velocities. K is 2..128, N is
1..16. The maximum is 33,792 bytes. CHUNK has upload id, byte offset, length
1..88 and zero-padded data. SEAL hashes exactly `byte_count` bytes. Times must
be finite, nonnegative and strictly increasing; initial release begins at zero.
Replacement keeps the existing clock anchor and every protected whole segment
byte-for-byte, including the cut knot q/qd; it promises C1 only. COMMIT carries
the exact protected cut and latest acceptable phase (exclusive). Initial
COMMIT uses both zero. At body offset 56, `selection_budget_ns` is a u64
initial-selection duration (zero means absent); bytes 64..103 remain reserved.
Replacement COMMITs require a zero selection budget. The TCP owner timestamps
the first successful SEAL, never its retries. After the SEAL ACK, the PC computes
its remaining `admit_by` duration; RT requires elapsed time since that SEAL to be
strictly less than the budget. Both queue and network delay consume this
conservative bound; PC/RT absolute clock values are never compared. A failed
budget returns REF_LATE without changing selection. Packet sizes stay unchanged.
ABORT names an upload; CANCEL names an execution reference
and retires active/pending revisions without granting delayed frames authority.

The v7 UDP dynamics envelope is 448 bytes. Its 32-byte common header retains
session, ARM epoch and command sequence; the 32-byte binding carries reference
id/revision, phase in seconds, PC elapsed compute/queue age in microseconds,
wall validity and phase window in microseconds. Three 16-double arrays contain
PC tau_ff, kp, kd. No q/qd are transmitted in this mode. RT must require both
`PC age + time since ingress <= validity` and
`0 <= selected phase - envelope phase <= phase window`, additionally bounded
by the existing command freshness policy. No PC/RT clock subtraction is valid.
Neither buffered reference nor upload traffic refreshes dynamic freshness.

The v7 state is 1,312 bytes: existing 888-byte state (version 7), 40 bytes of
reference/envelope identity and timing, and selected q/qd/qdd arrays (16 each).
Existing q_cmd/qd_cmd/tau_cmd and last_cmd_seq retain output/backend-acceptance
meaning. Selected reference is separate. Static assertions include IPv6+UDP
headers within a 1,500-byte MTU. Unused reference-array slots are zero.

Reference status is NONE=0, RUNNING=1, END=2, UNDERFLOW=3, CANCELLED=4. END is
neither UNDERFLOW nor COMPLETED: terminal selection holds endpoint with zero
velocity/acceleration while a matching fresh dynamics envelope remains valid.
UNDERFLOW retires the nonterminal revision, captures measured position once,
zeros velocity/feedforward, holds with captured gains and stays armed. It does
not immediately latch CMD_LOST; existing independent command/session deadlines
remain in force. No late upload heals retirement. These execution semantics
are approved requirements, **not implemented by the B1 codecs**.

Envelope expiry (decision (a)) retires a **RUNNING** reference — status
`CANCELLED` on the wire, a journal-only `reference envelope expired` cause — and
the arm holds its measured position with zero feedforward; only a NEW reference
admitted from the measured state resumes motion. A reference at its stationary
`END` is never retired, a normal revision switch is never retired, and an
ordinary PC CANCEL keeps its retained id/phase fresh so the envelopes the PC
keeps streaming still hold the deadman. A selected `(id, revision)` that admits
no envelope within `min(hold_ms, valid_for_us)` is retired with the same cause,
so a late first envelope cannot be sampled at the advanced phase. V1 declares
`valid_for_us = phase_window_us = 100 ms`; `--hold-ms` defaults to 100, so the
effective ceiling is 100 ms and lowering `--hold-ms` lowers it. Retirement is
journal/diagnostic only: no new `ReferenceStatus` value and no packet change.

**Revision handover (CTRL-034 B2).** For the same `reference_id`, an envelope
for revision − 1 is still admissible while its phase is ≤ the current revision's
`protected_until`: `prefix_matches` already proved that retained prefix
byte-identical, so the dynamics decision is unchanged. The previous revision's
envelope counts for the first-envelope bound too. Without it every revision
switch held the arm until the new revision's first envelope arrived — at
0.4 rad/s, one hold frame per switch. The PC side owns the stream with a
dedicated fixed-rate sender (`ReferenceHorizon`). The sender may repeat a
decision but never re-stamps it: every envelope carries the decision's own RT
sample time and reference phase, so the reported age grows while the decision is
reused and the sender stops once it passes the ceiling — a stalled caller makes
the RT expire and retire rather than feeding a heartbeat. After RT switches
revision the sender keeps the decision's original revision until the caller
republishes.

Offline schema check (no server/device):

```sh
cmake --build build --target reference_protocol_selfcheck
PYTHONPATH=. python tools/bench/rt/reference_protocol.py build/reference_protocol_selfcheck
```


### B2 upload implementation

The TCP thread now owns one bounded `ReferenceUpload` value (67,792 bytes on
the checked x86-64 build). A session binds the execution-token digest once;
BEGIN checks identity, epoch, revision and capacity before reserving it. CHUNK
uses aligned 88-byte pieces (only the final piece may be shorter); exact
retries are idempotent and altered/reordered/overlapping data refuses. SEAL
requires complete bytes and SHA-256, then validates finite increasing knots
and conservatively rejects overflowing interpolation arithmetic. Hashing,
parsing and validation are non-RT. libcrypto is the SHA-256 dependency; matched
binary/dependency packaging remains CTRL-060. No vendor model is loaded here.

The client has one upload caller and **one frame awaiting acknowledgement**;
there is no bulk frame queue. It releases the send lock before waiting. ARM,
DISARM and abort interrupt an in-progress sender; PING/PONG continues on the
existing control thread. Upload frames do not refresh either deadman. A local
socketpair check pauses the peer and floods 32 callers: exactly one upload
frame precedes the DISARM frame. This tests bounded queueing, not a guaranteed
servo-stop latency. A send remains bounded by the socket timeout.

`reference.hpp` owns the extracted pure native polynomial. The C ABI parity
wrapper delegates to it; no polynomial remains in that wrapper. Publication
and execution are B3/B4 work, so capability remains unadvertised in the server.
The test session executable alone enables it without a plant/servo thread.

```sh
build/reference_upload_selfcheck
PYTHONPATH=. python tools/bench/rt/reference_numerics.py build/reference_probe.so
PYTHONPATH=. python tools/bench/rt/reference_upload.py build/reference_session_selfcheck
```

### RT selection and PC provider

`ReferenceSelection` now owns the selected native curve and its RT clock
anchor. The TCP thread publishes one immutable `ReferenceCommit` via the
existing Mailbox and permits no second pending commit until RT acknowledges
its ticket. RT repeats session/epoch/generation/revision/lateness checks and
compares every retained protected knot byte; malformed/stale replacement
cannot change active selection. ABORT refuses a pending commit; CANCEL retires
both selected and pending generations. Exact COMMIT/CANCEL retries do not
reselect or recapture a hold. Internal reference ids increase monotonically
per ARM epoch, allowing a bounded retired-id high-water mark. A new admitted
id can replace a terminal endpoint at COMMIT; the old endpoint remains selected
through staging. A running reference requires same-id prefix-preserving revision
or explicit cancellation. Linear references can be initially selected; the C1
replacement promise requires Hermite velocity knots.

The largest commit is 34,040 bytes, its three-slot mailbox 102,160. Every copy
has a static capacity; prefix validation scans at most 128 knots with at most
16 q/qd values per knot. No RT allocation, hash, socket call or retry loop is
introduced. The selection check disables C++ allocation during initial and
replacement publication/read/selection. Host mean timings are diagnostic,
not RT-box worst-case timing qualification.

The RT dynamic-envelope check is shared by acceptance and continued use.
Wrong identity/revision, stale phase or expired age does not refresh command
authority. Selection and applied output remain distinct. Nonterminal exhaustion
captures measured q once and holds with captured gains/zero feedforward without
immediate CMD_LOST. The independent deadline can subsequently latch if valid
command traffic ceases. END retains planned q with zero qd/qdd; it is not a
completion receipt. Native state is sent only to the negotiated commander;
passive observers retain v5. Ordinary clients that do not opt in continue to
use existing command/state packets, regardless of the advertised capability.

```sh
build/reference_selection_selfcheck
PYTHONPATH=. python tools/bench/rt/native_reference.py build/reference_runtime_selfcheck
```

The model-bound supervisor binds `TrajectoryReference` to selected feedback:
no PC reference clock/interpolation runs for that execution. The existing PC
executor still computes all dynamics and integral terms. `ReferenceHorizon`
copies exact windows from the admitted plan and refills near half-window; a
missed window never predicts/extrapolates. It has one worker/candidate and retains
session/epoch/socket scope across queued operations. Cancelling an unselected
upload ABORTs it; a selected cancellation retires the reference. The existing
controller's new compensated hold target requires an explicit constant terminal
admission; underflow never triggers that automatic recovery.

```sh
PYTHONPATH=. python tools/bench/rt/reference_horizon.py build/reference_runtime_selfcheck
PYTHONPATH=. python tools/bench/rt/reference_supervisor.py build/reference_runtime_selfcheck
PYTHONPATH=. python tools/bench/check_sdk_supervisor.py --native --server build/arm_rt_server
```

Checks cover command parity at the same RT sample, receipt ordering, deadline
races, unselected upload cancellation, underflow/new-admission recovery, terminal
hold, stale envelopes, old epochs, owner loss, restart, upload flood and stalled
peers. v5 base/v6 pose bytes remain unchanged. The pre-integration v7 checkpoint
binaries are not a deployment interface: use a matched final source/binary pair.
No hardware deployment or worst-case RT timing qualification is implied.

For the CTRL-034 first-envelope check, RT selection immediately publishes a
measured-position compensation decision for the selected reference, using that
feedback sample's reference phase and original PC receipt time. The fixed-rate
sender repeats the decision without changing its age. The first tracking
decision is measured separately: its latency uses the first selected packet's
receipt time plus that packet's reference phase, so no PC/RT clock equality is
assumed. Full Hermite bounds are checked at staging; release rechecks current
observation and start pose without repeating the whole-curve calculation.
While a new candidate uploads or aborts, an older selected terminal endpoint
keeps receiving freshly computed compensation decisions from its own state
samples. Candidate interruption cannot silence that END reference.
