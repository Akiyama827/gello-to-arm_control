// Shared context between the four server threads:
//   udp_rx    (non-RT)  commands in  -> cmd_in mailbox, learns the peer addr
//   state_tx  (non-RT)  state_out mailbox -> UDP to the peer, paced
//   control   (non-RT)  TCP session: arm/disarm/ping + fault/status events
//   rt_loop   (RT)      backend read -> servo law -> backend write
//
// Channel topology (each Mailbox is exactly one writer, one reader):
//   cmd_in    udp_rx -> rt_loop      state_out  rt_loop -> state_tx
// state_tx fans the state out to the peer and observers from its own copy.
//
// The RT thread touches ONLY atomics and mailboxes; every blocking object
// (sockets) lives on the non-RT threads. latch() is lock-free: a CAS claims
// the one fault slot (two threads can latch concurrently — RT loop and the
// control thread), the winner writes the text, then release-stores the
// flag. Readers copy the buffer only after seeing fault==true.
#pragma once

#include <netinet/in.h>
#include <sys/types.h>

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "arm_rt/backend.hpp"
#include "arm_rt/protocol.hpp"
#include "arm_rt/mailbox.hpp"

namespace arm_rt {

// Fault codes (StatePacket.fault_code / CTL_FAULT.arg)
constexpr uint32_t FAULT_CMD_LOST = 1;  // command stream stale past --fault-ms
constexpr uint32_t FAULT_CTL_LOST = 2;  // TCP control session dropped while armed
constexpr uint32_t FAULT_PLANT = 3;     // backend read/write failed

struct ServerConfig {
  std::string backend = "fake";
  // Provisioned identity, checked by every commander's CLAIM before ARM and
  // stamped into the state stream for observers. A configured label, not an
  // attestation of which robot is on the other end of the FCI cable.
  std::string instance_id;               // required except for the fake backend
  std::vector<std::string> joint_names;  // required except for the fake backend
  std::string franka_ip = "172.16.0.3";
  // Extra payload bolted past the flange (wrist camera + its mount, a carried
  // module), declared to the robot with Robot::setLoad so its own gravity
  // compensation accounts for it. ADDITIVE to Desk's end-effector config: the
  // Franka Hand's 0.73 kg is m_ee and stays there, this is m_load. Leave at 0
  // and an undeclared 0.2 kg at the flange is ~1 N.m of permanent elbow torque
  // that a kp=0 float has nothing to hold against — the arm creeps into its
  // joint-4 limit (2026-08-06, cost a joint_velocity_violation reflex).
  double ee_mass = 0.0;          // kg
  double ee_com[3] = {0, 0, 0};  // load CoM in the FLANGE frame (m)
  // dm backend spec: "IF;ID:TYPE[:MST[:PMAX:VMAX:TMAX]],...". A bare
  // interface name has no motors and is refused by make_dm_backend.
  std::string can_if = "can0";
  int n = 7;                // joints (fake); franka fixes 7 itself
  uint16_t udp_port = 47800;
  uint16_t tcp_port = 47801;
  double state_hz = 250.0;
  double hold_ms = 100.0;   // stale commands -> hold current pose
  double fault_ms = 1000.0; // prolonged staleness -> latch (DISARM+ARM clears)
  double slew = 1.0;        // N.m per tick, the servo slew budget
  double tau_max = 0.0;     // optional operating cap; 0 = backend limits
  double hold_kp = 50.0;    // hold gains when no command was ever received
  double hold_kd = 5.0;
  uint32_t initial_active_mask = 0xFFFFu;
  int rt_priority = 80;     // SCHED_FIFO; failure to set is a warning, not fatal
  int rt_cpu = -1;          // pin the SERVO thread here (an isolcpus core);
                            // comms threads float on the housekeeping cores
  // Bind address for BOTH listeners. Default any: the box is dual-NIC (PC
  // direct link + robot LAN) and INADDR_ANY answers on the robot LAN too —
  // production units should pass --bind <direct-link-ip>.
  uint32_t bind_addr = 0;   // network order; 0 = INADDR_ANY
};

constexpr int MAX_OBSERVERS = 4;

// One servo iteration that took > LOOP_GAP_NS from read-return to read-return.
// The gap = previous iteration's compute+write+publish, then this one's read.
constexpr uint64_t LOOP_GAP_NS = 2'000'000;
struct LoopGap {
  uint64_t t_read_ns;   // this read's return, server clock (== StatePacket stamp domain)
  uint64_t period_ns;   // read-return to read-return
  uint64_t read_ns;     // this iteration: blocked inside backend->read()
  uint64_t compute_ns;  // previous iteration: read-return -> write start
  uint64_t write_ns;    // previous iteration: inside backend->write_command()
  uint64_t publish_ns;  // previous iteration: write end -> state mailbox written
  int64_t robot_dt_ms;  // robot clock advance between the two samples
  double cmd_success_rate;
  uint32_t state_seq;
  bool armed;
};
constexpr uint32_t LOOP_GAP_RING = 64;

// One accepted command, published as ONE value together with the facts that
// made it authority: its own receive time and the ARM generation it was
// validated against. The RT loop never pairs a payload with another packet's
// timestamp or epoch (they used to be published separately).
struct CommandEnvelope {
  PoseHoldCommandPacket packet;
  uint64_t rx_ns;       // server clock at INGRESS (recvfrom return), never
                        // re-dated by a handler that was delayed afterwards
  uint64_t arm_gen;     // generation this packet's arm_epoch was checked against
  uint64_t session_id;  // control session it was admitted under
};

// When an ARM generation began: the no-first-command deadline's baseline.
// Written by the ARM handler BEFORE it publishes the generation, so an RT
// tick that loads the new generation and then reads this sees its record.
struct ArmStart {
  uint64_t gen;
  uint64_t t_ns;
};

inline double clamped_age_ns(uint64_t now, uint64_t then) {
  // Signed, clamped: udp_rx and the ARM handler stamp in parallel with the
  // tick, so `then` can be nanoseconds in the future.
  const int64_t age = int64_t(now) - int64_t(then);
  return double(age < 0 ? 0 : age);
}

// The RT loop's own view of command authority, updated once per tick from
// the two mailboxes (after loading the current generation). Everything that
// decides tracking and the CMD_LOST deadline derives from here, scoped to
// ONE generation: a handler of a retired generation that publishes late can
// neither be tracked nor move this generation's deadline, and a delayed
// same-generation packet carries its own old ingress stamp.
struct CommandAuthority {
  CommandEnvelope current{};  // newest accepted envelope of `gen`
  bool have = false;
  uint64_t gen = 0;
  uint64_t arm_start_ns = 0;  // start of `gen`, 0 = not seen yet

  void observe(uint64_t gen_now, const ArmStart& start, uint64_t version,
               const CommandEnvelope& e) {
    if (gen_now != gen) {  // new generation: nothing of the old one survives
      gen = gen_now;
      have = false;
      arm_start_ns = 0;
    }
    if (start.gen == gen) arm_start_ns = start.t_ns;
    // A retired or not-yet-current generation is ignored; within this one,
    // the latest ingress wins (a late publish never moves time backwards).
    if (version != 0 && e.arm_gen == gen && (!have || e.rx_ns >= current.rx_ns)) {
      current = e;
      have = true;
    }
  }

  // CMD_LOST baseline: the later of this generation's start and its newest
  // accepted ingress. 0 = nothing of this generation yet (no deadline).
  uint64_t deadman_base() const {
    const uint64_t rx = have ? current.rx_ns : 0;
    return rx > arm_start_ns ? rx : arm_start_ns;
  }

  // Tracking authority for this tick: a command of THIS generation, from
  // the CURRENT claimed session, right joint count, own ingress within hold.
  bool fresh(uint64_t session, int n, uint64_t now, double hold_ns) const {
    return have && session != 0 && current.session_id == session &&
           current.packet.command.n == uint16_t(n) &&
           clamped_age_ns(now, current.rx_ns) <= hold_ns;
  }
};

struct Observer {
  sockaddr_in addr{};
  uint64_t expiry_ns = 0;  // 0 = free slot
};

struct ServerCtx {
  ServerConfig cfg;

  // Identity, fixed once the backend is up (read-only afterwards).
  char instance_id[ID_CHARS] = {};
  char model_id[ID_CHARS] = {};
  uint64_t joint_order_hash = 0;
  uint64_t identity_hash = 0;

  // Current control session. 0 = no client; a fresh random value per accepted
  // TCP connection. Commands must carry it, and it must have passed CLAIM.
  std::atomic<uint64_t> session_id{0};
  std::atomic<bool> claim_ok{false};
  // DISARM generation: bumped AFTER `armed` is cleared, so a servo tick that
  // reads this value first and `armed` second can only report a stop epoch
  // that covers the request (see rt_loop).
  std::atomic<uint64_t> disarm_gen{0};

  std::mutex obs_mu;
  Observer observers[MAX_OBSERVERS];

  Mailbox<CommandEnvelope> cmd_in;
  Mailbox<StatePacket> state_out;
  // control thread (CTL_ARM) -> rt_loop. The staleness clock starts AT ARM;
  // the RT loop owns the deadline (CommandAuthority), not a shared stamp.
  Mailbox<ArmStart> arm_start;
  // Commands from BEFORE the current ARM are never authority: each envelope
  // carries the generation it was validated under and the RT loop tracks only
  // the current one. Kills two live hazards: the client's zero-authority
  // address-teach packet being adopted as "the task's gains" (found on the
  // FR3 impedance rung: the spring never engaged), and a pre-fault leftover
  // q_des yanking the arm for the first hold-ms after a re-arm.
  // ARM generation: bumped by every accepted CTL_ARM. The RT loop compares
  // against the last generation it acted on, so a DISARM->ARM pair that
  // completes between two 1 kHz samples is still seen as an edge (sampling
  // the armed LEVEL misses it: prev_armed==armed==true) and the per-epoch
  // resets (plant retry, gains, hold pose, backend stop) still run.
  std::atomic<uint64_t> arm_gen{0};
  // IP of the current TCP control client. While armed, udp_rx only accepts
  // command datagrams from this address — any valid packet from anywhere
  // must not be able to steal the state stream or inject authority into an
  // armed arm (a stray tool's zero-gain prime packet was enough to kill the
  // hold spring before the gains snapshot; the pin closes the whole class).
  std::atomic<uint32_t> ctl_peer_ip{0};
  // The command FLOW pin: the IP check alone is not enough, because every
  // PC-side tool shares the control client's IP — a stray same-host tool's
  // zero-gain prime packet would be accepted as a fresh command and become
  // the gains snapshot of an ARMED arm (audit 2026-07-29, three independent
  // reviewers). ARM clears this; udp_rx pins the first post-ARM sender's
  // UDP source port and drops every other port until the next ARM. Stored
  // with the generation it was taken under (gen << 16 | port, network-order
  // port): a pin from another generation is no pin at all, so a packet that
  // raced an ARM can never lock the new flow out.
  std::atomic<uint64_t> cmd_owner{0};  // 0 = unpinned
  std::atomic<bool> armed{false};
  std::atomic<bool> fault{false};
  std::atomic<bool> fault_claim{false};  // CAS gate: exactly one latch writes
  std::atomic<uint32_t> fault_code{0};
  std::atomic<bool> fault_event_pending{false};  // control thread pushes CTL_FAULT
  std::atomic<bool> shutdown{false};
  std::atomic<bool> failed{false};   // startup/bind failure -> nonzero exit,
                                     // so systemd's Restart=on-failure retries
                                     // (a port race must not look like success)
  std::atomic<bool> backend_ready{false};
  std::atomic<bool> supports_pose_hold{false};
  std::atomic<int> backend_n{0};
  std::atomic<uint32_t> online_mask{0};
  std::atomic<uint32_t> active_mask{0};
  std::atomic<uint32_t> requested_active_mask{0};
  char backend_name[32] = {};

  // Latched fault reason. latch() copies into the fixed buffer then flips the
  // atomics — readers copy the buffer only after seeing fault==true, and the
  // buffer is written exactly once per latch (cleared only on DISARM).
  char fault_text[104] = {};

  // State-stream peer, learned from the last command datagram (direct link,
  // one client). Guarded: udp_rx writes, state_tx reads.
  // Tagged with the generation of its last writer: a handler of an older
  // generation cannot redirect the stream after an ARM (begin_arm raises it).
  std::mutex peer_mu;
  sockaddr_in peer = {};
  bool have_peer = false;
  uint64_t peer_gen = 0;

  std::atomic<int> udp_fd{-1};  // udp_rx writes, state_tx reads

  // ponytail: SPSC ring, rt_loop produces, state_tx drains to stderr; a full
  // ring drops the new record (counted), so the RT side never waits or allocates.
  LoopGap gaps[LOOP_GAP_RING] = {};
  std::atomic<uint32_t> gap_head{0}, gap_tail{0}, gap_dropped{0};

  void latch(uint32_t code, const char* text) {
    bool expected = false;  // first cause wins, decided by ONE atomic claim
    if (!fault_claim.compare_exchange_strong(expected, true,
                                             std::memory_order_acq_rel))
      return;
    std::snprintf(fault_text, sizeof(fault_text), "%s", text);
    fault_code.store(code, std::memory_order_release);
    fault.store(true, std::memory_order_release);
    fault_event_pending.store(true, std::memory_order_release);
  }
};

uint64_t mono_ns();

// CTL_ARM's state transition, in its required order; the control thread is
// its only caller in the server. The ARM-start record is published before
// the generation, the flow pin is cleared, the peer is scoped to the new
// generation, then authority is raised. Returns the new generation.
inline uint64_t begin_arm(ServerCtx& ctx) {
  const uint64_t gen = ctx.arm_gen.load(std::memory_order_relaxed) + 1;
  ctx.arm_start.write(ArmStart{gen, mono_ns()});
  ctx.cmd_owner.store(0);
  {
    std::lock_guard<std::mutex> lock(ctx.peer_mu);
    ctx.peer_gen = gen;
  }
  ctx.arm_gen.store(gen);
  ctx.armed.store(true);
  return gen;
}

// udp_rx's own admission memory (one flow, one thread).
struct CommandFlow {
  uint64_t seq_gen = 0;
  uint32_t seq_ip = 0, last_seq = 0;
  uint16_t seq_port = 0;
  bool have_seq = false;
  // Warn-once state for the drop logs.
  uint64_t warned_session = 0;
  uint32_t warned_epoch = 0;
  bool have_warned_session = false, have_warned_epoch = false;
  uint32_t dropped_ip = 0;
  uint16_t dropped_port = 0, warned_n = 0;
  uint64_t last_accept_rx = 0;  // accept-gap log only
  // Test seam: called after every check passed, before any side effect.
  // Null in the server; a check uses it to stall a handler across an ARM.
  void (*after_checks)(void*) = nullptr;
  void* seam_arg = nullptr;
};

// One received datagram: an observer lease or a command. Every check runs
// before any side effect (flow pin, sequence memory, peer, publication), so a
// rejected packet changes nothing. `ingress_ns` is the server clock when the
// datagram left recvfrom. True = published.
bool handle_datagram(ServerCtx& ctx, CommandFlow& flow, const void* data, ssize_t size,
                     const sockaddr_in& src, uint64_t ingress_ns);

void udp_rx_thread(ServerCtx& ctx);
void state_tx_thread(ServerCtx& ctx);
void control_thread(ServerCtx& ctx);
void rt_loop(ServerCtx& ctx);  // constructs the backend ON this thread

} // namespace arm_rt
