// Shared context between the four server threads:
//   udp_rx    (non-RT)  commands in  -> cmd_in seqlock, learns the peer addr
//   state_tx  (non-RT)  state_out seqlock -> UDP to the peer, paced
//   control   (non-RT)  TCP session: arm/disarm/ping + fault/status events
//   rt_loop   (RT)      backend read -> servo law -> backend write
//
// The RT thread touches ONLY atomics and seqlocks; every blocking object
// (sockets) lives on the non-RT threads. latch() is lock-free: a CAS claims
// the one fault slot (two threads can latch concurrently — RT loop and the
// control thread), the winner writes the text, then release-stores the
// flag. Readers copy the buffer only after seeing fault==true.
#pragma once

#include <netinet/in.h>

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <string>

#include "arm_rt/backend.hpp"
#include "arm_rt/protocol.hpp"
#include "arm_rt/seqlock.hpp"

namespace arm_rt {

// Fault codes (StatePacket.fault_code / CTL_FAULT.arg)
constexpr uint32_t FAULT_CMD_LOST = 1;  // command stream stale past --fault-ms
constexpr uint32_t FAULT_CTL_LOST = 2;  // TCP control session dropped while armed
constexpr uint32_t FAULT_PLANT = 3;     // backend read/write failed

struct ServerConfig {
  std::string backend = "fake";
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
  // dm backend spec: "IF;ID:TYPE[:MST],..." (--dm-spec / --can-if). A bare
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

struct ServerCtx {
  ServerConfig cfg;

  SeqLock<PoseHoldCommandPacket> cmd_in;
  SeqLock<StatePacket> state_out;

  std::atomic<uint64_t> last_cmd_rx_ns{0};
  // Commands from BEFORE the current ARM are never authority: ARM stores the
  // command seqlock's version here and the RT loop only tracks newer ones.
  // Kills two live hazards: the client's zero-authority address-teach packet
  // being adopted as "the task's gains" (found on the FR3 impedance rung:
  // the spring never engaged), and a pre-fault leftover q_des yanking the
  // arm for the first hold-ms after a re-arm.
  std::atomic<uint64_t> cmd_epoch{0};
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
  // UDP source port and drops every other port until the next ARM.
  std::atomic<uint16_t> cmd_owner_port{0};  // network order; 0 = unpinned
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
  std::mutex peer_mu;
  sockaddr_in peer = {};
  bool have_peer = false;

  std::atomic<int> udp_fd{-1};  // udp_rx writes, state_tx reads

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

void udp_rx_thread(ServerCtx& ctx);
void state_tx_thread(ServerCtx& ctx);
void control_thread(ServerCtx& ctx);
void rt_loop(ServerCtx& ctx);  // constructs the backend ON this thread

} // namespace arm_rt
