// Shared context between the four server threads:
//   udp_rx    (non-RT)  commands in  -> cmd_in seqlock, learns the peer addr
//   state_tx  (non-RT)  state_out seqlock -> UDP to the peer, paced
//   control   (non-RT)  TCP session: arm/disarm/ping + fault/status events
//   rt_loop   (RT)      backend read -> servo law -> backend write
//
// The RT thread touches ONLY atomics and seqlocks; every blocking object
// (sockets, the fault-text mutex from the writer side) lives on the non-RT
// threads. Fault text is written under a mutex by whoever latches, but the
// RT loop latches through a fixed-size copy + release-store flag so it never
// takes the lock on the tick path.
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
  std::string can_if = "can0";
  int n = 7;                // joints (fake); franka fixes 7 itself
  uint16_t udp_port = 47800;
  uint16_t tcp_port = 47801;
  double state_hz = 250.0;
  double hold_ms = 100.0;   // stale commands -> hold current pose
  double fault_ms = 1000.0; // prolonged staleness -> latch (DISARM+ARM clears)
  double slew = 1.0;        // N.m per tick, the servo slew budget
  double hold_kp = 50.0;    // hold gains when no command was ever received
  double hold_kd = 5.0;
  int rt_priority = 80;     // SCHED_FIFO; failure to set is a warning, not fatal
  int rt_cpu = -1;          // pin the SERVO thread here (an isolcpus core);
                            // comms threads float on the housekeeping cores
};

struct ServerCtx {
  ServerConfig cfg;

  SeqLock<CommandPacket> cmd_in;
  SeqLock<StatePacket> state_out;

  std::atomic<uint64_t> last_cmd_rx_ns{0};
  std::atomic<bool> armed{false};
  std::atomic<bool> fault{false};
  std::atomic<uint32_t> fault_code{0};
  std::atomic<bool> fault_event_pending{false};  // control thread pushes CTL_FAULT
  std::atomic<bool> shutdown{false};
  std::atomic<bool> failed{false};   // startup/bind failure -> nonzero exit,
                                     // so systemd's Restart=on-failure retries
                                     // (a port race must not look like success)
  std::atomic<bool> backend_ready{false};
  std::atomic<int> backend_n{0};
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

  int udp_fd = -1;

  void latch(uint32_t code, const char* text) {
    if (fault.load(std::memory_order_acquire)) return;  // first cause wins
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
