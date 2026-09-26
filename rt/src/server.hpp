// Shared context between the four server threads:
//   udp_rx    (non-RT)  commands in  -> cmd_in mailbox, learns the peer addr
//   state_tx  (non-RT)  state_out mailbox -> UDP to the peer, paced
//   control   (non-RT)  TCP session: arm/disarm/ping + fault/status events
//   rt_loop   (RT)      backend read -> servo law -> backend write
//
// Channel topology (each Mailbox is exactly one writer, one reader):
//   cmd_in    udp_rx -> rt_loop      state_out  rt_loop -> state_tx
//   arm_start control -> rt_loop
// state_tx fans the state out to the peer and observers from its own copy.
//
// The RT thread touches ONLY atomics and mailboxes; every blocking object
// (sockets) lives on the non-RT threads. The wire latch has ONE state word
// (latch_state): latch() claims FREE->WRITING by CAS, fills the slot and
// publishes SET last; clear_latch() (DISARM) takes SET/FREE->CLEARING, never
// a slot still being written; ARM is admitted only by FREE->ARMING on the
// same word (arm_unlatched). The servo never waits on it: a lost claim is a
// secondary cause. Readers copy the text only after seeing fault==true.
#pragma once

#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/types.h>

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

#include "arm_rt/backend.hpp"
#include "arm_rt/protocol.hpp"
#include "arm_rt/mailbox.hpp"
#include "arm_rt/reference_selection.hpp"

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
  int rt_priority = 80;     // SCHED_FIFO for the servo thread
  int rt_cpu = -1;          // pin the SERVO thread here (an isolcpus core);
                            // comms threads float on the housekeeping cores
  // Process-local RT setup (mlockall, SCHED_FIFO at rt_priority, rt_cpu when
  // given) gates ARM for real backends. The fake backend runs degraded unless
  // this asks it to gate too (device-free checks of the refusal path).
  bool require_rt_setup = false;
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

// Fault journal: the non-RT record of every latch's FIRST cause, of a
// different cause arriving while latched (once per code per latch), and of
// failed backend stops. state_tx drains it to stderr (journald). DISARM and a
// client disconnect clear the wire latch, never these records, so a closed
// GUI no longer means an unrecorded fault. Truncation is flagged, not silent.
constexpr int FAULT_RECORD_TEXT = 240;
enum FaultKind : uint8_t { FAULT_FIRST = 0, FAULT_SECONDARY = 1, FAULT_STOP_FAILED = 2 };
struct FaultRecord {
  uint64_t t_ns;        // server clock (StatePacket stamp domain)
  uint64_t session_id;
  uint32_t code;
  uint32_t arm_epoch;
  uint8_t kind;
  bool armed;
  bool truncated;
  char text[FAULT_RECORD_TEXT];
};
constexpr uint32_t FAULT_RING = 16;
// Multi-producer (RT loop, control thread), one consumer at a time (state_tx
// while running, then finish_fault_journal after every thread is joined). A
// producer claims an index by CAS on head only while head - tail < RING, so
// it never writes a slot the consumer has not released; `ready` = index + 1
// publishes that slot (release), the consumer copies after acquiring it and
// then advances tail (release). Full ring: the NEW record is dropped
// (fault_dropped). Bounded work per call: at most FAULT_CLAIM_TRIES claim
// attempts (a lost race past that drops the record, fault_contended) and one
// scan/copy of at most FAULT_RECORD_TEXT bytes -- bounded steps given
// lock-free atomics on the target, not an RT latency measurement.
constexpr int FAULT_CLAIM_TRIES = 4;
static_assert(std::atomic<uint32_t>::is_always_lock_free, "journal needs lock-free atomics");

// Copies at most cap-1 bytes of `src` and NUL-terminates; scans at most cap
// bytes. True = `src` did not fit (the copy is truncated).
inline bool bounded_copy(char* dst, size_t cap, const char* src) {
  const size_t len = strnlen(src, cap);
  const size_t n = len < cap ? len : cap - 1;
  std::memcpy(dst, src, n);
  dst[n] = '\0';
  return len >= cap;
}

struct FaultSlot {
  std::atomic<uint32_t> ready{0};
  FaultRecord rec{};
};

uint64_t mono_ns();

// One accepted command, published as ONE value together with the facts that
// made it authority: its own receive time and the ARM generation it was
// validated against. The RT loop never pairs a payload with another packet's
// timestamp or epoch (they used to be published separately).
struct ReferenceEnvelopeBinding {
  uint64_t reference_id=0;
  uint32_t revision=0, age_us=0;
  double phase=0;
  uint32_t valid_for_us=0, phase_window_us=0;
};

struct CommandEnvelope {
  PoseHoldCommandPacket packet;
  uint64_t rx_ns;       // server clock at INGRESS (recvfrom return), never
                        // re-dated by a handler that was delayed afterwards
  uint64_t arm_gen;     // generation this packet's arm_epoch was checked against
  uint64_t session_id;  // control session it was admitted under
  ReferenceEnvelopeBinding reference{};
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

// One envelope validity rule for admission and continued use. Wall age uses
// local elapsed intervals only; phase is the shared semantic coordinate.
inline bool reference_envelope_fresh(const ReferenceEnvelopeBinding& b,
                                    const ReferenceSelectionInfo& r,
                                    uint64_t now,uint64_t ingress,double hold_ns) {
  const double lag=r.phase-b.phase;
  return r.reference_id==b.reference_id && r.revision==b.revision && lag>=0 &&
    lag*1e6<=b.phase_window_us && lag*1e9<=hold_ns &&
    clamped_age_ns(now,ingress)+double(b.age_us)*1000<=
      std::min(hold_ns,double(b.valid_for_us)*1000);
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

struct ReferenceCancel {
  uint64_t session_id=0, reference_id=0, generation=0;
  uint32_t arm_epoch=0, revision=0;
};

struct Observer {
  sockaddr_in addr{};
  uint64_t expiry_ns = 0;  // 0 = free slot
};

// Process-local RT prerequisites this server establishes for its own servo
// thread. The syscalls are injectable so self-checks need no privileges. Host
// facts (PREEMPT_RT, IRQ routing, ksoftirqd, isolation, deployed artifacts)
// belong to the deployment package (install.py --check-host), not here.
enum : uint32_t {
  RT_SETUP_MLOCK = 1u << 0,     // mlockall(MCL_CURRENT | MCL_FUTURE)
  RT_SETUP_SCHED = 1u << 1,     // SCHED_FIFO at the requested priority
  RT_SETUP_AFFINITY = 1u << 2,  // servo thread on exactly --rt-cpu
};
struct RtSetupOps {
  int (*lock_memory)(int) = ::mlockall;
  int (*set_sched)(pthread_t, int, const sched_param*) = ::pthread_setschedparam;
  int (*get_sched)(pthread_t, int*, sched_param*) = ::pthread_getschedparam;
  int (*set_affinity)(pthread_t, size_t, const cpu_set_t*) = ::pthread_setaffinity_np;
  int (*get_affinity)(pthread_t, size_t, cpu_set_t*) = ::pthread_getaffinity_np;
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
  // Capability stays off until the complete native selection path is available.
  std::atomic<bool> supports_reference{false};
  std::atomic<uint64_t> reference_session{0};
  // DISARM generation: bumped AFTER `armed` is cleared, so a servo tick that
  // reads this value first and `armed` second can only report a stop epoch
  // that covers the request (see rt_loop).
  std::atomic<uint64_t> disarm_gen{0};

  std::mutex obs_mu;
  Observer observers[MAX_OBSERVERS];

  Mailbox<CommandEnvelope> cmd_in;
  Mailbox<StatePacket> state_out;
  Mailbox<ReferenceStatePacket> reference_state_out; // RT -> state_tx, negotiated peers only
  Mailbox<ReferenceCommit> reference_in;       // control -> RT, one pending
  Mailbox<ReferenceCancel> reference_cancel;   // control -> RT, latest retirement
  Mailbox<ReferenceSelectionInfo> reference_feedback; // RT -> control
  std::atomic<uint64_t> reference_published{0}, reference_done{0};
  std::atomic<uint32_t> reference_result{REF_NOT_ADMITTED};
  std::atomic<uint64_t> reference_cancel_generation{0};
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
  // Wire-latch lifecycle; see latch(), clear_latch(), arm_unlatched(). Only
  // the WRITING owner touches fault_text/fault_code/fault_gen/fault until it
  // publishes SET; only the CLEARING owner resets them; ARMING is CTL_ARM
  // preparing a generation. The control thread owns every transition except
  // the claim.
  enum : uint32_t {
    LATCH_FREE = 0, LATCH_WRITING = 1, LATCH_SET = 2, LATCH_CLEARING = 3, LATCH_ARMING = 4
  };
  std::atomic<uint32_t> latch_state{LATCH_FREE};
  std::atomic<uint64_t> fault_gen{0};  // ARM generation the latched fault arose in
  // Latch serial, advanced by each latch before it publishes SET. The control
  // thread delivers CTL_FAULT for a serial once (fault_notified is its own).
  std::atomic<uint64_t> fault_seq{0};
  uint64_t fault_notified = 0;
  // Test seam: called right after a successful claim. Null in the server.
  void (*latch_seam)(void*) = nullptr;
  void* latch_seam_arg = nullptr;
  // Test seam: called between arm_unlatched's two final stores, to hold that
  // ordering window open. Null in the server.
  void (*arm_seam)() = nullptr;
  std::atomic<uint32_t> fault_code{0};
  std::atomic<bool> shutdown{false};
  std::atomic<bool> failed{false};   // startup/bind failure -> nonzero exit,
                                     // so systemd's Restart=on-failure retries
                                     // (a port race must not look like success)
  std::atomic<bool> backend_ready{false};
  // Failed process-local RT prerequisites (RT_SETUP_*) and whether they gate
  // ARM. Written once by rt_loop before backend_ready; a failure lasts for
  // the process: fix the limits and restart the service.
  std::atomic<uint32_t> rt_setup_failed{0};
  std::atomic<bool> rt_setup_required{false};
  RtSetupOps rt_setup_ops;  // the real syscalls; self-checks inject a kernel
  bool rt_setup_blocks_arm() const {
    return rt_setup_required.load() && rt_setup_failed.load() != 0;
  }
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

  FaultSlot fault_ring[FAULT_RING];
  std::atomic<uint32_t> fault_head{0}, fault_tail{0}, fault_dropped{0}, fault_contended{0};
  FILE* journal_out = stderr;  // the consumer's sink (a check substitutes a buffer)
  // Codes already journaled as SECONDARY in this latch: a persisting cause
  // re-latches every tick and is not new information. Cleared by DISARM.
  std::atomic<uint32_t> secondary_codes{0};

  // `origin_gen` is the ARM generation the event arose in, never re-read at
  // enqueue time, so a record is never reassigned to a later epoch.
  void journal(FaultKind kind, uint32_t code, const char* text, uint64_t origin_gen) {
    uint32_t head = fault_head.load(std::memory_order_relaxed);
    for (int tries = 0;; ++tries) {
      if (head - fault_tail.load(std::memory_order_acquire) >= FAULT_RING) {
        fault_dropped.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      if (fault_head.compare_exchange_strong(head, head + 1, std::memory_order_acq_rel,
                                             std::memory_order_relaxed))
        break;
      if (tries + 1 >= FAULT_CLAIM_TRIES) {
        fault_contended.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
    FaultSlot& slot = fault_ring[head % FAULT_RING];
    FaultRecord& r = slot.rec;
    r.t_ns = mono_ns();
    r.session_id = session_id.load(std::memory_order_acquire);
    r.code = code;
    r.arm_epoch = uint32_t(origin_gen);
    r.kind = kind;
    r.armed = armed.load(std::memory_order_acquire);
    r.truncated = bounded_copy(r.text, sizeof(r.text), text);
    slot.ready.store(head + 1, std::memory_order_release);
  }

  void latch(uint32_t code, const char* text, uint64_t origin_gen) {
    uint32_t expected = LATCH_FREE;  // first cause wins, decided by ONE claim
    if (!latch_state.compare_exchange_strong(expected, LATCH_WRITING,
                                             std::memory_order_acq_rel)) {
      // Latched, being written, cleared or armed. Never waited on: a cause
      // that persists re-latches on a later tick once DISARM has cleared.
      const uint32_t bit = 1u << (code & 31u);
      if (code != fault_code.load(std::memory_order_acquire) &&
          !(secondary_codes.fetch_or(bit, std::memory_order_acq_rel) & bit))
        journal(FAULT_SECONDARY, code, text, origin_gen);
      return;
    }
    if (latch_seam) latch_seam(latch_seam_arg);
    if (bounded_copy(fault_text, sizeof(fault_text), text))  // the wire copy says so too
      std::memcpy(fault_text + sizeof(fault_text) - 4, "...", 4);
    fault_code.store(code, std::memory_order_release);
    fault_gen.store(origin_gen, std::memory_order_release);
    fault_seq.fetch_add(1, std::memory_order_acq_rel);
    fault.store(true, std::memory_order_release);
    journal(FAULT_FIRST, code, text, origin_gen);
    latch_state.store(LATCH_SET, std::memory_order_release);  // LAST
  }

  // DISARM's latch reset (control thread only). Waits for an in-progress
  // writer -- the RT loop fills a slot in microseconds -- until a monotonic
  // deadline LATCH_CLEAR_BUDGET_NS away, never clearing underneath it. False
  // = the writer did not finish; the latch is left intact (DISARM still drops
  // authority) and arm_unlatched() keeps refusing until a later DISARM
  // clears it. The journal records are never cleared.
  static constexpr uint64_t LATCH_CLEAR_BUDGET_NS = 100'000'000;
  bool clear_latch() {
    const uint64_t deadline = mono_ns() + LATCH_CLEAR_BUDGET_NS;
    for (;;) {
      uint32_t s = latch_state.load(std::memory_order_acquire);
      if (s == LATCH_WRITING) {
        if (mono_ns() >= deadline) return false;
        timespec pause{0, 50'000};
        nanosleep(&pause, nullptr);
        continue;
      }
      if (latch_state.compare_exchange_strong(s, LATCH_CLEARING,
                                              std::memory_order_acq_rel))
        break;
    }
    // An undelivered notice for this latch is simply no longer owed:
    // pending_fault_notice() reports only a SET latch. The journal holds it.
    fault.store(false, std::memory_order_release);
    fault_code.store(0, std::memory_order_release);
    fault_gen.store(0, std::memory_order_release);
    fault_text[0] = '\0';
    secondary_codes.store(0, std::memory_order_release);
    latch_state.store(LATCH_FREE, std::memory_order_release);
    return true;
  }

  // CTL_ARM's admission, coherent with the latch (control thread only). The
  // word goes FREE->ARMING by CAS: a latch still being written (including one
  // whose DISARM clear timed out before fault was published), a latched
  // fault, or a reset in progress refuses the ARM, and no latch can claim the
  // word while the new generation is prepared (an old-generation latch still
  // in flight is journaled as secondary with its own generation). Authority
  // is raised only after the word is FREE again, so every fault of the new
  // generation can latch. False = refused; nothing was prepared.
  template <class Prepare>
  bool arm_unlatched(Prepare prepare) {
    uint32_t s = LATCH_FREE;
    if (!latch_state.compare_exchange_strong(s, LATCH_ARMING, std::memory_order_acq_rel))
      return false;
    prepare();
    latch_state.store(LATCH_FREE, std::memory_order_release);
    if (arm_seam) arm_seam();
    armed.store(true, std::memory_order_release);
    return true;
  }

  // The CTL_FAULT still owed to the client, if any (control thread only): a
  // SET latch whose serial has not been delivered. Only a published latch is
  // owed -- one DISARM cleared, or one still being written, is not -- and its
  // text is stable while SET, because only this thread can clear it.
  struct FaultNotice {
    uint64_t seq;
    uint32_t code;
    char text[sizeof(fault_text)];
  };
  bool pending_fault_notice(FaultNotice& out) const {
    if (latch_state.load(std::memory_order_acquire) != LATCH_SET) return false;
    const uint64_t seq = fault_seq.load(std::memory_order_acquire);
    if (seq == fault_notified) return false;
    out.seq = seq;
    out.code = fault_code.load(std::memory_order_acquire);
    std::memcpy(out.text, fault_text, sizeof(out.text));
    return true;
  }
  // Delivered. Monotonic: acknowledging an older notice never discards a
  // newer one.
  void ack_fault_notice(uint64_t seq) {
    if (seq > fault_notified) fault_notified = seq;
  }
};

// Drain the fault journal to stderr; returns the number of records written.
// Non-RT only (state_tx): formatting and stdio stay off the servo thread.
inline int drain_fault_journal(ServerCtx& ctx, FILE* out) {
  static const char* const kinds[] = {"first cause", "secondary cause",
                                      "stop failed"};
  int written = 0;
  for (uint32_t tail = ctx.fault_tail.load(std::memory_order_relaxed);
       ctx.fault_ring[tail % FAULT_RING].ready.load(std::memory_order_acquire) ==
       tail + 1;
       ++tail, ++written) {
    const FaultRecord r = ctx.fault_ring[tail % FAULT_RING].rec;
    ctx.fault_tail.store(tail + 1, std::memory_order_release);
    std::fprintf(out,
                 "[rt] fault journal: %s code=%u t=%.6f arm_epoch=%u armed=%d "
                 "session=%016llx dropped=%u contended=%u text=\"%s\"%s\n",
                 kinds[r.kind < 3 ? r.kind : 1], r.code, r.t_ns / 1e9, r.arm_epoch,
                 int(r.armed), (unsigned long long)r.session_id,
                 ctx.fault_dropped.load(std::memory_order_relaxed),
                 ctx.fault_contended.load(std::memory_order_relaxed), r.text,
                 r.truncated ? " (truncated)" : "");
  }
  return written;
}

// CTL_ARM's state transition (control thread only; checks arm through it
// too). Admission is ONE transition on the latch word (arm_unlatched): a
// latched, unfinished or resetting latch refuses the ARM and nothing is
// prepared. Admitted, in order: the ARM-start record is published before the
// generation, the flow pin is cleared, the peer is scoped to the new
// generation, the generation is published, and authority is raised last,
// after the latch word is FREE again. Returns the new generation, 0 = refused.
// Applies the setup to the calling thread; returns what failed (0 = all
// established). Scheduling and affinity are read back, not assumed from a
// zero return. Affinity is a prerequisite only when rt_cpu is requested.
inline uint32_t establish_rt_setup(int rt_priority, int rt_cpu, const RtSetupOps& ops = {}) {
  uint32_t failed = 0;
  if (ops.lock_memory(MCL_CURRENT | MCL_FUTURE) != 0) failed |= RT_SETUP_MLOCK;
  sched_param want{}, got{};
  want.sched_priority = rt_priority;
  int policy = -1;
  if (ops.set_sched(pthread_self(), SCHED_FIFO, &want) != 0 ||
      ops.get_sched(pthread_self(), &policy, &got) != 0 || policy != SCHED_FIFO ||
      got.sched_priority != rt_priority)
    failed |= RT_SETUP_SCHED;
  if (rt_cpu >= 0) {
    cpu_set_t set, have;
    CPU_ZERO(&set);
    CPU_ZERO(&have);
    CPU_SET(rt_cpu, &set);
    if (ops.set_affinity(pthread_self(), sizeof(set), &set) != 0 ||
        ops.get_affinity(pthread_self(), sizeof(have), &have) != 0 || !CPU_EQUAL(&set, &have))
      failed |= RT_SETUP_AFFINITY;
  }
  return failed;
}

// "mlockall+sched_fifo+affinity" for the failed bits; "ok" when none.
inline void rt_setup_names(uint32_t failed, char* out, size_t cap) {
  static const char* const names[] = {"mlockall", "sched_fifo", "affinity"};
  size_t at = 0;
  out[0] = '\0';
  for (int b = 0; b < 3; ++b)
    if (failed & (1u << b) && at < cap)
      at += size_t(std::snprintf(out + at, cap - at, "%s%s", at ? "+" : "", names[b]));
  if (!failed) std::snprintf(out, cap, "ok");
}

// CTL_ARM's refusal text when the process-local setup gates ARM.
inline void rt_setup_refusal(const ServerCtx& ctx, char* out, size_t cap) {
  char names[48];
  rt_setup_names(ctx.rt_setup_failed.load(), names, sizeof(names));
  std::snprintf(out, cap, "refused: RT process setup failed (%s); fix limits, restart", names);
}

inline uint64_t begin_arm(ServerCtx& ctx) {
  // Checked first, before anything of a new generation is prepared: no ARM
  // until rt_loop has published the backend and its RT setup result (the
  // setup fields are permissive defaults until then), and none when that
  // setup failed.
  if (!ctx.backend_ready.load(std::memory_order_acquire) || ctx.rt_setup_blocks_arm()) return 0;
  uint64_t gen = 0;
  ctx.arm_unlatched([&] {
    gen = ctx.arm_gen.load(std::memory_order_relaxed) + 1;
    ctx.arm_start.write(ArmStart{gen, mono_ns()});
    ctx.cmd_owner.store(0);
    {
      std::lock_guard<std::mutex> lock(ctx.peer_mu);
      ctx.peer_gen = gen;
    }
    ctx.arm_gen.store(gen);
  });
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
// After every thread is joined: the only safe point to close the UDP socket
// both udp_rx (recvfrom) and state_tx (sendto) used.
void close_udp_socket(ServerCtx& ctx);
// Orderly-shutdown export. Call ONLY after every producer (rt_loop, control)
// and the running consumer (state_tx) have returned: it is then the one
// consumer, so records enqueued by the final backend stop or a last control
// fault still reach the sink. Demotes the calling thread to SCHED_OTHER
// first -- main IS the servo thread and still holds its FIFO priority. Best
// effort: SIGKILL, power loss or a failed sink still lose the tail.
void finish_fault_journal(ServerCtx& ctx);
void control_thread(ServerCtx& ctx);
void rt_loop(ServerCtx& ctx);  // constructs the backend ON this thread

} // namespace arm_rt
