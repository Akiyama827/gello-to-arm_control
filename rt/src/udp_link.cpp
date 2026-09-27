// UDP fast path: commands in (latest-wins into the mailbox), state out
// (paced re-publish of whatever the RT loop last wrote). Loss handling is
// the RT loop's staleness logic — nothing here retries anything.
#include <arpa/inet.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "server.hpp"

namespace arm_rt {

uint64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000ull + uint64_t(ts.tv_nsec);
}

bool handle_datagram(ServerCtx& ctx, CommandFlow& flow, const void* data, ssize_t got,
                     const sockaddr_in& src, uint64_t ingress_ns) {
  if (got == ssize_t(sizeof(ObserveRequest))) {
    // Read-only subscription: never touches the command peer, the deadman,
    // the flow pin or the mailbox. A full table refuses newcomers.
    ObserveRequest req{};
    std::memcpy(&req, data, sizeof(req));
    if (req.magic != MAGIC_OBS || req.version != VERSION ||
        req.lease_ms < 100 || req.lease_ms > 5000) return false;
    const uint64_t now = mono_ns();
    std::lock_guard<std::mutex> lock(ctx.obs_mu);
    Observer* slot = nullptr;
    for (auto& o : ctx.observers) {
      if (o.expiry_ns > now && o.addr.sin_addr.s_addr == src.sin_addr.s_addr &&
          o.addr.sin_port == src.sin_port) { slot = &o; break; }
    }
    if (!slot)
      for (auto& o : ctx.observers)
        if (o.expiry_ns <= now) { slot = &o; slot->addr = src; break; }
    if (slot) slot->expiry_ns = now + uint64_t(req.lease_ms) * 1'000'000ull;
    else std::fprintf(stderr, "[rt] observer table full; lease refused\n");
    return false;
  }
  PoseHoldCommandPacket decoded{};
  ReferenceEnvelopeBinding binding{};
  if(got==ssize_t(sizeof(ReferenceCommandPacket))) {
    ReferenceCommandPacket r{};
    const uint64_t session=ctx.session_id.load(std::memory_order_acquire);
    if(!decode_reference_command(data,size_t(got),
          session && ctx.reference_session.load()==session,r)) return false;
    auto& c=decoded.command;
    c.magic=r.magic; c.version=r.version; c.n=r.n; c.seq=r.seq; c.arm_epoch=r.arm_epoch;
    c.t_mono_ns=r.t_mono_ns; c.session_id=r.session_id;
    std::memcpy(c.tau_ff,r.tau_ff,sizeof(c.tau_ff));
    std::memcpy(c.kp,r.kp,sizeof(c.kp)); std::memcpy(c.kd,r.kd,sizeof(c.kd));
    binding={r.reference_id,r.revision,r.age_us,r.phase,r.valid_for_us,r.phase_window_us};
  } else if (got < 0 || !decode_command(data, size_t(got), ctx.supports_pose_hold.load(), decoded))
    return false;
  const auto& pkt = decoded.command;
  // Wrong joint count is dropped HERE, before anything is published: the
  // servo rejects such packets on its own n-check, but if they refreshed the
  // CMD_LOST deadline the deadman would never fire and the arm would park
  // silently forever behind a healthy-looking stream (audit 2026-07-29).
  const int want_n = ctx.backend_ready.load() ? ctx.backend_n.load() : ctx.cfg.n;
  if (pkt.n != uint16_t(want_n)) {
    if (flow.warned_n != pkt.n) {
      flow.warned_n = pkt.n;
      std::fprintf(stderr, "[rt] dropping commands with n=%u (server n=%d)\n",
                   pkt.n, want_n);
    }
    return false;
  }
  // Session binding: a command belongs to the CURRENT, CLAIMED control
  // session or it is nothing -- not authority, not a state-stream address,
  // not a deadman refresh. Replaces trusting whichever datagram came first.
  const uint64_t session = ctx.session_id.load(std::memory_order_acquire);
  if (session == 0 || pkt.session_id != session ||
      !ctx.claim_ok.load(std::memory_order_acquire)) {
    if (!flow.have_warned_session || flow.warned_session != pkt.session_id) {
      flow.have_warned_session = true;
      flow.warned_session = pkt.session_id;
      std::fprintf(stderr, "[rt] dropping commands: session %016llx is not the "
                           "claimed control session\n",
                   (unsigned long long)pkt.session_id);
    }
    return false;
  }
  if (!ctx.armed.load(std::memory_order_acquire)) {
    // Disarmed: the session may teach its state-stream address, nothing else.
    // Nothing is written to the servo's mailbox while there is no authority.
    const uint64_t gen = ctx.arm_gen.load(std::memory_order_acquire);
    if (src.sin_addr.s_addr == ctx.ctl_peer_ip.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(ctx.peer_mu);
      if (gen >= ctx.peer_gen) {  // not after an ARM this handler predates
        ctx.peer = src;
        ctx.have_peer = true;
        ctx.peer_gen = gen;
      }
    }
    return false;
  }
  // While ARMED, only the control client's FLOW has authority. The IP alone
  // is not enough — every PC-side tool shares it, and a stray tool's
  // zero-gain prime packet must not become the gains snapshot or steal the
  // state stream.
  const uint32_t owner_ip = ctx.ctl_peer_ip.load(std::memory_order_acquire);
  if (owner_ip != 0 && src.sin_addr.s_addr != owner_ip) {
    if (flow.dropped_ip != src.sin_addr.s_addr) {
      flow.dropped_ip = src.sin_addr.s_addr;
      char buf[INET_ADDRSTRLEN] = {};
      inet_ntop(AF_INET, &src.sin_addr, buf, sizeof(buf));
      std::fprintf(stderr, "[rt] dropping commands from %s (armed; owner"
                           " is the control client)\n", buf);
    }
    return false;
  }
  // The ARM epoch is checked BEFORE the flow pin: a stale-epoch packet from an
  // old port used to pin that port and then be dropped, locking the valid new
  // flow out for the whole ARM. Loading the generation first also means a
  // pin cleared by that ARM is visible below (ARM clears, then bumps).
  const uint64_t gen = ctx.arm_gen.load(std::memory_order_acquire);
  if (pkt.arm_epoch != uint32_t(gen)) {
    if (!flow.have_warned_epoch || flow.warned_epoch != pkt.arm_epoch) {
      flow.have_warned_epoch = true;
      flow.warned_epoch = pkt.arm_epoch;
      std::fprintf(stderr, "[rt] dropping commands from ARM epoch %u (current %u)\n",
                   pkt.arm_epoch, uint32_t(gen));
    }
    return false;
  }
  const uint64_t owner = ctx.cmd_owner.load(std::memory_order_acquire);
  const bool pinned = owner != 0 && (owner >> 16) == gen;
  const uint16_t pinned_port = uint16_t(owner & 0xFFFFu);
  if (pinned && src.sin_port != pinned_port) {
    if (flow.dropped_port != src.sin_port) {
      flow.dropped_port = src.sin_port;
      std::fprintf(stderr, "[rt] dropping commands from port %u (armed; "
                           "flow is pinned to %u)\n",
                   ntohs(src.sin_port), ntohs(pinned_port));
    }
    return false;
  }
  if (gen != flow.seq_gen || src.sin_addr.s_addr != flow.seq_ip ||
      src.sin_port != flow.seq_port)
    flow.have_seq = false;
  if (flow.have_seq &&
      (pkt.seq - flow.last_seq == 0 || pkt.seq - flow.last_seq >= 0x80000000u))
    return false;
  // Accepted. Side effects from here on only.
  if (flow.after_checks) flow.after_checks(flow.seam_arg);
  if (!pinned) {
    uint64_t expected = owner;
    if (!ctx.cmd_owner.compare_exchange_strong(expected, (gen << 16) | src.sin_port,
                                               std::memory_order_acq_rel))
      return false;  // an ARM cleared it meanwhile; the next packet re-pins
    std::fprintf(stderr, "[rt] command flow pinned to port %u for this ARM\n",
                 ntohs(src.sin_port));
  }
  flow.seq_gen = gen;
  flow.seq_ip = src.sin_addr.s_addr;
  flow.seq_port = src.sin_port;
  flow.last_seq = pkt.seq;
  flow.have_seq = true;
  {
    // The peer is scoped by generation under its own lock (begin_arm raises
    // peer_gen under it): a handler of a retired generation cannot redirect
    // the state stream of the current one.
    std::lock_guard<std::mutex> lock(ctx.peer_mu);
    if (gen >= ctx.peer_gen) {
      ctx.peer = src;
      ctx.have_peer = true;
      ctx.peer_gen = gen;
    }
  }
  if (flow.last_accept_rx != 0 && ingress_ns - flow.last_accept_rx > 300'000'000ull)
    std::fprintf(stderr, "[rt] cmd accept gap %.2fs (seq %u)\n",
                 double(ingress_ns - flow.last_accept_rx) / 1e9, pkt.seq);
  flow.last_accept_rx = ingress_ns;
  // Published with the generation and session it was checked against and its
  // ingress stamp. If an ARM or session change landed after the checks, the
  // RT loop's CommandAuthority neither tracks it nor lets it move the new
  // generation's deadline: there is no shared deadman stamp left to refresh.
  ctx.cmd_in.write(CommandEnvelope{decoded, ingress_ns, gen, session, binding});
  return true;
}

void udp_rx_thread(ServerCtx& ctx) {
  const int fd = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = ctx.cfg.bind_addr ? ctx.cfg.bind_addr : INADDR_ANY;
  addr.sin_port = htons(ctx.cfg.udp_port);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::perror("[rt] udp bind");
    ctx.failed.store(true);
    ctx.shutdown.store(true);
    return;
  }
  timeval tv{0, 100'000};  // 100 ms poll so shutdown is honoured
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ctx.udp_fd.store(fd);

  PoseHoldCommandPacket received{};
  CommandFlow flow;
  while (!ctx.shutdown.load()) {
    sockaddr_in src = {};
    socklen_t srclen = sizeof(src);
    const ssize_t got = recvfrom(fd, &received, sizeof(received), MSG_TRUNC,
                                 reinterpret_cast<sockaddr*>(&src), &srclen);
    const uint64_t ingress = mono_ns();  // the documented ingress boundary
    handle_datagram(ctx, flow, &received, got, src, ingress);
  }
  // Not closed here: state_tx may already hold this descriptor for a
  // sendto(), and unpublishing it first does not stop that. main closes it
  // after every thread is joined (close_udp_socket).
}

void close_udp_socket(ServerCtx& ctx) {
  const int fd = ctx.udp_fd.exchange(-1);
  if (fd >= 0) close(fd);
}

void finish_fault_journal(ServerCtx& ctx) {
  // Leave SCHED_FIFO before stdio. If that fails the servo has already
  // stopped, so exporting at FIFO priority is still better than losing the
  // records; the failure itself is reported first.
  const sched_param normal{};
  const int demoted = pthread_setschedparam(pthread_self(), SCHED_OTHER, &normal);
  if (demoted != 0)
    std::fprintf(ctx.journal_out, "[rt] fault journal: could not leave real-time "
                 "scheduling (%s); exporting anyway\n", std::strerror(demoted));
  drain_fault_journal(ctx, ctx.journal_out);
  const uint32_t dropped = ctx.fault_dropped.load(std::memory_order_relaxed);
  const uint32_t contended = ctx.fault_contended.load(std::memory_order_relaxed);
  if (dropped || contended)
    std::fprintf(ctx.journal_out, "[rt] fault journal: %u record(s) lost (%u ring full, "
                 "%u claim contention)\n", dropped + contended, dropped, contended);
  std::fflush(ctx.journal_out);
}

void state_tx_thread(ServerCtx& ctx) {
  const double hz = ctx.cfg.state_hz > 1.0 ? ctx.cfg.state_hz : 1.0;
  const long period_ns = long(1e9 / hz);
  StatePacket pkt;
  uint64_t sent_version = 0;
  int unchanged = 0;
  while (!ctx.shutdown.load()) {
    timespec ts{0, period_ns};
    nanosleep(&ts, nullptr);
    drain_fault_journal(ctx, ctx.journal_out);
    for (uint32_t tail = ctx.gap_tail.load(std::memory_order_relaxed);
         tail != ctx.gap_head.load(std::memory_order_acquire); ++tail) {
      const LoopGap g = ctx.gaps[tail % LOOP_GAP_RING];
      ctx.gap_tail.store(tail + 1, std::memory_order_release);
      // One bounded line per gap. The controller fields describe the
      // PREVIOUS write cycle (the compute+write the gap contains);
      // diag=absent means no servo computation ran in that cycle — the
      // fields are then absent, never zero-filled as if measured.
      if (g.diag_valid) {
        std::fprintf(stderr,
                     "[rt] loop gap t=%.6f seq=%u armed=%d period=%.3fms read=%.3fms "
                     "prev_compute=%.3fms prev_write=%.3fms prev_publish=%.3fms "
                     "robot_dt=%lldms cmd_success=%.3f dropped=%u "
                     "session=%016llx arm_epoch=%u last_cmd_seq=%u "
                     "q_err=%.6f qd_err=%.6f tau_ff=%.3f tau_raw=%.3f tau_out=%.3f "
                     "tau_ref=%.3f tau_meas=%.3f limiter_joint=%d clamp=%d slew=%d\n",
                     g.t_read_ns / 1e9, g.state_seq, int(g.armed), g.period_ns / 1e6,
                     g.read_ns / 1e6, g.compute_ns / 1e6, g.write_ns / 1e6,
                     g.publish_ns / 1e6, (long long)g.robot_dt_ms, g.cmd_success_rate,
                     ctx.gap_dropped.load(std::memory_order_relaxed),
                     (unsigned long long)g.session_id, g.arm_epoch, g.last_cmd_seq,
                     g.diag.q_err_max, g.diag.qd_err_max, g.diag.tau_ff_max,
                     g.diag.tau_raw_max, g.diag.tau_out_max, g.diag.tau_ref_max,
                     g.diag.tau_meas_max, g.diag.limiter_joint,
                     int(g.diag.clamp_active), int(g.diag.slew_active));
      } else {
        std::fprintf(stderr,
                     "[rt] loop gap t=%.6f seq=%u armed=%d period=%.3fms read=%.3fms "
                     "prev_compute=%.3fms prev_write=%.3fms prev_publish=%.3fms "
                     "robot_dt=%lldms cmd_success=%.3f dropped=%u "
                     "session=%016llx arm_epoch=%u last_cmd_seq=%u diag=absent\n",
                     g.t_read_ns / 1e9, g.state_seq, int(g.armed), g.period_ns / 1e6,
                     g.read_ns / 1e6, g.compute_ns / 1e6, g.write_ns / 1e6,
                     g.publish_ns / 1e6, (long long)g.robot_dt_ms, g.cmd_success_rate,
                     ctx.gap_dropped.load(std::memory_order_relaxed),
                     (unsigned long long)g.session_id, g.arm_epoch, g.last_cmd_seq);
      }
    }
    const int ufd = ctx.udp_fd.load();
    if (ufd < 0) continue;
    const uint64_t v = ctx.state_out.read(pkt);
    if (v == 0) continue;
    if (v == sent_version) {
      // Nothing new — the RT thread may be inside a legitimately blocking
      // plant call (franka session open takes seconds). Re-send the last
      // packet at ~10 Hz so the CLIENT can tell "link alive, servo busy"
      // (its stamp goes stale, arrival stays fresh) from "link dead".
      if (++unchanged < int(hz / 10.0) + 1) continue;
    }
    unchanged = 0;
    sent_version = v;
    sockaddr_in peer{};
    bool have_peer = false;
    {
      std::lock_guard<std::mutex> lock(ctx.peer_mu);
      have_peer = ctx.have_peer;
      peer = ctx.peer;
    }
    if (have_peer) {
      ReferenceStatePacket extended{};
      if(ctx.reference_session.load()==pkt.session_id && pkt.session_id &&
         ctx.reference_state_out.read(extended) && extended.state.session_id==pkt.session_id)
        sendto(ufd,&extended,sizeof(extended),MSG_DONTWAIT,
               reinterpret_cast<sockaddr*>(&peer),sizeof(peer));
      else
        sendto(ufd, &pkt, sizeof(pkt), MSG_DONTWAIT,
               reinterpret_cast<sockaddr*>(&peer), sizeof(peer));
    }
    Observer copy[MAX_OBSERVERS];
    {
      std::lock_guard<std::mutex> lock(ctx.obs_mu);
      std::memcpy(copy, ctx.observers, sizeof(copy));
    }
    const uint64_t now = mono_ns();
    for (const auto& o : copy) {
      if (o.expiry_ns <= now) continue;
      if (have_peer && o.addr.sin_addr.s_addr == peer.sin_addr.s_addr &&
          o.addr.sin_port == peer.sin_port) continue;
      sendto(ufd, &pkt, sizeof(pkt), MSG_DONTWAIT,
             reinterpret_cast<const sockaddr*>(&o.addr), sizeof(o.addr));
    }
  }
}

} // namespace arm_rt
