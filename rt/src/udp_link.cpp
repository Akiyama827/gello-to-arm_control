// UDP fast path: commands in (latest-wins into the seqlock), state out
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

  PoseHoldCommandPacket received{}, decoded{};
  uint64_t warned_session = 0;
  uint32_t warned_epoch = 0;
  bool have_warned_session = false, have_warned_epoch = false;
  sockaddr_in src = {};
  socklen_t srclen = sizeof(src);
  uint32_t dropped_ip = 0;
  uint16_t dropped_port = 0;
  uint16_t warned_n = 0;
  uint64_t seq_epoch = 0;
  uint32_t seq_ip = 0, last_seq = 0;
  uint16_t seq_port = 0;
  bool have_seq = false;
  while (!ctx.shutdown.load()) {
    srclen = sizeof(src);
    const ssize_t got = recvfrom(fd, &received, sizeof(received), MSG_TRUNC,
                                 reinterpret_cast<sockaddr*>(&src), &srclen);
    if (got == ssize_t(sizeof(ObserveRequest))) {
      // Read-only subscription: never touches the command peer, the deadman,
      // the flow pin or the seqlock. A full table refuses newcomers.
      ObserveRequest req{};
      std::memcpy(&req, &received, sizeof(req));
      if (req.magic != MAGIC_OBS || req.version != VERSION ||
          req.lease_ms < 100 || req.lease_ms > 5000) continue;
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
      continue;
    }
    if(got<0 || !decode_command(&received,size_t(got),ctx.supports_pose_hold.load(),decoded)) continue;
    const auto& pkt=decoded.command;
    // Wrong joint count is dropped HERE, before the rx-stamp: the servo
    // rejects such packets on its own n-check, but if they refresh
    // last_cmd_rx_ns the CMD_LOST deadman never fires and the arm parks
    // silently forever behind a healthy-looking stream (audit 2026-07-29).
    const int want_n = ctx.backend_ready.load() ? ctx.backend_n.load()
                                                : ctx.cfg.n;
    if (pkt.n != uint16_t(want_n)) {
      if (warned_n != pkt.n) {
        warned_n = pkt.n;
        std::fprintf(stderr, "[rt] dropping commands with n=%u (server n=%d)\n",
                     pkt.n, want_n);
      }
      continue;
    }
    // Session binding: a command belongs to the CURRENT, CLAIMED control
    // session or it is nothing -- not authority, not a state-stream address,
    // not a deadman refresh. Replaces trusting whichever datagram came first.
    const uint64_t session = ctx.session_id.load(std::memory_order_acquire);
    if (session == 0 || pkt.session_id != session ||
        !ctx.claim_ok.load(std::memory_order_acquire)) {
      if (!have_warned_session || warned_session != pkt.session_id) {
        have_warned_session = true;
        warned_session = pkt.session_id;
        std::fprintf(stderr, "[rt] dropping commands: session %016llx is not the "
                             "claimed control session\n",
                     (unsigned long long)pkt.session_id);
      }
      continue;
    }
    if (!ctx.armed.load(std::memory_order_acquire)) {
      // Disarmed: the session may teach its state-stream address, nothing else.
      // Nothing is written to the servo's buffer while there is no authority.
      if (src.sin_addr.s_addr == ctx.ctl_peer_ip.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(ctx.peer_mu);
        ctx.peer = src;
        ctx.have_peer = true;
      }
      continue;
    }
    // While ARMED, only the control client's FLOW has authority. The IP alone
    // is not enough — every PC-side tool shares it, and a stray tool's
    // zero-gain prime packet must not become the gains snapshot or steal the
    // state stream. First post-ARM sender pins the port; ARM resets the pin.
    if (ctx.armed.load(std::memory_order_acquire)) {
      const uint32_t owner = ctx.ctl_peer_ip.load(std::memory_order_acquire);
      if (owner != 0 && src.sin_addr.s_addr != owner) {
        if (dropped_ip != src.sin_addr.s_addr) {
          dropped_ip = src.sin_addr.s_addr;
          char buf[INET_ADDRSTRLEN] = {};
          inet_ntop(AF_INET, &src.sin_addr, buf, sizeof(buf));
          std::fprintf(stderr, "[rt] dropping commands from %s (armed; owner"
                               " is the control client)\n", buf);
        }
        continue;
      }
      uint16_t oport = ctx.cmd_owner_port.load(std::memory_order_acquire);
      if (oport == 0) {
        ctx.cmd_owner_port.store(src.sin_port, std::memory_order_release);
        oport = src.sin_port;
        std::fprintf(stderr, "[rt] command flow pinned to port %u for this "
                             "ARM\n", ntohs(oport));
      }
      if (src.sin_port != oport) {
        if (dropped_port != src.sin_port) {
          dropped_port = src.sin_port;
          std::fprintf(stderr, "[rt] dropping commands from port %u (armed; "
                               "flow is pinned to %u)\n",
                       ntohs(src.sin_port), ntohs(oport));
        }
        continue;
      }
    }
    const uint64_t epoch=ctx.arm_gen.load(std::memory_order_acquire);
    if (pkt.arm_epoch != uint32_t(epoch)) {
      if (!have_warned_epoch || warned_epoch != pkt.arm_epoch) {
        have_warned_epoch = true;
        warned_epoch = pkt.arm_epoch;
        std::fprintf(stderr, "[rt] dropping commands from ARM epoch %u (current %u)\n",
                     pkt.arm_epoch, uint32_t(epoch));
      }
      continue;
    }
    if(epoch!=seq_epoch || src.sin_addr.s_addr!=seq_ip || src.sin_port!=seq_port)
      have_seq=false;
    if(have_seq && (pkt.seq-last_seq==0 || pkt.seq-last_seq>=0x80000000u)) continue;
    seq_epoch=epoch; seq_ip=src.sin_addr.s_addr; seq_port=src.sin_port;
    last_seq=pkt.seq; have_seq=true;
    {
      std::lock_guard<std::mutex> lock(ctx.peer_mu);
      ctx.peer = src;
      ctx.have_peer = true;
    }
    const uint64_t rx = mono_ns();
    const uint64_t prev = ctx.last_cmd_rx_ns.load(std::memory_order_acquire);
    if (prev != 0 && rx - prev > 300'000'000ull)
      std::fprintf(stderr, "[rt] cmd accept gap %.2fs (seq %u)\n",
                   double(rx - prev) / 1e9, pkt.seq);
    ctx.cmd_in.write(decoded);
    ctx.last_cmd_rx_ns.store(rx, std::memory_order_release);
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
  const sched_param normal{};
  pthread_setschedparam(pthread_self(), SCHED_OTHER, &normal);
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
      std::fprintf(stderr,
                   "[rt] loop gap t=%.6f seq=%u armed=%d period=%.3fms read=%.3fms "
                   "prev_compute=%.3fms prev_write=%.3fms prev_publish=%.3fms "
                   "robot_dt=%lldms cmd_success=%.3f dropped=%u\n",
                   g.t_read_ns / 1e9, g.state_seq, int(g.armed), g.period_ns / 1e6,
                   g.read_ns / 1e6, g.compute_ns / 1e6, g.write_ns / 1e6,
                   g.publish_ns / 1e6, (long long)g.robot_dt_ms, g.cmd_success_rate,
                   ctx.gap_dropped.load(std::memory_order_relaxed));
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
    if (have_peer)
      sendto(ufd, &pkt, sizeof(pkt), MSG_DONTWAIT,
             reinterpret_cast<sockaddr*>(&peer), sizeof(peer));
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
