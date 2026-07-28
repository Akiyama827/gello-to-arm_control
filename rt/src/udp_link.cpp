// UDP fast path: commands in (latest-wins into the seqlock), state out
// (paced re-publish of whatever the RT loop last wrote). Loss handling is
// the RT loop's staleness logic — nothing here retries anything.
#include <arpa/inet.h>
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
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(ctx.cfg.udp_port);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::perror("[rt] udp bind");
    ctx.failed.store(true);
    ctx.shutdown.store(true);
    return;
  }
  timeval tv{0, 100'000};  // 100 ms poll so shutdown is honoured
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ctx.udp_fd = fd;

  CommandPacket pkt;
  sockaddr_in src = {};
  socklen_t srclen = sizeof(src);
  uint32_t dropped_ip = 0;
  bool nan_warned = false;
  while (!ctx.shutdown.load()) {
    const ssize_t got = recvfrom(fd, &pkt, sizeof(pkt), 0,
                                 reinterpret_cast<sockaddr*>(&src), &srclen);
    if (got != ssize_t(sizeof(pkt))) continue;  // timeout, runt, or junk
    if (pkt.magic != MAGIC_CMD || pkt.version != VERSION) continue;
    if (pkt.n == 0 || pkt.n > MAX_JOINTS) continue;
    // While ARMED, only the control client's address has authority — a valid
    // packet from anywhere else (stray tool, second graph) must not steal
    // the state stream or reach the seqlock. Disarmed keeps the open
    // teach-me behavior the bench tools rely on.
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
    }
    // NaN passes every clamp comparison — stop it here, on the non-RT thread.
    const auto all_finite = [](const double* a) {
      for (int j = 0; j < MAX_JOINTS; ++j)
        if (!std::isfinite(a[j])) return false;
      return true;
    };
    if (!(all_finite(pkt.q_des) && all_finite(pkt.qd_des) &&
          all_finite(pkt.tau_ff) && all_finite(pkt.kp) && all_finite(pkt.kd))) {
      if (!nan_warned) {  // once per stream: a silent drop path is a trap
        nan_warned = true;
        std::fprintf(stderr, "[rt] dropping non-finite command (seq %u)\n",
                     pkt.seq);
      }
      continue;
    }
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
    ctx.cmd_in.write(pkt);
    ctx.last_cmd_rx_ns.store(rx, std::memory_order_release);
  }
  close(fd);
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
    if (ctx.udp_fd < 0) continue;
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
    sockaddr_in peer;
    {
      std::lock_guard<std::mutex> lock(ctx.peer_mu);
      if (!ctx.have_peer) continue;
      peer = ctx.peer;
    }
    sendto(ctx.udp_fd, &pkt, sizeof(pkt), MSG_DONTWAIT,
           reinterpret_cast<sockaddr*>(&peer), sizeof(peer));
  }
}

} // namespace arm_rt
