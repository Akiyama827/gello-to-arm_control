// UDP fast path: commands in (latest-wins into the seqlock), state out
// (paced re-publish of whatever the RT loop last wrote). Loss handling is
// the RT loop's staleness logic — nothing here retries anything.
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

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
    ctx.shutdown.store(true);
    return;
  }
  timeval tv{0, 100'000};  // 100 ms poll so shutdown is honoured
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ctx.udp_fd = fd;

  CommandPacket pkt;
  sockaddr_in src = {};
  socklen_t srclen = sizeof(src);
  while (!ctx.shutdown.load()) {
    const ssize_t got = recvfrom(fd, &pkt, sizeof(pkt), 0,
                                 reinterpret_cast<sockaddr*>(&src), &srclen);
    if (got != ssize_t(sizeof(pkt))) continue;  // timeout, runt, or junk
    if (pkt.magic != MAGIC_CMD || pkt.version != VERSION) continue;
    if (pkt.n == 0 || pkt.n > MAX_JOINTS) continue;
    {
      std::lock_guard<std::mutex> lock(ctx.peer_mu);
      ctx.peer = src;
      ctx.have_peer = true;
    }
    ctx.cmd_in.write(pkt);
    ctx.last_cmd_rx_ns.store(mono_ns(), std::memory_order_release);
  }
  close(fd);
}

void state_tx_thread(ServerCtx& ctx) {
  const double hz = ctx.cfg.state_hz > 1.0 ? ctx.cfg.state_hz : 1.0;
  const long period_ns = long(1e9 / hz);
  StatePacket pkt;
  uint64_t sent_version = 0;
  while (!ctx.shutdown.load()) {
    timespec ts{0, period_ns};
    nanosleep(&ts, nullptr);
    if (ctx.udp_fd < 0) continue;
    const uint64_t v = ctx.state_out.read(pkt);
    if (v == 0 || v == sent_version) continue;  // nothing new
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
