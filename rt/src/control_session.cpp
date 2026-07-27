// TCP control channel: fixed 128-byte ControlPacket frames, one client at a
// time (the direct link has exactly one PC). Carries what must not be lost:
// arm/disarm, heartbeat, and the server's STATUS/FAULT events.
//
// Safety semantics live here and in the RT loop together:
//  - ARM is refused while a fault is latched — DISARM first (the explicit
//    DISARM->ARM cycle is the only path past a latch; nothing auto-re-arms).
//  - Losing the session while armed LATCHES and the arm HOLDS. It does not
//    drop authority: a crashed PC must leave the arm parked, not falling.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

#include "server.hpp"

namespace arm_rt {
namespace {

bool send_frame(int fd, const ControlPacket& pkt) {
  const char* p = reinterpret_cast<const char*>(&pkt);
  size_t left = sizeof(pkt);
  while (left > 0) {
    const ssize_t sent = send(fd, p, left, MSG_NOSIGNAL);
    if (sent <= 0) return false;
    p += sent;
    left -= size_t(sent);
  }
  return true;
}

ControlPacket make(uint16_t type, uint32_t arg, const char* text) {
  ControlPacket pkt = {};
  pkt.magic = MAGIC_CTL;
  pkt.version = VERSION;
  pkt.type = type;
  pkt.arg = arg;
  pkt.t_mono_ns = mono_ns();
  if (text) std::snprintf(pkt.text, sizeof(pkt.text), "%s", text);
  return pkt;
}

uint32_t flags_snapshot(const ServerCtx& ctx) {
  uint32_t f = 0;
  if (ctx.armed.load()) f |= FLAG_ARMED;
  if (ctx.fault.load()) f |= FLAG_FAULTED;
  return f;
}

void serve_client(ServerCtx& ctx, int fd) {
  const int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  send_frame(fd, make(CTL_HELLO, uint32_t(ctx.backend_n.load()), ctx.backend_name));

  ControlPacket rx;
  size_t have = 0;
  while (!ctx.shutdown.load()) {
    if (ctx.fault_event_pending.exchange(false)) {
      send_frame(fd, make(CTL_FAULT, ctx.fault_code.load(), ctx.fault_text));
    }
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, 100);
    if (ready < 0) break;
    if (ready == 0) continue;
    const ssize_t got = recv(fd, reinterpret_cast<char*>(&rx) + have,
                             sizeof(rx) - have, 0);
    if (got <= 0) break;  // client gone
    have += size_t(got);
    if (have < sizeof(rx)) continue;
    have = 0;
    if (rx.magic != MAGIC_CTL || rx.version != VERSION) continue;

    switch (rx.type) {
      case CTL_ARM:
        if (ctx.fault.load()) {
          // Refused: flags still show ARMED when the server is fault-holding
          // (authority is deliberately retained) — the FAULTED bit is the
          // refusal, and clients must check it, not just ARMED.
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), ctx.fault_text));
        } else {
          // The staleness clock starts AT ARM, not at the first command
          // (same semantics as the bench bridges): without this, re-arming
          // after any pause instantly re-latches on the OLD command age.
          ctx.last_cmd_rx_ns.store(mono_ns());
          ctx.armed.store(true);
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "armed"));
        }
        break;
      case CTL_DISARM:
        ctx.armed.store(false);
        ctx.fault.store(false);
        ctx.fault_code.store(0);
        ctx.fault_text[0] = '\0';
        send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "disarmed"));
        break;
      case CTL_PING:
        send_frame(fd, make(CTL_PONG, 0, nullptr));
        break;
      default:
        break;
    }
  }
  if (ctx.armed.load()) {
    ctx.latch(FAULT_CTL_LOST, "control session lost while armed");
  }
  close(fd);
}

} // namespace

void control_thread(ServerCtx& ctx) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  const int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(ctx.cfg.tcp_port);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(fd, 1) != 0) {
    std::perror("[rt] tcp bind/listen");
    ctx.failed.store(true);
    ctx.shutdown.store(true);
    return;
  }
  while (!ctx.shutdown.load()) {
    pollfd pfd{fd, POLLIN, 0};
    if (poll(&pfd, 1, 200) <= 0) continue;
    const int client = accept(fd, nullptr, nullptr);
    if (client < 0) continue;
    std::printf("[rt] control client connected\n");
    serve_client(ctx, client);
    std::printf("[rt] control client disconnected\n");
  }
  close(fd);
}

} // namespace arm_rt
