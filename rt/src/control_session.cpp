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
  return with_online_mask(f, ctx.online_mask.load());
}

void serve_client(ServerCtx& ctx, int fd) {
  const int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  // A peer that stops draining must not wedge this thread mid-send forever
  // (the deadman below never runs while send() blocks, and shutdown joins
  // hang until SIGKILL). A short send timeout turns that into session loss.
  timeval stv{0, 500'000};
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof(stv));
  // HELLO only once the backend exists — a client connecting during franka
  // construction (~1 s) must not be told "0 joints".
  for (int i = 0; i < 300 && !ctx.backend_ready.load() && !ctx.shutdown.load();
       ++i) {
    timespec ts{0, 10'000'000};
    nanosleep(&ts, nullptr);
  }
  // HELLO text carries the launch-time SAFETY flags, not just the backend
  // name: the client must be able to refuse a server whose deadman is
  // configured for hand-guiding (--fault-ms 3600000) — a commander graph
  // against that server has no staleness reflex at all (audit 2026-07-29).
  char hello[96];
  std::snprintf(hello, sizeof(hello),
                "%s hold_ms=%.0f fault_ms=%.0f active=0x%X%s",
                ctx.backend_name, ctx.cfg.hold_ms, ctx.cfg.fault_ms,
                ctx.active_mask.load(), ctx.supports_pose_hold.load() ? " pose_hold=2" : "");
  send_frame(fd, make(CTL_HELLO, uint32_t(ctx.backend_n.load()), hello));

  // Session deadman: the client answers CTL_PING with CTL_PONG, so a healthy
  // idle link always has traffic. A yanked cable / dead PC leaves a half-open
  // socket that poll() never reports — without this, FAULT_CTL_LOST could
  // take the kernel's 2-hour keepalive to fire.
  constexpr uint64_t PING_EVERY_NS = 250'000'000;   // 4 Hz
  constexpr uint64_t DEAD_AFTER_NS = 1'200'000'000; // ~5 missed pings
  uint64_t last_alive_ns = mono_ns();
  uint64_t last_ping_ns = 0;

  ControlPacket rx;
  size_t have = 0;
  while (!ctx.shutdown.load()) {
    if (ctx.fault_event_pending.load()) {
      // Clear only on a DELIVERED frame — consuming the flag before a failed
      // send loses the one CTL_FAULT event this latch will ever emit.
      if (!send_frame(fd, make(CTL_FAULT, ctx.fault_code.load(), ctx.fault_text)))
        break;
      ctx.fault_event_pending.store(false);
    }
    const uint64_t now = mono_ns();
    if (now - last_ping_ns > PING_EVERY_NS) {
      last_ping_ns = now;
      send_frame(fd, make(CTL_PING, 0, nullptr));
    }
    if (now - last_alive_ns > DEAD_AFTER_NS) break;  // half-open: treat as lost
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
    last_alive_ns = mono_ns();  // any valid frame (PONGs included) is life

    switch (rx.type) {
      case CTL_ARM:
        if (ctx.fault.load()) {
          // Refused: flags still show ARMED when the server is fault-holding
          // (authority is deliberately retained) — the FAULTED bit is the
          // refusal, and clients must check it, not just ARMED.
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), ctx.fault_text));
        } else if (ctx.armed.load()) {
          // Already armed: ACK without a generation bump. A duplicate ARM
          // (double click, client retry after a lost ack) must not tear down
          // and re-open the live torque session mid-flight — the arm_gen edge
          // makes the RT loop run backend->stop() + session re-open inside
          // one tick, a multi-ms gap libfranka can reject under load.
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "already armed"));
        } else {
          // The staleness clock starts AT ARM, not at the first command
          // (same semantics as the bench bridges): without this, re-arming
          // after any pause instantly re-latches on the OLD command age.
          ctx.last_cmd_rx_ns.store(mono_ns());
          // ...and so does the command epoch: whatever sits in the seqlock
          // (address-teach prime, pre-fault leftovers) is not authority.
          // sequence() — NOT read(): a bounded read() that collides with a
          // mid-write returns 0, and an epoch of 0 re-admits every pre-ARM
          // packet ever written. The raw counter can never be a spurious 0.
          ctx.cmd_epoch.store(ctx.cmd_in.sequence());
          // New ARM = new command flow: udp_rx re-pins the source port to
          // the first post-ARM sender (see server.hpp: cmd_owner_port).
          ctx.cmd_owner_port.store(0);
          // Generation bump makes this ARM visible to the RT loop even if a
          // DISARM->ARM pair fits between two 1 kHz samples of `armed`.
          ctx.arm_gen.fetch_add(1);
          ctx.armed.store(true);
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "armed"));
        }
        break;
      case CTL_DISARM:
        ctx.armed.store(false);
        ctx.fault.store(false);
        ctx.fault_code.store(0);
        ctx.fault_text[0] = '\0';
        // Release the latch claim LAST: a latch racing this clear is dropped
        // (its CAS fails) and simply re-fires on the next tick if the cause
        // persists — better than letting it scribble a half-cleared slot.
        ctx.fault_claim.store(false);
        send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "disarmed"));
        break;
      case CTL_SET_ACTIVE: {
        const int n = ctx.backend_n.load();
        const uint32_t configured = n >= 16 ? 0xFFFFu : ((1u << n) - 1u);
        const uint32_t desired = rx.arg;
        const uint32_t current = ctx.active_mask.load();
        if ((desired & ~configured) || (desired & ~ctx.online_mask.load())) {
          send_frame(fd, make(CTL_STATUS, current,
                              "active mask refused: slot offline or unconfigured"));
          break;
        }
        if (ctx.armed.load() && (current & ~desired)) {
          send_frame(fd, make(CTL_STATUS, current,
                              "active mask refused: disarm before removal"));
          break;
        }
        ctx.requested_active_mask.store(desired);
        for (int i = 0; i < 500 && ctx.active_mask.load() != desired; ++i) {
          timespec wait{0, 1'000'000};
          nanosleep(&wait, nullptr);
        }
        const uint32_t applied = ctx.active_mask.load();
        send_frame(fd, make(CTL_STATUS, applied,
                            applied == desired ? "active mask applied"
                                               : "active mask refused by backend"));
        break;
      }
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
  addr.sin_addr.s_addr = ctx.cfg.bind_addr ? ctx.cfg.bind_addr : INADDR_ANY;
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
    sockaddr_in peer = {};
    socklen_t peerlen = sizeof(peer);
    const int client =
        accept(fd, reinterpret_cast<sockaddr*>(&peer), &peerlen);
    if (client < 0) continue;
    // An ARMED (typically fault-holding after CTL_LOST) arm must not hand
    // authority to whoever connects next: only the SAME host that armed it
    // may reclaim the session (its graph restarting is the recovery path).
    // A different host gets a refusal and the pin stays untouched.
    const uint32_t owner = ctx.ctl_peer_ip.load();
    if (ctx.armed.load() && owner != 0 && peer.sin_addr.s_addr != owner) {
      char buf[INET_ADDRSTRLEN] = {};
      inet_ntop(AF_INET, &peer.sin_addr, buf, sizeof(buf));
      std::fprintf(stderr, "[rt] refusing control client %s (armed; owner "
                           "is the arming host)\n", buf);
      send_frame(client, make(CTL_STATUS, flags_snapshot(ctx),
                              "refused: armed by another host"));
      close(client);
      continue;
    }
    // The control client's address is the ONLY source udp_rx will accept
    // commands from while armed (see server.hpp: ctl_peer_ip).
    ctx.ctl_peer_ip.store(peer.sin_addr.s_addr);
    std::printf("[rt] control client connected\n");
    serve_client(ctx, client);
    std::printf("[rt] control client disconnected\n");
  }
  close(fd);
}

} // namespace arm_rt
