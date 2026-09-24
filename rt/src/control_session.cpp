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
#include <random>

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

ControlPacket make(uint16_t type, uint32_t arg, const char* text, uint32_t seq = 0) {
  ControlPacket pkt = {};
  pkt.magic = MAGIC_CTL;
  pkt.version = VERSION;
  pkt.type = type;
  pkt.seq = seq;
  pkt.arg = arg;
  pkt.t_mono_ns = mono_ns();
  if (text) std::snprintf(pkt.text, sizeof(pkt.text), "%s", text);
  return pkt;
}

uint32_t flags_snapshot(const ServerCtx& ctx) {
  uint32_t f = 0;
  if (ctx.armed.load()) f |= FLAG_ARMED;
  if (ctx.fault.load()) f |= FLAG_FAULTED;
  if (ctx.claim_ok.load()) f |= FLAG_CLAIMED;
  return with_online_mask(f, ctx.online_mask.load());
}

uint32_t capabilities(const ServerCtx& ctx) {
  // CAP_CALIBRATED_TASK_SPACE is never advertised: no backend evaluates a
  // deployment-approved model in the servo (see protocol.hpp).
  return CAP_OBSERVER | (ctx.supports_pose_hold.load() ? CAP_POSE_HOLD : 0u);
}

IdentityPayload identity(const ServerCtx& ctx, uint64_t session) {
  IdentityPayload id{};
  std::memcpy(id.instance_id, ctx.instance_id, ID_CHARS);
  std::memcpy(id.model_id, ctx.model_id, ID_CHARS);
  id.session_id = session;
  id.joint_order_hash = ctx.joint_order_hash;
  id.identity_hash = ctx.identity_hash;
  id.capabilities = capabilities(ctx);
  id.n = uint32_t(ctx.backend_n.load());
  return id;
}

// Why a CLAIM does not match, or nullptr. Every field a commander relies on
// is compared; the claim cannot ask for less than it will use.
const char* claim_mismatch(const ServerCtx& ctx, const IdentityPayload& want,
                           uint64_t session) {
  if (want.session_id != session) return "claim refused: wrong session";
  char instance[ID_CHARS] = {};
  std::memcpy(instance, want.instance_id, ID_CHARS);
  instance[ID_CHARS - 1] = '\0';
  if (std::memcmp(instance, ctx.instance_id, ID_CHARS) != 0)
    return "claim refused: instance id mismatch";
  if (want.n != uint32_t(ctx.backend_n.load())) return "claim refused: joint count mismatch";
  if (want.joint_order_hash != ctx.joint_order_hash)
    return "claim refused: joint order mismatch";
  if (want.model_id[0] != '\0' &&
      std::strncmp(want.model_id, ctx.model_id, ID_CHARS) != 0)
    return "claim refused: task model mismatch";
  if (want.capabilities & ~capabilities(ctx))
    return "claim refused: required capability not offered";
  return nullptr;
}

uint64_t fresh_session() {
  static std::random_device rd;
  uint64_t v = 0;
  while (v == 0) v = (uint64_t(rd()) << 32) ^ uint64_t(rd());
  return v;
}

void serve_client(ServerCtx& ctx, int fd, uint64_t session) {
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
  char capability[24] = {};
  if (ctx.supports_pose_hold.load())
    std::snprintf(capability, sizeof(capability), " pose_hold=%u", POSE_HOLD_VERSION);
  std::snprintf(hello, sizeof(hello),
                "%s hold_ms=%.0f fault_ms=%.0f active=0x%X%s",
                ctx.backend_name, ctx.cfg.hold_ms, ctx.cfg.fault_ms,
                ctx.active_mask.load(), capability);
  send_frame(fd, make(CTL_HELLO, uint32_t(ctx.backend_n.load()), hello));
  {
    ControlPacket idf = make(CTL_IDENTITY, uint32_t(ctx.backend_n.load()), nullptr);
    const IdentityPayload id = identity(ctx, session);
    std::memcpy(idf.text, &id, sizeof(id));
    send_frame(fd, idf);
  }

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
    ServerCtx::FaultNotice notice;
    if (ctx.pending_fault_notice(notice)) {
      // Acknowledge only a DELIVERED frame, and only its own serial: a failed
      // send keeps the notice owed, and a newer latch is never discarded by
      // the acknowledgement of an older one. A latch DISARM has cleared is
      // not owed at all (the journal keeps it).
      if (!send_frame(fd, make(CTL_FAULT, notice.code, notice.text))) break;
      ctx.ack_fault_notice(notice.seq);
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
      case CTL_STATUS:
        // Read-only inspection; never changes authority or clears a fault.
        send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "status"));
        break;
      case CTL_CLAIM: {
        IdentityPayload want{};
        std::memcpy(&want, rx.text, sizeof(want));
        const char* why = claim_mismatch(ctx, want, session);
        if (why) {
          // A failed claim never revokes an earlier successful one mid-flight
          // (it cannot change who is armed); it only reports the refusal.
          std::fprintf(stderr, "[rt] %s\n", why);
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), why));
        } else {
          ctx.claim_ok.store(true);
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "claim accepted"));
        }
        break;
      }
      case CTL_ARM:
        if (!ctx.claim_ok.load()) {
          // Flags as they are: without FLAG_ARMED set by this request, the
          // client's ack rule already reads this as a refusal.
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx),
                              "refused: no accepted CLAIM on this session"));
        } else if (ctx.armed.load() && !ctx.fault.load()) {
          // Already armed: ACK without a generation bump. A duplicate ARM
          // (double click, client retry after a lost ack) must not tear down
          // and re-open the live torque session mid-flight — the arm_gen edge
          // makes the RT loop run backend->stop() + session re-open inside
          // one tick, a multi-ms gap libfranka can reject under load.
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "already armed",
                              uint32_t(ctx.arm_gen.load())));
        } else {
          // Admission and publication are ONE transition on the latch word
          // (server.hpp: arm_unlatched): a latched fault, a latch still being
          // written -- even one whose DISARM clear timed out before fault was
          // published -- or a reset in progress refuses the ARM. Refused:
          // flags still show ARMED when the server is fault-holding (authority
          // is deliberately retained) — the FAULTED bit is the refusal, and
          // clients must check it, not just ARMED.
          uint64_t gen = 0;
          const bool admitted = ctx.arm_unlatched([&] {
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
            gen = ctx.arm_gen.fetch_add(1) + 1;
          });  // armed is raised inside, after the latch word is FREE again
          if (admitted) {
            // seq = the ARM epoch every command of this authority must carry.
            send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "armed", uint32_t(gen)));
          } else if (ctx.latch_state.load() == ServerCtx::LATCH_SET) {
            std::fprintf(stderr, "[rt] ARM refused: fault latched in ARM epoch %llu\n",
                         (unsigned long long)ctx.fault_gen.load());
            send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), ctx.fault_text));
          } else {
            send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx),
                                "refused: fault latch busy (DISARM again)"));
          }
        }
        break;
      case CTL_DISARM: {
        ctx.armed.store(false);
        // The latch is reset only when no writer holds it (server.hpp). A
        // latch racing this clear is not taken and re-fires on a later tick
        // if the cause persists. The journal keeps every first cause.
        const bool cleared = ctx.clear_latch();
        if (!cleared)
          std::fprintf(stderr, "[rt] DISARM: fault latch still being written; "
                               "not cleared\n");
        // AFTER armed=false (rt_loop reads the generation first). The ack is
        // RECEIPT; completion is StatePacket.stop_epoch reaching this value.
        const uint64_t dg = ctx.disarm_gen.fetch_add(1) + 1;
        send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx),
                            cleared ? "disarm requested"
                                    : "disarm requested; fault latch busy, not cleared",
                            uint32_t(dg)));
        break;
      }
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
    ctx.latch(FAULT_CTL_LOST, "control session lost while armed", ctx.arm_gen.load());
  }
  // The session is over: its commands are no longer admissible and the state
  // stream stops going to its address. Observers keep their own leases.
  ctx.claim_ok.store(false);
  ctx.session_id.store(0);
  {
    std::lock_guard<std::mutex> lock(ctx.peer_mu);
    ctx.have_peer = false;
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
    const uint64_t session = fresh_session();
    ctx.claim_ok.store(false);
    ctx.session_id.store(session);
    std::printf("[rt] control client connected (session %016llx)\n",
                (unsigned long long)session);
    serve_client(ctx, client, session);
    std::printf("[rt] control client disconnected\n");
  }
  close(fd);
}

} // namespace arm_rt
