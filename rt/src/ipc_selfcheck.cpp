// RT <-> non-RT transport (CTRL-012): the Mailbox primitive, command
// admission (handle_datagram) and RT-side authority (CommandAuthority). No
// sockets are opened: datagrams are handed to the production admission
// function directly. No backend, no devices.
#include <arpa/inet.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <thread>

#include "server.hpp"

namespace {
using namespace arm_rt;

// ---- Mailbox --------------------------------------------------------------

struct Pattern {  // every word equals the version: any tear is detectable
  uint64_t words[64];
};

void check_mailbox_sequential() {
  Mailbox<Pattern> box;
  Pattern out;
  std::memset(&out, 0xAB, sizeof(out));
  assert(box.read(out) == 0 && out.words[0] == 0);  // nothing published: T{}
  for (uint64_t v = 1; v <= 5; ++v) {
    Pattern p;
    for (auto& w : p.words) w = v;
    box.write(p);
  }
  assert(box.read(out) == 5 && out.words[63] == 5);  // latest wins
  assert(box.read(out) == 5 && out.words[0] == 5);   // nothing new: same again
  for (uint64_t v = 6; v <= 1'000'000; ++v) {        // stalled reader: writer
    Pattern p;                                       // never waits or fails
    for (auto& w : p.words) w = v;
    box.write(p);
  }
  assert(box.read(out) == 1'000'000 && out.words[31] == 1'000'000);
}

// Overlapping producer/consumer: the reader always gets a whole value, in
// non-decreasing version order, and eventually the last one.
void check_mailbox_concurrent() {
  Mailbox<Pattern> box;
  constexpr uint64_t LAST = 2'000'000;
  std::atomic<bool> go{false};
  std::thread writer([&] {
    while (!go.load()) {}
    Pattern p;
    for (uint64_t v = 1; v <= LAST; ++v) {
      for (auto& w : p.words) w = v;
      box.write(p);
    }
  });
  go.store(true);
  uint64_t prev = 0, reads = 0, distinct = 0;
  Pattern out;
  for (;;) {
    const uint64_t v = box.read(out);
    ++reads;
    for (const auto w : out.words) assert(w == v);  // never torn
    assert(v >= prev);
    distinct += v != prev;
    prev = v;
    if (v == LAST) break;
  }
  writer.join();
  std::printf("mailbox concurrent: %llu reads, %llu distinct versions seen\n",
              (unsigned long long)reads, (unsigned long long)distinct);
}

// ---- Admission ------------------------------------------------------------

constexpr uint64_t SESSION = 0x1234;
const uint32_t CLIENT_IP = htonl(0x7F000001);  // 127.0.0.1
const uint32_t OTHER_IP = htonl(0x7F000002);

sockaddr_in from(uint32_t ip, uint16_t port) {
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = ip;
  a.sin_port = htons(port);
  return a;
}

CommandPacket command(uint32_t epoch, uint32_t seq, uint64_t session = SESSION,
                      uint16_t n = 2) {
  CommandPacket c{};
  c.magic = MAGIC_CMD;
  c.version = VERSION;
  c.n = n;
  c.seq = seq;
  c.arm_epoch = epoch;
  c.session_id = session;
  c.kp[0] = double(seq);  // identifies the packet in the mailbox
  return c;
}

struct Bench {
  ServerCtx ctx;
  CommandFlow flow;
  // The RT side of one servo tick: the single reader of both mailboxes.
  CommandAuthority rt;
  ArmStart start{};
  CommandEnvelope envelope{};
  Bench() {
    ctx.cfg.n = 2;
    ctx.session_id.store(SESSION);
    ctx.claim_ok.store(true);
    ctx.ctl_peer_ip.store(CLIENT_IP);
  }
  uint64_t arm() { return begin_arm(ctx); }  // CTL_ARM's transition
  void disarm() { ctx.armed.store(false); }
  bool send(const CommandPacket& c, const sockaddr_in& src, uint64_t ingress = 0,
            CommandFlow* f = nullptr) {
    return handle_datagram(ctx, f ? *f : flow, &c, ssize_t(sizeof(c)), src,
                           ingress ? ingress : mono_ns());
  }
  uint16_t pin() const {
    const uint64_t o = ctx.cmd_owner.load();
    return (o >> 16) == ctx.arm_gen.load() ? ntohs(uint16_t(o & 0xFFFF)) : 0;
  }
  uint64_t tick() {  // rt_loop's order: generation first, then the mailboxes
    const uint64_t gen = ctx.arm_gen.load();
    ctx.arm_start.read(start);
    const uint64_t v = ctx.cmd_in.read(envelope);
    rt.observe(gen, start, v, envelope);
    return v;
  }
  bool fresh(uint64_t now = 0) const {
    return rt.fresh(ctx.session_id.load(), 2, now ? now : mono_ns(), 100e6);
  }
  uint16_t peer_port() {
    std::lock_guard<std::mutex> lock(ctx.peer_mu);
    return ntohs(ctx.peer.sin_port);
  }
};

void check_admission() {
  Bench b;
  const auto a = from(CLIENT_IP, 5000), c = from(CLIENT_IP, 5002);

  // Disarmed: the session may teach its address; nothing is published.
  assert(!b.send(command(0, 1), a));
  assert(b.ctx.have_peer && b.peer_port() == 5000);
  assert(b.tick() == 0 && !b.rt.have && b.rt.deadman_base() == 0);

  b.arm();  // epoch 1: its deadline starts now
  b.tick();
  const uint64_t armed_base = b.rt.deadman_base();
  assert(armed_base != 0 && armed_base == b.start.t_ns && b.start.gen == 1);
  // Rejected before any side effect: wrong n, wrong session, other host,
  // wrong epoch. None pins, publishes or moves the deadline.
  assert(!b.send(command(1, 2, SESSION, 3), a));
  assert(!b.send(command(1, 2, 0x999), a));
  assert(!b.send(command(1, 2), from(OTHER_IP, 5000)));
  assert(!b.send(command(7, 2), a));
  assert(b.pin() == 0 && b.tick() == 0 && b.rt.deadman_base() == armed_base);

  const uint64_t ingress = mono_ns();
  assert(b.send(command(1, 10), a, ingress));  // first valid sender pins its port
  assert(b.pin() == 5000);
  assert(b.tick() == 1 && b.rt.have && b.rt.current.packet.command.kp[0] == 10);
  assert(b.rt.current.rx_ns == ingress && b.rt.current.session_id == SESSION);
  assert(b.rt.deadman_base() == ingress && b.fresh());
  assert(!b.send(command(1, 11), from(CLIENT_IP, 5001)));  // other port
  assert(!b.send(command(1, 10), a));                      // duplicate seq
  assert(!b.send(command(1, 9), a));                       // older seq
  assert(b.send(command(1, 11), a));
  assert(b.tick() == 2);

  // F02: after DISARM -> ARM, a stale epoch-1 packet from the OLD port
  // arrives first. It must not pin: the valid new flow from another port wins.
  b.disarm();
  b.arm();  // epoch 2
  assert(!b.send(command(1, 12), a));
  assert(b.pin() == 0);
  b.tick();
  assert(!b.rt.have && !b.fresh());  // epoch 1's command is not authority now
  assert(b.send(command(2, 1), c));  // new flow, seq restarts in the new epoch
  assert(b.pin() == 5002);
  assert(b.tick() == 3 && b.rt.current.arm_gen == 2 && b.fresh());
  assert(!b.send(command(2, 13), a));  // the old port stays out

  // A pin taken under another generation is no pin: an ARM that races the
  // CAS (simulated by a stale owner word) cannot lock the new flow out.
  b.arm();  // epoch 3
  b.ctx.cmd_owner.store((uint64_t(2) << 16) | htons(5000));
  assert(b.send(command(3, 1), c) && b.pin() == 5002);

  // Observer leases never touch command state, even from the control host
  // while armed and unpinned.
  b.arm();  // epoch 4
  b.tick();
  const uint64_t base = b.rt.deadman_base();
  const uint16_t peer_before = b.peer_port();
  ObserveRequest obs{MAGIC_OBS, VERSION, 0, 1000, 0};
  assert(!handle_datagram(b.ctx, b.flow, &obs, ssize_t(sizeof(obs)),
                          from(CLIENT_IP, 6000), mono_ns()));
  assert(b.pin() == 0 && b.tick() == 4 && b.rt.deadman_base() == base);
  assert(b.peer_port() == peer_before);
  bool leased = false;
  for (const auto& o : b.ctx.observers) leased |= o.addr.sin_port == htons(6000);
  assert(leased);
}

// R1 (review 2026-09-24): a VALID epoch-N handler stalls after its checks,
// DISARM/ARM establishes N+1, time passes, then the handler resumes. It must
// not be tracked, must not refresh N+1's deadline, must not redirect the
// state stream, and must not keep N+1's own flow out.
struct Pause {
  std::mutex m;
  std::condition_variable cv;
  bool paused = false, released = false;
};
void stall(void* arg) {
  auto* p = static_cast<Pause*>(arg);
  std::unique_lock<std::mutex> lock(p->m);
  p->paused = true;
  p->cv.notify_all();
  p->cv.wait(lock, [p] { return p->released; });
}

void check_retired_handler() {
  Bench b;
  const auto a = from(CLIENT_IP, 5000), c = from(CLIENT_IP, 5002);
  b.arm();  // epoch 1
  assert(b.send(command(1, 1), a));
  b.tick();
  assert(b.fresh() && b.pin() == 5000 && b.peer_port() == 5000);

  Pause pause;
  b.flow.after_checks = stall;  // udp_rx itself stalls: nothing else admits
  b.flow.seam_arg = &pause;
  const uint64_t old_ingress = mono_ns();
  std::thread udp_rx([&] { assert(b.send(command(1, 2), a, old_ingress)); });
  {
    std::unique_lock<std::mutex> lock(pause.m);
    pause.cv.wait(lock, [&] { return pause.paused; });
  }
  // Control thread meanwhile: DISARM, the new client teaches port 5002 while
  // disarmed (its own flow state), ARM -> epoch 2. The servo keeps ticking.
  b.disarm();
  CommandFlow teach;
  assert(!b.send(command(0, 1), c, 0, &teach) && b.peer_port() == 5002);
  b.arm();
  b.tick();
  const uint64_t epoch2_base = b.rt.deadman_base();
  assert(b.start.gen == 2 && epoch2_base == b.start.t_ns && !b.rt.have);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));  // the stall lasts
  {
    std::lock_guard<std::mutex> lock(pause.m);
    pause.released = true;
  }
  pause.cv.notify_all();
  udp_rx.join();
  b.flow.after_checks = nullptr;

  // The late epoch-1 publish did happen (the mailbox holds it) ...
  assert(b.tick() == 2 && b.envelope.arm_gen == 1);
  // ... but it is not tracked, does not move epoch 2's deadline, and did not
  // redirect the state stream.
  assert(!b.rt.have && !b.fresh() && b.rt.deadman_base() == epoch2_base);
  assert(b.peer_port() == 5002 && b.ctx.peer_gen == 2);
  // Its CAS may have written an epoch-1 pin; that is no pin for epoch 2.
  assert(b.pin() == 0);
  // The valid epoch-2 flow is admitted, tracked, and owns the new deadline.
  const uint64_t ingress = mono_ns();
  assert(b.send(command(2, 1), c, ingress) && b.pin() == 5002);
  assert(b.tick() == 3 && b.rt.have && b.fresh() && b.rt.deadman_base() == ingress);
  assert(!b.send(command(2, 3), a));  // the retired port stays out

  // Owner/session loss inside the epoch: tracking stops at once (the
  // CTL_LOST latch follows separately), and a new session's claim does not
  // adopt the old session's commands.
  b.ctx.session_id.store(0);
  assert(!b.fresh());
  b.ctx.session_id.store(SESSION + 1);
  assert(!b.fresh());
  b.ctx.session_id.store(SESSION);
  assert(b.fresh());

  // Delayed ingress within one epoch: stamped at ingress, so a packet whose
  // handler ran late is stale by its own age and cannot extend the deadline.
  b.arm();  // epoch 3
  b.tick();
  const uint64_t epoch3_base = b.rt.deadman_base();
  assert(b.send(command(3, 1), c, epoch3_base - 200'000'000));
  assert(b.tick() == 4 && b.rt.have && !b.fresh(epoch3_base));
  assert(b.rt.deadman_base() == epoch3_base);
}

// ---- Freshness at consumption ----------------------------------------------

void check_freshness() {
  CommandAuthority rt;
  CommandEnvelope e{};
  e.packet.command = command(5, 1);
  e.rx_ns = 1'000'000'000;
  e.arm_gen = 5;
  e.session_id = SESSION;
  const double hold = 100e6;
  rt.observe(5, ArmStart{5, 900'000'000}, 0, e);  // nothing published yet
  assert(!rt.have && rt.deadman_base() == 900'000'000);
  rt.observe(5, ArmStart{5, 900'000'000}, 1, e);
  assert(rt.fresh(SESSION, 2, e.rx_ns + 50'000'000, hold));
  assert(rt.fresh(SESSION, 2, e.rx_ns - 1000, hold));            // future stamp
  assert(!rt.fresh(SESSION, 3, e.rx_ns, hold));                  // joint count
  assert(!rt.fresh(SESSION, 2, e.rx_ns + 150'000'000, hold));    // own age
  assert(!rt.fresh(0, 2, e.rx_ns, hold));                        // no session
  assert(rt.deadman_base() == e.rx_ns);
  CommandEnvelope older = e;
  older.rx_ns -= 10;  // a late publish never moves time backwards
  rt.observe(5, ArmStart{5, 900'000'000}, 2, older);
  assert(rt.current.rx_ns == e.rx_ns);
  rt.observe(6, ArmStart{5, 900'000'000}, 2, e);  // ARM raced it: new generation
  assert(!rt.have && rt.deadman_base() == 0);      // (its record not seen yet)
  rt.observe(6, ArmStart{6, 1'100'000'000}, 2, e);
  assert(!rt.have && rt.deadman_base() == 1'100'000'000);
}
}  // namespace

namespace arm_rt {
uint64_t mono_ns();  // udp_link.cpp (real clock)
}

int main() {
  check_mailbox_sequential();
  check_mailbox_concurrent();
  check_admission();
  check_retired_handler();
  check_freshness();
  std::puts("PASS RT IPC: mailbox latest-wins without tearing, admission before "
            "side effects (stale-port pin, epochs, observers), retired-handler and session "
            "scoping, generation-owned deadline");
}
