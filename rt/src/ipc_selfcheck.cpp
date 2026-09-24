// RT <-> non-RT transport (CTRL-012): the Mailbox primitive, command
// admission (handle_datagram) and tracking freshness (command_fresh). No
// sockets are opened: datagrams are handed to the production admission
// function directly. No backend, no devices.
#include <arpa/inet.h>

#include <atomic>
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
  Bench() {
    ctx.cfg.n = 2;
    ctx.session_id.store(SESSION);
    ctx.claim_ok.store(true);
    ctx.ctl_peer_ip.store(CLIENT_IP);
  }
  void arm() {  // CTL_ARM's order (control_session.cpp)
    ctx.last_cmd_rx_ns.store(mono_ns());
    ctx.cmd_owner.store(0);
    ctx.arm_gen.fetch_add(1);
    ctx.armed.store(true);
  }
  void disarm() { ctx.armed.store(false); }
  bool send(const CommandPacket& c, const sockaddr_in& src) {
    return handle_datagram(ctx, flow, &c, ssize_t(sizeof(c)), src);
  }
  uint16_t pin() const {
    const uint64_t o = ctx.cmd_owner.load();
    return (o >> 16) == ctx.arm_gen.load() ? ntohs(uint16_t(o & 0xFFFF)) : 0;
  }
  CommandEnvelope latest(uint64_t* version = nullptr) {  // test is the one reader
    CommandEnvelope e{};
    const uint64_t v = ctx.cmd_in.read(e);
    if (version) *version = v;
    return e;
  }
};

void check_admission() {
  Bench b;
  const auto a = from(CLIENT_IP, 5000), c = from(CLIENT_IP, 5002);

  // Disarmed: the session may teach its address; nothing is published.
  assert(!b.send(command(0, 1), a));
  assert(b.ctx.have_peer && b.ctx.peer.sin_port == htons(5000));
  uint64_t v = 0;
  b.latest(&v);
  assert(v == 0 && b.ctx.last_cmd_rx_ns == 0);

  b.arm();  // epoch 1
  const uint64_t armed_rx = b.ctx.last_cmd_rx_ns;
  // Rejected before any side effect: wrong n, wrong session, other host,
  // wrong epoch. None pins, refreshes the deadman or publishes.
  assert(!b.send(command(1, 2, SESSION, 3), a));
  assert(!b.send(command(1, 2, 0x999), a));
  assert(!b.send(command(1, 2), from(OTHER_IP, 5000)));
  assert(!b.send(command(7, 2), a));
  assert(b.pin() == 0 && b.ctx.last_cmd_rx_ns == armed_rx);
  b.latest(&v);
  assert(v == 0);

  assert(b.send(command(1, 10), a));  // first valid sender pins its port
  assert(b.pin() == 5000);
  CommandEnvelope e = b.latest(&v);
  assert(v == 1 && e.arm_gen == 1 && e.packet.command.kp[0] == 10);
  assert(e.rx_ns == b.ctx.last_cmd_rx_ns);  // payload and stamp are one value
  assert(!b.send(command(1, 11), from(CLIENT_IP, 5001)));  // other port
  assert(!b.send(command(1, 10), a));                      // duplicate seq
  assert(!b.send(command(1, 9), a));                       // older seq
  assert(b.send(command(1, 11), a));
  b.latest(&v);
  assert(v == 2);

  // F02: after DISARM -> ARM, a stale epoch-1 packet from the OLD port
  // arrives first. It must not pin: the valid new flow from another port wins.
  b.disarm();
  b.arm();  // epoch 2
  assert(!b.send(command(1, 12), a));
  assert(b.pin() == 0);
  assert(b.send(command(2, 1), c));  // new flow, seq restarts in the new epoch
  assert(b.pin() == 5002);
  e = b.latest(&v);
  assert(v == 3 && e.arm_gen == 2 && e.packet.command.kp[0] == 1);
  assert(!b.send(command(2, 13), a));  // the old port stays out

  // A pin taken under another generation is no pin: an ARM that races the
  // CAS (simulated by a stale owner word) cannot lock the new flow out.
  b.arm();  // epoch 3
  b.ctx.cmd_owner.store((uint64_t(2) << 16) | htons(5000));
  assert(b.send(command(3, 1), c) && b.pin() == 5002);

  // Observer leases never touch command state, even from the control host
  // while armed and unpinned.
  b.arm();  // epoch 4
  const uint64_t rx = b.ctx.last_cmd_rx_ns;
  const sockaddr_in peer_before = b.ctx.peer;
  ObserveRequest obs{MAGIC_OBS, VERSION, 0, 1000, 0};
  assert(!handle_datagram(b.ctx, b.flow, &obs, ssize_t(sizeof(obs)),
                          from(CLIENT_IP, 6000)));
  assert(b.pin() == 0 && b.ctx.last_cmd_rx_ns == rx);
  assert(b.ctx.peer.sin_port == peer_before.sin_port);
  bool leased = false;
  for (const auto& o : b.ctx.observers) leased |= o.addr.sin_port == htons(6000);
  assert(leased);
}

// ---- Freshness at consumption ----------------------------------------------

void check_freshness() {
  CommandEnvelope e{};
  e.packet.command = command(5, 1);
  e.rx_ns = 1'000'000'000;
  e.arm_gen = 5;
  const double hold = 100e6;
  assert(command_fresh(1, e, 5, 2, e.rx_ns + 50'000'000, hold));
  assert(!command_fresh(0, e, 5, 2, e.rx_ns, hold));                // unpublished
  assert(!command_fresh(1, e, 6, 2, e.rx_ns, hold));                // ARM raced it
  assert(!command_fresh(1, e, 5, 3, e.rx_ns, hold));                // joint count
  assert(!command_fresh(1, e, 5, 2, e.rx_ns + 150'000'000, hold));  // own age
  assert(command_fresh(1, e, 5, 2, e.rx_ns - 1000, hold));          // future stamp
}
}  // namespace

namespace arm_rt {
uint64_t mono_ns();  // udp_link.cpp (real clock)
}

int main() {
  check_mailbox_sequential();
  check_mailbox_concurrent();
  check_admission();
  check_freshness();
  std::puts("PASS RT IPC: mailbox latest-wins without tearing, admission before "
            "side effects (stale-port pin, epochs, observers), envelope freshness");
}
