// dock_bridge — the Dock_Control CAN nodes re-exported as a TCP line
// protocol on the RT box. Pattern-copied from hand_bridge.cpp (same
// single-client line-parse loop); the CAN side is a straight port of
// arm_control/dock_bench.py, the E1b bench tool. Wire facts from
// Dock_Control PROTOCOL.md, bench-verified 2026-08-03: command to an ACTIVE
// node = CAN id DOCK_ID-0x10, 1 byte (0x01 latch / 0x00 unlatch / 0x02
// state query); query to a PASSIVE node = DOCK_ID-0x20, any byte; replies
// arrive on DOCK_ID — a 1-byte verb echo, a 1-byte sensor bit field, or the
// 4-byte [0x02, state, pos_lo, pos_hi] state reply. All frames FD+BRS,
// standard ids.
//
//   PC -> "LATCH F2\n" / "UNLATCH F2\n"  <- "OK" | "ERR timeout"
//      -> "STATE F2\n"   <- "STATE LATCHED 77.5"   (deg = pos * 270/1024;
//                            "-" when pos reads 0xFFFF) | "ERR timeout"
//      -> "SENSE F3\n"   <- "SENSE 08" (hex bit field) | "SENSE NONE"
//      -> "SCAN\n"       <- "F2 ACTIVE LATCHED 77.5" / "F3 PASSIVE 08" ...
//                            then "OK <count>"
//
// DUMB PIPE by design (decision fixed 2026-08-02): no debounce, no clocking
// math, and NO deadman — the latch changes state only on explicit verbs; on
// client loss this process does nothing. Dock policy lives PC-side.
// Single client, newest-wins: a new connection evicts the old one (between
// verbs — each verb is a short synchronous CAN round-trip).
//
// Shares can0 with the DM joint server via disjoint kernel filters: this
// socket accepts ONLY the dock response band 0xF0-0xFF (filter 0x0F0 mask
// 0x7F0), so motor replies and our own looped-back TX never reach it.
#include <arpa/inet.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace {

// Wire constants from Dock_Control/PROTOCOL.md (mirrors dock_bench.py).
constexpr unsigned CMD_UNLATCH = 0x00;
constexpr unsigned CMD_LATCH = 0x01;
constexpr unsigned CMD_STATE_QUERY = 0x02;
constexpr unsigned ACTIVE_CMD_OFFSET = 0x10;    // command id = DOCK_ID - 0x10
constexpr unsigned PASSIVE_QUERY_OFFSET = 0x20; // query id = DOCK_ID - 0x20
constexpr unsigned DOCK_ID_LO = 0xF0, DOCK_ID_HI = 0xFF;
constexpr int REPLY_WAIT_MS = 50;  // per-query deadline (read ~3 ms + slack)

const char* STATE_NAMES[4] = {"UNKNOWN", "LATCHED", "UNLATCHED", "READ_ERROR"};

int g_can = -1;

double mono_s() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

int open_can(const char* iface) {
  const int fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (fd < 0) {
    std::perror("[dock] socket(PF_CAN)");
    return -1;
  }
  const int on = 1;
  if (setsockopt(fd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &on, sizeof on) != 0) {
    std::perror("[dock] CAN_RAW_FD_FRAMES");
    close(fd);
    return -1;
  }
  // Response band only — same filter the bench used.
  can_filter flt{0x0F0, 0x7F0};
  if (setsockopt(fd, SOL_CAN_RAW, CAN_RAW_FILTER, &flt, sizeof flt) != 0) {
    std::perror("[dock] CAN_RAW_FILTER");
    close(fd);
    return -1;
  }
  ifreq ifr{};
  std::snprintf(ifr.ifr_name, IFNAMSIZ, "%s", iface);
  if (ioctl(fd, SIOCGIFINDEX, &ifr) != 0) {
    std::fprintf(stderr,
                 "[dock] no CAN interface '%s' (%s) — is the link up?  "
                 "sudo ip link set %s up type can bitrate 1000000 "
                 "dbitrate 4000000 fd on\n",
                 iface, std::strerror(errno), iface);
    close(fd);
    return -1;
  }
  sockaddr_can addr{};
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
    std::perror("[dock] bind(can)");
    close(fd);
    return -1;
  }
  return fd;
}

bool can_send1(unsigned can_id, uint8_t byte) {
  canfd_frame f{};
  f.can_id = can_id;
  f.len = 1;
  f.flags = CANFD_FDF | CANFD_BRS;
  f.data[0] = byte;
  return write(g_can, &f, CANFD_MTU) == ssize_t(CANFD_MTU);
}

// Drop queued frames (stale replies from a timed-out verb must not satisfy
// the next one).
void can_drain() {
  canfd_frame f;
  while (recv(g_can, &f, sizeof f, MSG_DONTWAIT) > 0) {}
}

// Next frame from dock_id before the ABSOLUTE deadline, or -1. Payload into
// out (>= 64 bytes), returns its length. Non-matching frames are skipped
// without extending the window.
int can_await(unsigned dock_id, uint8_t* out, double deadline) {
  for (;;) {
    const int left = int((deadline - mono_s()) * 1e3);
    if (left <= 0) return -1;
    pollfd p{g_can, POLLIN, 0};
    if (poll(&p, 1, left) <= 0) return -1;
    canfd_frame f;
    const ssize_t got = recv(g_can, &f, sizeof f, MSG_DONTWAIT);
    if (got <= 0) continue;
    if (f.can_id != dock_id) continue;
    std::memcpy(out, f.data, f.len);
    return f.len;
  }
}

// Latch/unlatch: send the verb, wait for the 1-byte echo.
bool dock_command(unsigned dock_id, unsigned verb) {
  can_drain();
  if (!can_send1(dock_id - ACTIVE_CMD_OFFSET, uint8_t(verb))) return false;
  const double deadline = mono_s() + REPLY_WAIT_MS * 1e-3;
  uint8_t buf[64];
  for (;;) {
    const int len = can_await(dock_id, buf, deadline);
    if (len < 0) return false;
    if (len == 1 && buf[0] == verb) return true;
  }
}

// 0x02 state query -> state index + raw position (0xFFFF = read error).
bool dock_state(unsigned dock_id, unsigned* state, unsigned* pos) {
  can_drain();
  if (!can_send1(dock_id - ACTIVE_CMD_OFFSET, CMD_STATE_QUERY)) return false;
  const double deadline = mono_s() + REPLY_WAIT_MS * 1e-3;
  uint8_t buf[64];
  for (;;) {
    const int len = can_await(dock_id, buf, deadline);
    if (len < 0) return false;
    if (len >= 4 && buf[0] == CMD_STATE_QUERY) {
      *state = buf[1];
      *pos = unsigned(buf[2]) | (unsigned(buf[3]) << 8);
      return true;
    }
  }
}

// Passive query -> sensor bit field, -1 on timeout.
int dock_sense(unsigned dock_id) {
  can_drain();
  if (!can_send1(dock_id - PASSIVE_QUERY_OFFSET, 0x00)) return -1;
  uint8_t buf[64];
  const int len = can_await(dock_id, buf, mono_s() + REPLY_WAIT_MS * 1e-3);
  return len >= 1 ? buf[0] : -1;
}

// "STATE LATCHED 77.5" tail: name + degrees (pos * 270/1024, "-" on 0xFFFF).
void format_state(unsigned state, unsigned pos, char* out, size_t out_len) {
  char deg[16] = "-";
  if (pos != 0xFFFF)
    std::snprintf(deg, sizeof deg, "%.1f", double(pos) * 270.0 / 1024.0);
  if (state < 4)
    std::snprintf(out, out_len, "%s %s", STATE_NAMES[state], deg);
  else
    std::snprintf(out, out_len, "?%u %s", state, deg);
}

bool sendline(int fd, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));
bool sendline(int fd, const char* fmt, ...) {
  char out[128];
  va_list ap;
  va_start(ap, fmt);
  const int n = std::vsnprintf(out, sizeof out - 1, fmt, ap);
  va_end(ap);
  out[n] = '\n';
  return send(fd, out, size_t(n) + 1, MSG_NOSIGNAL) == ssize_t(n) + 1;
}

// One verb line -> one (or more, for SCAN) reply lines. False = client gone.
bool handle_line(int fd, const char* line) {
  unsigned id = 0;
  if (std::sscanf(line, "LATCH %x", &id) == 1 ||
      std::sscanf(line, "UNLATCH %x", &id) == 1) {
    if (id < DOCK_ID_LO || id > DOCK_ID_HI) return sendline(fd, "ERR badid");
    const unsigned verb = line[0] == 'L' ? CMD_LATCH : CMD_UNLATCH;
    std::printf("[dock] %s 0x%02X\n", verb == CMD_LATCH ? "latch" : "unlatch",
                id);
    return dock_command(id, verb) ? sendline(fd, "OK")
                                  : sendline(fd, "ERR timeout");
  }
  if (std::sscanf(line, "STATE %x", &id) == 1) {
    if (id < DOCK_ID_LO || id > DOCK_ID_HI) return sendline(fd, "ERR badid");
    unsigned state, pos;
    if (!dock_state(id, &state, &pos)) return sendline(fd, "ERR timeout");
    char tail[32];
    format_state(state, pos, tail, sizeof tail);
    return sendline(fd, "STATE %s", tail);
  }
  if (std::sscanf(line, "SENSE %x", &id) == 1) {
    if (id < DOCK_ID_LO || id > DOCK_ID_HI) return sendline(fd, "ERR badid");
    const int bits = dock_sense(id);
    return bits < 0 ? sendline(fd, "SENSE NONE")
                    : sendline(fd, "SENSE %02X", unsigned(bits));
  }
  if (!std::strcmp(line, "SCAN")) {
    // Read-only probe of every dock id (state query, then passive query) —
    // dock_bench cmd_scan. Worst case ~16 x 2 x 50 ms.
    int found = 0;
    for (unsigned did = DOCK_ID_LO; did <= DOCK_ID_HI; ++did) {
      unsigned state, pos;
      if (dock_state(did, &state, &pos)) {
        char tail[32];
        format_state(state, pos, tail, sizeof tail);
        if (!sendline(fd, "%02X ACTIVE %s", did, tail)) return false;
        ++found;
        continue;
      }
      const int bits = dock_sense(did);
      if (bits >= 0) {
        if (!sendline(fd, "%02X PASSIVE %02X", did, unsigned(bits)))
          return false;
        ++found;
      }
    }
    std::printf("[dock] scan: %d node(s)\n", found);
    return sendline(fd, "OK %d", found);
  }
  if (line[0] == '\0') return true;  // ignore blank lines
  return sendline(fd, "ERR badcmd");
}

} // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // journald must not sit in a 4K buffer
  const char* iface = argc > 1 ? argv[1] : "can0";
  const long port_arg = argc > 2 ? std::atol(argv[2]) : 47803;
  if (port_arg < 1 || port_arg > 65535) {
    std::fprintf(stderr, "[dock] port must be 1..65535\n");
    return 2;
  }
  // Optional bind address, hand_bridge convention: the box is dual-NIC and
  // INADDR_ANY would answer on the robot LAN too.
  in_addr bind_ip{};
  bind_ip.s_addr = INADDR_ANY;
  if (argc > 3 && inet_pton(AF_INET, argv[3], &bind_ip) != 1) {
    std::fprintf(stderr, "[dock] bind arg must be a dotted IPv4 address\n");
    return 2;
  }

  g_can = open_can(iface);
  if (g_can < 0) return 1;

  const int lfd = socket(AF_INET, SOCK_STREAM, 0);
  const int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr = bind_ip;
  addr.sin_port = htons(uint16_t(port_arg));
  if (bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 ||
      listen(lfd, 1) != 0) {
    std::perror("[dock] bind/listen");
    return 1;
  }
  std::printf("[dock] dock_bridge up: %s, port %ld\n", iface, port_arg);

  int cfd = -1;
  char buf[256];
  size_t have = 0;
  for (;;) {
    pollfd pfds[2] = {{lfd, POLLIN, 0}, {cfd, POLLIN, 0}};
    if (poll(pfds, cfd >= 0 ? 2 : 1, -1) < 0) continue;
    if (pfds[0].revents & POLLIN) {
      const int fresh = accept(lfd, nullptr, nullptr);
      if (fresh >= 0) {
        if (cfd >= 0) {
          std::printf("[dock] client evicted by new connection\n");
          close(cfd);
        }
        const int nd = 1;
        setsockopt(fresh, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof nd);
        cfd = fresh;
        have = 0;
        std::printf("[dock] client connected\n");
      }
    }
    if (cfd >= 0 && (pfds[1].revents & (POLLIN | POLLHUP | POLLERR))) {
      const ssize_t got = recv(cfd, buf + have, sizeof buf - have - 1, 0);
      if (got <= 0) {
        close(cfd);
        cfd = -1;
        std::printf("[dock] client disconnected\n");
        continue;
      }
      have += size_t(got);
      buf[have] = '\0';
      char* line = buf;
      bool alive = true;
      for (char* nl; alive && (nl = std::strchr(line, '\n')) != nullptr;
           line = nl + 1) {
        *nl = '\0';
        if (nl > line && nl[-1] == '\r') nl[-1] = '\0';  // tolerate CRLF (nc)
        alive = handle_line(cfd, line);
      }
      if (!alive) {
        close(cfd);
        cfd = -1;
        std::printf("[dock] client disconnected\n");
        continue;
      }
      have = std::strlen(line);
      std::memmove(buf, line, have);
      if (have == sizeof buf - 1) {
        std::printf("[dock] oversized line — resyncing\n");
        have = 0;
      }
    }
  }
}
