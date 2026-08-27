// DM-motor FDCAN backend over SocketCAN — the modular BASE's joints.
//
// Topology (fixed 2026-08-03): the RT box owns the dmcan USB2FDCAN, which
// enumerates as candleLight/gs_usb -> kernel `can0`; this backend runs in a
// SECOND arm_rt_server instance (`--backend dm`, ports 47810/47811) sharing
// that one bus with dock_bridge via DISJOINT kernel CAN filters (motors
// reply on their Master ID, docks on 0xF0-0xFF). Motors run MIT mode as
// PURE TORQUE devices: every control frame carries kp=kd=0, q_des=v_des=0 —
// the on-motor PD is bypassed and rt_loop's servo law (which already
// computed PD + ff) is the single authority. Gains ride the UDP
// CommandPacket, never the CAN frame.
//
// Ctor spec (one string, via --dm-spec / --can-if):
//   "can0;1:4340,2:4340"        iface ; comma list of id:type[:mst]
// id   = motor CAN id, 1..15 (replies carry it in the payload LOW nibble),
// type = 4310 | 4310p | 4340 | 4340p (per-type encode limits below, ported
//        verbatim from arm_control/hardware/dm_backend.py),
// mst  = the CAN id the motor's firmware REPLIES on (Master ID). Default 0 —
//        the DM factory default ("反馈帧ID…默认为0", DM-J4310 manual) and
//        this project's motors (hardware.yaml: master_id null throughout).
//        Replies are routed by the payload nibble exactly like the Python
//        backend, so a shared mst collides nowhere; a motor flashed with a
//        different Master ID is a spec edit, not a code change.
//
// Pacing: tick 2 ms (500 Hz). read() drains feedback while sleeping toward
// an ABSOLUTE deadline (fake-backend discipline, ppoll instead of a blind
// clock_nanosleep so RX is decoded as it lands), then faults if any motor
// has been silent > 50 ms, naming it. DM motors only speak when spoken to,
// so a read with no write() since the previous read sends a zero-torque MIT
// frame per motor (the Python listen_step) — this keeps q streaming while
// DISARMED. Bench evidence says DISABLED motors ack those too
// (rt_handguide.py:130: safe_stop "verified: ack + state stream"); if some
// firmware doesn't, the disarmed silence fault will say so loudly.
//
// Enable: DM enable frame per motor on the FIRST read(), and lazily on the
// first write() after a stop() (franka's lazy-session pattern). stop()
// sends zero-torque + disable per motor — the Python
// _zero_torque_and_disable sequence — idempotent, never throws. tau_ref
// self-echoes the last written (clamped) torque: fake-backend semantics,
// DM has no accepted-torque echo.
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "arm_rt/backend.hpp"

namespace arm_rt {
namespace {

constexpr double TICK_S = 0.002;  // 500 Hz
constexpr int64_t TICK_NS = 2'000'000;
constexpr int64_t SILENCE_NS = 50'000'000;  // per-motor feedback deadman

constexpr uint8_t DM_ENABLE_FRAME[8] = {0xFF, 0xFF, 0xFF, 0xFF,
                                        0xFF, 0xFF, 0xFF, 0xFC};
constexpr uint8_t DM_DISABLE_FRAME[8] = {0xFF, 0xFF, 0xFF, 0xFF,
                                         0xFF, 0xFF, 0xFF, 0xFD};

// Per-motor-TYPE MIT encode limits — protocol constants, ported EXACTLY from
// dm_backend.py DEFAULT_LIMITS_BY_TYPE (same precedent as FR3_TAU_LIMIT).
struct DmLimits {
  double p_lo, p_hi, v_lo, v_hi, kp_lo, kp_hi, kd_lo, kd_hi, t_lo, t_hi;
};
constexpr DmLimits LIM_4310 = {-12.5, 12.5, -30, 30, 0, 500, 0, 5, -10, 10};
constexpr DmLimits LIM_4310P = {-12.5, 12.5, -50, 50, 0, 500, 0, 5, -10, 10};
constexpr DmLimits LIM_4340 = {-12.5, 12.5, -10, 10, 0, 500, 0, 5, -28, 28};
// 4340P: same electrical specs as 4340 (cross-roller bearing variant)

bool limits_for(const std::string& type, DmLimits& out) {
  if (type == "4310") out = LIM_4310;
  else if (type == "4310p") out = LIM_4310P;
  else if (type == "4340" || type == "4340p") out = LIM_4340;
  else return false;
  return true;
}

// float<->uint scaling, bit-for-bit with the Python _float_to_uint /
// _uint_to_float (same operand order, same truncation toward zero).
uint32_t f2u(double v, double lo, double hi, int bits) {
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return uint32_t((v - lo) * double((1u << bits) - 1) / (hi - lo));
}
double u2f(uint32_t raw, double lo, double hi, int bits) {
  return double(raw) * (hi - lo) / double((1u << bits) - 1) + lo;
}

// MIT control frame: pos16 | vel12 | kp12 | kd12 | tau12 (dm_backend.py
// pack_mit_control_frame). No gain validation here: the RT path never
// throws, and this backend hardwires kp=kd=0 — the clip in f2u is the
// backstop, rt_loop's clamp against tau_limit() the real ceiling.
void pack_mit(double pos, double vel, double kp, double kd, double tau,
              const DmLimits& L, uint8_t out[8]) {
  const uint32_t p = f2u(pos, L.p_lo, L.p_hi, 16);
  const uint32_t v = f2u(vel, L.v_lo, L.v_hi, 12);
  const uint32_t s = f2u(kp, L.kp_lo, L.kp_hi, 12);
  const uint32_t d = f2u(kd, L.kd_lo, L.kd_hi, 12);
  const uint32_t t = f2u(tau, L.t_lo, L.t_hi, 12);
  out[0] = uint8_t((p >> 8) & 0xFF);
  out[1] = uint8_t(p & 0xFF);
  out[2] = uint8_t((v >> 4) & 0xFF);
  out[3] = uint8_t(((v & 0x0F) << 4) | ((s >> 8) & 0x0F));
  out[4] = uint8_t(s & 0xFF);
  out[5] = uint8_t((d >> 4) & 0xFF);
  out[6] = uint8_t(((d & 0x0F) << 4) | ((t >> 8) & 0x0F));
  out[7] = uint8_t(t & 0xFF);
}

// MIT reply: [err<<4|id, pos16, vel12, tau12, T_mos, T_rotor]
// (dm_backend.py decode_mit_reply).
struct MitReply {
  int id, err;
  double q, dq, tau, t_mos, t_rotor;
};
bool decode_mit(const uint8_t* d, int len, const DmLimits& L, MitReply& r) {
  if (len < 8) return false;
  r.id = d[0] & 0x0F;
  r.err = (d[0] >> 4) & 0x0F;
  const uint32_t pos_u = uint32_t(d[1] << 8) | d[2];
  const uint32_t vel_u = uint32_t(d[3] << 4) | uint32_t(d[4] >> 4);
  const uint32_t tau_u = (uint32_t(d[4] & 0x0F) << 8) | d[5];
  r.q = u2f(pos_u, L.p_lo, L.p_hi, 16);
  r.dq = u2f(vel_u, L.v_lo, L.v_hi, 12);
  r.tau = u2f(tau_u, L.t_lo, L.t_hi, 12);
  r.t_mos = double(d[6]);   // surfaced by the Python motor_health(); no
  r.t_rotor = double(d[7]); // PlantState slot — dropped here for now
  return true;
}

struct Motor {
  int id = 0;
  int mst = 0;  // reply CAN id (Master ID)
  std::string type;
  DmLimits lim{};
  double q = 0, dq = 0, tau = 0;
  uint64_t last_rx_ns = 0;
  bool seen = false;
};

uint64_t ts_ns(const timespec& t) {
  return uint64_t(t.tv_sec) * 1'000'000'000ull + uint64_t(t.tv_nsec);
}
void ts_add(timespec& t, int64_t ns) {
  t.tv_nsec += ns;
  while (t.tv_nsec >= 1'000'000'000) {
    t.tv_nsec -= 1'000'000'000;
    t.tv_sec += 1;
  }
}

// "iface;id:type[:mst],..." -> iface + motors. Errors to `err`.
bool parse_spec(const std::string& spec, std::string& iface,
                std::vector<Motor>& motors, std::string& err) {
  const size_t semi = spec.find(';');
  iface = spec.substr(0, semi);
  if (iface.empty() || iface.size() >= IFNAMSIZ) {
    err = "bad interface name in spec";
    return false;
  }
  if (semi == std::string::npos || semi + 1 >= spec.size()) {
    err = "no motors in spec — want \"can0;1:4340,2:4340\" (id:type[:mst])";
    return false;
  }
  std::string rest = spec.substr(semi + 1);
  for (size_t pos = 0; pos < rest.size();) {
    size_t comma = rest.find(',', pos);
    if (comma == std::string::npos) comma = rest.size();
    const std::string item = rest.substr(pos, comma - pos);
    pos = comma + 1;
    Motor m;
    char type_buf[16] = {};
    int mst = 0;
    const int fields = std::sscanf(item.c_str(), "%i:%15[^:]:%i", &m.id,
                                   type_buf, &mst);
    if (fields < 2) {
      err = "bad motor entry '" + item + "' — want id:type[:mst]";
      return false;
    }
    m.type = type_buf;
    m.mst = fields >= 3 ? mst : 0x00;  // DM factory default Master ID
    if (m.id < 1 || m.id > 15) {
      err = "motor id must be 1..15 (payload-nibble routed): '" + item + "'";
      return false;
    }
    if (!limits_for(m.type, m.lim)) {
      err = "unknown motor type '" + m.type + "' (4310|4310p|4340|4340p)";
      return false;
    }
    for (const Motor& o : motors)
      if (o.id == m.id) {
        err = "duplicate motor id in spec: '" + item + "'";
        return false;
      }
    motors.push_back(m);
  }
  if (motors.empty() || int(motors.size()) > MAX_JOINTS) {
    err = "need 1..16 motors in spec";
    return false;
  }
  return true;
}

class DmBackend final : public Backend {
public:
  DmBackend(const std::string& iface, std::vector<Motor> motors)
      : iface_(iface), motors_(std::move(motors)) {
    fault_.reserve(256);  // latch path must not allocate on the RT thread
    for (size_t j = 0; j < motors_.size(); ++j)
      tau_limit_[j] = motors_[j].lim.t_hi;

    fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (fd_ < 0) {
      fail("socket(PF_CAN): %s", std::strerror(errno));
      return;
    }
    const int on = 1;
    if (setsockopt(fd_, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &on, sizeof on) != 0) {
      fail("CAN_RAW_FD_FRAMES: %s (kernel/driver without CAN FD?)",
           std::strerror(errno));
      return;
    }
    // Only the motors' reply ids reach this socket — dock traffic
    // (0xD0-0xFF bands) and dock_bridge's own TX loopback never do.
    std::vector<can_filter> filters;
    for (const Motor& m : motors_) {
      bool dup = false;
      for (const can_filter& f : filters) dup |= f.can_id == canid_t(m.mst);
      if (!dup) filters.push_back({canid_t(m.mst), CAN_SFF_MASK});
    }
    if (setsockopt(fd_, SOL_CAN_RAW, CAN_RAW_FILTER, filters.data(),
                   socklen_t(filters.size() * sizeof(can_filter))) != 0) {
      fail("CAN_RAW_FILTER: %s", std::strerror(errno));
      return;
    }
    ifreq ifr{};
    std::snprintf(ifr.ifr_name, IFNAMSIZ, "%s", iface_.c_str());
    if (ioctl(fd_, SIOCGIFINDEX, &ifr) != 0) {
      fail("no CAN interface '%s': %s (ip link set %s up type can "
           "bitrate 1000000 dbitrate 4000000 fd on)",
           iface_.c_str(), std::strerror(errno), iface_.c_str());
      return;
    }
    sockaddr_can addr{};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      fail("bind(%s): %s", iface_.c_str(), std::strerror(errno));
      return;
    }
    clock_gettime(CLOCK_MONOTONIC, &next_);
    ok_ = true;
  }

  ~DmBackend() override {
    if (fd_ >= 0) ::close(fd_);
  }

  bool ok() const { return ok_; }
  const char* name() const override { return "dm"; }
  int n() const override { return int(motors_.size()); }
  double tick_s() const override { return TICK_S; }
  const double* tau_limit() const override { return tau_limit_; }

  bool read(PlantState& out) override {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    ts_add(next_, TICK_NS);
    if (ts_ns(next_) < ts_ns(now)) next_ = now;  // ponytail: resnap after a
    // stall (rt_loop paces failed reads at 100 ms) — banked ticks would
    // replay as a zero-sleep elicit burst and flood the bus.

    if (!started_) {
      started_ = true;
      if (!enable_all()) return false;
      const uint64_t t0 = ts_ns(now);
      for (Motor& m : motors_) m.last_rx_ns = t0;  // silence grace from here
    }
    if (!wrote_ && !elicit_all()) return false;  // keepalive: solicit replies
    wrote_ = false;

    // Collect feedback while sleeping toward the absolute deadline.
    for (;;) {
      if (!drain()) return false;
      clock_gettime(CLOCK_MONOTONIC, &now);
      const uint64_t now_ns = ts_ns(now), due_ns = ts_ns(next_);
      if (now_ns >= due_ns) break;
      timespec left{};
      left.tv_sec = time_t((due_ns - now_ns) / 1'000'000'000ull);
      left.tv_nsec = long((due_ns - now_ns) % 1'000'000'000ull);
      pollfd p{fd_, POLLIN, 0};
      ppoll(&p, 1, &left, nullptr);  // wakes on RX or deadline; EINTR fine
    }

    // Startup only: never publish zeros as a pose — block past the deadline
    // (bounded by the silence budget) until every motor has spoken once.
    while (!all_seen_) {
      all_seen_ = true;
      const Motor* quiet = nullptr;
      for (const Motor& m : motors_)
        if (!m.seen) {
          all_seen_ = false;
          quiet = &m;
        }
      if (all_seen_) break;
      clock_gettime(CLOCK_MONOTONIC, &now);
      if (int64_t(ts_ns(now) - quiet->last_rx_ns) > SILENCE_NS) {
        fail("motor %d (%s) never replied on %s (Master ID 0x%02X — set "
             "id:type:mst in --dm-spec if the firmware differs)",
             quiet->id, quiet->type.c_str(), iface_.c_str(), quiet->mst);
        return false;
      }
      timespec brief{0, 1'000'000};
      pollfd p{fd_, POLLIN, 0};
      ppoll(&p, 1, &brief, nullptr);
      if (!drain()) return false;
    }

    const uint64_t now_ns = ts_ns(now);
    for (const Motor& m : motors_)
      if (int64_t(now_ns - m.last_rx_ns) > SILENCE_NS) {
        fail("motor %d (%s) silent %lld ms on %s", m.id, m.type.c_str(),
             (long long)((now_ns - m.last_rx_ns) / 1'000'000ull),
             iface_.c_str());
        return false;
      }

    out.n = int(motors_.size());
    for (size_t j = 0; j < motors_.size(); ++j) {
      out.q[j] = motors_[j].q;
      out.dq[j] = motors_[j].dq;
      out.tau[j] = motors_[j].tau;
      out.tau_ref[j] = last_tau_[j];  // self-echo (fake semantics)
    }
    // No EE geometry here (the base's kinematics live PC-side): explicit
    // zeros + false flags, same decision the fake backend documents.
    for (int i = 0; i < 6; ++i) out.wrench[i] = 0.0;
    out.wrench_valid = false;
    out.jacobian_valid = false;
    return true;
  }

  bool write(const double* tau, int n) override {
    if (!enabled_ && !enable_all()) return false;  // lazy re-arm after stop()
    uint8_t buf[8];
    for (int j = 0; j < n && j < int(motors_.size()); ++j) {
      const DmLimits& L = motors_[j].lim;
      double t = tau[j];
      if (t < L.t_lo) t = L.t_lo;
      if (t > L.t_hi) t = L.t_hi;
      pack_mit(0, 0, 0, 0, t, L, buf);  // pure torque: motor PD bypassed
      if (!send8(motors_[j].id, buf)) return false;
      last_tau_[j] = t;
    }
    wrote_ = true;
    return true;
  }

  void stop() override {
    // Zero-gain hold then DISABLE per motor — the Python
    // _zero_torque_and_disable sequence. Errors ignored: never throws, and
    // the disarmed elicit keeps state flowing afterwards.
    if (fd_ < 0) return;
    uint8_t zt[8];
    for (const Motor& m : motors_) {
      pack_mit(0, 0, 0, 0, 0, m.lim, zt);
      send8(m.id, zt);
      send8(m.id, DM_DISABLE_FRAME);
    }
    for (size_t j = 0; j < motors_.size(); ++j) last_tau_[j] = 0.0;
    enabled_ = false;
  }

  const std::string& fault_text() const override { return fault_; }

private:
  bool send8(int can_id, const uint8_t* payload) {
    canfd_frame f{};
    f.can_id = canid_t(can_id);
    f.len = 8;
    f.flags = CANFD_FDF | CANFD_BRS;  // FD + bit-rate switch, like the bench
    std::memcpy(f.data, payload, 8);
    const ssize_t sent = ::write(fd_, &f, CANFD_MTU);
    if (sent != ssize_t(CANFD_MTU)) {
      fail("can send id 0x%02X on %s: %s", can_id, iface_.c_str(),
           std::strerror(errno));
      return false;
    }
    return true;
  }

  bool enable_all() {
    for (const Motor& m : motors_)
      if (!send8(m.id, DM_ENABLE_FRAME)) return false;
    enabled_ = true;
    return true;
  }

  bool elicit_all() {
    uint8_t buf[8];
    for (const Motor& m : motors_) {
      pack_mit(0, 0, 0, 0, 0, m.lim, buf);  // kp=kd=0: reply, no authority
      if (!send8(m.id, buf)) return false;
    }
    return true;
  }

  // Decode everything queued; route by the payload LOW NIBBLE (Python
  // semantics — the reply CAN id is the shared Master ID, not the motor).
  bool drain() {
    for (;;) {
      canfd_frame f;
      const ssize_t got = ::recv(fd_, &f, sizeof f, MSG_DONTWAIT);
      if (got < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
        if (errno == EINTR) continue;
        fail("can recv on %s: %s", iface_.c_str(), std::strerror(errno));
        return false;
      }
      if (got != ssize_t(CAN_MTU) && got != ssize_t(CANFD_MTU)) continue;
      if (f.can_id & CAN_ERR_FLAG) continue;
      if (f.len < 8) continue;
      const int nib = f.data[0] & 0x0F;
      for (Motor& m : motors_) {
        if (m.id != nib) continue;
        MitReply r;
        if (!decode_mit(f.data, f.len, m.lim, r)) break;
        m.q = r.q;
        m.dq = r.dq;
        m.tau = r.tau;
        timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        m.last_rx_ns = ts_ns(now);
        m.seen = true;
        break;
      }
    }
  }

  void fail(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
    char buf[224];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fault_.assign("dm: ").append(buf);
  }

  std::string iface_;
  std::vector<Motor> motors_;
  double tau_limit_[MAX_JOINTS] = {};
  double last_tau_[MAX_JOINTS] = {};
  int fd_ = -1;
  bool ok_ = false;
  bool started_ = false;   // first read() done (enable + first elicit)
  bool enabled_ = false;   // motors enabled since the last stop()
  bool wrote_ = false;     // a write() happened since the last read()
  bool all_seen_ = false;  // every motor has replied at least once
  timespec next_ = {};
  std::string fault_;
};

} // namespace

std::unique_ptr<Backend> make_dm_backend(const std::string& spec) {
  std::string iface, err;
  std::vector<Motor> motors;
  if (!parse_spec(spec, iface, motors, err)) {
    std::fprintf(stderr, "[rt] dm backend: %s (spec '%s')\n", err.c_str(),
                 spec.c_str());
    return nullptr;
  }
  auto backend = std::make_unique<DmBackend>(iface, std::move(motors));
  if (!backend->ok()) {
    // fault_text already carries the "dm: " prefix (it feeds rt_loop's latch).
    std::fprintf(stderr, "[rt] %s\n", backend->fault_text().c_str());
    return nullptr;
  }
  return backend;
}

// --mit-check: golden MIT frames + decodes for fixed inputs. The Python
// mirror (pack_mit_control_frame / decode_mit_reply through the SAME inputs)
// must print byte-identical lines — the dm equivalent of protocol_selfcheck.
int dm_mit_selfcheck() {
  struct PackCase {
    const char* type;
    double p, v, kp, kd, t;
  };
  const PackCase packs[] = {
      {"4340", 1.234, -2.5, 0, 0, 13.579},
      {"4340", 0, 0, 0, 0, 0},
      {"4340", -12.5, -10, 0, 0, -28},
      {"4340", 12.5, 10, 500, 5, 28},
      {"4310", -3.3, 7.77, 123.4, 2.71, -9.87},
      {"4310p", 5, 41, 250, 1, 3.3},
  };
  struct DecCase {
    const char* type;
    uint8_t d[8];
  };
  const DecCase decs[] = {
      {"4340", {0x01, 0x80, 0x00, 0x80, 0x08, 0x00, 0x19, 0x22}},
      {"4340", {0xA2, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00}},
      {"4310", {0x01, 0x12, 0x34, 0x56, 0x78, 0x9A, 0x28, 0x3C}},
      {"4310p", {0x03, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32}},
  };
  for (const PackCase& c : packs) {
    DmLimits L;
    limits_for(c.type, L);
    uint8_t out[8];
    pack_mit(c.p, c.v, c.kp, c.kd, c.t, L, out);
    std::printf("PACK %s %.9g %.9g %.9g %.9g %.9g ", c.type, c.p, c.v, c.kp,
                c.kd, c.t);
    for (int i = 0; i < 8; ++i) std::printf("%02x", out[i]);
    std::printf("\n");
  }
  for (const DecCase& c : decs) {
    DmLimits L;
    limits_for(c.type, L);
    MitReply r;
    decode_mit(c.d, 8, L, r);
    std::printf("DEC %s ", c.type);
    for (int i = 0; i < 8; ++i) std::printf("%02x", c.d[i]);
    std::printf(" %d %d %.9g %.9g %.9g %.9g %.9g\n", r.id, r.err, r.q, r.dq,
                r.tau, r.t_mos, r.t_rotor);
  }
  return 0;
}

} // namespace arm_rt
