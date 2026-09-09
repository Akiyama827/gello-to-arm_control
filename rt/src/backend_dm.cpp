// DM-motor FDCAN backend over SocketCAN.
//
// Each backend instance owns a SocketCAN socket. Exact Master-ID filters
// isolate motor replies from other devices sharing the bus. Motors run MIT mode as
// PURE TORQUE devices: every control frame carries kp=kd=0, q_des=v_des=0 —
// the on-motor PD is bypassed and rt_loop's servo law (which already
// computed PD + ff) is the single authority. Gains ride the UDP
// CommandPacket, never the CAN frame.
//
// Ctor spec (one string, via --dm-spec / --can-if):
//   "can0;1:4340,2:4340"    iface ; list of id:type[:mst[:pmax:vmax:tmax]]
// id   = motor CAN id, 1..15 (replies carry it in the payload LOW nibble),
// type = 4310 | 4310p | 4340 | 4340p (per-type encode limits below, ported
//        verbatim from arm_control/plants/dm/backend.py),
// mst  = the CAN id the motor's firmware REPLIES on (Master ID). Default 0 —
//        the DM factory default ("反馈帧ID…默认为0", DM-J4310 manual).
//        Replies are routed by the payload nibble exactly like the Python
//        backend, so a shared mst collides nowhere; a motor flashed with a
//        different Master ID is a spec edit, not a code change.
// pmax/vmax/tmax = optional symmetric MIT half-ranges (rad, rad/s, N.m).
//        Supply all three and mst together; omission keeps the type defaults.
//        Host mappings must match firmware; this does not program registers.
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

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
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

// Default per-type MIT limits, matching dm_backend.py DEFAULT_LIMITS_BY_TYPE.
// A per-motor spec override changes the host's encoding and feedback decoding.
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
// backstop. Explicit caps additionally constrain the final torque code below.
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
  double tau_cap = 0;  // zero keeps the legacy wire encoding
  uint32_t tau_code_min = 0, tau_code_max = 4095;
  double q = 0, dq = 0, tau = 0;
  uint64_t last_rx_ns = 0;
  bool seen = false;
};

// Run before constructing the backend: an impossible cap must never open CAN.
bool set_torque_cap(Motor& m, double tau_max, std::string& err) {
  if (tau_max == 0) return true;
  if (!std::isfinite(tau_max) || tau_max <= 0) {
    err = "torque cap must be finite and positive";
    return false;
  }
  m.tau_cap = std::min(tau_max, m.lim.t_hi);
  const DmLimits& L = m.lim;
  m.tau_code_min = f2u(-m.tau_cap, L.t_lo, L.t_hi, 12);
  m.tau_code_max = f2u(m.tau_cap, L.t_lo, L.t_hi, 12);
  // Use the actual decoder at boundaries: floating-point inverse rounding
  // and the MIT encoder's truncation can otherwise exceed a negative cap.
  while (m.tau_code_min > 0 &&
         u2f(m.tau_code_min - 1, L.t_lo, L.t_hi, 12) >= -m.tau_cap)
    --m.tau_code_min;
  while (m.tau_code_min < 4095 &&
         u2f(m.tau_code_min, L.t_lo, L.t_hi, 12) < -m.tau_cap)
    ++m.tau_code_min;
  while (m.tau_code_max < 4095 &&
         u2f(m.tau_code_max + 1, L.t_lo, L.t_hi, 12) <= m.tau_cap)
    ++m.tau_code_max;
  while (m.tau_code_max > 0 &&
         u2f(m.tau_code_max, L.t_lo, L.t_hi, 12) > m.tau_cap)
    --m.tau_code_max;
  if (m.tau_code_min > m.tau_code_max) {
    err = "torque cap has no representable MIT code for motor " + m.type;
    return false;
  }
  return true;
}

void clamp_torque_field(uint8_t out[8], const Motor& m) {
  if (m.tau_cap == 0) return;
  const uint32_t raw = (uint32_t(out[6] & 0x0F) << 8) | out[7];
  const uint32_t t = std::clamp(raw, m.tau_code_min, m.tau_code_max);
  out[6] = uint8_t((out[6] & 0xF0) | (t >> 8));
  out[7] = uint8_t(t & 0xFF);
}

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

// "iface;id:type[:mst[:pmax:vmax:tmax]],..." -> iface + motors. Errors to `err`.
bool parse_spec(const std::string& spec, std::string& iface,
                std::vector<Motor>& motors, std::string& err) {
  const size_t semi = spec.find(';');
  iface = spec.substr(0, semi);
  if (iface.empty() || iface.size() >= IFNAMSIZ) {
    err = "bad interface name in spec";
    return false;
  }
  if (semi == std::string::npos || semi + 1 >= spec.size()) {
    err = "no motors in spec — want iface;id:type[:mst[:pmax:vmax:tmax]],...";
    return false;
  }
  std::string rest = spec.substr(semi + 1);
  auto parse_id = [](const std::string& text, int lo, int hi, int& out) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text.c_str(), &end, 0);
    if (errno || end == text.c_str() || end != text.c_str() + text.size() ||
        value < lo || value > hi)
      return false;
    out = int(value);
    return true;
  };
  for (size_t pos = 0; pos <= rest.size();) {
    size_t comma = rest.find(',', pos);
    if (comma == std::string::npos) comma = rest.size();
    const std::string item = rest.substr(pos, comma - pos);
    pos = comma + 1;
    std::vector<std::string> fields;
    for (size_t start = 0; start <= item.size();) {
      size_t colon = item.find(':', start);
      if (colon == std::string::npos) colon = item.size();
      fields.push_back(item.substr(start, colon - start));
      start = colon + 1;
    }
    Motor m;
    if (fields.size() != 2 && fields.size() != 3 && fields.size() != 6) {
      err = "bad motor entry '" + item + "' — want id:type[:mst[:pmax:vmax:tmax]]";
      return false;
    }
    m.type = fields[1];
    if (!parse_id(fields[0], 1, 15, m.id)) {
      err = "motor id must be 1..15 (payload-nibble routed): '" + item + "'";
      return false;
    }
    if (fields.size() >= 3 && !parse_id(fields[2], 0, 0x7ff, m.mst)) {
      err = "master id must be 0..0x7ff: '" + item + "'";
      return false;
    }
    if (!limits_for(m.type, m.lim)) {
      err = "unknown motor type '" + m.type + "' (4310|4310p|4340|4340p)";
      return false;
    }
    if (fields.size() == 6) {
      double ranges[3];
      for (int i = 0; i < 3; ++i) {
        const std::string& text = fields[i + 3];
        char* end = nullptr;
        errno = 0;
        ranges[i] = std::strtod(text.c_str(), &end);
        const double span = 2 * ranges[i];
        const double levels = i == 0 ? 65535 : 4095;
        if (errno || end == text.c_str() || end != text.c_str() + text.size() ||
            !std::isfinite(ranges[i]) || ranges[i] < std::numeric_limits<double>::min() ||
            !std::isfinite(span) || !std::isfinite(span * levels)) {
          err = "MIT half-ranges must be finite, positive, and representable: '" + item + "'";
          return false;
        }
      }
      m.lim.p_lo = -ranges[0]; m.lim.p_hi = ranges[0];
      m.lim.v_lo = -ranges[1]; m.lim.v_hi = ranges[1];
      m.lim.t_lo = -ranges[2]; m.lim.t_hi = ranges[2];
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
  DmBackend(const std::string& iface, std::vector<Motor> motors,
            uint32_t active_mask)
      : iface_(iface), motors_(std::move(motors)) {
    fault_.reserve(256);  // latch path must not allocate on the RT thread
    for (size_t j = 0; j < motors_.size(); ++j)
      tau_limit_[j] = motors_[j].tau_cap == 0 ? motors_[j].lim.t_hi
                                            : motors_[j].tau_cap;
    const uint32_t configured = motors_.size() >= 16
                                    ? 0xFFFFu
                                    : ((1u << motors_.size()) - 1u);
    active_mask_ = active_mask & configured;

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
    // Only the configured motors' reply ids reach this socket; unrelated
    // device traffic on a shared bus is excluded by exact-id filters.
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
  uint32_t online_mask() const override {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    const uint64_t stamp = ts_ns(now);
    uint32_t mask = 0;
    for (size_t j = 0; j < motors_.size(); ++j)
      if (motors_[j].seen && stamp - motors_[j].last_rx_ns <= uint64_t(SILENCE_NS))
        mask |= 1u << j;
    return mask;
  }
  uint32_t active_mask() const override { return active_mask_; }
  bool set_active_mask(uint32_t mask) override {
    const uint32_t configured = motors_.size() >= 16
                                    ? 0xFFFFu
                                    : ((1u << motors_.size()) - 1u);
    if (mask & ~configured || mask & ~online_mask()) return false;
    const uint32_t added = mask & ~active_mask_;
    const uint32_t removed = active_mask_ & ~mask;
    uint8_t zero[8];
    for (size_t j = 0; j < motors_.size(); ++j) {
      if (removed & (1u << j)) {
        pack_mit(0, 0, 0, 0, 0, motors_[j].lim, zero);
        if (!send8(motors_[j].id, zero) ||
            !send8(motors_[j].id, DM_DISABLE_FRAME))
          return false;
        last_tau_[j] = 0.0;
      } else if (enabled_ && (added & (1u << j))) {
        if (!send8(motors_[j].id, DM_ENABLE_FRAME)) return false;
      }
    }
    active_mask_ = mask;
    return true;
  }

  bool read(PlantState& out) override {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    ts_add(next_, TICK_NS);
    if (ts_ns(next_) < ts_ns(now)) next_ = now;  // ponytail: resnap after a
    // stall (rt_loop paces failed reads at 100 ms) — banked ticks would
    // replay as a zero-sleep elicit burst and flood the bus.

    if (!started_) {
      started_ = true;
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
      for (size_t j = 0; j < motors_.size(); ++j)
        if ((active_mask_ & (1u << j)) && !motors_[j].seen) {
          all_seen_ = false;
          quiet = &motors_[j];
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
    for (size_t j = 0; j < motors_.size(); ++j) {
      const Motor& m = motors_[j];
      if ((active_mask_ & (1u << j)) &&
          int64_t(now_ns - m.last_rx_ns) > SILENCE_NS) {
        fail("motor %d (%s) silent %lld ms on %s", m.id, m.type.c_str(),
             (long long)((now_ns - m.last_rx_ns) / 1'000'000ull),
             iface_.c_str());
        return false;
      }
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
      if (!(active_mask_ & (1u << j))) continue;
      const DmLimits& L = motors_[j].lim;
      double t = tau[j];
      if (t < L.t_lo) t = L.t_lo;
      if (t > L.t_hi) t = L.t_hi;
      pack_mit(0, 0, 0, 0, t, L, buf);  // pure torque: motor PD bypassed
      clamp_torque_field(buf, motors_[j]);
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
    for (size_t j = 0; j < motors_.size(); ++j)
      if ((active_mask_ & (1u << j)) &&
          !send8(motors_[j].id, DM_ENABLE_FRAME))
        return false;
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
  uint32_t active_mask_ = 0;
  timespec next_ = {};
  std::string fault_;
};

} // namespace

std::unique_ptr<Backend> make_dm_backend(const std::string& spec,
                                         uint32_t active_mask, double tau_max) {
  std::string iface, err;
  std::vector<Motor> motors;
  if (!parse_spec(spec, iface, motors, err)) {
    std::fprintf(stderr, "[rt] dm backend: %s (spec '%s')\n", err.c_str(),
                 spec.c_str());
    return nullptr;
  }
  for (Motor& m : motors) {
    if (!set_torque_cap(m, tau_max, err)) {
      std::fprintf(stderr, "[rt] dm backend: %s\n", err.c_str());
      return nullptr;
    }
  }
  auto backend =
      std::make_unique<DmBackend>(iface, std::move(motors), active_mask);
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
  // Explicit failures keep these checks active in release/NDEBUG builds.
  std::string iface, err;
  std::vector<Motor> mapped;
  if (!parse_spec("can0;1:4340:0x11:12.5:20:28", iface, mapped, err) ||
      mapped.size() != 1 || iface != "can0" || mapped[0].mst != 0x11) {
    std::fprintf(stderr, "MIT map check: explicit map rejected: %s\n", err.c_str());
    return 1;
  }
  if (mapped[0].lim.v_lo != -20 || mapped[0].lim.v_hi != 20) {
    std::fprintf(stderr, "MIT map check: explicit VMAX 20 parsed as %.17g\n",
                 mapped[0].lim.v_hi);
    return 1;
  }
  std::vector<Motor> mixed;
  if (!parse_spec("can0;1:4340,2:4340:0x12,0xf:4310p:0x7ff,03:4340p:00,"
                  "4:4310,5:4340:0:8:15:16,6:4310:0:6:7:8",
                  iface, mixed, err) || mixed.size() != 7 ||
      mixed[0].mst != 0 || mixed[0].lim.v_hi != 10 ||
      mixed[1].mst != 0x12 || mixed[1].lim.v_hi != 10 ||
      mixed[2].id != 15 || mixed[2].mst != 0x7ff || mixed[2].lim.v_hi != 50 ||
      mixed[3].id != 3 || mixed[3].mst != 0 || mixed[3].lim.v_hi != 10 ||
      mixed[4].lim.v_hi != 30 ||
      mixed[5].lim.p_hi != 8 || mixed[5].lim.v_hi != 15 || mixed[5].lim.t_hi != 16 ||
      mixed[6].lim.p_hi != 6 || mixed[6].lim.v_hi != 7 || mixed[6].lim.t_hi != 8) {
    std::fprintf(stderr, "MIT map check: legacy or independent maps changed: %s\n",
                 err.c_str());
    return 1;
  }
  for (const char* spec : {
      "", ";1:4340", "can0", "can0;", "can0;,1:4340", "can0;1:4340,",
      "can0;1:4340,,2:4340", "can0;1:4340,1:4310", "can0;1:4340,01:4310",
      "can0;1", "can0;:4340", "can0;1:", "can0;1:9999", "can0;1:4340:",
      "can0;1:4340:0:12.5", "can0;1:4340:0:12.5:20",
      "can0;1:4340::12.5:20:28", "can0;1:4340:0::20:28",
      "can0;1:4340:0:12.5::28", "can0;1:4340:0:12.5:20:",
      "can0;1:4340:0:12.5:20:28:", "can0;1:4340:0:12.5:20:28:1",
      "can0;0:4340", "can0;16:4340", "can0;-1:4340", "can0;1x:4340",
      "can0;1.0:4340", "can0;08:4340", "can0;0x1g:4340",
      "can0;999999999999999999999999:4340", "can0;1:4340:-1",
      "can0;1:4340:0x800", "can0;1:4340:0x11x", "can0;1:4340:1.5",
      "can0;1:4340:999999999999999999999999"}) {
    std::vector<Motor> invalid;
    err.clear();
    if (parse_spec(spec, iface, invalid, err) || err.empty()) {
      std::fprintf(stderr, "MIT map check: malformed spec accepted: '%s'\n", spec);
      return 1;
    }
  }
  for (const char* value : {"nan", "inf", "-inf", "0", "-1", "1e-999",
                            "1e-310", "0x1p-1074", "1e309", "1e308", "8e307",
                            "20junk", "20 "}) {
    for (int field = 0; field < 3; ++field) {
      std::string spec = "can0;1:4340:0";
      for (int j = 0; j < 3; ++j) spec += ":" + std::string(j == field ? value : "20");
      std::vector<Motor> invalid;
      err.clear();
      if (parse_spec(spec, iface, invalid, err) || err.empty()) {
        std::fprintf(stderr, "MIT map check: invalid half-range accepted: '%s'\n",
                     spec.c_str());
        return 1;
      }
    }
  }
  const uint8_t reply[8] = {0xA1, 0x80, 0x00, 0x99, 0x98, 0x00, 25, 34};
  MitReply corrected, legacy;
  if (!decode_mit(reply, 8, mapped[0].lim, corrected) ||
      !decode_mit(reply, 8, mixed[0].lim, legacy) ||
      corrected.dq != 4 || legacy.dq != 2 || corrected.q != legacy.q ||
      corrected.tau != legacy.tau || corrected.id != 1 || corrected.err != 10 ||
      corrected.t_mos != 25 || corrected.t_rotor != 34) {
    std::fprintf(stderr, "MIT map check: reply velocity code 2457 must decode as 4/2\n");
    return 1;
  }
  const double velocities[] = {-20, -4, 0, 4, 20};
  const uint8_t expected[5][8] = {
      {0x8c, 0xa2, 0x00, 0x03, 0xf2, 0x8a, 0xbb, 0xe0},
      {0x8c, 0xa2, 0x66, 0x63, 0xf2, 0x8a, 0xbb, 0xe0},
      {0x8c, 0xa2, 0x7f, 0xf3, 0xf2, 0x8a, 0xbb, 0xe0},
      {0x8c, 0xa2, 0x99, 0x93, 0xf2, 0x8a, 0xbb, 0xe0},
      {0x8c, 0xa2, 0xff, 0xf3, 0xf2, 0x8a, 0xbb, 0xe0},
  };
  for (int i = 0; i < 5; ++i) {
    uint8_t out[8];
    pack_mit(1.234, velocities[i], 123.4, 2.71, 13.579, mapped[0].lim, out);
    if (std::memcmp(out, expected[i], sizeof out)) {
      std::fprintf(stderr, "MIT map check: VMAX 20 golden changed at velocity %.17g\n",
                   velocities[i]);
      return 1;
    }
  }
  for (Motor m : {mapped[0], mixed[5], mixed[6]}) {
    const DmLimits& L = m.lim;
    if (L.p_lo != -L.p_hi || L.v_lo != -L.v_hi || L.t_lo != -L.t_hi ||
        L.kp_lo != 0 || L.kp_hi != 500 || L.kd_lo != 0 || L.kd_hi != 5) {
      std::fprintf(stderr, "MIT map check: range symmetry or gains changed\n");
      return 1;
    }
    for (int sign : {-1, 1}) {
      uint8_t out[8];
      pack_mit(sign * L.p_hi, sign * L.v_hi, sign > 0 ? 500 : 0,
               sign > 0 ? 5 : 0, sign * L.t_hi, L, out);
      for (uint8_t byte : out) {
        if (byte != (sign > 0 ? 0xff : 0)) {
          std::fprintf(stderr, "MIT map check: signed endpoint encoding changed\n");
          return 1;
        }
      }
      const uint8_t edge = sign > 0 ? 0xff : 0;
      const uint8_t feedback[8] = {1, edge, edge, edge, edge, edge, 25, 34};
      MitReply decoded;
      if (!decode_mit(feedback, 8, L, decoded) || decoded.q != sign * L.p_hi ||
          decoded.dq != sign * L.v_hi || decoded.tau != sign * L.t_hi) {
        std::fprintf(stderr, "MIT map check: signed endpoint decoding changed\n");
        return 1;
      }
    }
    if (!set_torque_cap(m, 27, err) || m.tau_cap != std::min(27.0, L.t_hi)) {
      std::fprintf(stderr, "MIT map check: operating cap not based on mapped TMAX\n");
      return 1;
    }
    for (double tau : {-1000.0, -27.0, 0.0, 27.0, 1000.0}) {
      uint8_t original[8], out[8];
      pack_mit(1.234, 4, 123.4, 2.71, tau, L, original);
      std::memcpy(out, original, sizeof out);
      clamp_torque_field(out, m);
      const double decoded = u2f((uint32_t(out[6] & 0x0f) << 8) | out[7],
                                 L.t_lo, L.t_hi, 12);
      if (decoded < -m.tau_cap || decoded > m.tau_cap ||
          std::memcmp(out, original, 6) || (out[6] & 0xf0) != (original[6] & 0xf0) ||
          (L.v_hi == 20 && ((uint32_t(out[2]) << 4) | (out[3] >> 4)) != 2457)) {
        std::fprintf(stderr, "MIT map check: encoding or operating cap changed fields\n");
        return 1;
      }
    }
  }
  for (const char* type : {"4340", "4340p", "4310", "4310p"}) {
    DmLimits L;
    limits_for(type, L);
    const double nearest = std::min(-u2f(2047, L.t_lo, L.t_hi, 12),
                                    u2f(2048, L.t_lo, L.t_hi, 12));
    for (double cap : {0.0, 27.0, 0.01, nearest, 1000.0}) {
      Motor m;
      m.type = type;
      m.lim = L;
      std::string err;
      if (!set_torque_cap(m, cap, err)) {
        std::fprintf(stderr, "MIT cap check: %s cap %.17g rejected: %s\n",
                     type, cap, err.c_str());
        return 1;
      }
      const double effective = cap == 0 ? L.t_hi : std::min(cap, L.t_hi);
      if ((cap == 0 && m.tau_cap != 0) ||
          (cap != 0 && m.tau_cap != effective)) {
        std::fprintf(stderr, "MIT cap check: %s incorrect effective cap\n", type);
        return 1;
      }
      for (double tau : {-2 * L.t_hi, -effective, 0.0, effective, 2 * L.t_hi}) {
        uint8_t legacy[8], out[8];
        pack_mit(1.234, -2.5, 123.4, 2.71, tau, L, legacy);
        std::memcpy(out, legacy, sizeof out);
        clamp_torque_field(out, m);
        const uint32_t raw = (uint32_t(out[6] & 0x0F) << 8) | out[7];
        const double decoded = u2f(raw, L.t_lo, L.t_hi, 12);
        if (decoded < -effective || decoded > effective) {
          std::fprintf(stderr, "MIT cap check: %s cap %.17g torque %.17g "
                               "encoded as %.17g\n", type, cap, tau, decoded);
          return 1;
        }
        if (std::memcmp(out, legacy, 6) || (out[6] & 0xF0) != (legacy[6] & 0xF0) ||
            ((cap == 0 || cap >= L.t_hi) && std::memcmp(out, legacy, 8))) {
          std::fprintf(stderr, "MIT cap check: %s changed legacy fields\n", type);
          return 1;
        }
      }
    }
    for (double cap : {1e-12, std::nextafter(nearest, 0.0), -1.0,
                       std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
      Motor m;
      m.type = type;
      m.lim = L;
      std::string err;
      if (set_torque_cap(m, cap, err) ||
          (cap > 0 && cap < nearest && err.find("representable") == std::string::npos)) {
        std::fprintf(stderr, "MIT cap check: %s invalid cap %.17g accepted "
                             "or wrong error\n", type, cap);
        return 1;
      }
    }
  }
  struct PackCase {
    const char* type;
    double p, v, kp, kd, t;
    uint8_t expected[8];
  };
  const PackCase packs[] = {
      {"4340", 1.234, -2.5, 0, 0, 13.579,
       {0x8c, 0xa2, 0x5f, 0xf0, 0x00, 0x00, 0x0b, 0xe0}},
      {"4340", 0, 0, 0, 0, 0,
       {0x7f, 0xff, 0x7f, 0xf0, 0x00, 0x00, 0x07, 0xff}},
      {"4340", -12.5, -10, 0, 0, -28,
       {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
      {"4340", 12.5, 10, 500, 5, 28,
       {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
      {"4310", -3.3, 7.77, 123.4, 2.71, -9.87,
       {0x5e, 0x34, 0xa1, 0x13, 0xf2, 0x8a, 0xb0, 0x1a}},
      {"4310p", 5, 41, 250, 1, 3.3,
       {0xb3, 0x32, 0xe8, 0xe7, 0xff, 0x33, 0x3a, 0xa3}},
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
    if (std::memcmp(out, c.expected, sizeof out)) {
      std::fprintf(stderr, "MIT cap check: %s legacy golden changed\n", c.type);
      return 1;
    }
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
