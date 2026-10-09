#pragma once
// The DM gripper behind hand_bridge (operator 2026-10-09, remaining-migrations
// design A2): ONE DM motor (the DM arm's gripper, CAN id 7) on its own
// SocketCAN socket. The arm's server owns the other motors on the same bus;
// raw CAN sockets coexist and each routes replies by the payload id nibble.
//
// Law (operator decisions, nothing else is assumed -- every value is a flag):
// * MOVE: on-motor MIT position PD at kp/kd (40/2 decided), the target ramped
//   at the commanded width speed through the mimic map; every frame clamps the
//   target to within tau_cap/kp of the MEASURED angle, so kp*error <= tau_cap
//   (3 N.m decided). The kd term is not part of that bound.
// * Success = the measured width settles (0.2 s window, 0.5 mm) within 1 mm of
//   the goal -- the sim Hand's rule. Blocked short of it: the action fails and
//   the target becomes the measured angle (no squeeze is kept). An action that
//   never settles fails after its travel time + 2 s and holds where it is.
// * GSTOP / disconnect with committed work: hold the measured angle.
// * HOME: nothing to do (absolute encoder). GRASP: refused (no N->torque map
//   until bench rung 11; the bridge rejects it before it reaches here).
// * Enabled on the first action, then it keeps holding, independent of the
//   arm's ARM. A silent motor (50 ms, the arm backend's bound) or shutdown():
//   zero torque, then disable.
// Width = 2 * finger_stroke * (q - motor_closed) / (motor_open - motor_closed).
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dm_mit.hpp"
#include "hand_device.hpp"

struct CanPort {
  virtual ~CanPort() = default;
  virtual bool send(uint32_t can_id, const uint8_t data[8]) = 0;
  // One frame into data (8 bytes): 1, 0 when none arrived within timeout_ms, -1 on error.
  virtual int recv(uint8_t data[8], int timeout_ms) = 0;
};

class SocketCanPort final : public CanPort {
public:
  // Only frames on the motor's reply id (its Master ID) reach this socket.
  SocketCanPort(const std::string& iface, uint32_t reply_id) {
    fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
    const int on = 1;
    can_filter filter{canid_t(reply_id), CAN_SFF_MASK};
    ifreq ifr{};
    std::snprintf(ifr.ifr_name, IFNAMSIZ, "%s", iface.c_str());
    sockaddr_can addr{};
    addr.can_family = AF_CAN;
    if (fd_ < 0 ||
        setsockopt(fd_, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &on, sizeof on) != 0 ||
        setsockopt(fd_, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, sizeof filter) != 0 ||
        ioctl(fd_, SIOCGIFINDEX, &ifr) != 0 ||
        (addr.can_ifindex = ifr.ifr_ifindex,
         ::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0)) {
      const std::string why = std::strerror(errno);
      if (fd_ >= 0) ::close(fd_);
      throw DeviceLost("CAN " + iface + ": " + why);
    }
  }
  ~SocketCanPort() override { ::close(fd_); }
  bool send(uint32_t can_id, const uint8_t data[8]) override {
    canfd_frame f{};
    f.can_id = canid_t(can_id);
    f.len = 8;
    f.flags = CANFD_FDF | CANFD_BRS;  // as the arm backend (FD + bit-rate switch)
    std::memcpy(f.data, data, 8);
    return ::write(fd_, &f, CANFD_MTU) == ssize_t(CANFD_MTU);
  }
  int recv(uint8_t data[8], int timeout_ms) override {
    pollfd p{fd_, POLLIN, 0};
    const int ready = ::poll(&p, 1, timeout_ms);
    if (ready <= 0) return ready < 0 && errno != EINTR ? -1 : 0;
    canfd_frame f{};
    const ssize_t got = ::read(fd_, &f, sizeof f);
    if (got != ssize_t(CAN_MTU) && got != ssize_t(CANFD_MTU)) return got < 0 ? -1 : 0;
    if ((f.can_id & CAN_ERR_FLAG) || f.len < 8) return 0;
    std::memcpy(data, f.data, 8);
    return 1;
  }

private:
  int fd_ = -1;
};

struct DmGripperConfig {
  int id = 0;               // motor CAN id (replies carry it in the payload low nibble)
  arm_rt::DmLimits lim{};   // the motor type's MIT ranges
  double motor_open = 0, motor_closed = 0, finger_stroke_m = 0;  // the mimic map
  double kp = 0, kd = 0, tau_cap = 0, speed_max_mps = 0;

  HandLimits limits() const { return {2 * finger_stroke_m, speed_max_mps, 0, 0}; }
  double width(double q) const {
    return 2 * finger_stroke_m * (q - motor_closed) / (motor_open - motor_closed);
  }
  double motor(double width_m) const {
    return motor_closed + width_m / (2 * finger_stroke_m) * (motor_open - motor_closed);
  }
  const char* invalid() const {
    if (id < 1 || id > 15) return "motor id must be 1..15";
    if (!(std::isfinite(motor_open) && std::isfinite(motor_closed)) || motor_open == motor_closed)
      return "motor_open and motor_closed must differ";
    if (!(finger_stroke_m > 0)) return "finger stroke must be positive";
    if (!(kp > 0 && kp <= lim.kp_hi && kd >= 0 && kd <= lim.kd_hi)) return "kp/kd outside the motor's MIT range";
    if (!(tau_cap > 0 && tau_cap <= lim.t_hi)) return "tau_cap must be positive and within the motor's torque range";
    if (!(speed_max_mps > 0)) return "speed_max must be positive";
    return nullptr;
  }
};

class DmGripper final : public HandDevice {
public:
  static constexpr double kPeriodS = 0.01;      // servo frame period
  static constexpr double kSilenceS = 0.05;     // the arm backend's per-motor deadman
  static constexpr double kSettleWindowS = 0.2, kSettleTolM = 5e-4, kMoveTolM = 1e-3;  // the sim Hand's
  static constexpr double kMoveGraceS = 2.0;    // past the travel time, an unsettled move fails
  static constexpr double kEdgeTolM = 2e-3;     // measured width clamped to the stroke within this

  DmGripper(const DmGripperConfig& cfg, std::unique_ptr<CanPort> port)
      : cfg_(cfg), port_(std::move(port)) {
    if (const char* why = cfg_.invalid()) throw std::invalid_argument(why);
    servo_ = std::thread([this] { servo(); });
    std::unique_lock<std::mutex> lk(m_);
    if (!cv_.wait_for(lk, std::chrono::milliseconds(200), [this] { return seq_ > 0 || !lost_.empty(); }) ||
        !lost_.empty()) {
      lk.unlock();
      shutdown();
      throw DeviceLost("DM motor " + std::to_string(cfg_.id) + " never replied");
    }
  }
  ~DmGripper() override { shutdown(); }

  HandSample read() override {
    std::unique_lock<std::mutex> lk(m_);
    if (!cv_.wait_for(lk, std::chrono::milliseconds(200),
                      [this] { return seq_ != read_seq_ || !lost_.empty(); }))
      throw DeviceLost("no DM sample in 200 ms");
    if (!lost_.empty()) throw DeviceLost(lost_);
    read_seq_ = seq_;
    double w = cfg_.width(q_);
    const double wmax = 2 * cfg_.finger_stroke_m;
    if (w > -kEdgeTolM && w < 0) w = 0;
    if (w > wmax && w < wmax + kEdgeTolM) w = wmax;
    return {w, false, seq_};
  }

  bool move(double width_m, double speed_mps) override {
    std::unique_lock<std::mutex> lk(m_);
    if (!lost_.empty()) throw DeviceLost(lost_);
    const double goal = cfg_.motor(width_m);
    const double rate = speed_mps * std::fabs(cfg_.motor_open - cfg_.motor_closed) / (2 * cfg_.finger_stroke_m);
    enable_ = true;
    ramping_ = true;
    target_ = q_;  // ramp from where the fingers ARE
    goal_ = goal;
    goal_w_ = width_m;
    rate_ = rate;
    deadline_ = now_s() + std::fabs(goal - q_) / rate + kMoveGraceS;
    hist_.clear();
    verdict_ = -1;
    const uint64_t action = ++action_seq_;
    cv_.wait(lk, [&] { return verdict_ >= 0 || action_seq_ != action || !lost_.empty() || closed_; });
    if (!lost_.empty() || closed_) throw DeviceLost(lost_.empty() ? "closed" : lost_);
    return action_seq_ == action && verdict_ == 1;
  }

  bool homing() override {
    std::lock_guard<std::mutex> lk(m_);
    if (!lost_.empty()) throw DeviceLost(lost_);
    return true;  // absolute encoder: nothing to find
  }

  bool grasp(double, double, double, double, double) override {
    throw ActionEnded("uncalibrated_force");  // no N->torque map until bench rung 11
  }

  bool stop() override {
    std::lock_guard<std::mutex> lk(m_);
    ++action_seq_;  // a running move returns false
    ramping_ = false;
    target_ = q_;   // hold the measured angle (nothing to hold before the first action)
    cv_.notify_all();
    return lost_.empty();
  }

  // Zero torque, then disable; idempotent. Bridge exit and loss.
  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
      cv_.notify_all();
    }
    if (servo_.joinable()) servo_.join();
    std::lock_guard<std::mutex> lk(m_);
    if (!released_) {
      released_ = true;
      zero_and_disable();
    }
  }

  // For the self-check: the frame law's inputs.
  double tau_cap() const { return cfg_.tau_cap; }

private:
  static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void zero_and_disable() {  // under m_, servo stopped or the caller IS the servo
    uint8_t zero[8];
    arm_rt::pack_mit(0, 0, 0, 0, 0, cfg_.lim, zero);
    port_->send(uint32_t(cfg_.id), zero);
    port_->send(uint32_t(cfg_.id), arm_rt::DM_DISABLE_FRAME);
    enabled_ = false;
  }

  void servo() {
    auto next = std::chrono::steady_clock::now();
    double last_rx = now_s();
    for (;;) {
      uint8_t frame[8];
      bool send_enable = false;
      {
        std::lock_guard<std::mutex> lk(m_);
        if (closed_ || !lost_.empty()) return;
        if (enable_ && !enabled_) send_enable = enabled_ = true;
        if (!enabled_) {
          arm_rt::pack_mit(0, 0, 0, 0, 0, cfg_.lim, frame);  // kp=kd=0: a reply, no authority
        } else {
          if (ramping_) {
            const double step = rate_ * kPeriodS;
            target_ = target_ < goal_ ? std::min(goal_, target_ + step) : std::max(goal_, target_ - step);
          }
          const double reach = cfg_.tau_cap / cfg_.kp;
          const double q_des = std::clamp(target_, q_ - reach, q_ + reach);
          arm_rt::pack_mit(q_des, 0, cfg_.kp, cfg_.kd, 0, cfg_.lim, frame);
        }
      }
      if ((send_enable && !port_->send(uint32_t(cfg_.id), arm_rt::DM_ENABLE_FRAME)) ||
          !port_->send(uint32_t(cfg_.id), frame)) {
        fail("CAN send failed");
        return;
      }
      next += std::chrono::microseconds(int64_t(kPeriodS * 1e6));
      for (;;) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            next - std::chrono::steady_clock::now()).count();
        uint8_t data[8];
        const int got = port_->recv(data, int(std::max<int64_t>(left, 0)));
        if (got < 0) {
          fail("CAN receive failed");
          return;
        }
        if (got == 0) break;
        arm_rt::MitReply r;
        if ((data[0] & 0x0F) != cfg_.id || !arm_rt::decode_mit(data, 8, cfg_.lim, r)) continue;
        last_rx = now_s();
        std::lock_guard<std::mutex> lk(m_);
        q_ = r.q;
        ++seq_;
        cv_.notify_all();
      }
      const double now = now_s();
      if (now - last_rx > kSilenceS) {
        fail("DM motor " + std::to_string(cfg_.id) + " silent");
        return;
      }
      std::lock_guard<std::mutex> lk(m_);
      if (ramping_ && target_ == goal_) settle(now);
      if (std::chrono::steady_clock::now() > next + std::chrono::milliseconds(50))
        next = std::chrono::steady_clock::now();  // a stall: no banked burst of frames
    }
  }

  void settle(double now) {  // under m_: the sim Hand's verdict on the measured width
    const double w = cfg_.width(q_);
    hist_.emplace_back(now, w);
    while (!hist_.empty() && now - hist_.front().first > kSettleWindowS) hist_.erase(hist_.begin());
    double lo = w, hi = w;
    for (const auto& [t, x] : hist_) lo = std::min(lo, x), hi = std::max(hi, x);
    const bool settled = hist_.size() >= 2 && hist_.back().first - hist_.front().first >= 0.5 * kSettleWindowS &&
                         hi - lo < kSettleTolM;
    if (!settled && now < deadline_) return;
    ramping_ = false;
    verdict_ = settled && std::fabs(w - goal_w_) <= kMoveTolM ? 1 : 0;
    if (!verdict_) target_ = q_;  // blocked or unsettled: stop where the fingers are
    cv_.notify_all();
  }

  void fail(const std::string& why) {  // servo thread
    std::lock_guard<std::mutex> lk(m_);
    lost_ = why;
    std::printf("[hand] %s -- zero torque, disable\n", why.c_str());
    zero_and_disable();
    released_ = true;
    cv_.notify_all();
  }

  const DmGripperConfig cfg_;
  std::unique_ptr<CanPort> port_;
  std::mutex m_;
  std::condition_variable cv_;
  std::thread servo_;
  std::string lost_;
  bool closed_ = false, released_ = false;
  bool enable_ = false, enabled_ = false, ramping_ = false;
  double q_ = 0, target_ = 0, goal_ = 0, goal_w_ = 0, rate_ = 0, deadline_ = 0;
  uint64_t seq_ = 0, read_seq_ = 0, action_seq_ = 0;
  int verdict_ = -1;
  std::vector<std::pair<double, double>> hist_;
};
