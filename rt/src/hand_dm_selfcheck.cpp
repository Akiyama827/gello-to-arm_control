// The DM gripper device against a fake DM4310 on a fake CAN port: no socket,
// interface, privilege or motor. The fake decodes every MIT frame the device
// sends, moves a quasi-static finger motor (optionally against an object) and
// replies like the firmware (payload low nibble = motor id).
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <future>

#include "hand_dm.hpp"

using namespace std::chrono_literals;

namespace {

struct Frame {
  bool enable = false, disable = false;
  double q_des = 0, kp = 0, kd = 0, tau = 0;
  double q_seen = 0;  // the angle last REPORTED to the device when this frame arrived
  std::chrono::steady_clock::time_point t;
};

class FakeMotor final : public CanPort {
public:
  explicit FakeMotor(const arm_rt::DmLimits& lim) : lim_(lim) {}
  std::mutex mx;
  std::vector<Frame> frames;
  double q = 0;                 // rad; 0 = open
  double block_q = -1e9;        // an object stops closing (decreasing q) here
  bool silent = false, self_silent = false, foreign = true;
  bool enabled = false;
  double jitter = 0;            // rad, alternating on every reply (never settles)
  int early_zeros = 0;          // recv returns 0 at once this many times (odd frames on the bus)
  int* sends = nullptr;         // counts frames into storage that outlives the port (a refused session destroys it)

  bool send(uint32_t can_id, const uint8_t d[8]) override {
    assert(can_id == 7);
    std::lock_guard<std::mutex> lk(mx);
    if (sends) ++*sends;
    Frame f;
    f.q_seen = reported_;
    f.t = std::chrono::steady_clock::now();
    if (std::all_of(d, d + 7, [](uint8_t b) { return b == 0xFF; }) && d[7] == 0xFC) {
      f.enable = enabled = true;
    } else if (std::all_of(d, d + 7, [](uint8_t b) { return b == 0xFF; }) && d[7] == 0xFD) {
      f.disable = true;
      enabled = false;
    } else {
      f.q_des = arm_rt::u2f(uint32_t(d[0] << 8 | d[1]), lim_.p_lo, lim_.p_hi, 16);
      f.kp = arm_rt::u2f(uint32_t((d[3] & 0x0F) << 8 | d[4]), lim_.kp_lo, lim_.kp_hi, 12);
      f.kd = arm_rt::u2f(uint32_t(d[5] << 4 | d[6] >> 4), lim_.kd_lo, lim_.kd_hi, 12);
      f.tau = arm_rt::u2f(uint32_t((d[6] & 0x0F) << 8 | d[7]), lim_.t_lo, lim_.t_hi, 12);
      if (enabled) step(f);
      if (!silent) {
        if (!self_silent) queue_reply(7, q + (jitter_sign_ = -jitter_sign_) * jitter);
        if (foreign) queue_reply(3, 1.0);  // another motor on the same Master ID, AFTER ours: must not overwrite it
      }
    }
    frames.push_back(f);
    return true;
  }
  void queue_before_any_frame() { queue_reply(7, 0.0); }  // a motor someone else is already driving
  int recv(uint8_t d[8], int timeout_ms) override {
    std::unique_lock<std::mutex> lk(mx);
    if (early_zeros > 0) {
      --early_zeros;
      return 0;
    }
    if (replies_.empty()) {
      lk.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
      return 0;
    }
    std::copy_n(replies_.front().data(), 8, d);
    if ((d[0] & 0x0F) == 7) reported_ = arm_rt::u2f(uint32_t(d[1] << 8 | d[2]), lim_.p_lo, lim_.p_hi, 16);
    replies_.pop_front();
    return 1;
  }
  std::vector<Frame> snapshot() {
    std::lock_guard<std::mutex> lk(mx);
    return frames;
  }

private:
  void step(const Frame& f) {  // quasi-static: velocity ~ net torque, static friction, a hard object
    const double dt = DmGripper::kPeriodS, b = 0.5, friction = 0.05;
    const double tau = f.kp * (f.q_des - q) + f.tau;
    if (std::fabs(tau) > friction) q += (tau - std::copysign(friction, tau)) / (b + f.kd) * dt;
    q = std::max(q, block_q);
  }
  void queue_reply(int id, double pos) {
    std::array<uint8_t, 8> d{};
    const uint32_t p = arm_rt::f2u(pos, lim_.p_lo, lim_.p_hi, 16);
    const uint32_t v = arm_rt::f2u(0, lim_.v_lo, lim_.v_hi, 12);
    const uint32_t t = arm_rt::f2u(0, lim_.t_lo, lim_.t_hi, 12);
    d[0] = uint8_t(1 << 4 | id);
    d[1] = uint8_t(p >> 8);
    d[2] = uint8_t(p & 0xFF);
    d[3] = uint8_t(v >> 4);
    d[4] = uint8_t((v & 0x0F) << 4 | t >> 8);
    d[5] = uint8_t(t & 0xFF);
    d[6] = d[7] = 30;
    replies_.push_back(d);
  }
  const arm_rt::DmLimits lim_;
  std::deque<std::array<uint8_t, 8>> replies_;
  double reported_ = 0;
  double jitter_sign_ = 1;
};

DmGripperConfig config() {
  DmGripperConfig c;
  c.id = 7;
  c.lim = arm_rt::LIM_4310;
  c.motor_open = 0.0;        // the assembler's mimic (hardware.yaml joint_mimics)
  c.motor_closed = -4.985;
  c.finger_stroke_m = 0.0439;
  c.kp = 40;                 // operator 2026-10-09: MOVE 40/2, 3 N.m target clamp
  c.kd = 2;
  c.tau_cap = 3;
  c.speed_max_mps = 0.2;     // a test value: the deployed one is the operator's
  return c;
}

// Every authority frame keeps kp * |q_des - q_reported| <= tau_cap (+ one pos16 code).
void assert_clamped(const std::vector<Frame>& frames, const DmGripperConfig& c) {
  const double code = (c.lim.p_hi - c.lim.p_lo) / 65535.0;
  for (const Frame& f : frames)
    if (f.kp > 0) assert(f.kp * std::fabs(f.q_des - f.q_seen) <= c.tau_cap + c.kp * 2 * code);
}

}  // namespace

int main() {
  const DmGripperConfig c = config();
  assert(std::fabs(c.width(0) - .0878) < 1e-12 && std::fabs(c.width(c.motor(.05)) - .05) < 1e-12);
  {
    DmGripperConfig bad = c;
    bad.tau_cap = 0;
    assert(bad.invalid());
    bad = c;
    bad.kp = 600;  // past the DM4310's MIT kp range
    assert(bad.invalid());
  }

  // A motor that never replies: no session.
  {
    auto port = std::make_unique<FakeMotor>(c.lim);
    port->silent = true;
    bool lost = false;
    try {
      DmGripper g(c, std::move(port));
    } catch (const DeviceLost&) {
      lost = true;
    }
    assert(lost);
  }

  // Another driver already commands motor 7 (its replies flow before we send): refuse, transmit nothing.
  {
    auto port = std::make_unique<FakeMotor>(c.lim);
    int sends = 0;
    port->sends = &sends;
    port->queue_before_any_frame();
    bool refused = false;
    try {
      DmGripper g(c, std::move(port));
    } catch (const DeviceLost& e) {
      refused = std::string(e.what()).find("another driver") != std::string::npos;
    }
    assert(refused && sends == 0);
  }

  // GSTOP before any action: nothing to stop, no enable, no authority.
  {
    auto port = std::make_unique<FakeMotor>(c.lim);
    FakeMotor& m = *port;
    DmGripper g(c, std::move(port));
    assert(g.stop());
    std::this_thread::sleep_for(60ms);
    for (const Frame& f : m.snapshot()) assert(!f.enable && f.kp == 0);
  }

  auto port_owner = std::make_unique<FakeMotor>(c.lim);
  FakeMotor& motor = *port_owner;
  auto g = std::make_unique<DmGripper>(c, std::move(port_owner));
  HandSample s = g->read();
  assert(std::fabs(s.width - .0878) < 1e-5 && !s.grasped);  // one pos16 code is ~3 um of width
  const uint64_t first = s.stamp;
  assert(g->read().stamp > first);
  // Before any action: no enable, no authority (kp = kd = 0), a reply per frame.
  for (const Frame& f : motor.snapshot()) assert(!f.enable && f.kp == 0 && f.kd == 0 && std::fabs(f.tau) < 0.01);
  assert(g->homing());

  // MOVE: enabled once, ramps, settles within 1 mm, every frame clamped.
  const auto t0 = std::chrono::steady_clock::now();
  assert(g->move(.06, .05));
  const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  assert(took >= (.0878 - .06) / .05 * 0.9);  // no faster than the commanded speed
  assert(std::fabs(g->read().width - .06) <= 1e-3);
  {
    const auto frames = motor.snapshot();
    int enables = 0;
    bool authority_before_enable = false;
    for (const Frame& f : frames) {
      enables += f.enable;
      if (f.kp > 0 && enables == 0) authority_before_enable = true;
    }
    assert(enables == 1 && !authority_before_enable);
    assert_clamped(frames, c);
    for (const Frame& f : frames)
      if (f.kp > 0) assert(std::fabs(f.kp - 40) < 0.2 && std::fabs(f.kd - 2) < 0.01 && std::fabs(f.tau) < 0.01);
  }

  // The ramp: at a speed the finger can follow, the target advances at most rate * period per frame (+ codes).
  {
    const size_t from = motor.snapshot().size();
    assert(g->move(.055, .01));
    const auto frames = motor.snapshot();
    const double code = (c.lim.p_hi - c.lim.p_lo) / 65535.0;
    const double step = .01 * (c.motor_open - c.motor_closed) / (2 * c.finger_stroke_m) * DmGripper::kPeriodS;
    for (size_t i = from + 1; i < frames.size(); ++i)
      if (frames[i].kp > 0 && frames[i - 1].kp > 0) assert(std::fabs(frames[i].q_des - frames[i - 1].q_des) <= step + 2 * code);
  }

  // Odd frames on the bus (an early empty receive) do not pace the servo: frames stay ~one period apart, no false loss.
  {
    const size_t from = motor.snapshot().size();
    {
      std::lock_guard<std::mutex> lk(motor.mx);
      motor.early_zeros = 50;
    }
    std::this_thread::sleep_for(300ms);
    const auto frames = motor.snapshot();
    for (size_t i = from + 1; i < frames.size(); ++i) assert(frames[i].t - frames[i - 1].t > 5ms);
    g->read();  // still a session
  }

  // A foreign reply arriving after ours never moves the reported width.
  for (int i = 0; i < 5; ++i) assert(std::fabs(g->read().width - .055) < 2e-3);

  // Blocked by an object at 50 mm: the move fails, then it holds the measured angle (no squeeze kept).
  motor.block_q = c.motor(.05);
  assert(!g->move(.02, .05));
  const double w_block = g->read().width;
  assert(std::fabs(w_block - .05) < 2e-3);
  std::this_thread::sleep_for(100ms);
  {
    const auto frames = motor.snapshot();
    const Frame& last = frames.back();
    assert(last.kp > 0 && last.kp * std::fabs(last.q_des - last.q_seen) < 0.1);
    assert_clamped(frames, c);
  }
  // GSTOP while squeezing a blocking object mid-ramp: the move fails and the squeeze goes (hold measured).
  {
    auto squeezing = std::async(std::launch::async, [&] { return g->move(.0, .05); });
    std::this_thread::sleep_for(400ms);  // blocked at 50 mm, target still ramping toward 0
    assert(g->stop());
    assert(squeezing.wait_for(200ms) == std::future_status::ready && !squeezing.get());
    std::this_thread::sleep_for(100ms);
    const Frame last = motor.snapshot().back();
    assert(last.kp > 0 && last.kp * std::fabs(last.q_des - last.q_seen) < 0.1);
  }
  motor.block_q = -1e9;

  // GSTOP mid-move: the move returns false promptly and the target becomes the measured angle.
  auto moving = std::async(std::launch::async, [&] { return g->move(.0, .02); });
  std::this_thread::sleep_for(300ms);
  const auto t_stop = std::chrono::steady_clock::now();
  assert(g->stop());
  assert(moving.wait_for(200ms) == std::future_status::ready && !moving.get());
  assert(std::chrono::steady_clock::now() - t_stop < 200ms);
  std::this_thread::sleep_for(100ms);
  const double w_stopped = g->read().width;
  std::this_thread::sleep_for(200ms);
  assert(std::fabs(g->read().width - w_stopped) < 5e-4);  // held, not still closing

  // GRASP never runs here (no N->torque map).
  bool refused = false;
  try {
    g->grasp(.04, .05, 40, .02, .02);
  } catch (const ActionEnded& e) {
    refused = std::string(e.what()) == "uncalibrated_force";
  }
  assert(refused);

  // A move that never settles fails at its travel time + 2 s, and holds.
  {
    {
      std::lock_guard<std::mutex> lk(motor.mx);
      motor.jitter = 0.03;  // ~0.5 mm of width, every reply
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double w0 = g->read().width;
    auto unsettled = std::async(std::launch::async, [&] { return g->move(w0 + .002, .1); });
    if (unsettled.wait_for(4s) != std::future_status::ready) {
      std::fputs("FAIL: an unsettled move did not end at its deadline\n", stderr);
      std::_Exit(1);  // the move thread is still blocked: no orderly exit
    }
    assert(!unsettled.get());
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    assert(took > DmGripper::kMoveGraceS - .1 && took < DmGripper::kMoveGraceS + .6);
    std::lock_guard<std::mutex> lk(motor.mx);
    motor.jitter = 0;
  }

  // Only OTHER motors answering is silence for ours.
  {
    auto port = std::make_unique<FakeMotor>(c.lim);
    FakeMotor& m = *port;
    DmGripper h(c, std::move(port));
    {
      std::lock_guard<std::mutex> lk(m.mx);
      m.self_silent = true;
    }
    bool gone = false;
    try {
      for (int i = 0; i < 20; ++i) h.read();
    } catch (const DeviceLost&) {
      gone = true;
    }
    assert(gone);
  }

  // Silence: the session is lost, and the motor gets zero torque then disable.
  motor.silent = true;
  bool lost = false;
  try {
    for (int i = 0; i < 20; ++i) g->read();
  } catch (const DeviceLost&) {
    lost = true;
  }
  assert(lost);
  {
    const auto frames = motor.snapshot();
    assert(frames.back().disable && frames[frames.size() - 2].kp == 0 && std::fabs(frames[frames.size() - 2].tau) < 0.01);
  }
  bool move_lost = false;
  try {
    g->move(.05, .05);
  } catch (const DeviceLost&) {
    move_lost = true;
  }
  assert(move_lost);
  g.reset();

  // Shutdown (bridge exit): zero torque, then disable, exactly once.
  {
    auto port = std::make_unique<FakeMotor>(c.lim);
    FakeMotor& m = *port;
    DmGripper h(c, std::move(port));
    assert(h.move(.07, .05));
    h.shutdown();
    h.shutdown();
    const auto frames = m.snapshot();
    assert(frames.back().disable && frames[frames.size() - 2].kp == 0);
    int disables = 0;
    for (const Frame& f : frames) disables += f.disable;
    assert(disables == 1);
  }
  std::puts("PASS DM gripper: map, config bounds, no session without replies, no authority before the first action, "
            "MOVE ramp rate/settle/clamp, blocked move holds measured, GSTOP mid-move, mid-squeeze and before any action, "
            "unsettled move fails at travel + grace, GRASP refused, silence (also with only foreign replies) and shutdown "
            "release (zero torque, disable), foreign replies ignored, odd frames do not pace the servo, a motor another "
            "driver commands is refused");
}
