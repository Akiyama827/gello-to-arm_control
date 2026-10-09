// hand_bridge — gripper ownership ON the robot LAN (the RT box).
//
// Backends (one protocol, admission and stop path for both):
//   hand_bridge [robot_ip [port [bind_ip]]]            the Franka Hand (libfranka)
//   hand_bridge --backend dm --can-if IF --motor ID:TYPE:MST
//       --mimic OPEN,CLOSED,STROKE --kp KP --kd KD --tau-cap N.M --speed-max M/S
//       [--port P] [--bind IP]                         a DM motor gripper (hand_dm.hpp)
// Every DM value is a flag (nothing assumed). The bridge states its gripper's
// envelope on LIMITS; the DM gripper has no force grasp yet, so its GRASP is
// REJECTed `uncalibrated_force`. SIGTERM/SIGINT: the DM gripper gets zero torque,
// then disable, before exit.
//
// The Hand mirrors the robot's protocol split: commands over TCP, cyclic
// state PUSHED over UDP. Through the PC-side NAT those pushed datagrams die
// (a server-initiated UDP flow has no conntrack entry, and the advertised
// return port is ephemeral — unforwardable), so any pylibfranka Gripper on
// the PC moves fine and then times out on every read. Same physics that put
// the torque loop on this box: FCI-class UDP never crosses the NAT.
//
// This daemon owns the Hand where the UDP can reach and re-exports a dumb
// TCP line protocol over the NAT-free direct link:
//
//   PC -> "CMD <epoch> MOVE <width_m> <speed_mps>\n" | "CMD <epoch> HOME\n"
//      | "CMD <epoch> GRASP <width_m> <speed_mps> <force_n> <eps_in_m> <eps_out_m>\n"
//      | "GSTOP\n" (cancel pending and actively stop a running move/grasp)
//   <-  "LIMITS <width_max_m> <speed_max_mps> <force_min_n> <force_max_n>\n"
//       (~10 Hz, first in every report, before STATE; a bridge that omits it
//       is the Franka Hand. FROZEN at four fields: clients refuse any other
//       count, so a new field goes on a new line)
//   <-  "STATE <width_m> <0|1>\n"   (~10 Hz, streams DURING moves too)
//   <-  "GDONE <0|1>\n"             (once per completed GRASP: the
//                                    franka::Gripper::grasp verdict — object
//                                    held within the epsilon band. A GRASP
//                                    killed by GSTOP sends nothing; the PC
//                                    client owns the timeout.)
//   <-  "META 2 <epoch> <available> <busy> <measured> <sample_seq>\n"
//   <-  "CAPS active_stop\n" (the older mailbox-only bridge omits this)
//   <-  "STOPPING <epoch>\n" (request admitted, NOT physical stop completion)
//   <-  "ACCEPT|REJECT <command_epoch> <reason>\n" | "DONE <command_epoch> <ok>\n"
// Epoch changes on every admission/cancellation/client or robot disconnect.
// Legacy STATE/GDONE framing is retained; unversioned actuator commands fail.
//
// ONE franka::Gripper session: the actor owns reads and actions; a separate
// worker calls its thread-safe stop() concurrently. A dual-session design
// was tried and is IMPOSSIBLE: the Hand
// server is single-client with NEWEST-WINS EVICTION — a second connect is
// "accepted" by starving the first session's UDP and then RST-ing its TCP
// (bench 2026-07-29: reader and actor evicted each other in a ping-pong,
// one reconnect per move; the earlier "accepts a second client" constructor
// test was measuring the eviction, not coexistence).
//
// Width still streams while move() blocks: the Hand travels at the
// commanded speed, so the serve loop SYNTHESIZES width from (start, goal,
// speed, t0) during an action and snaps to the first real readOnce after
// it. Synthetic width is a kinematic estimate — an early stall (jaws meet
// an object) shows as the goal until the post-move sample corrects it, so
// contact logic must key on real samples (grasp()/is_grasped), never on
// the mid-move stream.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <thread>

#include <csignal>
#include <cstdlib>
#include <functional>
#include <string>

#include "hand_device.hpp"
#include "hand_dm.hpp"
#ifdef HAND_WITH_FRANKA
#include <franka/exception.h>
#include <franka/gripper.h>
#endif

namespace {

double mono_s() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

// One action mailbox + shared Hand state, all under one mutex.
std::mutex mx;
std::condition_variable cv;
HandAdmission admission;
std::shared_ptr<HandDevice> hand_session; // protected by mx
HandLimits bridge_limits = franka_hand;   // the served gripper's envelope (set once, before serving)
uint64_t grasp_seq = 0;  // bumped per completed GRASP; serve() sends GDONE on change
uint64_t done_seq = 0, done_token = 0, sample_seq = 0;
bool done_ok = false;
bool grasp_ok = false;   // verdict of the grasp behind grasp_seq
bool have_real = false;  // last real sample valid
double real_t = 0.0;
uint64_t robot_sample_time = 0;
bool seen_robot_sample = false;
double real_w = 0.0;
bool real_grasped = false;
double act_from = 0.0, act_goal = 0.0, act_speed = 0.0, act_t0 = 0.0;

// Measurement-only endpoint allowance: real closed jaws report about -2 um.
// This 10 um envelope is not a vendor accuracy rating or a command allowance.
constexpr double width_measurement_tolerance_m = 1e-5;

// The OPEN end needs a much wider allowance than the closed end. Full stroke
// is whatever the homing sweep measured, and this Hand reports 80.76 mm there
// -- 760 um past the nominal 80 mm. The closed-end 10 um figure is endpoint
// noise and does not transfer. real_w is clamped to [0, .08] regardless, so
// accepting an overshoot costs no accuracy, whereas rejecting it withheld
// STATE for the whole session and rendered the fingers as CLOSED.
constexpr double width_open_tolerance_m = 2e-3;

// Called under mx. Only a new device timestamp can refresh measured freshness.
void observe_state(const HandSample& st) {
  if (admission.stopping) return; // Never attest freshness during stop dispatch.
  const auto stamp = st.stamp;
  if (seen_robot_sample && stamp <= robot_sample_time) return;
  if (!std::isfinite(st.width) || st.width < -width_measurement_tolerance_m ||
      st.width > bridge_limits.width_max_m + width_open_tolerance_m) {
    // Throttled, NOT gated on have_real: the first bad sample clears the
    // very flag such a gate would test, so a Hand that never once reports a
    // usable width stays completely silent -- which is the failure that is
    // hardest to diagnose, because STATE is withheld and nothing is logged.
    static double last_reject_print = -1e9;
    if (mono_s() - last_reject_print > 2.0) {
      last_reject_print = mono_s();
      std::printf("[hand] rejected measured width %.12g m at device time %llu ms"
                  " (no STATE is sent until a sample is usable)\n",
                  st.width, (unsigned long long)stamp);
    }
    have_real = false;
    return;
  }
  seen_robot_sample = true;
  robot_sample_time = stamp;
  have_real = true;
  real_t = mono_s();
  ++sample_seq;
  admission.sampled();
  real_w = std::clamp(st.width, 0.0, bridge_limits.width_max_m);
  real_grasped = st.grasped;
}

// GSTOP is STICKY: clearing the mailbox alone loses the race where the
// actor already drained the goal but has not executed it yet (it can be
// inside a reconnect backoff). The sequence number kills any action taken
// before the stop, wherever it is in the actor's pipeline. The stop worker
// interrupts blocking move/grasp on this SAME session (libfranka is thread-safe).
uint64_t gstop_pending() {
  std::lock_guard<std::mutex> lk(mx);
  admission.request_stop();
  have_real = false;
  cv.notify_all();
  return admission.epoch;
}

// Called with mx locked; drop it across the SDK call so neither TCP handling
// nor action completion waits on stop(). The caller pins the session lifetime.
template<class Stop>
void perform_hand_stop(std::unique_lock<std::mutex>& lk, Stop stop) {
  const auto serial = admission.stop_serial;
  const bool was_acting = admission.acting;
  lk.unlock();
  bool ok = false;
  try {
    ok = stop();
  } catch (const std::exception& e) {
    std::printf("[hand] stop failed: %s\n", e.what());
  }
  lk.lock();
  admission.finish_stop(serial, was_acting, ok);
  have_real = false;
  if (!admission.stopping) {
    std::printf("[hand] stop %s; fresh measured state required\n",
                ok ? "acknowledged" : "FAILED — commands inhibited");
  }
}

void stopper() {
  std::unique_lock<std::mutex> lk(mx);
  for (;;) {
    cv.wait(lk, [] { return admission.stopping; });
    const auto hand = hand_session; // Cannot be destroyed during stop().
    perform_hand_stop(lk, [&] { return hand && hand->stop(); });
    if (admission.stopping) {
      // Retry until canceled action exit, including the pre-SDK-entry race.
      // This is a retry interval, NOT a guaranteed physical stopping latency.
      cv.wait_for(lk, std::chrono::milliseconds(20));
    }
  }
}

void actor(std::function<std::shared_ptr<HandDevice>()> open_device, std::string label) {
  const char* robot_ip = label.c_str();
  std::shared_ptr<HandDevice> hand;
  double next_connect = 0.0;
  bool warned = false;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mx);
      cv.wait_for(lk, std::chrono::milliseconds(100),
                  [] { return admission.pending.kind != 0 && !admission.stopping; });
      // Never replace a session until concurrent stop calls have drained.
      if (!hand && admission.stopping) continue;
    }
    if (!hand) {
      if (mono_s() < next_connect) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      try {
        hand = open_device();
        std::printf("[hand] connected to %s\n", robot_ip);
        std::lock_guard<std::mutex> lk(mx);
        hand_session = hand;
        seen_robot_sample = false;
        warned = false;
        if (admission.stop_fault)
          std::printf("[hand] retrying the failed stop on the new session; "
                      "commands stay inhibited until it succeeds and a fresh "
                      "sample follows\n");
        admission.session_opened();
        cv.notify_all();
      } catch (const std::exception& e) {
        if (!warned) {
          warned = true;
          std::printf("[hand] no Hand at %s (%s) — retrying\n", robot_ip,
                      e.what());
        }
        next_connect = mono_s() + 2.0;
        continue; // No actions are admitted until a fresh read succeeds.
      }
    }
    HandAction a{};
    {
      std::lock_guard<std::mutex> lk(mx);
      admission.take(a);
    }
    if (a.kind != 0) {
      {
        // Re-check under the lock immediately before executing: a GSTOP
        // that landed while we were connecting outranks the carried action.
        std::lock_guard<std::mutex> lk(mx);
        if (!admission.begin_action(a)) {
          std::printf("[hand] action dropped (GSTOP outranks it)\n");
          admission.acting = false;
          admission.dispatch_committed = false;
          continue;
        }
        act_from = have_real ? real_w : a.w;
        act_goal = a.kind == 2 ? act_from : a.w;  // homing: no width model
        act_speed = a.kind == 2 ? 0.0 : a.s;
        act_t0 = mono_s();
      }
      const bool is_grasp = a.kind == 3;
      bool ok = false;
      try {
        if (a.kind == 1) {
          std::printf("[hand] move -> %.4f m @ %.2f m/s\n", a.w, a.s);
          ok = hand->move(a.w, a.s); // blocking; serve() synthesizes meanwhile
        } else if (a.kind == 2) {
          std::printf("[hand] homing\n");
          ok = hand->homing();
        } else if (a.kind == 3) {
          std::printf("[hand] grasp -> %.4f m @ %.2f m/s, %.1f N, eps %.3f/%.3f\n",
                      a.w, a.s, a.f, a.ei, a.eo);
          ok = hand->grasp(a.w, a.s, a.f, a.ei, a.eo);
          std::printf("[hand] grasp verdict: %s\n", ok ? "HELD" : "not held");
        }
      } catch (const DeviceLost& e) {
        std::printf("[hand] session lost (%s) — reconnecting\n", e.what());
        hand.reset();
        next_connect = mono_s() + 2.0;
      } catch (const ActionEnded& e) {
        // e.g. jaws met an obstacle. Session is fine.
        std::printf("[hand] action ended: %s\n", e.what());
      } catch (const std::exception& e) {
        // Anything else escaping a detached thread is std::terminate.
        std::printf("[hand] unexpected error (%s) — resetting\n", e.what());
        hand.reset();
        next_connect = mono_s() + 2.0;
      }
      std::lock_guard<std::mutex> lk(mx);
      admission.acting = false;
      admission.dispatch_committed = false;
      have_real = false; // Require a new measured sample before another action.
      if (!hand) {
        hand_session.reset();
        admission.connected = false;
        admission.cancel();
      }
      if (admission.current(a)) {
        done_token = a.token;
        done_ok = ok;
        ++done_seq;
      }
      if (is_grasp && admission.current(a)) {
        // Every executed GRASP reports — including exception paths, where the
        // verdict stays false: the PC client is blocked on this answer.
        grasp_ok = ok;
        ++grasp_seq;
      }
      continue;  // loop straight into a real readOnce to truth the width
    }
    // Idle: the actor remains the only readOnce caller. The stop worker may
    // concurrently call stop(), but never opens a second Hand session.
    uint64_t read_stop_serial;
    bool read_during_stop;
    {
      std::lock_guard<std::mutex> lk(mx);
      read_stop_serial = admission.stop_serial;
      read_during_stop = admission.stopping;
    }
    try {
      const HandSample st = hand->read();
      std::lock_guard<std::mutex> lk(mx);
      if (!read_during_stop && read_stop_serial == admission.stop_serial)
        observe_state(st);
    } catch (const std::exception& e) {
      std::printf("[hand] read failed (%s) — reconnecting\n", e.what());
      {
        std::lock_guard<std::mutex> lk(mx);
        have_real = false;
        admission.connected = false;
        admission.cancel();
        hand_session.reset();
      }
      hand.reset();
      next_connect = mono_s() + 2.0;
    }
  }
}

void serve(int fd) {
  // Disconnect interrupts outstanding work, not an already completed grasp.
  struct Disconnect {
    ~Disconnect() {
      std::lock_guard<std::mutex> lk(mx);
      admission.client = false;
      if (admission.dispatch_committed || admission.stopping) {
        admission.request_stop();
        have_real = false;
        cv.notify_all();
      } else {
        // An open may be queued or carried but not committed to SDK dispatch.
        // Invalidate it without stopping the existing grasp's holding force.
        admission.cancel();
      }
    }
  } disconnect;
  double last_state = 0.0;
  char buf[256];
  size_t have = 0;
  uint64_t seen_grasp;
  uint64_t seen_done;
  {
    // Results from before this client connected are nobody's to consume.
    std::lock_guard<std::mutex> lk(mx);
    seen_grasp = grasp_seq;
    seen_done = done_seq;
    admission.cancel();
    admission.client = true;
  }

  for (;;) {
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, 50);
    if (ready < 0) return;
    if (ready > 0) {
      const ssize_t got = recv(fd, buf + have, sizeof(buf) - have - 1, 0);
      if (got <= 0) return;  // client gone
      have += size_t(got);
      buf[have] = '\0';
      char* line = buf;
      for (char* nl; (nl = std::strchr(line, '\n')) != nullptr; line = nl + 1) {
        *nl = '\0';
        HandAction action;
        if (!std::strcmp(line, "GSTOP")) {
          const auto epoch = gstop_pending();
          char out[64];
          const int n = std::snprintf(out, sizeof(out), "STOPPING %llu\n",
                                     (unsigned long long)epoch);
          if (send(fd, out, size_t(n), MSG_NOSIGNAL) != n) return;
        } else {
          const char* error = "invalid_command";
          if (parse_hand_command(line, action, bridge_limits)) {
            std::lock_guard<std::mutex> lk(mx);
            error = admission.admit(action, have_real && mono_s() - real_t < .6);
          } else if (action.kind == 3 && !(bridge_limits.force_max_n > 0)) {
            error = "uncalibrated_force";  // no force grasp on this gripper (yet)
          }
          if (!error) cv.notify_all();
          char out[128];
          const int n = std::snprintf(out, sizeof(out), "%s %llu %s\n",
              error ? "REJECT" : "ACCEPT", (unsigned long long)action.token,
              error ? error : "accepted");
          if (send(fd, out, size_t(n), MSG_NOSIGNAL) != n) return;
        }
      }
      have = std::strlen(line);
      std::memmove(buf, line, have);
      if (have == sizeof(buf) - 1) {
        // A full buffer with no newline would make the next recv length 0 —
        // recv returns 0, and that reads as "client disconnected". Resync
        // instead of dropping a healthy session over one garbage line.
        std::printf("[hand] oversized line — disconnecting\n");
        return;
      }
    }

    {
      // GRASP verdicts push promptly (poll granularity), not on the state tick.
      uint64_t gs;
      bool gok;
      {
        std::lock_guard<std::mutex> lk(mx);
        gs = grasp_seq;
        gok = grasp_ok;
      }
      if (gs != seen_grasp) {
        seen_grasp = gs;
        char out[32];
        const int n = std::snprintf(out, sizeof(out), "GDONE %d\n", gok ? 1 : 0);
        if (send(fd, out, size_t(n), MSG_NOSIGNAL) != n) return;
      }
      uint64_t ds, token;
      bool ok;
      {
        std::lock_guard<std::mutex> lk(mx);
        ds = done_seq; token = done_token; ok = done_ok;
      }
      if (ds != seen_done) {
        seen_done = ds;
        char out[64];
        const int n = std::snprintf(out, sizeof(out), "DONE %llu %d\n",
            (unsigned long long)token, ok ? 1 : 0);
        if (send(fd, out, size_t(n), MSG_NOSIGNAL) != n) return;
      }
    }

    if (mono_s() - last_state >= 0.1) {
      last_state = mono_s();
      double w;
      bool grasped, valid;
      bool available, busy, measured;
      uint64_t epoch, seq;
      {
        std::lock_guard<std::mutex> lk(mx);
        valid = have_real;
        available = admission.connected && !admission.stop_fault;
        busy = admission.acting || admission.pending.kind || admission.stopping;
        measured = available && have_real && mono_s() - real_t < .6 && !busy;
        epoch = admission.epoch;
        seq = sample_seq;
        grasped = real_grasped;
        if (admission.acting && act_speed > 0.0) {
          // Mid-move: the Hand travels at the commanded speed — integrate
          // toward the goal. Corrected by the first real post-move sample.
          const double travelled = act_speed * (mono_s() - act_t0);
          const double dist = std::fabs(act_goal - act_from);
          const double frac = dist > 1e-9 ? std::fmin(travelled / dist, 1.0) : 1.0;
          w = act_from + (act_goal - act_from) * frac;
          valid = true;
        } else {
          w = real_w;
        }
      }
      // The envelope first: a client validates STATE against it, and one that
      // has not seen it assumes the Franka Hand's.
      char limits[96];
      const int nl = std::snprintf(limits, sizeof(limits), "LIMITS %.5f %.5f %.3f %.3f\n",
          bridge_limits.width_max_m, bridge_limits.speed_max_mps, bridge_limits.force_min_n,
          bridge_limits.force_max_n);
      if (send(fd, limits, size_t(nl), MSG_NOSIGNAL) != nl) return;
      if (valid) {
        char out[64];
        const int n = std::snprintf(out, sizeof(out), "STATE %.5f %d\n", w,
                                    grasped ? 1 : 0);
        if (send(fd, out, size_t(n), MSG_NOSIGNAL) != n) return;
      }
      char meta[160];
      const int n = std::snprintf(meta, sizeof(meta), "CAPS active_stop\nMETA 2 %llu %d %d %d %llu\n",
          (unsigned long long)epoch, available ? 1 : 0, busy ? 1 : 0,
          measured ? 1 : 0, (unsigned long long)seq);
      if (send(fd, meta, size_t(n), MSG_NOSIGNAL) != n) return;
    }
  }
}

} // namespace

#ifdef HAND_WITH_FRANKA
// The Franka Hand: libfranka's Gripper, its exceptions mapped onto the seam.
class FrankaHand final : public HandDevice {
public:
  explicit FrankaHand(const char* ip) : g_(open(ip)) {}
  HandSample read() override {
    try {
      const franka::GripperState st = g_.readOnce();
      return {st.width, st.is_grasped, uint64_t(st.time.toMSec())};
    } catch (const franka::Exception& e) {
      throw DeviceLost(e.what());
    }
  }
  bool move(double w, double s) override { return act([&] { return g_.move(w, s); }); }
  bool homing() override { return act([&] { return g_.homing(); }); }
  bool grasp(double w, double s, double f, double ei, double eo) override {
    return act([&] { return g_.grasp(w, s, f, ei, eo); });
  }
  bool stop() override { return g_.stop(); }  // thread-safe; interrupts move/grasp

private:
  static franka::Gripper open(const char* ip) {
    try {
      return franka::Gripper(ip);
    } catch (const franka::Exception& e) {
      throw DeviceLost(e.what());
    }
  }
  template <class F> static bool act(F f) {
    try {
      return f();
    } catch (const franka::NetworkException& e) {
      throw DeviceLost(e.what());
    } catch (const franka::Exception& e) {
      throw ActionEnded(e.what());  // CommandException: the session is fine
    }
  }
  franka::Gripper g_;
};
#endif

// One served device, so SIGTERM/SIGINT can release it (DM: zero torque, then disable).
std::function<void()> release_device = [] {};

const char* const kFlags[] = {"--backend", "--port", "--bind", "--robot-ip", "--can-if", "--motor", "--mimic",
                              "--kp", "--kd", "--tau-cap", "--speed-max"};

// Flag mode: every argument is a known flag followed by its value, each flag at most once; a typo must not fall back
// to a default (a misspelt --bind would expose the jaws on every interface).
void check_flags(int argc, char** argv) {
  std::vector<std::string> seen;
  for (int i = 1; i < argc; i += 2) {
    const bool known = std::any_of(std::begin(kFlags), std::end(kFlags),
                                   [&](const char* f) { return !std::strcmp(argv[i], f); });
    if (!known || i + 1 >= argc || !std::strncmp(argv[i + 1], "--", 2) ||
        std::find(seen.begin(), seen.end(), argv[i]) != seen.end()) {
      std::fprintf(stderr, "[hand] bad argument '%s': flags are --name value, each known flag once\n", argv[i]);
      std::exit(2);
    }
    seen.push_back(argv[i]);
  }
}

bool flag(int argc, char** argv, const char* name, std::string& out) {
  for (int i = 1; i + 1 < argc; i += 2)
    if (!std::strcmp(argv[i], name)) {
      out = argv[i + 1];
      return true;
    }
  return false;
}

std::vector<std::string> split(const std::string& text, char sep) {
  std::vector<std::string> out;
  for (size_t at = 0;;) {
    const size_t end = text.find(sep, at);
    out.push_back(text.substr(at, end - at));
    if (end == std::string::npos) return out;
    at = end + 1;
  }
}

double number(const std::string& text, const char* what) {
  char* end = nullptr;
  const double v = std::strtod(text.c_str(), &end);
  if (text.empty() || *end || !std::isfinite(v)) {
    std::fprintf(stderr, "[hand] %s must be a number, got '%s'\n", what, text.c_str());
    std::exit(2);
  }
  return v;
}

// "--motor ID:TYPE:MST" and "--mimic OPEN,CLOSED,STROKE" into a DM gripper config; exits on error.
DmGripperConfig dm_config(int argc, char** argv, std::string& iface, uint32_t& reply_id) {
  std::string motor, mimic, kp, kd, cap, speed;
  for (auto [name, out] : {std::pair{"--can-if", &iface}, {"--motor", &motor}, {"--mimic", &mimic},
                           {"--kp", &kp}, {"--kd", &kd}, {"--tau-cap", &cap}, {"--speed-max", &speed}})
    if (!flag(argc, argv, name, *out)) {
      std::fprintf(stderr, "[hand] --backend dm needs %s (nothing is assumed)\n", name);
      std::exit(2);
    }
  DmGripperConfig c;
  const std::vector<std::string> m = split(motor, ':');
  if (m.size() != 3 || !arm_rt::limits_for(m[1], c.lim)) {
    std::fprintf(stderr, "[hand] --motor ID:TYPE:MST (TYPE 4310|4310p|4340|4340p; MST = the reply CAN id)\n");
    std::exit(2);
  }
  const double id = number(m[0], "motor id"), mst = number(m[2], "Master ID");
  if (id != std::floor(id) || id < 1 || id > 15 || mst != std::floor(mst) || mst < 0 || mst > 0x7FF) {
    std::fprintf(stderr, "[hand] motor id must be an integer 1..15 and Master ID 0..0x7FF\n");
    std::exit(2);
  }
  c.id = int(id);
  reply_id = uint32_t(mst);
  const std::vector<std::string> v = split(mimic, ',');
  if (v.size() != 3) {
    std::fprintf(stderr, "[hand] --mimic OPEN_RAD,CLOSED_RAD,FINGER_STROKE_M\n");
    std::exit(2);
  }
  c.motor_open = number(v[0], "--mimic OPEN_RAD");
  c.motor_closed = number(v[1], "--mimic CLOSED_RAD");
  c.finger_stroke_m = number(v[2], "--mimic FINGER_STROKE_M");
  c.kp = number(kp, "--kp");
  c.kd = number(kd, "--kd");
  c.tau_cap = number(cap, "--tau-cap");
  c.speed_max_mps = number(speed, "--speed-max");
  if (const char* why = c.invalid()) {
    std::fprintf(stderr, "[hand] %s\n", why);
    std::exit(2);
  }
  return c;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // journald/file logs must not sit in a 4K buffer
  std::string backend = "franka", port_text = "47802", bind_text, robot_text = "172.16.0.3";
  const bool flags = argc > 1 && !std::strncmp(argv[1], "--", 2);
  if (flags) {
    check_flags(argc, argv);
    if (!flag(argc, argv, "--backend", backend)) {
      std::fprintf(stderr, "[hand] flag mode needs --backend franka|dm\n");
      return 2;
    }
    flag(argc, argv, "--port", port_text);
    flag(argc, argv, "--bind", bind_text);
    flag(argc, argv, "--robot-ip", robot_text);
  } else {  // the deployed unit's positional form: robot_ip [port [bind_ip]]
    if (argc > 1) robot_text = argv[1];
    if (argc > 2) port_text = argv[2];
    if (argc > 3) bind_text = argv[3];
  }
  const long port_arg = std::atol(port_text.c_str());
  if (port_arg < 1 || port_arg > 65535) {
    std::fprintf(stderr, "[hand] port must be 1..65535\n");
    return 2;
  }
  const uint16_t port = uint16_t(port_arg);
  // Optional bind address: this box is dual-NIC and INADDR_ANY answers on
  // the robot LAN too — anything routable there could drive the jaws.
  in_addr bind_ip{};
  bind_ip.s_addr = INADDR_ANY;
  if (!bind_text.empty() && inet_pton(AF_INET, bind_text.c_str(), &bind_ip) != 1) {
    std::fprintf(stderr, "[hand] bind arg must be a dotted IPv4 address\n");
    return 2;
  }

  std::function<std::shared_ptr<HandDevice>()> open_device;
  std::string label;
  if (backend == "dm") {
    std::string iface;
    uint32_t reply_id = 0;
    const DmGripperConfig cfg = dm_config(argc, argv, iface, reply_id);
    bridge_limits = cfg.limits();
    label = iface + " motor " + std::to_string(cfg.id);
    open_device = [cfg, iface, reply_id] {
      return std::make_shared<DmGripper>(cfg, std::make_unique<SocketCanPort>(iface, reply_id));
    };
    release_device = [] {
      std::shared_ptr<HandDevice> hand;
      {
        std::lock_guard<std::mutex> lk(mx);
        hand = hand_session;
      }
      if (auto dm = std::dynamic_pointer_cast<DmGripper>(hand)) dm->shutdown();
    };
  } else if (backend == "franka") {
#ifdef HAND_WITH_FRANKA
    label = robot_text;
    open_device = [robot_text] { return std::make_shared<FrankaHand>(robot_text.c_str()); };
#else
    std::fprintf(stderr, "[hand] built without libfranka: only --backend dm\n");
    return 2;
#endif
  } else {
    std::fprintf(stderr, "[hand] --backend franka|dm\n");
    return 2;
  }

  // Signals go to one waiting thread (blocked everywhere else), which releases the device and exits.
  sigset_t stop_signals;
  sigemptyset(&stop_signals);
  sigaddset(&stop_signals, SIGTERM);
  sigaddset(&stop_signals, SIGINT);
  pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);

  const int lfd = socket(AF_INET, SOCK_STREAM, 0);
  const int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr = bind_ip;
  addr.sin_port = htons(port);
  if (bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(lfd, 1) != 0) {
    std::perror("[hand] bind/listen");
    return 1;
  }
  std::printf("[hand] hand_bridge up: %s %s, port %u, limits %.5f m %.5f m/s %.1f-%.1f N\n",
              backend.c_str(), label.c_str(), port, bridge_limits.width_max_m,
              bridge_limits.speed_max_mps, bridge_limits.force_min_n, bridge_limits.force_max_n);
  std::thread([stop_signals] {
    int sig = 0;
    sigwait(&stop_signals, &sig);
    std::printf("[hand] signal %d: releasing the gripper\n", sig);
    release_device();
    std::_Exit(0);
  }).detach();
  std::thread(stopper).detach();
  std::thread(actor, open_device, label).detach();
  for (;;) {
    const int client = accept(lfd, nullptr, nullptr);
    if (client < 0) continue;
    const int nd = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
    const timeval send_timeout{0, 200000};
    setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
    std::printf("[hand] client connected\n");
    serve(client);
    close(client);
    std::printf("[hand] client disconnected\n");
  }
}
