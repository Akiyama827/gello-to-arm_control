// hand_bridge — Franka Hand ownership ON the robot LAN (the RT box).
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
//      | "GSTOP\n" (cancel pending; cannot interrupt a blocking SDK action)
//   <-  "STATE <width_m> <0|1>\n"   (~10 Hz, streams DURING moves too)
//   <-  "GDONE <0|1>\n"             (once per completed GRASP: the
//                                    franka::Gripper::grasp verdict — object
//                                    held within the epsilon band. A GRASP
//                                    killed by GSTOP sends nothing; the PC
//                                    client owns the timeout.)
//   <-  "META 2 <epoch> <available> <busy> <measured> <sample_seq>\n"
//   <-  "ACCEPT|REJECT <command_epoch> <reason>\n" | "DONE <command_epoch> <ok>\n"
// Epoch changes on every admission/cancellation/client or robot disconnect.
// Legacy STATE/GDONE framing is retained; unversioned actuator commands fail.
//
// ONE franka::Gripper session, owned entirely by the actor thread (reads AND
// actions). A dual-session design was tried and is IMPOSSIBLE: the Hand
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

#include <franka/exception.h>
#include <franka/gripper.h>
#include "hand_admission.hpp"

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

// Called under mx. Only a new device timestamp can refresh measured freshness.
void observe_state(const franka::GripperState& st) {
  const auto stamp = st.time.toMSec();
  if (seen_robot_sample && stamp <= robot_sample_time) return;
  if (!std::isfinite(st.width) || st.width < -width_measurement_tolerance_m ||
      st.width > .08 + width_measurement_tolerance_m) {
    if (have_real) {
      std::printf("[hand] rejected measured width %.12g m at device time %llu ms\n",
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
  real_w = std::clamp(st.width, 0.0, .08);
  real_grasped = st.is_grasped;
}

// GSTOP is STICKY: clearing the mailbox alone loses the race where the
// actor already drained the goal but has not executed it yet (it can be
// inside a reconnect backoff). The sequence number kills any action taken
// before the stop, wherever it is in the actor's pipeline. A single session
// cannot abort its OWN blocking move/grasp. Cancellation prevents replay;
// it does not promise an immediate physical stop of the blocking SDK call.
void gstop_pending() {
  std::lock_guard<std::mutex> lk(mx);
  admission.cancel();
}

void actor(const char* robot_ip) {
  std::unique_ptr<franka::Gripper> hand;
  double next_connect = 0.0;
  bool warned = false;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mx);
      cv.wait_for(lk, std::chrono::milliseconds(100),
                  [] { return admission.pending.kind != 0; });
    }
    if (!hand) {
      if (mono_s() < next_connect) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      try {
        hand = std::make_unique<franka::Gripper>(robot_ip);
        std::printf("[hand] connected to %s\n", robot_ip);
        std::lock_guard<std::mutex> lk(mx);
        admission.connected = true;
        seen_robot_sample = false;
        warned = false;
      } catch (const franka::Exception& e) {
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
        if (!admission.current(a)) {
          std::printf("[hand] action dropped (GSTOP outranks it)\n");
          admission.acting = false;
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
      } catch (const franka::NetworkException& e) {
        std::printf("[hand] session lost (%s) — reconnecting\n", e.what());
        hand.reset();
        next_connect = mono_s() + 2.0;
      } catch (const franka::Exception& e) {
        // CommandException — e.g. jaws met an obstacle. Session is fine.
        std::printf("[hand] action ended: %s\n", e.what());
      } catch (const std::exception& e) {
        // Anything else escaping a detached thread is std::terminate.
        std::printf("[hand] unexpected error (%s) — resetting\n", e.what());
        hand.reset();
        next_connect = mono_s() + 2.0;
      }
      std::lock_guard<std::mutex> lk(mx);
      admission.acting = false;
      have_real = false; // Require a new measured sample before another action.
      if (!hand) {
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
    // Idle: refresh the real sample (same thread, same session — the ONLY
    // Hand access; the serve loop never touches the session).
    try {
      const franka::GripperState st = hand->readOnce();
      std::lock_guard<std::mutex> lk(mx);
      observe_state(st);
    } catch (const franka::Exception& e) {
      std::printf("[hand] read failed (%s) — reconnecting\n", e.what());
      {
        std::lock_guard<std::mutex> lk(mx);
        have_real = false;
        admission.connected = false;
        admission.cancel();
      }
      hand.reset();
      next_connect = mono_s() + 2.0;
    }
  }
}

void serve(int fd) {
  // Every return cancels pending/carried work; a blocking action may finish.
  struct Disconnect {
    ~Disconnect() {
      std::lock_guard<std::mutex> lk(mx);
      admission.client = false;
      admission.cancel();
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
          gstop_pending();
        } else {
          const char* error = "invalid_command";
          if (parse_hand_command(line, action)) {
            std::lock_guard<std::mutex> lk(mx);
            error = admission.admit(action, have_real && mono_s() - real_t < .6);
          }
          if (!error) cv.notify_one();
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
        available = admission.connected;
        busy = admission.acting || admission.pending.kind;
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
      if (valid) {
        char out[64];
        const int n = std::snprintf(out, sizeof(out), "STATE %.5f %d\n", w,
                                    grasped ? 1 : 0);
        if (send(fd, out, size_t(n), MSG_NOSIGNAL) != n) return;
      }
      char meta[160];
      const int n = std::snprintf(meta, sizeof(meta), "META 2 %llu %d %d %d %llu\n",
          (unsigned long long)epoch, available ? 1 : 0, busy ? 1 : 0,
          measured ? 1 : 0, (unsigned long long)seq);
      if (send(fd, meta, size_t(n), MSG_NOSIGNAL) != n) return;
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // journald/file logs must not sit in a 4K buffer
  const char* robot_ip = argc > 1 ? argv[1] : "172.16.0.3";
  const long port_arg = argc > 2 ? std::atol(argv[2]) : 47802;
  if (port_arg < 1 || port_arg > 65535) {
    std::fprintf(stderr, "[hand] port must be 1..65535\n");
    return 2;
  }
  const uint16_t port = uint16_t(port_arg);
  // Optional bind address: this box is dual-NIC and INADDR_ANY answers on
  // the robot LAN too — anything routable there could drive the jaws.
  in_addr bind_ip{};
  bind_ip.s_addr = INADDR_ANY;
  if (argc > 3 && inet_pton(AF_INET, argv[3], &bind_ip) != 1) {
    std::fprintf(stderr, "[hand] bind arg must be a dotted IPv4 address\n");
    return 2;
  }

  std::thread(actor, robot_ip).detach();

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
  std::printf("[hand] hand_bridge up: robot %s, port %u\n", robot_ip, port);
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
