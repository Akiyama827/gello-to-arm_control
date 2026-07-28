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
//   PC -> "MOVE <width_m> <speed_mps>\n" | "HOME\n" | "GSTOP\n"
//   <-  "STATE <width_m> <0|1>\n"   (~10 Hz, streams DURING actions too)
//
// TWO franka::Gripper sessions on purpose (the Hand server accepts
// concurrent clients — verified on the bench 2026-07-28):
//   - reader: owned by the serve loop, readOnce only. Never blocks on an
//     action, so width keeps streaming WHILE the jaws travel — with the old
//     single session the state froze exactly when the truth was changing
//     (move() blocks for the full travel) and every ghost snapped to the new
//     width only after the command finished.
//   - actor: owned by a worker thread, executes blocking move()/homing()
//     from a latest-wins mailbox (each move is a full physical travel;
//     replaying queued intermediate widths is the bench 'gripper lag').
// Each session reconnects independently; a dead Hand never kills the daemon.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <thread>

#include <franka/exception.h>
#include <franka/gripper.h>

namespace {

double mono_s() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

// Latest-wins action mailbox: the actor drains to the newest goal only.
struct Action {
  int kind = 0;  // 0 = none, 1 = MOVE, 2 = HOME
  double w = 0, s = 0;
};
std::mutex act_mx;
std::condition_variable act_cv;
Action act_pending;

void post(int kind, double w = 0, double s = 0) {
  {
    std::lock_guard<std::mutex> lk(act_mx);
    act_pending = {kind, w, s};
  }
  act_cv.notify_one();
}

void clear_pending() {
  std::lock_guard<std::mutex> lk(act_mx);
  act_pending = {};
}

void actor(const char* robot_ip) {
  std::unique_ptr<franka::Gripper> hand;
  Action a{};
  bool warned = false;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(act_mx);
      if (act_pending.kind != 0) {
        a = act_pending;  // a newer goal replaces a retry-pending one
        act_pending = {};
      } else if (a.kind == 0) {
        act_cv.wait(lk, [] { return act_pending.kind != 0; });
        a = act_pending;
        act_pending = {};
      }
    }
    if (!hand) {
      try {
        hand = std::make_unique<franka::Gripper>(robot_ip);
        std::printf("[hand] cmd session connected to %s\n", robot_ip);
        warned = false;
      } catch (const franka::Exception& e) {
        if (!warned) {
          warned = true;
          std::printf("[hand] cmd session: no Hand at %s (%s) — retrying\n",
                      robot_ip, e.what());
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;  // `a` stays pending; a newer goal may replace it above
      }
    }
    try {
      if (a.kind == 1) {
        std::printf("[hand] move -> %.4f m @ %.2f m/s\n", a.w, a.s);
        hand->move(a.w, a.s);  // blocking; reader keeps streaming meanwhile
      } else if (a.kind == 2) {
        std::printf("[hand] homing\n");
        hand->homing();
      }
      a = {};
    } catch (const franka::NetworkException& e) {
      std::printf("[hand] cmd session lost (%s) — reconnecting\n", e.what());
      hand.reset();
      a = {};  // never retry-loop a poisoned action
    } catch (const franka::Exception& e) {
      // CommandException — e.g. a GSTOP aborted the move. Session is fine.
      std::printf("[hand] action ended: %s\n", e.what());
      a = {};
    }
  }
}

void serve(int fd, const char* robot_ip) {
  std::unique_ptr<franka::Gripper> hand;  // READER session: readOnce only
  double last_state = 0.0;
  double next_connect = 0.0;
  char buf[256];
  size_t have = 0;
  bool warned = false;

  for (;;) {
    if (!hand && mono_s() >= next_connect) {
      try {
        hand = std::make_unique<franka::Gripper>(robot_ip);
        std::printf("[hand] read session connected to %s\n", robot_ip);
        warned = false;
      } catch (const franka::Exception& e) {
        if (!warned) {
          warned = true;
          std::printf("[hand] read session: no Hand at %s (%s) — retrying\n",
                      robot_ip, e.what());
        }
        next_connect = mono_s() + 2.0;
      }
    }

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
        double w = 0, s = 0;
        if (std::sscanf(line, "MOVE %lf %lf", &w, &s) == 2) {
          post(1, w, s);
        } else if (!std::strcmp(line, "HOME")) {
          post(2);
        } else if (!std::strcmp(line, "GSTOP")) {
          clear_pending();
          if (hand) {
            try {
              // Cross-session stop: aborts the actor's in-flight move (the
              // actor logs the resulting CommandException and moves on).
              hand->stop();
            } catch (const franka::Exception& e) {
              std::printf("[hand] stop failed (%s) — resetting read session\n",
                          e.what());
              hand.reset();
              next_connect = mono_s() + 2.0;
            }
          }
        }
      }
      have = std::strlen(line);
      std::memmove(buf, line, have);
    }

    if (hand && mono_s() - last_state >= 0.1) {
      last_state = mono_s();
      try {
        const franka::GripperState st = hand->readOnce();
        char out[64];
        const int n = std::snprintf(out, sizeof(out), "STATE %.5f %d\n",
                                    st.width, st.is_grasped ? 1 : 0);
        if (send(fd, out, size_t(n), MSG_NOSIGNAL) != n) return;
      } catch (const franka::Exception& e) {
        std::printf("[hand] read failed (%s) — reconnecting\n", e.what());
        hand.reset();
        next_connect = mono_s() + 2.0;
      }
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // journald/file logs must not sit in a 4K buffer
  const char* robot_ip = argc > 1 ? argv[1] : "172.16.0.3";
  const uint16_t port = argc > 2 ? uint16_t(std::atoi(argv[2])) : 47802;

  std::thread(actor, robot_ip).detach();

  const int lfd = socket(AF_INET, SOCK_STREAM, 0);
  const int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
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
    std::printf("[hand] client connected\n");
    serve(client, robot_ip);
    close(client);
    std::printf("[hand] client disconnected\n");
  }
}
