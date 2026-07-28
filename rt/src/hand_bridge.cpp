// hand_bridge — Franka Hand ownership ON the robot LAN (the RT box).
//
// The Hand mirrors the robot's protocol split: commands over TCP, cyclic
// state PUSHED over UDP. Through the PC-side NAT those pushed datagrams die
// (a server-initiated UDP flow has no conntrack entry, and the advertised
// return port is ephemeral — unforwardable), so any pylibfranka Gripper on
// the PC moves fine and then times out on every read. Same physics that put
// the torque loop on this box: FCI-class UDP never crosses the NAT.
//
// This daemon owns franka::Gripper where the UDP can reach and re-exports a
// dumb TCP line protocol over the NAT-free direct link:
//
//   PC -> "MOVE <width_m> <speed_mps>\n" | "HOME\n" | "GSTOP\n"
//   <-  "STATE <width_m> <0|1>\n"   (~5 Hz; paused during blocking actions)
//
// Single-threaded ON PURPOSE: one strictly sequential session with the Hand
// — the interleaved-read/move corruption that killed the PC-side attempt
// cannot exist here. Not RT: the Hand is an action device, seconds-scale.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>

#include <franka/exception.h>
#include <franka/gripper.h>

namespace {

double mono_s() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

void serve(int fd, const char* robot_ip) {
  std::unique_ptr<franka::Gripper> hand;
  double last_state = 0.0;
  double next_connect = 0.0;
  char buf[256];
  size_t have = 0;
  bool warned = false;

  for (;;) {
    if (!hand && mono_s() >= next_connect) {
      try {
        hand = std::make_unique<franka::Gripper>(robot_ip);
        std::printf("[hand] connected to %s\n", robot_ip);
        warned = false;
      } catch (const franka::Exception& e) {
        if (!warned) {
          warned = true;
          std::printf("[hand] no Hand at %s (%s) — retrying\n", robot_ip, e.what());
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
        try {
          if (std::sscanf(line, "MOVE %lf %lf", &w, &s) == 2 && hand) {
            hand->move(w, s);  // blocking, seconds — state pauses, by design
          } else if (!std::strcmp(line, "HOME") && hand) {
            std::printf("[hand] homing\n");
            hand->homing();
          } else if (!std::strcmp(line, "GSTOP") && hand) {
            hand->stop();
          }
        } catch (const franka::Exception& e) {
          std::printf("[hand] action failed (%s) — reconnecting\n", e.what());
          hand.reset();
          next_connect = mono_s() + 2.0;
        }
      }
      have = std::strlen(line);
      std::memmove(buf, line, have);
    }

    if (hand && mono_s() - last_state >= 0.2) {
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
  const char* robot_ip = argc > 1 ? argv[1] : "172.16.0.3";
  const uint16_t port = argc > 2 ? uint16_t(std::atoi(argv[2])) : 47802;

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
