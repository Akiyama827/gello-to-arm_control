// Exercise production serve() with local sockets, never create a Gripper.
#define main hand_bridge_daemon_main
#include "hand_bridge.cpp"
#undef main
#include <cassert>
#include <limits>
#include <string>

int main() {
  franka::GripperState sample{};
  sample.width = .04;
  sample.time = franka::Duration(10);
  observe_state(sample);
  const auto first_seq = sample_seq;
  const auto first_time = real_t;
  observe_state(sample);
  assert(sample_seq == first_seq && real_t == first_time);
  sample.time = franka::Duration(11);
  observe_state(sample);
  assert(sample_seq == first_seq + 1);
  // Actual closed-Hand observation: -1.97 um is endpoint noise, not stale data.
  uint64_t stamp = 12;
  for (double width : {-1.97000008484e-6, -1e-5, .080002, .08001}) {
    sample.width = width;
    sample.time = franka::Duration(stamp++);
    const auto seq = sample_seq;
    observe_state(sample);
    assert(have_real && sample_seq == seq + 1);
    assert(real_w == (width < 0 ? 0 : .08));
    observe_state(sample);
    assert(sample_seq == seq + 1); // Normalization must not refresh duplicates.
  }
  for (double width : {-1.01e-5, .0800101,
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
    sample.width = width;
    sample.time = franka::Duration(stamp++);
    const auto seq = sample_seq;
    const auto observed_at = real_t;
    observe_state(sample);
    assert(!have_real && sample_seq == seq && real_t == observed_at);
  }
  HandAction command;
  assert(!parse_hand_command("CMD 1 MOVE -0.000002 0.05", command));
  assert(!parse_hand_command("CMD 1 MOVE 0.080002 0.05", command));
  int pair[2];
  assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
  {
    std::lock_guard<std::mutex> lock(mx);
    admission.connected = true;
    have_real = true;
    real_t = mono_s();
    real_w = .075;
    sample_seq = 7;
  }
  std::thread server([&] { serve(pair[1]); close(pair[1]); });
  auto until = [&](const std::string& expected) {
    std::string response;
    for (int i = 0; i < 30; ++i) {
      pollfd event{pair[0], POLLIN, 0};
      if (poll(&event, 1, 50) > 0) {
        char buf[512];
        const auto got = recv(pair[0], buf, sizeof(buf), 0);
        assert(got > 0);
        response.append(buf, size_t(got));
        if (response.find(expected) != std::string::npos) return;
      }
    }
    assert(false && "bridge response timeout");
  };
  auto send_line = [&](const std::string& line) {
    assert(send(pair[0], line.data(), line.size(), MSG_NOSIGNAL) == ssize_t(line.size()));
  };
  until("META 2");
  uint64_t epoch;
  { std::lock_guard<std::mutex> lock(mx); epoch = admission.epoch; }
  send_line("CMD " + std::to_string(epoch) + " MOVE nan 0.05\n");
  until("REJECT");
  { std::lock_guard<std::mutex> lock(mx); assert(admission.pending.kind == 0); }
  send_line("CMD " + std::to_string(epoch) + " GRASP 0.04 0.05 55 0.02 0.02\n");
  until("ACCEPT");
  { std::lock_guard<std::mutex> lock(mx); assert(admission.pending.f == 55); }
  send_line("CMD " + std::to_string(epoch) + " MOVE 0.08 0.05\n");
  until("REJECT");
  close(pair[0]);
  server.join();
  {
    std::lock_guard<std::mutex> lock(mx);
    assert(!admission.client && !admission.pending.kind && admission.epoch > epoch);
  }
  std::puts("PASS production Hand serve invalid rejection, admission, disconnect cancellation");
}
