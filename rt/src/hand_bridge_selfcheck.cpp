// Exercise production serve() with local sockets, never create a Gripper.
#define main hand_bridge_daemon_main
#include "hand_bridge.cpp"
#undef main
#include <cassert>
#include <future>
#include <limits>
#include <string>

void check_stop_dispatch() {
  using namespace std::chrono_literals;
  admission = {};
  admission.client = admission.connected = true;
  admission.acting = true; // Action taken, but has not entered the SDK yet.
  gstop_pending();
  {
    std::unique_lock<std::mutex> lock(mx);
    perform_hand_stop(lock, [] { return true; }); // Too early to interrupt.
    assert(admission.stopping && admission.acting);
  }
  std::promise<void> entered, released;
  auto release = released.get_future();
  auto sdk_move = std::async(std::launch::async, [&] {
    entered.set_value();
    assert(release.wait_for(2s) == std::future_status::ready);
    std::lock_guard<std::mutex> lock(mx);
    admission.acting = false;
  });
  assert(entered.get_future().wait_for(2s) == std::future_status::ready);
  {
    std::unique_lock<std::mutex> lock(mx);
    perform_hand_stop(lock, [&] {
      released.set_value();
      // Action completion must acquire mx while stop() is outstanding.
      assert(sdk_move.wait_for(2s) == std::future_status::ready);
      sdk_move.get();
      return true;
    });
    assert(admission.stopping && !admission.acting);
    perform_hand_stop(lock, [] { return false; });
    assert(!admission.stopping && admission.stop_fault && !have_real);
  }
  gstop_pending();
  {
    std::unique_lock<std::mutex> lock(mx);
    perform_hand_stop(lock, []() -> bool { throw std::runtime_error("fake SDK failure"); });
    assert(admission.stop_fault);
  }
  gstop_pending();
  std::promise<void> stop_entered, stop_release;
  auto finish = stop_release.get_future();
  auto sdk_stop = std::async(std::launch::async, [&] {
    std::unique_lock<std::mutex> lock(mx);
    perform_hand_stop(lock, [&] {
      stop_entered.set_value();
      assert(finish.wait_for(2s) == std::future_status::ready);
      return true;
    });
  });
  assert(stop_entered.get_future().wait_for(2s) == std::future_status::ready);
  {
    std::lock_guard<std::mutex> lock(mx); // A slow stop must not block TCP admission.
    HandAction action;
    assert(parse_hand_command("CMD 1 MOVE 0.04 0.05", action));
    action.token = admission.epoch;
    assert(admission.admit(action, true) != nullptr);
  }
  stop_release.set_value();
  assert(sdk_stop.wait_for(2s) == std::future_status::ready);
  sdk_stop.get();
  assert(!admission.stopping && !admission.stop_fault && !have_real);
  admission = {};
  std::puts("PASS production stop dispatch: active/pre-entry race, unlocked SDK, failure latch");
}

int main() {
  check_stop_dispatch();
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
  admission.client = admission.connected = true;
  gstop_pending();
  assert(parse_hand_command("CMD 1 MOVE 0.04 0.05", command));
  command.token = admission.epoch;
  assert(admission.admit(command, true) != nullptr &&
         "GSTOP must block new motion until physical stopping completes");
  admission = {};
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
  HandAction active;
  {
    std::lock_guard<std::mutex> lock(mx);
    assert(admission.take(active) && admission.begin_action(active));
  }
  send_line("GSTOP\n");
  until("STOPPING");
  {
    std::unique_lock<std::mutex> lock(mx);
    assert(admission.stopping && !admission.current(active) && !have_real);
    perform_hand_stop(lock, [] { return true; });
    assert(admission.stopping);
    admission.acting = false;
    admission.dispatch_committed = false;
    perform_hand_stop(lock, [] { return true; });
    assert(!have_real);
    sample.width = .06;
    sample.time = franka::Duration(stamp++);
    observe_state(sample);
    assert(have_real);
    epoch = admission.epoch;
  }
  send_line("CMD " + std::to_string(epoch) + " MOVE 0.075 0.05\n");
  until("ACCEPT");
  {
    std::lock_guard<std::mutex> lock(mx);
    assert(admission.take(active) && admission.begin_action(active));
  }
  close(pair[0]);
  server.join();
  {
    std::lock_guard<std::mutex> lock(mx);
    assert(!admission.client && !admission.pending.kind && admission.epoch > epoch);
    assert(admission.stopping && !admission.current(active));
  }
  // A completed held grasp is not implicitly released on client disconnect.
  for (bool carried : {false, true}) {
    admission = {};
    admission.connected = true;
    real_grasped = true;
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    std::thread idle_server([&] { serve(pair[1]); close(pair[1]); });
    until("CAPS active_stop");
    {
      std::lock_guard<std::mutex> lock(mx);
      have_real = true;
      real_t = mono_s();
      epoch = admission.epoch;
    }
    send_line("CMD " + std::to_string(epoch) + " MOVE 0.075 0.05\n");
    until("ACCEPT"); // Queued: disconnect must not release the held object.
    if (carried) {
      std::lock_guard<std::mutex> lock(mx);
      assert(admission.take(active)); // Final SDK dispatch not committed.
    }
    close(pair[0]);
    idle_server.join();
    assert(!admission.stopping && real_grasped);
    if (carried) assert(!admission.begin_action(active));
  }
  std::puts("PASS production Hand socket STOP, active disconnect, completed grasp preservation");
}
