// arm_rt_server — the RT machine's whole userland: four threads, one binary.
//
//   ./arm_rt_server --backend fake --n 7            # loopback plant, any PC
//   ./arm_rt_server --backend franka                # FR3 (build WITH_FRANKA)
//
// Design rule the flags encode: this process is a SERVO WITH REFLEXES, never
// a brain. Torque ceilings, staleness thresholds and slew are launch-time
// safety config; everything task-shaped (gains schedules, grasp policy,
// phases, planning) stays on the PC and arrives per-tick in CommandPackets.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "server.hpp"

namespace {

arm_rt::ServerCtx* g_ctx = nullptr;

void on_signal(int) {
  if (g_ctx) g_ctx->shutdown.store(true);
}

void usage(const char* argv0) {
  std::printf(
      "usage: %s [--backend fake|franka|dm] [--n N] [--udp-port P] [--tcp-port P]\n"
      "          [--state-hz HZ] [--hold-ms MS] [--fault-ms MS] [--slew NM]\n"
      "          [--hold-kp V] [--hold-kd V] [--franka-ip IP] [--can-if IF]\n"
      "          [--rt-priority N]\n",
      argv0);
}

} // namespace

int main(int argc, char** argv) {
  arm_rt::ServerCtx ctx;
  auto& cfg = ctx.cfg;
  for (int i = 1; i < argc; ++i) {
    auto next = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--backend")) cfg.backend = next("--backend");
    else if (!std::strcmp(argv[i], "--n")) cfg.n = std::atoi(next("--n"));
    else if (!std::strcmp(argv[i], "--udp-port")) cfg.udp_port = uint16_t(std::atoi(next("--udp-port")));
    else if (!std::strcmp(argv[i], "--tcp-port")) cfg.tcp_port = uint16_t(std::atoi(next("--tcp-port")));
    else if (!std::strcmp(argv[i], "--state-hz")) cfg.state_hz = std::atof(next("--state-hz"));
    else if (!std::strcmp(argv[i], "--hold-ms")) cfg.hold_ms = std::atof(next("--hold-ms"));
    else if (!std::strcmp(argv[i], "--fault-ms")) cfg.fault_ms = std::atof(next("--fault-ms"));
    else if (!std::strcmp(argv[i], "--slew")) cfg.slew = std::atof(next("--slew"));
    else if (!std::strcmp(argv[i], "--hold-kp")) cfg.hold_kp = std::atof(next("--hold-kp"));
    else if (!std::strcmp(argv[i], "--hold-kd")) cfg.hold_kd = std::atof(next("--hold-kd"));
    else if (!std::strcmp(argv[i], "--franka-ip")) cfg.franka_ip = next("--franka-ip");
    else if (!std::strcmp(argv[i], "--can-if")) cfg.can_if = next("--can-if");
    else if (!std::strcmp(argv[i], "--rt-priority")) cfg.rt_priority = std::atoi(next("--rt-priority"));
    else {
      usage(argv[0]);
      return 2;
    }
  }
  if (cfg.n < 1 || cfg.n > arm_rt::MAX_JOINTS) {
    std::fprintf(stderr, "--n must be 1..%d\n", arm_rt::MAX_JOINTS);
    return 2;
  }

  g_ctx = &ctx;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  std::printf("[rt] arm_rt_server: backend=%s udp=%u tcp=%u state=%.0fHz "
              "hold=%.0fms fault=%.0fms\n",
              cfg.backend.c_str(), cfg.udp_port, cfg.tcp_port, cfg.state_hz,
              cfg.hold_ms, cfg.fault_ms);

  std::thread udp(arm_rt::udp_rx_thread, std::ref(ctx));
  std::thread tx(arm_rt::state_tx_thread, std::ref(ctx));
  std::thread ctl(arm_rt::control_thread, std::ref(ctx));
  arm_rt::rt_loop(ctx);  // RT thread = main thread; backend constructed here
  ctx.shutdown.store(true);
  udp.join();
  tx.join();
  ctl.join();
  std::printf("[rt] down\n");
  return 0;
}
