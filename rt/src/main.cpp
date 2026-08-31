// arm_rt_server — the RT machine's whole userland: four threads, one binary.
//
//   ./arm_rt_server --backend fake --n 7            # loopback plant, any PC
//   ./arm_rt_server --backend franka                # FR3 (build WITH_FRANKA)
//
// Design rule the flags encode: this process is a SERVO WITH REFLEXES, never
// a brain. Torque ceilings, staleness thresholds and slew are launch-time
// safety config; everything task-shaped (gains schedules, grasp policy,
// phases, planning) stays on the PC and arrives per-tick in CommandPackets.
#include <arpa/inet.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "server.hpp"

namespace arm_rt {
int dm_mit_selfcheck();  // backend_dm.cpp — MIT wire-format goldens
}

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
      "          [--ee-mass KG] [--ee-com X,Y,Z]\n"
      "          [--dm-spec IF;ID:TYPE[:MST],...] [--rt-priority N]\n"
      "          [--active-mask MASK] [--rt-cpu CPU] [--bind IP] [--mit-check]\n"
      "\n"
      "  --ee-mass  payload past the flange (wrist camera + mount, carried\n"
      "             module) in kg, ADDED to Desk's end-effector config.\n"
      "  --ee-com   that payload's centre of mass in the FLANGE frame, metres.\n",
      argv0);
}

uint16_t parse_port(const char* flag, const char* value) {
  const long p = std::atol(value);
  if (p < 1 || p > 65535) {  // atoi-into-uint16 silently truncates 70000->4464
    std::fprintf(stderr, "%s must be 1..65535 (got %s)\n", flag, value);
    std::exit(2);
  }
  return uint16_t(p);
}

} // namespace

int main(int argc, char** argv) {
  // MIT wire-format parity check (see backend_dm.cpp): golden frames the
  // Python pack_mit_control_frame/decode_mit_reply must reproduce exactly.
  if (argc > 1 && !std::strcmp(argv[1], "--mit-check"))
    return arm_rt::dm_mit_selfcheck();

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
    else if (!std::strcmp(argv[i], "--udp-port")) cfg.udp_port = parse_port("--udp-port", next("--udp-port"));
    else if (!std::strcmp(argv[i], "--tcp-port")) cfg.tcp_port = parse_port("--tcp-port", next("--tcp-port"));
    else if (!std::strcmp(argv[i], "--bind")) {
      in_addr a{};
      if (inet_pton(AF_INET, next("--bind"), &a) != 1) {
        std::fprintf(stderr, "--bind needs a dotted IPv4 address\n");
        return 2;
      }
      cfg.bind_addr = a.s_addr;
    }
    else if (!std::strcmp(argv[i], "--state-hz")) cfg.state_hz = std::atof(next("--state-hz"));
    else if (!std::strcmp(argv[i], "--hold-ms")) cfg.hold_ms = std::atof(next("--hold-ms"));
    else if (!std::strcmp(argv[i], "--fault-ms")) cfg.fault_ms = std::atof(next("--fault-ms"));
    else if (!std::strcmp(argv[i], "--slew")) cfg.slew = std::atof(next("--slew"));
    else if (!std::strcmp(argv[i], "--hold-kp")) cfg.hold_kp = std::atof(next("--hold-kp"));
    else if (!std::strcmp(argv[i], "--hold-kd")) cfg.hold_kd = std::atof(next("--hold-kd"));
    else if (!std::strcmp(argv[i], "--franka-ip")) cfg.franka_ip = next("--franka-ip");
    else if (!std::strcmp(argv[i], "--ee-mass")) cfg.ee_mass = std::atof(next("--ee-mass"));
    else if (!std::strcmp(argv[i], "--ee-com")) {
      const char* v = next("--ee-com");
      if (std::sscanf(v, "%lf,%lf,%lf", &cfg.ee_com[0], &cfg.ee_com[1],
                      &cfg.ee_com[2]) != 3) {
        std::fprintf(stderr, "--ee-com needs X,Y,Z in metres (got %s)\n", v);
        return 2;
      }
    }
    else if (!std::strcmp(argv[i], "--can-if")) cfg.can_if = next("--can-if");
    // dm backend: iface + motor list in one string ("can0;1:4340,2:4340").
    // Same cfg field as --can-if — this is the full spelling of it.
    else if (!std::strcmp(argv[i], "--dm-spec")) cfg.can_if = next("--dm-spec");
    else if (!std::strcmp(argv[i], "--active-mask")) {
      char* end = nullptr;
      const char* value = next("--active-mask");
      const unsigned long mask = std::strtoul(value, &end, 0);
      if (!end || *end || mask > 0xFFFFu) {
        std::fprintf(stderr, "--active-mask must fit 16 bits (got %s)\n", value);
        return 2;
      }
      cfg.initial_active_mask = uint32_t(mask);
    }
    else if (!std::strcmp(argv[i], "--rt-priority")) cfg.rt_priority = std::atoi(next("--rt-priority"));
    else if (!std::strcmp(argv[i], "--rt-cpu")) cfg.rt_cpu = std::atoi(next("--rt-cpu"));
    else {
      usage(argv[0]);
      return 2;
    }
  }
  if (cfg.n < 1 || cfg.n > arm_rt::MAX_JOINTS) {
    std::fprintf(stderr, "--n must be 1..%d\n", arm_rt::MAX_JOINTS);
    return 2;
  }
  // A misconfigured safety flag must refuse to start, not silently disable a
  // reflex (hold-ms >= fault-ms makes the command deadman unreachable;
  // slew <= 0 pins torque to tau_ref = an armed arm with zero authority).
  if (cfg.hold_ms <= 0 || cfg.fault_ms <= cfg.hold_ms) {
    std::fprintf(stderr, "need 0 < --hold-ms < --fault-ms (got %.0f/%.0f)\n",
                 cfg.hold_ms, cfg.fault_ms);
    return 2;
  }
  if (cfg.slew <= 0 || cfg.hold_kp < 0 || cfg.hold_kd < 0 ||
      cfg.state_hz < 1 || cfg.state_hz > 1000) {
    std::fprintf(stderr, "need --slew > 0, hold gains >= 0, --state-hz 1..1000\n");
    return 2;
  }
  // FR3 rated payload is 3 kg. A value outside that is a unit slip (grams for
  // kg is the easy one) and would put the robot's own gravity compensation
  // further off than declaring nothing — refuse rather than fight it.
  if (cfg.ee_mass < 0 || cfg.ee_mass > 3.0) {
    std::fprintf(stderr, "--ee-mass must be 0..3 kg (got %.3f) — kg, not grams\n",
                 cfg.ee_mass);
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
  return ctx.failed.load() ? 1 : 0;
}
