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
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

// Thread boundary: an exception escaping a server thread was std::terminate
// (no unwinding, no diagnosable cause). Report it and shut the whole process
// down nonzero instead -- never resume a thread whose invariants are unknown.
template <class Fn>
void guarded(const char* thread, arm_rt::ServerCtx& ctx, Fn fn) {
  try {
    fn(ctx);
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "[rt] %s thread exception: %s — shutting down\n", thread,
                 exc.what());
    ctx.failed.store(true);
    ctx.shutdown.store(true);
  } catch (...) {
    std::fprintf(stderr, "[rt] %s thread: unknown exception — shutting down\n",
                 thread);
    ctx.failed.store(true);
    ctx.shutdown.store(true);
  }
}

void usage(const char* argv0) {
  std::printf(
      "usage: %s [--backend fake|franka|dm] [--instance-id ID] [--joint-names A,B,..]\n"
      "          [--n N] [--fake-q0 Q1,Q2,..] [--udp-port P] [--tcp-port P]\n"
      "          [--state-hz HZ] [--hold-ms MS] [--fault-ms MS] [--slew NM]\n"
      "          [--hold-kp V] [--hold-kd V] [--franka-ip IP] [--can-if IF]\n"
      "          [--ee-mass KG] [--ee-com X,Y,Z] [--tau-max NM]\n"
      "          [--dm-spec IF;ID:TYPE[:MST[:PMAX:VMAX:TMAX]],...] [--rt-priority N]\n"
      "          [--active-mask MASK] [--rt-cpu CPU] [--bind IP] [--mit-check]\n"
      "          [--require-rt-setup] [--enable-reference]\n"
      "          [--machine-id-file PATH] [--boot-id-file PATH]\n"
      "          [--group-shm NAME --group-member K --group-members N\n"
      "           --group-peer-max-age-ms MS --group-min-lead-ms MS]\n"
      "\n"
      "  --instance-id  provisioned arm instance id ([A-Za-z0-9_.-], <32 chars);\n"
      "             required for franka/dm. Commanders CLAIM it before ARM.\n"
      "  --joint-names  comma-separated joint order the backend's slots mean;\n"
      "             required for franka/dm, count must equal the backend's.\n"
      "  --ee-mass  payload past the flange (wrist camera + mount, carried\n"
      "             module) in kg, ADDED to Desk's end-effector config.\n"
      "  --ee-com   that payload's centre of mass in the FLANGE frame, metres.\n"
      "  --dm-spec DM host MIT maps; optional symmetric PMAX/VMAX/TMAX must\n"
      "             match firmware (rad, rad/s, N.m). Omit for type defaults.\n"
      "  --tau-max  positive operating torque cap per joint (N.m), never above\n"
      "             backend limits; omitted keeps the backend defaults.\n"
      "  --require-rt-setup  refuse ARM if mlockall, SCHED_FIFO or --rt-cpu fails\n"
      "             (always on for franka/dm; the fake backend runs degraded).\n"
      "  --enable-reference  advertise and accept the v7 native-reference\n"
      "             contract. OFF by default: the native path is not qualified\n"
      "             for any deployment, so it must be opted into per instance\n"
      "             (the deployment registry renders this flag).\n",
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
    else if (!std::strcmp(argv[i], "--instance-id")) cfg.instance_id = next("--instance-id");
    else if (!std::strcmp(argv[i], "--joint-names")) {
      std::string list = next("--joint-names");
      cfg.joint_names.clear();
      size_t start = 0;
      while (start <= list.size()) {
        const size_t comma = list.find(',', start);
        const std::string name = list.substr(start, comma == std::string::npos
                                                        ? std::string::npos
                                                        : comma - start);
        if (name.empty()) {
          std::fprintf(stderr, "--joint-names has an empty entry\n");
          return 2;
        }
        cfg.joint_names.push_back(name);
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
    }
    else if (!std::strcmp(argv[i], "--n")) cfg.n = std::atoi(next("--n"));
    else if (!std::strcmp(argv[i], "--fake-q0")) {
      // test double only: the fake plant's start pose, so a client whose envelope excludes zero can ARM it
      std::string list = next("--fake-q0");
      cfg.fake_q0.clear();
      for (size_t start = 0; start <= list.size();) {
        const size_t comma = list.find(',', start);
        const std::string item = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        char* end = nullptr;
        const double v = std::strtod(item.c_str(), &end);
        if (item.empty() || *end || !std::isfinite(v)) {
          std::fprintf(stderr, "--fake-q0 needs comma-separated radians (got %s)\n", list.c_str());
          return 2;
        }
        cfg.fake_q0.push_back(v);
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
    }
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
    else if (!std::strcmp(argv[i], "--machine-id-file")) cfg.machine_id_file = next("--machine-id-file");
    else if (!std::strcmp(argv[i], "--boot-id-file")) cfg.boot_id_file = next("--boot-id-file");
    else if (!std::strcmp(argv[i], "--group-shm")) cfg.group_shm = next("--group-shm");
    else if (!std::strcmp(argv[i], "--group-member")) cfg.group_member = std::atoi(next("--group-member"));
    else if (!std::strcmp(argv[i], "--group-members")) cfg.group_members = uint32_t(std::atoi(next("--group-members")));
    else if (!std::strcmp(argv[i], "--group-peer-max-age-ms")) cfg.group_peer_max_age_ms = std::atof(next("--group-peer-max-age-ms"));
    else if (!std::strcmp(argv[i], "--group-min-lead-ms")) cfg.group_min_lead_ms = std::atof(next("--group-min-lead-ms"));
    else if (!std::strcmp(argv[i], "--hold-ms")) cfg.hold_ms = std::atof(next("--hold-ms"));
    else if (!std::strcmp(argv[i], "--fault-ms")) cfg.fault_ms = std::atof(next("--fault-ms"));
    else if (!std::strcmp(argv[i], "--slew")) cfg.slew = std::atof(next("--slew"));
    else if (!std::strcmp(argv[i], "--tau-max")) {
      const char* value = next("--tau-max");
      char* end = nullptr;
      errno = 0;
      const double cap = std::strtod(value, &end);
      if (errno || end == value || *end || !std::isfinite(cap) || cap <= 0) {
        std::fprintf(stderr, "--tau-max must be finite and > 0 N.m (got '%s')\n", value);
        return 2;
      }
      cfg.tau_max = cap;
    }
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
    else if (!std::strcmp(argv[i], "--require-rt-setup")) cfg.require_rt_setup = true;
    else if (!std::strcmp(argv[i], "--enable-reference")) cfg.enable_reference = true;
    else {
      usage(argv[0]);
      return 2;
    }
  }
  // Identity is provisioned deployment data. A hardware server without it
  // would accept any commander's claim -- refuse to start instead.
  if (cfg.backend != "fake" && (cfg.instance_id.empty() || cfg.joint_names.empty())) {
    std::fprintf(stderr, "--instance-id and --joint-names are required for --backend %s\n",
                 cfg.backend.c_str());
    return 2;
  }
  if (cfg.backend == "fake" && cfg.instance_id.empty()) cfg.instance_id = "fake";
  {
    char probe[arm_rt::ID_CHARS] = {};
    if (cfg.instance_id.size() >= sizeof(probe)) {
      std::fprintf(stderr, "--instance-id must be shorter than %d chars\n", arm_rt::ID_CHARS);
      return 2;
    }
    std::memcpy(probe, cfg.instance_id.data(), cfg.instance_id.size());
    if (!arm_rt::valid_id(probe)) {
      std::fprintf(stderr, "--instance-id must be 1..31 of [A-Za-z0-9_.-] (got '%s')\n",
                   cfg.instance_id.c_str());
      return 2;
    }
  }
  if (cfg.n < 1 || cfg.n > arm_rt::MAX_JOINTS) {
    std::fprintf(stderr, "--n must be 1..%d\n", arm_rt::MAX_JOINTS);
    return 2;
  }
  if (!cfg.fake_q0.empty() && (cfg.backend != "fake" || int(cfg.fake_q0.size()) != cfg.n)) {
    std::fprintf(stderr, "--fake-q0 is for the fake backend and needs exactly --n values\n");
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

  // Group execution is all-or-nothing and has no default timing constants.
  const bool any_group = !cfg.group_shm.empty() || cfg.group_member >= 0 || cfg.group_members ||
                         cfg.group_peer_max_age_ms > 0 || cfg.group_min_lead_ms > 0;
  if (any_group) {
    if (cfg.group_shm.empty() || cfg.group_shm[0] != '/' || cfg.group_members < 2 ||
        cfg.group_members > arm_rt::GROUP_MAX_MEMBERS || cfg.group_member < 0 ||
        uint32_t(cfg.group_member) >= cfg.group_members || cfg.group_peer_max_age_ms <= 0 ||
        cfg.group_min_lead_ms <= 0 || !cfg.enable_reference) {
      std::fprintf(stderr, "group execution needs --enable-reference and ALL of --group-shm /NAME, "
                           "--group-member K < --group-members N (2..%u), "
                           "--group-peer-max-age-ms > 0, --group-min-lead-ms > 0\n",
                   arm_rt::GROUP_MAX_MEMBERS);
      return 2;
    }
    if (!ctx.group.open(cfg.group_shm, cfg.group_members)) {
      std::fprintf(stderr, "cannot open group segment %s\n", cfg.group_shm.c_str());
      return 2;
    }
    ctx.group.reset_slot(uint32_t(cfg.group_member));  // a killed predecessor may have left it mid-write
    ctx.group_enabled.store(true);
  }

  g_ctx = &ctx;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // Off unless this launch opted in: the capability word and the v7 mode
  // gate both read this, so a bare start cannot expose the native path.
  ctx.supports_reference.store(cfg.enable_reference);

  std::printf("[rt] arm_rt_server: backend=%s udp=%u tcp=%u state=%.0fHz "
              "hold=%.0fms fault=%.0fms reference=%s\n",
              cfg.backend.c_str(), cfg.udp_port, cfg.tcp_port, cfg.state_hz,
              cfg.hold_ms, cfg.fault_ms, cfg.enable_reference ? "on" : "off");

  std::thread udp(guarded<void (*)(arm_rt::ServerCtx&)>, "udp_rx", std::ref(ctx),
                  arm_rt::udp_rx_thread);
  std::thread tx(guarded<void (*)(arm_rt::ServerCtx&)>, "state_tx", std::ref(ctx),
                 arm_rt::state_tx_thread);
  std::thread ctl(guarded<void (*)(arm_rt::ServerCtx&)>, "control", std::ref(ctx),
                  arm_rt::control_thread);
  // RT thread = main thread; backend constructed here
  guarded("rt_loop", ctx, arm_rt::rt_loop);
  ctx.shutdown.store(true);
  udp.join();
  tx.join();
  ctl.join();
  arm_rt::close_udp_socket(ctx);      // no thread can still use it
  arm_rt::finish_fault_journal(ctx);  // every producer and consumer is joined
  std::printf("[rt] down\n");
  return ctx.failed.load() ? 1 : 0;
}
