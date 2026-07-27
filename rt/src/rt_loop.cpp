// The 1 kHz servo thread. Constructs the backend HERE (libfranka raises the
// calling thread's scheduling in the Robot constructor), elevates itself
// (mlockall + SCHED_FIFO — warnings, not failures, so the fake runs on any
// desktop), then ticks: read plant -> staleness policy -> servo law -> write.
//
// Authority ladder, most-alive first:
//   armed + fresh command        -> track the command (its kp/kd/tau_ff)
//   armed + stale (> hold_ms)    -> HOLD the pose captured at staleness
//   armed + stale (> fault_ms)   -> still holding, but LATCHED (cmds ignored
//                                   until an explicit DISARM+ARM cycle)
//   armed + fault (any source)   -> hold, latched
//   disarmed                     -> backend.stop() once, no writes
//
// Holding is position PD around the captured pose with the last command's
// gains (or --hold-kp/--hold-kd if no command ever arrived — the state right
// after arming, which IS "hold where you are"). tau_ff is dropped in hold;
// the slew limiter ramps it out instead of stepping it.
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <cstdio>
#include <cstring>

#include "arm_rt/servo_law.hpp"
#include "server.hpp"

namespace arm_rt {

std::unique_ptr<Backend> make_fake_backend(int n);
std::unique_ptr<Backend> make_franka_backend(const std::string& ip);
std::unique_ptr<Backend> make_dm_backend(const std::string& can_if);

namespace {

void elevate(const ServerConfig& cfg) {
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
    std::perror("[rt] mlockall (continuing unpinned)");
  sched_param sp{};
  sp.sched_priority = cfg.rt_priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
    std::perror("[rt] SCHED_FIFO (continuing at normal priority)");
}

} // namespace

void rt_loop(ServerCtx& ctx) {
  std::unique_ptr<Backend> backend;
  if (ctx.cfg.backend == "fake") {
    backend = make_fake_backend(ctx.cfg.n);
  } else if (ctx.cfg.backend == "franka") {
    backend = make_franka_backend(ctx.cfg.franka_ip);
    if (!backend) std::fprintf(stderr, "[rt] built without libfranka (-DWITH_FRANKA=ON)\n");
  } else if (ctx.cfg.backend == "dm") {
    backend = make_dm_backend(ctx.cfg.can_if);
    if (!backend) std::fprintf(stderr, "[rt] dm backend not implemented yet\n");
  } else {
    std::fprintf(stderr, "[rt] unknown backend %s\n", ctx.cfg.backend.c_str());
  }
  if (!backend) {
    ctx.shutdown.store(true);
    return;
  }
  const int n = backend->n();
  ctx.backend_n.store(n);
  std::snprintf(ctx.backend_name, sizeof(ctx.backend_name), "%s", backend->name());
  ctx.backend_ready.store(true);
  elevate(ctx.cfg);
  std::printf("[rt] %s backend up: %d joints, tick %.1f ms, slew %.2f N.m/tick\n",
              backend->name(), n, backend->tick_s() * 1e3, ctx.cfg.slew);

  const double hold_ns = ctx.cfg.hold_ms * 1e6;
  const double fault_ns = ctx.cfg.fault_ms * 1e6;

  PlantState ps;
  CommandPacket cmd = {};
  StatePacket out = {};
  bool prev_armed = false;
  bool holding = false;
  bool have_cmd_gains = false;
  double q_hold[MAX_JOINTS] = {};
  double hold_kp[MAX_JOINTS], hold_kd[MAX_JOINTS];
  double zeros[MAX_JOINTS] = {};
  double tau_out[MAX_JOINTS] = {};
  for (int j = 0; j < MAX_JOINTS; ++j) {
    hold_kp[j] = ctx.cfg.hold_kp;
    hold_kd[j] = ctx.cfg.hold_kd;
  }
  uint32_t state_seq = 0;
  uint32_t last_cmd_seq = 0;

  while (!ctx.shutdown.load()) {
    if (!backend->read(ps)) {
      ctx.latch(FAULT_PLANT, backend->fault_text().c_str());
      timespec ts{0, long(backend->tick_s() * 1e9)};
      nanosleep(&ts, nullptr);
      continue;
    }
    const uint64_t now = mono_ns();
    const bool armed = ctx.armed.load(std::memory_order_acquire);
    bool faulted = ctx.fault.load(std::memory_order_acquire);

    const uint64_t cmd_v = ctx.cmd_in.read(cmd);
    const uint64_t rx_ns = ctx.last_cmd_rx_ns.load(std::memory_order_acquire);
    const double age_ns = rx_ns == 0 ? 1e18 : double(now - rx_ns);
    const bool fresh = cmd_v > 0 && cmd.n == uint16_t(n) && age_ns <= hold_ns;

    const double* q_des = zeros;
    const double* qd_des = zeros;
    const double* tau_ff = zeros;
    const double* kp = hold_kp;
    const double* kd = hold_kd;

    if (armed) {
      if (!faulted && fresh) {
        holding = false;
        have_cmd_gains = true;
        last_cmd_seq = cmd.seq;
        q_des = cmd.q_des;
        qd_des = cmd.qd_des;
        tau_ff = cmd.tau_ff;
        kp = cmd.kp;
        kd = cmd.kd;
      } else {
        if (!holding) {
          holding = true;
          std::memcpy(q_hold, ps.q, sizeof(double) * size_t(n));
        }
        if (!faulted && rx_ns != 0 && age_ns > fault_ns) {
          ctx.latch(FAULT_CMD_LOST, "command stream lost (staleness > fault-ms)");
          faulted = true;
        }
        q_des = q_hold;
        if (have_cmd_gains) {  // hold with the authority the task last chose
          kp = cmd.kp;
          kd = cmd.kd;
        }
      }
      servo_torque(n, ps.q, ps.dq, q_des, qd_des, tau_ff, kp, kd, ps.tau_ref,
                   backend->tau_limit(), ctx.cfg.slew, tau_out);
      if (!backend->write(tau_out, n)) {
        ctx.latch(FAULT_PLANT, backend->fault_text().c_str());
      }
    } else {
      if (prev_armed) backend->stop();
      holding = false;
      std::memset(tau_out, 0, sizeof(tau_out));
    }
    prev_armed = armed;

    out.magic = MAGIC_STATE;
    out.version = VERSION;
    out.n = uint16_t(n);
    out.state_seq = ++state_seq;
    out.last_cmd_seq = last_cmd_seq;
    out.t_mono_ns = now;
    out.flags = (armed ? FLAG_ARMED : 0u) |
                (ctx.fault.load(std::memory_order_acquire) ? FLAG_FAULTED : 0u) |
                (holding ? FLAG_HOLDING : 0u);
    out.fault_code = ctx.fault_code.load(std::memory_order_acquire);
    std::memcpy(out.q, ps.q, sizeof(out.q));
    std::memcpy(out.dq, ps.dq, sizeof(out.dq));
    std::memcpy(out.tau, ps.tau, sizeof(out.tau));
    std::memcpy(out.tau_cmd, tau_out, sizeof(out.tau_cmd));
    std::memcpy(out.q_cmd, q_des, sizeof(double) * size_t(n));
    ctx.state_out.write(out);
  }
  backend->stop();
}

} // namespace arm_rt
