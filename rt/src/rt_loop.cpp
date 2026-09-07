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
//   armed + fault (comms source) -> hold, latched — the plant is healthy, so
//                                   parking the arm is meaningful
//   armed + PLANT fault          -> latched with NO writes at all: the
//                                   session/bus is gone and the robot's own
//                                   safety owns the arm. Re-arming re-opens
//                                   the plant exactly once per ARM.
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
#include "arm_rt/pose_hold.hpp"
#include "server.hpp"

namespace arm_rt {

std::unique_ptr<Backend> make_fake_backend(int n);
std::unique_ptr<Backend> make_franka_backend(const std::string& ip, double ee_mass,
                                             const double ee_com[3]);
std::unique_ptr<Backend> make_dm_backend(const std::string& can_if,
                                         uint32_t active_mask);

namespace {

void elevate(const ServerConfig& cfg) {
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
    std::perror("[rt] mlockall (continuing unpinned)");
  sched_param sp{};
  sp.sched_priority = cfg.rt_priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
    std::perror("[rt] SCHED_FIFO (continuing at normal priority)");
  if (cfg.rt_cpu >= 0) {
    // Only THIS thread moves to the isolated core; the comms threads keep the
    // default mask (the housekeeping cores, once isolcpus removes this one).
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cfg.rt_cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
      std::perror("[rt] rt-cpu pin (continuing unpinned)");
    else
      std::printf("[rt] servo thread pinned to cpu %d\n", cfg.rt_cpu);
  }
}

} // namespace

void rt_loop(ServerCtx& ctx) {
  std::unique_ptr<Backend> backend;
  if (ctx.cfg.backend == "fake") {
    backend = make_fake_backend(ctx.cfg.n);
  } else if (ctx.cfg.backend == "franka") {
    backend = make_franka_backend(ctx.cfg.franka_ip, ctx.cfg.ee_mass, ctx.cfg.ee_com);
    if (!backend)
      std::fprintf(stderr, "[rt] franka backend failed to start "
                           "(reason above; RT perms? FCI on? robot reachable?)\n");
  } else if (ctx.cfg.backend == "dm") {
    backend = make_dm_backend(ctx.cfg.can_if, ctx.cfg.initial_active_mask);
    if (!backend)
      std::fprintf(stderr, "[rt] dm backend failed to start (reason above; "
                           "--dm-spec? link up? fd on?)\n");
  } else {
    std::fprintf(stderr, "[rt] unknown backend %s\n", ctx.cfg.backend.c_str());
  }
  if (!backend) {
    ctx.failed.store(true);
    ctx.shutdown.store(true);
    return;
  }
  const int n = backend->n();
  ctx.backend_n.store(n);
  ctx.online_mask.store(backend->online_mask());
  ctx.active_mask.store(backend->active_mask());
  ctx.requested_active_mask.store(backend->active_mask());
  std::snprintf(ctx.backend_name, sizeof(ctx.backend_name), "%s", backend->name());
  ctx.supports_pose_hold.store(backend->supports_pose_hold());
  ctx.backend_ready.store(true);
  elevate(ctx.cfg);
  std::printf("[rt] %s backend up: %d joints, tick %.1f ms, slew %.2f N.m/tick\n",
              backend->name(), n, backend->tick_s() * 1e3, ctx.cfg.slew);

  const double hold_ns = ctx.cfg.hold_ms * 1e6;
  const double fault_ns = ctx.cfg.fault_ms * 1e6;

  PlantState ps;
  PoseHoldCommandPacket incoming = {};
  const CommandPacket& cmd=incoming.command;
  PoseHold pose_hold;
  double pose_tau[MAX_JOINTS]{};
  StatePacket out = {};
  bool prev_armed = false;
  bool holding = false;
  bool have_cmd_gains = false;
  double q_hold[MAX_JOINTS] = {};
  // Gains SNAPSHOTTED at command acceptance. The hold branch must never
  // dereference the live seqlock buffer: any datagram that lands there
  // (address-teach prime, wrong-n, post-fault stray) would otherwise become
  // the hold spring's authority — a zero-gain packet silently un-springs a
  // parked arm (found by adversarial review after the FR3 impedance rung).
  double cmd_kp[MAX_JOINTS] = {}, cmd_kd[MAX_JOINTS] = {};
  double hold_kp[MAX_JOINTS], hold_kd[MAX_JOINTS];
  double zeros[MAX_JOINTS] = {};
  double tau_out[MAX_JOINTS] = {};
  for (int j = 0; j < MAX_JOINTS; ++j) {
    hold_kp[j] = ctx.cfg.hold_kp;
    hold_kd[j] = ctx.cfg.hold_kd;
  }
  uint32_t state_seq = 0;
  uint32_t last_cmd_seq = 0;
  uint32_t last_active_mask = backend->active_mask();

  bool plant_ok = true;  // false after any backend failure; reset only by ARM
  uint64_t seen_gen = ctx.arm_gen.load(std::memory_order_acquire);
  int read_fail_streak = 0;
  uint64_t cmd_v_seen = 0;  // survives a seqlock read-collision tick (read()
                            // returns 0 on retry exhaustion; without the cache
                            // that tick reads as "stale" and flaps into hold)

  while (!ctx.shutdown.load()) {
    if (!backend->read(ps)) {
      // Latch only while ARMED: a transient read hiccup on a DISARMED idle
      // server was blocking the NEXT arm (nothing is being written, so there
      // is nothing to protect — and the first armed write re-latches anyway
      // if the plant is really down).
      if (ctx.armed.load(std::memory_order_acquire)) {
        ctx.latch(FAULT_PLANT, backend->fault_text().c_str());
      } else if (read_fail_streak == 0) {
        std::fprintf(stderr, "[rt] plant read failed while disarmed (%s) — "
                             "not latching\n", backend->fault_text().c_str());
      }
      plant_ok = false;
      // A dead plant retried at 1 kHz is a 1 kHz exception-throw loop on the
      // SCHED_FIFO thread (allocation + unwinding, heap growth over hours).
      // Pace failures at 100 ms; nothing is being written anyway. Past ~5 s
      // of continuous failure, exit nonzero: franka::Robot is constructed
      // once and never rebuilt, so a control-box power cycle bricks this
      // process forever — systemd restarting us is the only recovery that
      // works unattended.
      if (++read_fail_streak >= 50) {
        std::fprintf(stderr, "[rt] plant unreachable for %.0fs — exiting for "
                             "a clean restart\n", 50 * 0.1);
        ctx.failed.store(true);
        ctx.shutdown.store(true);
        return;
      }
      timespec ts{0, 100'000'000};
      nanosleep(&ts, nullptr);
      continue;
    }
    read_fail_streak = 0;
    ctx.online_mask.store(backend->online_mask(), std::memory_order_release);
    const uint32_t requested =
        ctx.requested_active_mask.load(std::memory_order_acquire);
    if (requested != backend->active_mask() && backend->set_active_mask(requested)) {
      const uint32_t added = requested & ~last_active_mask;
      for (int j = 0; j < n; ++j)
        if (added & (1u << j)) q_hold[j] = ps.q[j];
      last_active_mask = requested;
      ctx.active_mask.store(requested, std::memory_order_release);
    }
    const uint64_t now = mono_ns();
    const bool armed = ctx.armed.load(std::memory_order_acquire);
    bool faulted = ctx.fault.load(std::memory_order_acquire);

    // A failed seqlock retry can have copied a partial payload; only publish
    // a successful read to the cached command used for this servo tick.
    PoseHoldCommandPacket candidate{};
    const uint64_t cmd_v_raw = ctx.cmd_in.read(candidate);
    if (cmd_v_raw != 0) { incoming=candidate; cmd_v_seen = cmd_v_raw; }
    const uint64_t cmd_v = cmd_v_seen;
    const uint64_t rx_ns = ctx.last_cmd_rx_ns.load(std::memory_order_acquire);
    // SIGNED, clamped age. rx_ns is stamped by udp_rx (or the ARM handler)
    // in parallel with this tick: a packet stamped between our `now` sample
    // above and this load is nanoseconds IN THE FUTURE, and the old unsigned
    // `now - rx_ns` underflowed to ~1.8e19 ns — an instant spurious CMD_LOST
    // latch on a perfectly healthy 100 Hz stream. Odds ~ packet rate x the
    // sample-to-load window (sub-us pinned, up to ms when this thread is
    // preemptible), i.e. one false latch per minutes of armed time — found
    // live on rung 3 and root-caused with a fake-plant soak + process
    // autopsy after every stream instrument reported healthy flow.
    const int64_t raw_age = int64_t(now) - int64_t(rx_ns);
    const double age_ns = rx_ns == 0 ? 1e18 : double(raw_age < 0 ? 0 : raw_age);
    // Only commands from AFTER the current ARM are authority (cmd_epoch).
    const bool fresh = cmd_v > ctx.cmd_epoch.load(std::memory_order_acquire) &&
                       cmd.n == uint16_t(n) && age_ns <= hold_ns;

    const double* q_des = zeros;
    const double* qd_des = zeros;
    const double* tau_ff = zeros;
    const double* kp = hold_kp;
    const double* kd = hold_kd;

    // The ARM edge is a GENERATION, not a level: a DISARM->ARM pair that
    // completes between two 1 kHz samples still changes arm_gen, so the
    // per-epoch resets can never be skipped by fast recovery cycles.
    const uint64_t gen = ctx.arm_gen.load(std::memory_order_acquire);
    if (gen != seen_gen) {
      seen_gen = gen;
      if (prev_armed) backend->stop();  // the DISARM we never sampled
      plant_ok = true;         // ARM = explicit plant retry
      have_cmd_gains = false;  // fresh epoch: hold gains until the task speaks
      holding = false;         // re-capture the hold pose in this epoch
      pose_hold.reset();
    }
    if(armed) {
      for(int j=0;j<n;++j) {
        if(!std::isfinite(ps.q[j]) || !std::isfinite(ps.dq[j]) || !std::isfinite(ps.tau_ref[j])) {
          ctx.latch(FAULT_PLANT,"non-finite measured joint state; torque writes suppressed");
          plant_ok=false;
          break;
        }
      }
    }
    if (armed && !plant_ok) {
      pose_hold.reset();
      holding = false;  // nothing is held — the robot's own safety has it
      std::memset(tau_out, 0, sizeof(tau_out));
    } else if (armed) {
      if (!faulted && fresh) {
        holding = false;
        have_cmd_gains = true;
        last_cmd_seq = cmd.seq;
        std::memcpy(cmd_kp, cmd.kp, sizeof(cmd_kp));  // accepted -> snapshot
        std::memcpy(cmd_kd, cmd.kd, sizeof(cmd_kd));
        q_des = cmd.q_des;
        qd_des = cmd.qd_des;
        tau_ff = cmd.tau_ff;
        kp = cmd_kp;
        kd = cmd_kd;
        if(cmd.version==POSE_HOLD_VERSION) {
          if(!backend->supports_pose_hold() || !ps.pose_valid || !ps.jacobian_valid ||
             !pose_hold.torque(n,ps.q,ps.dq,ps.jacobian,ps.pose,ps.coriolis,
                               incoming.pose_hold,backend->tick_s(),pose_tau)) {
            ctx.latch(FAULT_PLANT,"pose hold requires valid measured pose/J/model state");
            faulted=true;
          } else {
            q_des=ps.q; qd_des=zeros; tau_ff=pose_tau; kp=zeros; kd=zeros;
          }
        } else pose_hold.reset();
      }
      if(faulted || !fresh) {
        pose_hold.reset();
        if (!holding) {
          holding = true;
          std::memcpy(q_hold, ps.q, sizeof(double) * size_t(n));
        }
        if (!faulted && rx_ns != 0 && age_ns > fault_ns) {
          ctx.latch(FAULT_CMD_LOST, "command stream lost (staleness > fault-ms)");
          faulted = true;
        }
        q_des = q_hold;
        qd_des=zeros; tau_ff=zeros; kp=hold_kp; kd=hold_kd;
        if (have_cmd_gains) {  // hold with the authority the task last chose
          kp = cmd_kp;         // the SNAPSHOT — never the live buffer
          kd = cmd_kd;
        }
      }
      servo_torque(n, ps.q, ps.dq, q_des, qd_des, tau_ff, kp, kd, ps.tau_ref,
                   backend->tau_limit(), ctx.cfg.slew, tau_out);
      if (!backend->write(tau_out, n)) {
        ctx.latch(FAULT_PLANT, backend->fault_text().c_str());
        plant_ok = false;
      }
    } else {
      pose_hold.reset();
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
    out.flags = with_online_mask(
        (armed ? FLAG_ARMED : 0u) |
            (ctx.fault.load(std::memory_order_acquire) ? FLAG_FAULTED : 0u) |
            (holding ? FLAG_HOLDING : 0u) |
            (ps.wrench_valid ? FLAG_WRENCH_VALID : 0u),
        ctx.online_mask.load(std::memory_order_acquire));
    out.fault_code = ctx.fault_code.load(std::memory_order_acquire);
    std::memcpy(out.q, ps.q, sizeof(out.q));
    std::memcpy(out.dq, ps.dq, sizeof(out.dq));
    std::memcpy(out.tau, ps.tau, sizeof(out.tau));
    std::memcpy(out.tau_cmd, tau_out, sizeof(out.tau_cmd));
    std::memcpy(out.q_cmd, q_des, sizeof(double) * size_t(n));
    // The wrench field has been reserved in StatePacket since v1 and
    // FLAG_WRENCH_VALID has always gated it — filling it is NOT a wire
    // change, so no VERSION bump. Copied unconditionally: a backend with no
    // estimate leaves zeros AND clears the flag, so a client that ignores the
    // flag still reads zeros rather than stale numbers.
    std::memcpy(out.wrench, ps.wrench, sizeof(out.wrench));
    ctx.state_out.write(out);
  }
  backend->stop();
}

} // namespace arm_rt
