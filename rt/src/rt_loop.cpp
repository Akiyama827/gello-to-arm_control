// The 1 kHz servo thread. Constructs the backend HERE (libfranka raises the
// calling thread's scheduling in the Robot constructor), establishes its
// process-local RT setup (mlockall + SCHED_FIFO + --rt-cpu; a failure refuses
// ARM on real backends, while the fake runs degraded on any desktop), then
// ticks: read plant -> staleness policy -> servo law -> write.
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
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "arm_rt/servo_law.hpp"
#include "arm_rt/pose_hold.hpp"
#include "server.hpp"

namespace arm_rt {

std::unique_ptr<Backend> make_fake_backend(int n);
std::unique_ptr<Backend> make_franka_backend(const std::string& ip, double ee_mass,
                                             const double ee_com[3]);
std::unique_ptr<Backend> make_dm_backend(const std::string& can_if,
                                         uint32_t active_mask, double tau_max);

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
    backend = make_dm_backend(ctx.cfg.can_if, ctx.cfg.initial_active_mask,
                              ctx.cfg.tau_max);
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
  // Identity: joint order is provisioned per deployment; the fake backend
  // alone may default it (j0..j{n-1}) so loopback tests need no registry.
  std::vector<std::string> joints = ctx.cfg.joint_names;
  if (joints.empty() && ctx.cfg.backend == "fake")
    for (int j = 0; j < n; ++j) joints.push_back("j" + std::to_string(j));
  if (int(joints.size()) != n) {
    std::fprintf(stderr, "[rt] --joint-names lists %zu joints, %s backend drives %d\n",
                 joints.size(), backend->name(), n);
    ctx.failed.store(true);
    ctx.shutdown.store(true);
    return;
  }
  {
    std::string joined;
    for (size_t j = 0; j < joints.size(); ++j) joined += (j ? "\n" : "") + joints[j];
    ctx.joint_order_hash = fnv1a64(joined.data(), joined.size());
    std::snprintf(ctx.instance_id, sizeof(ctx.instance_id), "%s", ctx.cfg.instance_id.c_str());
    std::snprintf(ctx.model_id, sizeof(ctx.model_id), "%s", backend->task_model());
    ctx.identity_hash = identity_hash(ctx.instance_id, ctx.joint_order_hash);
    std::printf("[rt] instance '%s' joints [%s] identity %016llx task model %s\n",
                ctx.instance_id, ctx.cfg.joint_names.empty() ? "default" : "provisioned",
                (unsigned long long)ctx.identity_hash, ctx.model_id);
  }
  double tau_limits[MAX_JOINTS] = {};
  std::printf("[rt] effective torque limits (N.m):");
  for (int j = 0; j < n; ++j) {
    tau_limits[j] = ctx.cfg.tau_max > 0
        ? std::min(ctx.cfg.tau_max, backend->tau_limit()[j])
        : backend->tau_limit()[j];
    std::printf(" %.9g", tau_limits[j]);
  }
  std::printf("\n");
  // A launch flag may only be MORE conservative than the plant's torque-rate
  // limit -- main.cpp checks --slew > 0 and nothing else, so the ceiling has
  // to land here, where the backend is known. Same shape as the tau_max
  // clamp above: min(), announced, never a silent acceptance.
  const double slew = std::min(ctx.cfg.slew, backend->tau_slew_max());
  if (slew < ctx.cfg.slew)
    std::printf("[rt] --slew %.4g N.m/tick exceeds the %s torque-rate limit; "
                "clamped to %.4g\n", ctx.cfg.slew, backend->name(), slew);
  ctx.backend_n.store(n);
  ctx.online_mask.store(backend->online_mask());
  ctx.active_mask.store(backend->active_mask());
  ctx.requested_active_mask.store(backend->active_mask());
  std::snprintf(ctx.backend_name, sizeof(ctx.backend_name), "%s", backend->name());
  ctx.supports_pose_hold.store(backend->supports_pose_hold());
  // Only THIS thread moves to the isolated core; the comms threads keep the
  // default mask (the housekeeping cores, once isolcpus removes this one).
  // Published before backend_ready, so no ARM is admitted ahead of it.
  const uint32_t rt_failed = establish_rt_setup(ctx.cfg.rt_priority, ctx.cfg.rt_cpu, ctx.rt_setup_ops);
  const bool rt_required = ctx.cfg.backend != "fake" || ctx.cfg.require_rt_setup;
  ctx.rt_setup_failed.store(rt_failed);
  ctx.rt_setup_required.store(rt_required);
  {
    char names[48];
    rt_setup_names(rt_failed, names, sizeof(names));
    if (!rt_failed)
      std::printf("[rt] process RT setup ok: mlockall, SCHED_FIFO %d%s\n", ctx.cfg.rt_priority,
                  ctx.cfg.rt_cpu >= 0 ? (", cpu " + std::to_string(ctx.cfg.rt_cpu)).c_str() : "");
    else
      std::fprintf(stderr, "[rt] process RT setup FAILED: %s -- %s (check LimitMEMLOCK, "
                   "LimitRTPRIO/--rt-priority, --rt-cpu)\n", names,
                   rt_required ? "ARM will be refused; state stays observable"
                               : "running degraded (fake backend)");
  }
  ctx.backend_ready.store(true);
  std::printf("[rt] %s backend up: %d joints, tick %.1f ms, slew %.2f N.m/tick\n",
              backend->name(), n, backend->tick_s() * 1e3, slew);

  const double hold_ns = ctx.cfg.hold_ms * 1e6;
  const double fault_ns = ctx.cfg.fault_ms * 1e6;

  PlantState ps;
  CommandEnvelope envelope = {};  // this tick's mailbox copy
  ArmStart arm_start = {};
  CommandAuthority authority;     // RT-owned; tracking + deadline derive here
  const PoseHoldCommandPacket& incoming = authority.current.packet;
  const CommandPacket& cmd=incoming.command;
  PoseHold pose_hold;
  double pose_tau[MAX_JOINTS]{};
  StatePacket out = {};
  bool prev_armed = false;
  bool holding = false;
  bool have_cmd_gains = false;
  double q_hold[MAX_JOINTS] = {};
  // Gains SNAPSHOTTED at command acceptance. The hold branch must never
  // use the latest mailbox value: any datagram that lands there
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
  uint32_t stop_epoch = 0;
  uint32_t last_active_mask = backend->active_mask();

  bool plant_ok = true;  // false after any backend failure; reset only by ARM
  uint64_t seen_gen = ctx.arm_gen.load(std::memory_order_acquire);
  int read_fail_streak = 0;

  // Loop-gap diagnostics: stamps of the previous iteration's phases.
  uint64_t prev_t_read = 0, prev_compute = 0, prev_write = 0, prev_publish = 0;
  uint64_t prev_robot_ms = 0;
  while (!ctx.shutdown.load()) {
    const uint64_t t_top = mono_ns();
    if (!backend->read(ps)) {
      prev_t_read = 0;  // a failed read is logged by the fault path already
      // Latch only while ARMED: a transient read hiccup on a DISARMED idle
      // server was blocking the NEXT arm (nothing is being written, so there
      // is nothing to protect — and the first armed write re-latches anyway
      // if the plant is really down).
      if (ctx.armed.load(std::memory_order_acquire)) {
        // Before this tick's generation edge: the generation armed now.
        ctx.latch(FAULT_PLANT, backend->fault_text().c_str(),
                  ctx.arm_gen.load(std::memory_order_acquire));
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
    const uint64_t t_read = mono_ns();
    if (prev_t_read != 0 && t_read - prev_t_read > LOOP_GAP_NS) {
      const uint32_t head = ctx.gap_head.load(std::memory_order_relaxed);
      if (head - ctx.gap_tail.load(std::memory_order_acquire) < LOOP_GAP_RING) {
        ctx.gaps[head % LOOP_GAP_RING] = LoopGap{
            t_read, t_read - prev_t_read, t_read - t_top, prev_compute, prev_write,
            prev_publish, int64_t(ps.robot_time_ms) - int64_t(prev_robot_ms),
            ps.cmd_success_rate, state_seq + 1, ctx.armed.load()};
        ctx.gap_head.store(head + 1, std::memory_order_release);
      } else {
        ctx.gap_dropped.fetch_add(1, std::memory_order_relaxed);
      }
    }
    prev_t_read = t_read;
    prev_robot_ms = ps.robot_time_ms;
    uint64_t t_w0 = 0, t_w1 = 0;
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
    // Read the DISARM generation BEFORE the armed level: the control thread
    // clears `armed` first and bumps the generation second, so a generation
    // seen here is always covered by the level read below.
    const uint64_t disarm_gen = ctx.disarm_gen.load(std::memory_order_acquire);
    const bool armed = ctx.armed.load(std::memory_order_acquire);
    bool faulted = ctx.fault.load(std::memory_order_acquire);

    // Generation FIRST, then the mailboxes: begin_arm publishes the
    // ARM-start record before the generation, so a new generation seen here
    // comes with its record. The mailbox reads cannot fail or tear; one read
    // each per tick, whatever udp_rx or the control thread does.
    const uint64_t gen = ctx.arm_gen.load(std::memory_order_acquire);
    ctx.arm_start.read(arm_start);
    const uint64_t cmd_v = ctx.cmd_in.read(envelope);
    authority.observe(gen, arm_start, cmd_v, envelope);
    const uint64_t rx_ns = authority.deadman_base();
    // Deadman age (CMD_LOST), from this generation only. SIGNED, clamped age:
    // stamps are taken in parallel with this tick: a packet stamped between our `now` sample
    // above and this load is nanoseconds IN THE FUTURE, and the old unsigned
    // `now - rx_ns` underflowed to ~1.8e19 ns — an instant spurious CMD_LOST
    // latch on a perfectly healthy 100 Hz stream. Odds ~ packet rate x the
    // sample-to-load window (sub-us pinned, up to ms when this thread is
    // preemptible), i.e. one false latch per minutes of armed time — found
    // live on rung 3 and root-caused with a fake-plant soak + process
    // autopsy after every stream instrument reported healthy flow.
    const double age_ns = rx_ns == 0 ? 1e18 : clamped_age_ns(now, rx_ns);
    // Only commands of the CURRENT generation and session, received within
    // hold by their OWN ingress stamp, are authority: a newer packet's stamp
    // can no longer freshen an older payload, and a retired handler's late
    // publish is neither tracked nor able to move this generation's deadline.
    const bool fresh = authority.fresh(ctx.session_id.load(std::memory_order_acquire),
                                       n, now, hold_ns);

    const double* q_des = zeros;
    const double* qd_des = zeros;
    const double* tau_ff = zeros;
    const double* kp = hold_kp;
    const double* kd = hold_kd;

    // The ARM edge is a GENERATION, not a level: a DISARM->ARM pair that
    // completes between two 1 kHz samples still changes arm_gen, so the
    // per-epoch resets can never be skipped by fast recovery cycles.
    if (gen != seen_gen) {
      const uint64_t stopped_gen = seen_gen;
      seen_gen = gen;
      if (prev_armed && !backend->stop())  // the DISARM we never sampled
        ctx.journal(FAULT_STOP_FAILED, FAULT_PLANT, backend->fault_text().c_str(),
                    stopped_gen);
      plant_ok = true;         // ARM = explicit plant retry
      have_cmd_gains = false;  // fresh epoch: hold gains until the task speaks
      holding = false;         // re-capture the hold pose in this epoch
      pose_hold.reset();
    }
    if(armed) {
      for(int j=0;j<n;++j) {
        if(!std::isfinite(ps.q[j]) || !std::isfinite(ps.dq[j]) || !std::isfinite(ps.tau_ref[j])) {
          ctx.latch(FAULT_PLANT, "non-finite measured joint state; torque writes suppressed", seen_gen);
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
            ctx.latch(FAULT_PLANT, "pose hold requires valid measured pose/J/model state", seen_gen);
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
          ctx.latch(FAULT_CMD_LOST, "command stream lost (staleness > fault-ms)", seen_gen);
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
                   tau_limits, slew, tau_out);
      // Preserve the authority-selected fields for native MIT. In particular,
      // tau_out contains PD already; it must NOT become an extra MIT tau_ff.
      CommandPacket selected{};
      selected.n = uint16_t(n);
      std::copy_n(q_des, n, selected.q_des);
      std::copy_n(qd_des, n, selected.qd_des);
      std::copy_n(tau_ff, n, selected.tau_ff);
      std::copy_n(kp, n, selected.kp);
      std::copy_n(kd, n, selected.kd);
      t_w0 = mono_ns();
      const bool wrote = backend->write_command(selected, slew, tau_out);
      t_w1 = mono_ns();
      if (!wrote) {
        ctx.latch(FAULT_PLANT, backend->fault_text().c_str(), seen_gen);
        plant_ok = false;
        holding = false;
        std::fill_n(tau_out, n, 0.0);  // rejected output is not commanded torque
      }
    } else {
      pose_hold.reset();
      if (prev_armed && !backend->stop())
        ctx.journal(FAULT_STOP_FAILED, FAULT_PLANT, backend->fault_text().c_str(), seen_gen);
      holding = false;
      std::memset(tau_out, 0, sizeof(tau_out));
      // Disarmed at this sample and any stop() returned: the request is done
      // as far as this process can know. Not a claim about the robot's brakes.
      stop_epoch = uint32_t(disarm_gen);
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
    out.session_id = ctx.session_id.load(std::memory_order_acquire);
    out.identity_hash = ctx.identity_hash;
    out.arm_epoch = uint32_t(seen_gen);
    out.stop_epoch = stop_epoch;
    std::memcpy(out.q, ps.q, sizeof(out.q));
    std::memcpy(out.dq, ps.dq, sizeof(out.dq));
    std::memcpy(out.tau, ps.tau, sizeof(out.tau));
    std::memcpy(out.tau_cmd, tau_out, sizeof(out.tau_cmd));
    std::memcpy(out.q_cmd, q_des, sizeof(double) * size_t(n));
    // Echo the reference selected AFTER hold/fault/pose-hold handling, never
    // the latest UDP packet (which may no longer have command authority).
    std::memcpy(out.qd_cmd, qd_des, sizeof(double) * size_t(n));
    // The wrench field has been reserved in StatePacket since v1 and
    // FLAG_WRENCH_VALID has always gated it — filling it is NOT a wire
    // change, so no VERSION bump. Copied unconditionally: a backend with no
    // estimate leaves zeros AND clears the flag, so a client that ignores the
    // flag still reads zeros rather than stale numbers.
    std::memcpy(out.wrench, ps.wrench, sizeof(out.wrench));
    ctx.state_out.write(out);
    const uint64_t t_pub = mono_ns();
    if (t_w0 == 0) t_w0 = t_w1 = t_pub;  // no write this tick (disarmed/plant down)
    prev_compute = t_w0 - t_read;
    prev_write = t_w1 - t_w0;
    prev_publish = t_pub - t_w1;
  }
  if (!backend->stop())
    ctx.journal(FAULT_STOP_FAILED, FAULT_PLANT, backend->fault_text().c_str(), seen_gen);
}

} // namespace arm_rt
