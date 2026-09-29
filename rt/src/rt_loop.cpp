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
#include "arm_rt/controller.hpp"
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
  Controller controller;
  StatePacket out = {};
  ReferenceSelection reference;
  ReferenceCommit reference_commit{};
  ReferenceCancel reference_cancel{};
  uint64_t reference_seen=0, cancel_seen=0;
  ArmMode mode;
  ModeSelectRequest mode_request{};
  uint64_t mode_seen=0;
  double reference_point[1+3*MAX_JOINTS]{};
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
  double tau_raw[MAX_JOINTS] = {};  // pre-clamp law output, diagnostics only
  // Bounded controller context of the LAST finished servo computation
  // (CTRL-010). Stamped into a loop-gap record at the NEXT read-return, so a
  // gap record describes exactly the cycle whose compute+write it contains.
  CtrlDiag diag{};
  bool diag_valid = false;
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
    uint32_t retire_cause = 0;  // CTRL-034: set when a reference is retired this tick
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
            ps.cmd_success_rate, state_seq + 1, ctx.armed.load(),
            diag_valid, ctx.session_id.load(std::memory_order_acquire),
            uint32_t(seen_gen), last_cmd_seq, diag};
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
    const uint64_t session=ctx.session_id.load(std::memory_order_acquire);
    // --- ARM-epoch mode authority (CAP_MODE_SELECT) ------------------------
    // ARM seeds generation 0: MODE_REFERENCE iff this session's disarmed
    // CTL_REFERENCE_MODE opt-in was accepted, else MODE_STREAMED. DISARM and a
    // new epoch discard the whole state; MODE_UNSELECTED never survives an ARM.
    if (armed && (gen != mode.epoch || session != mode.session_id)) {
      const uint16_t seed =
          (session && ctx.supports_reference.load(std::memory_order_acquire) &&
           ctx.reference_session.load(std::memory_order_acquire) == session)
              ? MODE_REFERENCE : MODE_STREAMED;
      mode.arm(session, uint32_t(gen), seed);
      ctx.mode_generation.store(mode.generation, std::memory_order_release);
      ctx.mode_effective.store(mode.mode, std::memory_order_release);
    } else if (!armed && mode.mode != MODE_UNSELECTED) {
      mode.disarm();
      ctx.mode_generation.store(0, std::memory_order_release);
      ctx.mode_effective.store(MODE_UNSELECTED, std::memory_order_release);
    }
    // A request published at acceptance is answered within this tick. While it
    // is unconfirmed both families are fenced (the arm holds); confirmation
    // moves the floor to `now`, so the envelope already in the Mailbox never
    // actuates. RT is the only confirmer.
    const uint64_t mode_pub = ctx.mode_published.load(std::memory_order_acquire);
    if (mode_pub != mode_seen) {
      ctx.mode_in.read(mode_request);
      if (mode_request.ticket == mode_pub) {
        ModeSelectInfo info{};
        ModeSelectReplyPayload& r = info.reply;
        const ModeSelectPayload& mreq = mode_request.request;
        r.session_id = mreq.session_id;
        r.generation = mreq.generation;
        r.predecessor = mode.generation;
        r.arm_epoch = mreq.arm_epoch;
        r.requested_mode = mreq.desired_mode;
        r.selected_mode = mode.mode;
        uint32_t result;
        const bool latched =
            ctx.latch_state.load(std::memory_order_acquire) == ServerCtx::LATCH_SET;
        if (!armed || faulted || latched) {
          result = MODE_SEL_REFUSED;
        } else if (!mode.belongs(session, uint32_t(gen)) ||
                   mreq.session_id != session || mreq.arm_epoch != gen) {
          // The epoch this request named has ended: it can never be confirmed.
          result = MODE_SEL_REFUSED;
        } else if (mreq.desired_mode == MODE_REFERENCE &&
                   ctx.reference_session.load(std::memory_order_acquire) != session) {
          result = MODE_SEL_UNSUPPORTED;
        } else if (mreq.generation == mode.generation &&
                   mreq.desired_mode == mode.mode) {
          result = MODE_SEL_OK;  // exact retry of the confirmed generation
        } else if (mreq.generation == mode.generation + 1 &&
                   mreq.expected_predecessor == mode.generation) {
          // Retire the outgoing law through its existing owner BEFORE the new
          // one becomes authoritative: no old reference, envelope or candidate
          // survives, and `holding=false` re-captures the measured hold pose.
          reference.retire();
          holding = false;
          mode.confirm(mreq, now);
          ctx.mode_generation.store(mode.generation, std::memory_order_release);
          ctx.mode_effective.store(mode.mode, std::memory_order_release);
          r.selected_mode = mode.mode;
          result = MODE_SEL_OK;
        } else {
          result = MODE_SEL_ORDER;
        }
        r.result = result;
        ctx.mode_feedback.write(info);
        ctx.mode_done.store(mode_pub, std::memory_order_release);
        mode_seen = mode_pub;
      }
    }
    reference.reset(session,uint32_t(gen));
    const uint64_t published=ctx.reference_published.load(std::memory_order_acquire);
    if(published!=reference_seen) ctx.reference_in.read(reference_commit);
    const uint64_t cancelled=ctx.reference_cancel_generation.load(std::memory_order_acquire);
    if(cancelled!=cancel_seen) {
      ctx.reference_cancel.read(reference_cancel);
      if(reference_cancel.generation==cancelled) {
        if(reference_cancel.session_id==session && reference_cancel.arm_epoch==gen) {
          reference.retire(reference_cancel.reference_id);
          holding=false; // capture measured position once at this event
        }
        cancel_seen=cancelled;
      }
    }
    if(!armed || faulted) reference.retire();
    // Exhaustion is classified before considering a new admission.
    const uint32_t previous_reference_status=reference.info().status;
    reference.evaluate(now,reference_point);
    if(published!=reference_seen && reference_commit.ticket==published) {
      ReferenceResult result=REF_IDENTITY;
      if(armed && !faulted && ctx.reference_session.load()==session) {
        // A reference is admitted only into the native law. A mode change
        // retires an in-flight COMMIT through the cancel generation the control
        // thread publishes at acceptance, so nothing from the old law lands.
        if(mode.mode!=MODE_REFERENCE)
          result=REF_NOT_ADMITTED;
        else
          result=reference.commit(reference_commit,now,cancelled);
      }
      ctx.reference_result.store(result,std::memory_order_relaxed);
      reference_seen=published;
      ctx.reference_done.store(published,std::memory_order_release);
    }
    reference.evaluate(now,reference_point);
    if(reference.info().status==REFERENCE_UNDERFLOW && previous_reference_status!=REFERENCE_UNDERFLOW)
      holding=false; // underflow captures measured q once, independently of prior staleness hold
    ctx.reference_feedback.write(reference.info());
    // Wrong reference identity/phase never refreshes the command deadline.
    // A retired revision can report a live transport, but cannot regain selection.
    // Family validity — the baseline predicate with ONE mode-aware exception.
    // A reference id is RETAINED after retire(), so in the streamed law it must
    // not discard the new law's own envelopes: dropping them would starve the
    // deadman and latch CMD_LOST on a healthy, streaming PC (found by the
    // native->Soft transition case). Wrong reference identity/phase still never
    // refreshes the deadline, and a bare server never has an id to retain.
    const bool envelope_current = cmd_v != 0 && envelope.arm_gen == gen &&
        envelope.session_id == session;
    const bool incoming_native = envelope_current &&
        envelope.packet.command.version==REFERENCE_VERSION;
    // CTRL-034 decision (a), gap bound: a selected (reference_id, revision)
    // that has not admitted ANY envelope within the freshness ceiling is about
    // to be retired. This is evaluated BEFORE admission so the late first
    // envelope cannot be accepted at the advanced reference phase: that is
    // exactly the 107 mrad single-frame step the reviewer measured when the
    // first envelope arrived ~300 ms after COMMIT.
    const bool reference_gap =
        mode.mode==MODE_REFERENCE && reference.info().status==REFERENCE_RUNNING &&
        reference.admitted_at_ns()!=0 && !reference.envelope_seen() &&
        clamped_age_ns(now,reference.admitted_at_ns()) >
          std::min(hold_ns,REFERENCE_FIRST_ENVELOPE_NS);
    const double reference_cut = reference.protected_until();
    bool admit_envelope=true;
    if(incoming_native) {
      const bool for_active = reference_envelope_revision_ok(
          envelope.reference,reference.info(),reference_cut);
      admit_envelope = !reference.retired_by_expiry() &&
          !reference_gap &&
          reference_envelope_fresh(envelope.reference,reference.info(),reference_cut,
                                   now,envelope.rx_ns,hold_ns);
      if(admit_envelope && for_active && reference.active())
        reference.mark_envelope();
    } else if(reference.info().reference_id && mode.mode!=MODE_STREAMED) {
      admit_envelope=false;
    }
    authority.observe(gen, arm_start, admit_envelope?cmd_v:0, envelope);
    // --- retire on expiry (CTRL-034 decision (a)) --------------------------
    // In MODE_REFERENCE a RUNNING reference whose accepted envelope has expired
    // is FINISHED, not merely held: the arm holds its measured position with
    // the captured gains and zero feedforward, and motion resumes only through
    // a NEW reference the PC plans from the measured position. The guard is
    // `held_for_active`: an envelope for the CURRENT (reference_id, revision)
    // must already have been accepted, which is exactly what keeps a normal
    // revision switch from retiring a healthy reference; the no-envelope gap
    // is bounded separately by `reference_gap`. END is deliberately excluded:
    // a terminal endpoint is stationary, so retiring it buys no safety and
    // would drop the compensated endpoint hold the PC still relies on.
    // MODE_STREAMED never retires here.
    if (mode.mode == MODE_REFERENCE
        && reference.info().status == REFERENCE_RUNNING) {
      const ReferenceSelectionInfo& info = reference.info();
      const bool incoming_for_active = incoming_native &&
          reference_envelope_revision_ok(envelope.reference,info,reference_cut);
      const bool held_for_active = authority.have &&
          authority.current.arm_gen == gen &&
          authority.current.packet.command.version == REFERENCE_VERSION &&
          reference_envelope_revision_ok(authority.current.reference,info,reference_cut);
      bool expired = reference_gap;
      const char* expiry_rule = "first-envelope gap";
      const CommandEnvelope* expired_envelope = nullptr;
      if (!expired && incoming_for_active) {
        expired = !reference_envelope_fresh(envelope.reference, info, reference_cut,
                                            now, envelope.rx_ns, hold_ns);
        if (expired) {
          expiry_rule = "incoming envelope stale";
          expired_envelope = &envelope;
        }
      } else if (!expired && held_for_active) {
        expired = !reference_envelope_fresh(authority.current.reference, info,
                                            reference_cut, now,
                                            authority.current.rx_ns, hold_ns);
        if (expired) {
          expiry_rule = "held envelope stale";
          expired_envelope = &authority.current;
        }
      }
      if (expired) {
        reference.retire(0, /*expired=*/true);
        retire_cause = FAULT_REFERENCE_EXPIRED;
        holding = false;  // capture the measured position once, at this event
        char detail[FAULT_RECORD_TEXT]{};
        if (expired_envelope) {
          const auto& b = expired_envelope->reference;
          const double age_us =
              clamped_age_ns(now, expired_envelope->rx_ns) / 1000.0 + b.age_us;
          std::snprintf(detail, sizeof(detail),
                        "reference envelope expired: %s arm_gen=%llu "
                        "session_id=%016llx reference_id=%llu revision=%u age_us=%.0f",
                        expiry_rule,
                        static_cast<unsigned long long>(expired_envelope->arm_gen),
                        static_cast<unsigned long long>(expired_envelope->session_id),
                        static_cast<unsigned long long>(b.reference_id), b.revision, age_us);
        } else {
          std::snprintf(detail, sizeof(detail),
                        "reference envelope expired: %s reference_id=%llu revision=%u "
                        "gap_age_us=%.0f",
                        expiry_rule, static_cast<unsigned long long>(info.reference_id),
                        info.revision,
                        clamped_age_ns(now, reference.admitted_at_ns()) / 1000.0);
        }
        ctx.journal(FAULT_EVENT, FAULT_REFERENCE_EXPIRED, detail, gen);
      }
    }
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
    bool fresh = authority.fresh(ctx.session_id.load(std::memory_order_acquire),
                                n, now, hold_ns);

    const bool native_command=cmd.version==REFERENCE_VERSION;
    if(native_command) {
      // Which law may actuate is the MODE's decision, not the reference ID's.
      // A retired family, or an envelope already in the Mailbox when the floor
      // was raised, is not authority and holds instead.
      fresh=fresh && mode.admits(true, authority.current.rx_ns, session, uint32_t(gen))
            && reference.active() && reference_envelope_fresh(
        authority.current.reference,reference.info(),reference_cut,now,
        authority.current.rx_ns,hold_ns);
    } else {
      fresh=fresh && mode.admits(false, authority.current.rx_ns,
                                 session, uint32_t(gen));
    }

    const double* q_des = zeros;
    const double* qd_des = zeros;

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
      controller.reset();
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
      controller.reset();
      holding = false;  // nothing is held — the robot's own safety has it
      std::memset(tau_out, 0, sizeof(tau_out));
      diag_valid = false;  // no servo computation ran: context is absent
    } else if (armed) {
      if (!faulted && fresh) {
        holding = false;
        have_cmd_gains = true;
        last_cmd_seq = cmd.seq;
        std::memcpy(cmd_kp, cmd.kp, sizeof(cmd_kp));  // accepted -> snapshot
        std::memcpy(cmd_kd, cmd.kd, sizeof(cmd_kd));
        if(cmd.version==POSE_HOLD_VERSION) {
          if(!controller.pose(n, ps, incoming.pose_hold, backend->tick_s(),
                              backend->supports_pose_hold())) {
            ctx.latch(FAULT_PLANT, "pose hold requires valid measured pose/J/model state", seen_gen);
            faulted=true;
          }
        } else {
          controller.joint(n, native_command?reference_point+1:cmd.q_des,
                           native_command?reference_point+1+n:cmd.qd_des,
                           cmd.tau_ff, cmd_kp, cmd_kd);
        }
      }
      if(faulted || !fresh) {
        if (!holding) {
          holding = true;
          std::memcpy(q_hold, ps.q, sizeof(double) * size_t(n));
        }
        if (!faulted && rx_ns != 0 && age_ns > fault_ns) {
          ctx.latch(FAULT_CMD_LOST, "command stream lost (staleness > fault-ms)", seen_gen);
          faulted = true;
        }
        // Retain the accepted gain snapshot, never the live ingress buffer.
        controller.joint(n, q_hold, zeros, zeros,
                         have_cmd_gains?cmd_kp:hold_kp, have_cmd_gains?cmd_kd:hold_kd);
      }
      const auto& selected = controller.command();
      q_des = selected.q_des;
      qd_des = selected.qd_des;
      servo_torque(n, ps.q, ps.dq, selected.q_des, selected.qd_des, selected.tau_ff,
                   selected.kp, selected.kd, ps.tau_ref, tau_limits, slew, tau_out,
                   tau_raw);
      t_w0 = mono_ns();
      const bool wrote = backend->write_command(selected, slew, tau_out);
      t_w1 = mono_ns();
      diag = capture_ctrl_diag(n, ps.q, ps.dq, selected.q_des, selected.qd_des,
                               selected.tau_ff, tau_raw, tau_out, ps.tau_ref,
                               ps.tau, tau_limits, slew);
      diag.reference_retire_cause = retire_cause;
      diag_valid = true;
      if (!wrote) {
        ctx.latch(FAULT_PLANT, backend->fault_text().c_str(), seen_gen);
        plant_ok = false;
        holding = false;
        std::fill_n(tau_out, n, 0.0);  // rejected output is not commanded torque
      }
    } else {
      controller.reset();
      if (prev_armed && !backend->stop())
        ctx.journal(FAULT_STOP_FAILED, FAULT_PLANT, backend->fault_text().c_str(), seen_gen);
      holding = false;
      std::memset(tau_out, 0, sizeof(tau_out));
      diag_valid = false;  // no servo computation ran: context is absent
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
    // Confirmed effective law, reported ONLY where CAP_MODE_SELECT is
    // advertised: a bare (enable_reference: false) server's flags stay exactly
    // as they were, so its state stream is byte-identical.
    uint32_t mode_flags = 0;
    if (ctx.supports_reference.load(std::memory_order_acquire)) {
      if (mode.mode == MODE_REFERENCE) mode_flags = FLAG_MODE_REFERENCE;
      else if (mode.mode == MODE_STREAMED) mode_flags = FLAG_MODE_STREAMED;
    }
    out.flags = with_online_mask(
        (armed ? FLAG_ARMED : 0u) |
            (ctx.fault.load(std::memory_order_acquire) ? FLAG_FAULTED : 0u) |
            (holding ? FLAG_HOLDING : 0u) |
            (ps.wrench_valid ? FLAG_WRENCH_VALID : 0u) | mode_flags,
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
    if(session && ctx.reference_session.load()==session) {
      ReferenceStatePacket extended{};
      extended.state=out; extended.state.version=REFERENCE_VERSION;
      const auto& r=reference.info();
      extended.reference_id=r.reference_id; extended.revision=r.revision;
      extended.status=r.status; extended.phase=r.phase;
      extended.envelope_age_us=UINT32_MAX;
      if(authority.have && native_command) {
        extended.envelope_phase=authority.current.reference.phase;
        extended.envelope_seq=cmd.seq;
        extended.envelope_age_us=uint32_t(std::min(double(UINT32_MAX),
          clamped_age_ns(now,authority.current.rx_ns)/1000+authority.current.reference.age_us));
      }
      if(reference.active()) {
        std::copy_n(reference_point+1,n,extended.q_ref);
        std::copy_n(reference_point+1+n,n,extended.qd_ref);
        std::copy_n(reference_point+1+2*n,n,extended.qdd_ref);
      }
      ctx.reference_state_out.write(extended);
    }
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
