// TCP control channel: fixed 128-byte ControlPacket frames, one client at a
// time (the direct link has exactly one PC). Carries what must not be lost:
// arm/disarm, heartbeat, and the server's STATUS/FAULT events.
//
// Safety semantics live here and in the RT loop together:
//  - ARM is refused while a fault is latched — DISARM first (the explicit
//    DISARM->ARM cycle is the only path past a latch; nothing auto-re-arms).
//  - Losing the session while armed LATCHES and the arm HOLDS. It does not
//    drop authority: a crashed PC must leave the arm parked, not falling.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <optional>
#include "reference_upload.hpp"

#include "clock_domain.hpp"
#include "server.hpp"

namespace arm_rt {
namespace {

bool send_frame(int fd, const ControlPacket& pkt) {
  const char* p = reinterpret_cast<const char*>(&pkt);
  size_t left = sizeof(pkt);
  while (left > 0) {
    const ssize_t sent = send(fd, p, left, MSG_NOSIGNAL);
    if (sent <= 0) return false;
    p += sent;
    left -= size_t(sent);
  }
  return true;
}

ControlPacket make(uint16_t type, uint32_t arg, const char* text, uint32_t seq = 0) {
  ControlPacket pkt = {};
  pkt.magic = MAGIC_CTL;
  pkt.version = VERSION;
  pkt.type = type;
  pkt.seq = seq;
  pkt.arg = arg;
  pkt.t_mono_ns = mono_ns();
  if (text) std::snprintf(pkt.text, sizeof(pkt.text), "%s", text);
  return pkt;
}

uint32_t flags_snapshot(const ServerCtx& ctx) {
  uint32_t f = 0;
  if (ctx.armed.load()) f |= FLAG_ARMED;
  if (ctx.fault.load()) f |= FLAG_FAULTED;
  if (ctx.claim_ok.load()) f |= FLAG_CLAIMED;
  return with_online_mask(f, ctx.online_mask.load());
}

uint32_t capabilities(const ServerCtx& ctx) {
  // CAP_CALIBRATED_TASK_SPACE is never advertised: no backend evaluates a
  // deployment-approved model in the servo (see protocol.hpp). CAP_MODE_SELECT
  // rides on native-reference support, so a bare (enable_reference: false)
  // deployment neither offers the opcode nor sets the mode flags.
  return CAP_OBSERVER | (ctx.supports_pose_hold.load() ? CAP_POSE_HOLD : 0u) |
    (ctx.supports_reference.load()
       ? uint32_t(CAP_REFERENCE_HORIZON | CAP_MODE_SELECT) : 0u);
}

// Bounded wait for RT's mode-selection acknowledgement. The control thread does
// NOT block: it keeps servicing the socket and answers when `mode_done` matches
// its ticket, or reports PENDING once this budget expires. RT confirms within
// its next tick normally. While a reply is outstanding, the control thread
// polls at 1 ms so acknowledgement is not delayed by the idle 100 ms wait.
constexpr uint64_t MODE_SELECT_WAIT_NS = 250'000'000;

ControlPacket mode_reply(uint16_t type, const ModeSelectReplyPayload& r,
                         uint32_t seq) {
  ControlPacket pkt = make(type, 0, nullptr, seq);
  std::memcpy(pkt.text, &r, sizeof(r));
  return pkt;
}

IdentityPayload identity(const ServerCtx& ctx, uint64_t session) {
  IdentityPayload id{};
  std::memcpy(id.instance_id, ctx.instance_id, ID_CHARS);
  std::memcpy(id.model_id, ctx.model_id, ID_CHARS);
  id.session_id = session;
  id.joint_order_hash = ctx.joint_order_hash;
  id.identity_hash = ctx.identity_hash;
  id.capabilities = capabilities(ctx);
  id.n = uint32_t(ctx.backend_n.load());
  return id;
}

// Why a CLAIM does not match, or nullptr. Every field a commander relies on
// is compared; the claim cannot ask for less than it will use.
const char* claim_mismatch(const ServerCtx& ctx, const IdentityPayload& want,
                           uint64_t session) {
  if (want.session_id != session) return "claim refused: wrong session";
  char instance[ID_CHARS] = {};
  std::memcpy(instance, want.instance_id, ID_CHARS);
  instance[ID_CHARS - 1] = '\0';
  if (std::memcmp(instance, ctx.instance_id, ID_CHARS) != 0)
    return "claim refused: instance id mismatch";
  if (want.n != uint32_t(ctx.backend_n.load())) return "claim refused: joint count mismatch";
  if (want.joint_order_hash != ctx.joint_order_hash)
    return "claim refused: joint order mismatch";
  if (want.model_id[0] != '\0' &&
      std::strncmp(want.model_id, ctx.model_id, ID_CHARS) != 0)
    return "claim refused: task model mismatch";
  if (want.capabilities & ~capabilities(ctx))
    return "claim refused: required capability not offered";
  return nullptr;
}

uint64_t fresh_session() {
  static std::random_device rd;
  uint64_t v = 0;
  while (v == 0) v = (uint64_t(rd()) << 32) ^ uint64_t(rd());
  return v;
}

void serve_client(ServerCtx& ctx, int fd, uint64_t session) {
  const int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  // A peer that stops draining must not wedge this thread mid-send forever
  // (the deadman below never runs while send() blocks, and shutdown joins
  // hang until SIGKILL). A short send timeout turns that into session loss.
  timeval stv{0, 500'000};
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof(stv));
  // HELLO only once the backend exists and its RT setup is published — a
  // client connecting during franka construction (~1 s) must not be told
  // "0 joints" or rt=ok before the setup ran. No timeout: a server that never
  // becomes ready says nothing (the client's own timeout reports it).
  while (!ctx.backend_ready.load() && !ctx.shutdown.load()) {
    timespec ts{0, 10'000'000};
    nanosleep(&ts, nullptr);
  }
  if (!ctx.backend_ready.load()) {  // shutting down before ready: no session was used
    ctx.session_id.store(0);
    close(fd);
    return;
  }
  // HELLO text carries the launch-time SAFETY flags, not just the backend
  // name: the client must be able to refuse a server whose deadman is
  // configured for hand-guiding (--fault-ms 3600000) — a commander graph
  // against that server has no staleness reflex at all (audit 2026-07-29).
  // rt= reports the process-local RT setup ("ok" or the failed parts); an
  // extra key=value token, so existing clients ignore it.
  char hello[sizeof(ControlPacket{}.text)];
  char capability[24] = {};
  char clock_domain[48] = {};  // " clk=mono:<host12>:<boot12>" (CTRL-041), before rt= so a long rt= truncates first
  char rt_setup[48];
  {
    const std::string token = clock_domain_token(ctx.cfg.machine_id_file, ctx.cfg.boot_id_file);
    if (!token.empty()) std::snprintf(clock_domain, sizeof(clock_domain), " %s", token.c_str());
  }
  if (ctx.supports_pose_hold.load())
    std::snprintf(capability, sizeof(capability), " pose_hold=%u", POSE_HOLD_VERSION);
  rt_setup_names(ctx.rt_setup_failed.load(), rt_setup, sizeof(rt_setup));
  std::snprintf(hello, sizeof(hello),
                "%s hold_ms=%.0f fault_ms=%.0f active=0x%X%s%s rt=%s",
                ctx.backend_name, ctx.cfg.hold_ms, ctx.cfg.fault_ms,
                ctx.active_mask.load(), capability, clock_domain, rt_setup);
  send_frame(fd, make(CTL_HELLO, uint32_t(ctx.backend_n.load()), hello));
  {
    ControlPacket idf = make(CTL_IDENTITY, uint32_t(ctx.backend_n.load()), nullptr);
    const IdentityPayload id = identity(ctx, session);
    std::memcpy(idf.text, &id, sizeof(id));
    send_frame(fd, idf);
  }

  // Session deadman: the client answers CTL_PING with CTL_PONG, so a healthy
  // idle link always has traffic. A yanked cable / dead PC leaves a half-open
  // socket that poll() never reports — without this, FAULT_CTL_LOST could
  // take the kernel's 2-hour keepalive to fire.
  constexpr uint64_t PING_EVERY_NS = 250'000'000;   // 4 Hz
  constexpr uint64_t DEAD_AFTER_NS = 1'200'000'000; // ~5 missed pings
  uint64_t last_alive_ns = mono_ns();
  uint64_t last_ping_ns = 0;

  std::optional<ReferenceUpload> upload;
  ReferenceModePayload bound{};
  ReferenceSelectionInfo selection{};
  ReferenceCommit pending{};
  ControlPacket commit_frame{};
  ReferenceReplyPayload commit_reply{};
  ReferenceRetirePayload last_cancel{};
  bool commit_waiting=false, have_commit_reply=false;
  // Armed-only mode selection: the request outstanding until RT answers, and
  // whether we already told the client PENDING (we do not send a second frame;
  // a later retry reconciles through RT's idempotent generation).
  ModeSelectPayload mode_request{};
  ControlPacket mode_frame{};
  ModeSelectInfo mode_info{};
  uint64_t mode_ticket=0, mode_deadline_ns=0;
  bool mode_outstanding=false, mode_pending_sent=false;
  ControlPacket rx;
  size_t have = 0;
  while (!ctx.shutdown.load()) {
    ctx.reference_feedback.read(selection);
    // Bounded wait for the RT mode-selection acknowledgement. Never blocks the
    // servo and never blocks this socket: we keep polling and answer when the
    // ticket completes, or report PENDING once and keep reconciling.
    if (mode_outstanding &&
        ctx.mode_done.load(std::memory_order_acquire)==mode_ticket) {
      ctx.mode_feedback.read(mode_info);
      if (!mode_pending_sent &&
          !send_frame(fd, mode_reply(CTL_MODE_STATUS, mode_info.reply,
                                     mode_frame.seq)))
        break;
      mode_outstanding=false;
      mode_pending_sent=false;
    } else if (mode_outstanding && !mode_pending_sent &&
               mono_ns()>=mode_deadline_ns) {
      ModeSelectReplyPayload r{};
      r.session_id=mode_request.session_id;
      r.generation=mode_request.generation;
      r.arm_epoch=mode_request.arm_epoch;
      r.requested_mode=mode_request.desired_mode;
      r.selected_mode=ctx.mode_effective.load(std::memory_order_acquire);
      r.predecessor=ctx.mode_generation.load(std::memory_order_acquire);
      r.result=MODE_SEL_PENDING;
      if(!send_frame(fd, mode_reply(CTL_MODE_STATUS,r,mode_frame.seq))) break;
      mode_pending_sent=true;
    }
    if(commit_waiting && ctx.reference_done.load(std::memory_order_acquire)==pending.ticket) {
      commit_reply.result=ctx.reference_result.load(std::memory_order_relaxed);
      ControlPacket answer=make(CTL_REFERENCE_STATUS,0,nullptr,commit_frame.seq);
      answer.version=REFERENCE_VERSION;
      std::memcpy(answer.text,&commit_reply,sizeof(commit_reply));
      if(!send_frame(fd,answer)) break;
      commit_waiting=false; have_commit_reply=true;
      if(commit_reply.result==REF_OK && upload) upload->discard();
    }
    ServerCtx::FaultNotice notice;
    if (ctx.pending_fault_notice(notice)) {
      // Acknowledge only a DELIVERED frame, and only its own serial: a failed
      // send keeps the notice owed, and a newer latch is never discarded by
      // the acknowledgement of an older one. A latch DISARM has cleared is
      // not owed at all (the journal keeps it).
      if (!send_frame(fd, make(CTL_FAULT, notice.code, notice.text))) break;
      ctx.ack_fault_notice(notice.seq);
    }
    const uint64_t now = mono_ns();
    if (now - last_ping_ns > PING_EVERY_NS) {
      last_ping_ns = now;
      send_frame(fd, make(CTL_PING, 0, nullptr));
    }
    if (now - last_alive_ns > DEAD_AFTER_NS) break;  // half-open: treat as lost
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, (commit_waiting || mode_outstanding) ? 1 : 100);
    if (ready < 0) break;
    if (ready == 0) continue;
    const ssize_t got = recv(fd, reinterpret_cast<char*>(&rx) + have,
                             sizeof(rx) - have, 0);
    if (got <= 0) break;  // client gone
    have += size_t(got);
    if (have < sizeof(rx)) continue;
    have = 0;
    if (rx.magic != MAGIC_CTL) continue;
    if (rx.version==REFERENCE_VERSION) {
      ctx.reference_feedback.read(selection); // refresh after a possibly blocking poll
      // One frame per iteration: deadline/fault processing is never skipped by
      // a producer flood. There is no batch drain or unbounded upload queue.
      ControlPacket checked{};
      if(!upload || !decode_reference_control(&rx,sizeof(rx),true,checked)) continue;
      upload->observe_epoch(ctx.armed.load()?uint32_t(ctx.arm_gen.load()):0);
      ReferenceResult result=REF_NOT_ADMITTED;
      ReferenceReplyPayload reply{}; reply.operation=rx.type;
      switch(rx.type) {
        case CTL_REFERENCE_BEGIN: {
          ReferenceBeginPayload b; std::memcpy(&b,rx.text,sizeof(b));
          reply.upload_id=b.upload_id; reply.reference_id=b.reference_id; reply.revision=b.revision;
          const bool active=selection.session_id==session && selection.arm_epoch==ctx.arm_gen.load() &&
            (selection.status==REFERENCE_RUNNING || selection.status==REFERENCE_END);
          const bool new_after_end=active && selection.status==REFERENCE_END &&
            b.reference_id>selection.reference_id && b.expected_revision==0;
          result=commit_waiting?REF_BUSY:upload->begin(b,selection.reference_id,
                                                      active&&!new_after_end?selection.revision:0);
          break;
        }
        case CTL_REFERENCE_CHUNK: {
          ReferenceChunkPayload c; std::memcpy(&c,rx.text,sizeof(c));
          reply.upload_id=c.upload_id; result=upload->chunk(c); break;
        }
        case CTL_REFERENCE_SEAL: {
          ReferenceSealPayload s; std::memcpy(&s,rx.text,sizeof(s));
          reply.upload_id=s.upload_id; result=upload->seal(s,mono_ns()); break;
        }
        case CTL_REFERENCE_ABORT: {
          ReferenceRetirePayload a; std::memcpy(&a,rx.text,sizeof(a));
          reply.upload_id=a.id;
          result=a.session_id==session && a.arm_epoch==ctx.arm_gen.load() && ctx.armed.load()
              ? (commit_waiting?REF_BUSY:upload->abort(a.id)) : REF_IDENTITY;
          break;
        }
        case CTL_REFERENCE_COMMIT: {
          ReferenceCommitPayload c; std::memcpy(&c,rx.text,sizeof(c));
          reply.upload_id=c.upload_id; reply.reference_id=c.reference_id; reply.revision=c.revision;
          if(c.session_id!=session || c.arm_epoch!=ctx.arm_gen.load() || !ctx.armed.load()) {
            result=REF_IDENTITY; break;
          }
          // A mode selection owns the law until RT confirms it: a COMMIT
          // arriving in that window must not land, and must not be silently
          // queued behind the handoff either.
          if(mode_outstanding) { result=REF_BUSY; break; }
          if(commit_waiting) {
            if(!std::memcmp(commit_frame.text,rx.text,sizeof(rx.text))) {
              commit_frame.seq=rx.seq; continue; // same operation, latest correlation
            }
            result=REF_BUSY; break;
          }
          if(have_commit_reply && !std::memcmp(commit_frame.text,rx.text,sizeof(rx.text))) {
            reply=commit_reply;
            result=pending.cancel_generation==ctx.reference_cancel_generation.load()
              ? ReferenceResult(reply.result):REF_RETIRED;
            break;
          }
          const auto* admitted=upload->admitted();
          if(!admitted || admitted->identity.upload_id!=c.upload_id) break;
          pending={}; pending.admitted=*admitted; pending.request=c;
          pending.ticket=ctx.reference_published.load()+1;
          pending.cancel_generation=ctx.reference_cancel_generation.load();
          commit_frame=rx; commit_reply=reply; commit_waiting=true; have_commit_reply=false;
          ctx.reference_in.write(pending);
          ctx.reference_published.store(pending.ticket,std::memory_order_release);
          continue; // RT acknowledgement is delivered without blocking this thread
        }
        case CTL_CANCEL_REFERENCE: {
          ReferenceRetirePayload c; std::memcpy(&c,rx.text,sizeof(c));
          reply.reference_id=c.id; reply.revision=c.revision;
          const auto* candidate=upload->candidate_identity();
          const bool selected=selection.session_id==session && selection.arm_epoch==c.arm_epoch &&
                              c.id==selection.reference_id && c.revision==selection.revision;
          const bool staged=candidate && c.id==candidate->reference_id &&
                            c.revision==candidate->revision;
          const bool waiting=commit_waiting && c.id==pending.request.reference_id &&
                             c.revision==pending.request.revision;
          if(c.session_id!=session || c.arm_epoch!=ctx.arm_gen.load() || !ctx.armed.load())
            result=REF_IDENTITY;
          else if(!std::memcmp(&c,&last_cancel,sizeof(c))) result=REF_OK;
          else if(!selected && !staged && !waiting) result=REF_REVISION;
          else {
            uint64_t retire_id=c.id;
            if(candidate) retire_id=std::max(retire_id,candidate->reference_id);
            if(commit_waiting) retire_id=std::max(retire_id,pending.request.reference_id);
            const uint64_t generation=ctx.reference_cancel_generation.load()+1;
            ctx.reference_cancel.write(ReferenceCancel{session,retire_id,generation,
                                                        c.arm_epoch,c.revision});
            ctx.reference_cancel_generation.store(generation,std::memory_order_release);
            upload->discard(); last_cancel=c; result=REF_OK;
          }
          break;
        }
        default: break;
      }
      reply.result=result;
      ControlPacket answer=make(CTL_REFERENCE_STATUS,0,nullptr,rx.seq);
      answer.version=REFERENCE_VERSION; std::memcpy(answer.text,&reply,sizeof(reply));
      if(!send_frame(fd,answer)) break;
      // Upload traffic does not renew session or command deadmen. The client
      // must still service the ordinary PING/PONG and fresh dynamic envelopes.
      continue;
    }
    if (rx.version != VERSION) continue;
    last_alive_ns = mono_ns();  // any valid frame (PONGs included) is life

    switch (rx.type) {
      case CTL_REFERENCE_MODE: {
        ControlPacket checked{};
        const bool valid=decode_reference_control(&rx,sizeof(rx),false,checked);
        ReferenceModePayload want{}; std::memcpy(&want,rx.text,sizeof(want));
        const char* why=nullptr;
        if(!ctx.supports_reference.load()) why="reference refused: capability unavailable";
        else if(!valid || !ctx.claim_ok.load() || want.session_id!=session ||
                want.joint_order_hash!=ctx.joint_order_hash)
          why="reference refused: identity/malformed mode";
        else if(ctx.armed.load()) why="reference refused: negotiate while disarmed";
        else if(upload && std::memcmp(&want,&bound,sizeof(want)))
          why="reference refused: reconnect to change model binding";
        if(!why && !upload) {
          bound=want;
          upload.emplace(session,ctx.joint_order_hash,uint16_t(ctx.backend_n.load()),want.model_digest);
          ctx.reference_session.store(session);
        }
        send_frame(fd,make(CTL_STATUS,flags_snapshot(ctx),why?why:"reference mode accepted",rx.seq));
        break;
      }
      case CTL_MODE_SELECT: {
        ControlPacket checked{};
        ModeSelectPayload want{};
        std::memcpy(&want,rx.text,sizeof(want));
        const bool valid=decode_mode_select(&rx,sizeof(rx),CTL_MODE_SELECT,checked);
        // An unresolved request owns the generation: an identical retry is
        // still PENDING, a different one is BUSY until RT answers.
        if (mode_outstanding) {
          ModeSelectReplyPayload r{};
          r.session_id=session; r.arm_epoch=want.arm_epoch;
          r.requested_mode=want.desired_mode;
          r.selected_mode=ctx.mode_effective.load(std::memory_order_acquire);
          r.predecessor=ctx.mode_generation.load(std::memory_order_acquire);
          if (!std::memcmp(&want,&mode_request,sizeof(want))) {
            r.generation=want.generation; r.result=MODE_SEL_PENDING;
          } else {
            r.generation=mode_request.generation; r.result=MODE_SEL_BUSY;
          }
          send_frame(fd,mode_reply(CTL_MODE_STATUS,r,rx.seq));
          break;
        }
        ModeSelectReplyPayload r{};
        r.session_id=session; r.generation=want.generation;
        r.arm_epoch=want.arm_epoch; r.requested_mode=want.desired_mode;
        r.selected_mode=ctx.mode_effective.load(std::memory_order_acquire);
        r.predecessor=ctx.mode_generation.load(std::memory_order_acquire);
        uint32_t result;
        if(!ctx.supports_reference.load())
          result=MODE_SEL_UNSUPPORTED;      // capability off: never offered
        else if(!valid || !ctx.claim_ok.load() || want.session_id!=session)
          result=MODE_SEL_MALFORMED;
        else if(!ctx.armed.load() || ctx.fault.load() ||
                ctx.latch_state.load()==ServerCtx::LATCH_SET ||
                want.arm_epoch!=ctx.arm_gen.load())
          result=MODE_SEL_REFUSED;          // faulted, latched or disarmed
        else if(want.desired_mode==MODE_REFERENCE &&
                (ctx.reference_session.load()!=session || !upload))
          result=MODE_SEL_UNSUPPORTED;      // requires the disarmed opt-in
        else if(want.desired_mode==MODE_REFERENCE &&
                std::memcmp(want.model_digest,bound.model_digest,32)!=0)
          result=MODE_SEL_IDENTITY;         // digest must equal the bound digest
        else
          result=MODE_SEL_OK;
        if (result!=MODE_SEL_OK) {
          r.result=result;
          send_frame(fd,mode_reply(CTL_MODE_STATUS,r,rx.seq));
          break;
        }
        // Ordering is decided against the RT-CONFIRMED generation, BEFORE
        // anything is published. Only the exact next generation is a new
        // request; an identical retry of the confirmed generation is an
        // idempotent OK. Both leave the running reference untouched — a
        // repeated or mis-ordered request must never retire a live trajectory.
        // RT re-checks ordering (defence in depth), but by then a published
        // cancel would already have retired the reference.
        const uint64_t confirmed_gen=ctx.mode_generation.load(std::memory_order_acquire);
        const uint16_t confirmed_mode=ctx.mode_effective.load(std::memory_order_acquire);
        if(want.generation==confirmed_gen && want.desired_mode==confirmed_mode) {
          r.generation=confirmed_gen; r.predecessor=confirmed_gen;
          r.selected_mode=confirmed_mode; r.result=MODE_SEL_OK;
          send_frame(fd,mode_reply(CTL_MODE_STATUS,r,rx.seq));
          break;
        }
        if(want.generation!=confirmed_gen+1 || want.expected_predecessor!=confirmed_gen) {
          r.predecessor=confirmed_gen; r.result=MODE_SEL_ORDER;
          send_frame(fd,mode_reply(CTL_MODE_STATUS,r,rx.seq));
          break;
        }
        // The fence at ACCEPTANCE for a valid next generation: the outgoing
        // law's execution authority is retired through its existing owner (the
        // staged candidate is discarded and a cancel generation is published,
        // which retires the running reference), and the arm holds its measured
        // position until RT confirms — RT does so within its next tick and
        // moves the ingress floor then, which is what stops the outgoing
        // family's envelopes from actuating.
        uint64_t retire_id=selection.session_id==session?selection.reference_id:0;
        if(upload) {
          if(const auto* cand=upload->candidate_identity())
            retire_id=std::max(retire_id,cand->reference_id);
          upload->discard();
        }
        const uint64_t generation=ctx.reference_cancel_generation.load()+1;
        // `revision` is only wire-CANCEL correlation; this internal fence does
        // not consume it, so 0 is explicit rather than a stale selection value.
        ctx.reference_cancel.write(ReferenceCancel{session,retire_id,generation,
                                                   want.arm_epoch,0u});
        ctx.reference_cancel_generation.store(generation,std::memory_order_release);
        mode_request=want; mode_frame=rx;
        mode_ticket=ctx.mode_published.load()+1;
        mode_outstanding=true; mode_pending_sent=false;
        mode_deadline_ns=mono_ns()+MODE_SELECT_WAIT_NS;
        ctx.mode_in.write(ModeSelectRequest{mode_request,mode_ticket});
        ctx.mode_published.store(mode_ticket,std::memory_order_release);
        break;  // answered by the bounded wait at the top of the loop
      }
      case CTL_STATUS:
        // Read-only inspection; never changes authority or clears a fault.
        send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "status"));
        break;
      case CTL_CLAIM: {
        IdentityPayload want{};
        std::memcpy(&want, rx.text, sizeof(want));
        const char* why = claim_mismatch(ctx, want, session);
        if (why) {
          // A failed claim never revokes an earlier successful one mid-flight
          // (it cannot change who is armed); it only reports the refusal.
          std::fprintf(stderr, "[rt] %s\n", why);
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), why));
        } else {
          ctx.claim_ok.store(true);
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "claim accepted"));
        }
        break;
      }
      case CTL_ARM:
        if (!ctx.claim_ok.load()) {
          // Flags as they are: without FLAG_ARMED set by this request, the
          // client's ack rule already reads this as a refusal.
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx),
                              "refused: no accepted CLAIM on this session"));
        } else if (ctx.armed.load() && !ctx.fault.load()) {
          // Already armed: ACK without a generation bump. A duplicate ARM
          // (double click, client retry after a lost ack) must not tear down
          // and re-open the live torque session mid-flight — the arm_gen edge
          // makes the RT loop run backend->stop() + session re-open inside
          // one tick, a multi-ms gap libfranka can reject under load.
          send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "already armed",
                              uint32_t(ctx.arm_gen.load())));
        } else {
          // Admission and publication are ONE transition on the latch word
          // (server.hpp: begin_arm -> arm_unlatched): a latched fault, a latch
          // still being written -- even one whose DISARM clear timed out
          // before fault was published -- or a reset in progress refuses the
          // ARM, and nothing of the new generation is prepared. Admitted:
          // the ARM-start record (the staleness clock starts AT ARM), the
          // cleared flow pin and the new peer scope are published before the
          // generation; pre-ARM leftovers in cmd_in carry an older generation
          // and are not authority (CommandAuthority); the generation bump
          // makes this ARM visible to the RT loop even if a DISARM->ARM pair
          // fits between two 1 kHz samples of `armed`. Refused: flags still
          // show ARMED when the server is fault-holding (authority is
          // deliberately retained) — the FAULTED bit is the refusal, and
          // clients must check it, not just ARMED.
          const uint64_t gen = begin_arm(ctx);
          if (gen == 0 && !ctx.backend_ready.load()) {
            send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx),
                                "refused: backend not ready (RT setup pending)"));
          } else if (gen == 0 && ctx.rt_setup_blocks_arm()) {
            char why[sizeof(ControlPacket{}.text)];
            rt_setup_refusal(ctx, why, sizeof(why));
            std::fprintf(stderr, "[rt] ARM %s\n", why);
            send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), why));
          } else if (gen != 0) {
            // seq = the ARM epoch every command of this authority must carry.
            send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), "armed", uint32_t(gen)));
          } else if (ctx.latch_state.load() == ServerCtx::LATCH_SET) {
            std::fprintf(stderr, "[rt] ARM refused: fault latched in ARM epoch %llu\n",
                         (unsigned long long)ctx.fault_gen.load());
            send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx), ctx.fault_text));
          } else {
            send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx),
                                "refused: fault latch busy (DISARM again)"));
          }
        }
        break;
      case CTL_DISARM: {
        ctx.armed.store(false);
        if(upload) upload->observe_epoch(0);
        // The latch is reset only when no writer holds it (server.hpp). A
        // latch racing this clear is not taken and re-fires on a later tick
        // if the cause persists. The journal keeps every first cause.
        const bool cleared = ctx.clear_latch();
        if (!cleared)
          std::fprintf(stderr, "[rt] DISARM: fault latch still being written; "
                               "not cleared\n");
        // AFTER armed=false (rt_loop reads the generation first). The ack is
        // RECEIPT; completion is StatePacket.stop_epoch reaching this value.
        const uint64_t dg = ctx.disarm_gen.fetch_add(1) + 1;
        send_frame(fd, make(CTL_STATUS, flags_snapshot(ctx),
                            cleared ? "disarm requested"
                                    : "disarm requested; fault latch busy, not cleared",
                            uint32_t(dg)));
        break;
      }
      case CTL_SET_ACTIVE: {
        const int n = ctx.backend_n.load();
        const uint32_t configured = n >= 16 ? 0xFFFFu : ((1u << n) - 1u);
        const uint32_t desired = rx.arg;
        const uint32_t current = ctx.active_mask.load();
        if ((desired & ~configured) || (desired & ~ctx.online_mask.load())) {
          send_frame(fd, make(CTL_STATUS, current,
                              "active mask refused: slot offline or unconfigured"));
          break;
        }
        if (ctx.armed.load() && (current & ~desired)) {
          send_frame(fd, make(CTL_STATUS, current,
                              "active mask refused: disarm before removal"));
          break;
        }
        ctx.requested_active_mask.store(desired);
        for (int i = 0; i < 500 && ctx.active_mask.load() != desired; ++i) {
          timespec wait{0, 1'000'000};
          nanosleep(&wait, nullptr);
        }
        const uint32_t applied = ctx.active_mask.load();
        send_frame(fd, make(CTL_STATUS, applied,
                            applied == desired ? "active mask applied"
                                               : "active mask refused by backend"));
        break;
      }
      case CTL_PING:
        send_frame(fd, make(CTL_PONG, 0, nullptr));
        break;
      default:
        break;
    }
  }
  if (ctx.armed.load()) {
    ctx.latch(FAULT_CTL_LOST, "control session lost while armed", ctx.arm_gen.load());
  }
  // The session is over: its commands are no longer admissible and the state
  // stream stops going to its address. Observers keep their own leases.
  ctx.claim_ok.store(false);
  ctx.reference_session.store(0);
  ctx.session_id.store(0);
  {
    std::lock_guard<std::mutex> lock(ctx.peer_mu);
    ctx.have_peer = false;
  }
  close(fd);
}

} // namespace

void control_thread(ServerCtx& ctx) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  const int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = ctx.cfg.bind_addr ? ctx.cfg.bind_addr : INADDR_ANY;
  addr.sin_port = htons(ctx.cfg.tcp_port);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(fd, 1) != 0) {
    std::perror("[rt] tcp bind/listen");
    ctx.failed.store(true);
    ctx.shutdown.store(true);
    return;
  }
  while (!ctx.shutdown.load()) {
    pollfd pfd{fd, POLLIN, 0};
    if (poll(&pfd, 1, 200) <= 0) continue;
    sockaddr_in peer = {};
    socklen_t peerlen = sizeof(peer);
    const int client =
        accept(fd, reinterpret_cast<sockaddr*>(&peer), &peerlen);
    if (client < 0) continue;
    // An ARMED (typically fault-holding after CTL_LOST) arm must not hand
    // authority to whoever connects next: only the SAME host that armed it
    // may reclaim the session (its graph restarting is the recovery path).
    // A different host gets a refusal and the pin stays untouched.
    const uint32_t owner = ctx.ctl_peer_ip.load();
    if (ctx.armed.load() && owner != 0 && peer.sin_addr.s_addr != owner) {
      char buf[INET_ADDRSTRLEN] = {};
      inet_ntop(AF_INET, &peer.sin_addr, buf, sizeof(buf));
      std::fprintf(stderr, "[rt] refusing control client %s (armed; owner "
                           "is the arming host)\n", buf);
      send_frame(client, make(CTL_STATUS, flags_snapshot(ctx),
                              "refused: armed by another host"));
      close(client);
      continue;
    }
    // The control client's address is the ONLY source udp_rx will accept
    // commands from while armed (see server.hpp: ctl_peer_ip).
    ctx.ctl_peer_ip.store(peer.sin_addr.s_addr);
    const uint64_t session = fresh_session();
    ctx.claim_ok.store(false);
    ctx.session_id.store(session);
    std::printf("[rt] control client connected (session %016llx)\n",
                (unsigned long long)session);
    serve_client(ctx, client, session);
    std::printf("[rt] control client disconnected\n");
  }
  close(fd);
}

} // namespace arm_rt
