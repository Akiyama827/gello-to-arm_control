// RT-owned ARM-epoch mode selection. One generation per accepted selection,
// subordinate to the session/ARM epoch. The control thread only validates,
// publishes a request and correlates; the RT loop owns the effective mode, the
// ingress floor and the retirement of the outgoing law, so no command family
// can actuate without the RT loop having confirmed the mode that authorizes it.
#pragma once
#include <cstdint>
#include <cstring>
#include "arm_rt/protocol.hpp"

namespace arm_rt {

// Control thread -> RT. Published at ACCEPTANCE of a valid request, which is
// what raises the fence: an unconfirmed request authorizes neither family, so
// the arm holds its measured position until RT confirms (amendment a).
struct ModeSelectRequest {
  ModeSelectPayload request{};
  uint64_t ticket = 0;
};

// RT -> control. `reply.result` is a ModeSelectResult. Written before
// `mode_done` is stored, so a reader that sees done also sees this record.
struct ModeSelectInfo {
  ModeSelectReplyPayload reply{};
};

// The RT loop's own view of command authority for one ARM epoch: which law may
// actuate, which generation is confirmed, and the server-clock ingress floor
// below which no streamed envelope may actuate.
struct ArmMode {
  uint64_t session_id = 0;
  uint32_t epoch = 0;
  uint16_t mode = MODE_UNSELECTED;
  uint64_t generation = 0;
  uint64_t floor_ns = 0;
  bool pending = false;

  // ARM edge. `seed` is MODE_REFERENCE when this session's disarmed
  // CTL_REFERENCE_MODE opt-in was accepted, else MODE_STREAMED. A new epoch is
  // a new authority: generation restarts at 0 and no state survives.
  void arm(uint64_t session, uint32_t gen, uint16_t seed) {
    session_id = session;
    epoch = gen;
    mode = seed;
    generation = 0;
    floor_ns = 0;
    pending = false;
  }
  void disarm() {
    session_id = 0;
    epoch = 0;
    mode = MODE_UNSELECTED;
    generation = 0;
    floor_ns = 0;
    pending = false;
  }
  bool belongs(uint64_t session, uint32_t gen) const {
    return session_id != 0 && session_id == session && epoch == gen;
  }
  // Confirmed generation, or 0 when this epoch has none yet.
  bool confirmed() const { return mode != MODE_UNSELECTED && !pending; }
  // Command gate, evaluated BEFORE authority.observe():
  //  * a pending/absent selection authorizes neither family: the arm holds;
  //  * MODE_REFERENCE admits only native (v7) envelopes; their phase/age
  //    freshness remains a separate upstream check;
  //  * MODE_STREAMED admits only non-native envelopes with ingress ABOVE the
  //    confirmed floor, so an envelope already in the Mailbox when the floor
  //    was raised never actuates;
  //  * MODE_UNSELECTED is never effective while armed.
  bool admits(bool native, uint64_t rx_ns, uint64_t session, uint32_t gen) const {
    if (!belongs(session, gen) || pending) return false;
    if (mode == MODE_REFERENCE) return native;
    if (mode == MODE_STREAMED) return !native && rx_ns > floor_ns;
    return false;
  }
  void confirm(const ModeSelectPayload& p, uint64_t now) {
    mode = p.desired_mode;
    generation = p.generation;
    floor_ns = now;
    pending = false;
  }
};

}  // namespace arm_rt
