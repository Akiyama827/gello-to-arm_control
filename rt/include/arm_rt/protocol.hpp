// Wire protocol between the PC's rt_backend.py and the RT server.
//
// Mirrored BY HAND in arm_control/rt_protocol.py — the two must byte-match.
// Parity is enforced, not hoped for: `protocol_selfcheck` prints canonical
// golden packets as hex and `python -m arm_control.rt_protocol --hex` prints
// the same; rt_backend's _demo diffs them automatically. Any field change
// bumps VERSION and edits both files in one commit.
//
// Layout rules that make the mirror trivial:
//  - little-endian POD, natural alignment, and field order chosen so there is
//    NO padding (static_asserts below pin the sizes);
//  - fixed MAX_JOINTS arrays with a live `n` — a joint count is data, not a
//    layout parameter;
//  - doubles start at an 8-aligned offset by construction.
//
// Transport split (deliberate, FCI-shaped):
//  - CommandPacket / StatePacket ride UDP, sequence-numbered, LATEST-WINS.
//    A lost datagram is dropped, never retransmitted late.
//  - ControlPacket rides TCP as fixed 128-byte frames (same-size framing =
//    no length prefix): arm/disarm, heartbeat, status + fault events.
#pragma once

#include <cstdint>

namespace arm_rt {

constexpr uint32_t MAGIC_CMD = 0x444D4341;   // bytes "ACMD" on the wire
constexpr uint32_t MAGIC_STATE = 0x41545341; // bytes "ASTA"
constexpr uint32_t MAGIC_CTL = 0x4C544341;   // bytes "ACTL"
constexpr uint16_t VERSION = 1;
constexpr int MAX_JOINTS = 16;

// StatePacket.flags bits
constexpr uint32_t FLAG_ARMED = 1u << 0;
constexpr uint32_t FLAG_FAULTED = 1u << 1;
constexpr uint32_t FLAG_HOLDING = 1u << 2;      // staleness hold active
constexpr uint32_t FLAG_WRENCH_VALID = 1u << 3; // reserved FT-sensor fields live
constexpr uint32_t ONLINE_MASK_SHIFT = 16;
constexpr uint32_t ONLINE_MASK_BITS = 0xFFFFu << ONLINE_MASK_SHIFT;

constexpr uint32_t with_online_mask(uint32_t flags, uint32_t mask) {
  return (flags & ~ONLINE_MASK_BITS) | ((mask & 0xFFFFu) << ONLINE_MASK_SHIFT);
}
constexpr uint32_t online_mask(uint32_t flags) {
  return (flags & ONLINE_MASK_BITS) >> ONLINE_MASK_SHIFT;
}

// ControlPacket.type
enum CtlType : uint16_t {
  CTL_HELLO = 1,  // server -> client on connect: arg = n, text = backend name
  CTL_ARM = 2,    // client -> server; STATUS answers
  CTL_DISARM = 3, // client -> server; also clears a fault latch (the explicit
                  // DISARM->ARM cycle is the ONLY way past a latch)
  CTL_PING = 4,   // either direction
  CTL_PONG = 5,
  CTL_STATUS = 6, // server -> client: arg = flags snapshot, text = note
  CTL_FAULT = 7,  // server -> client event: arg = fault code, text = reason
  CTL_SET_ACTIVE = 8, // client -> server: arg = desired fixed-slot mask
};

#pragma pack(push, 1)

// PC -> RT, UDP, latest-wins. The full bridge contract word — the same
// (q_des, qd_des, tau_ff, kp, kd) the Dora graph's motor_command carries.
struct CommandPacket {
  uint32_t magic;      // MAGIC_CMD
  uint16_t version;    // VERSION
  uint16_t n;          // valid joints
  uint32_t seq;        // sender-monotonic; server keeps the freshest only
  uint32_t flags;      // reserved, 0
  uint64_t t_mono_ns;  // sender clock, diagnostics only (clocks NOT synced)
  double q_des[MAX_JOINTS];
  double qd_des[MAX_JOINTS];
  double tau_ff[MAX_JOINTS];
  double kp[MAX_JOINTS];
  double kd[MAX_JOINTS];
};

// RT -> PC, UDP, streamed at --state-hz.
struct StatePacket {
  uint32_t magic;        // MAGIC_STATE
  uint16_t version;      // VERSION
  uint16_t n;
  uint32_t state_seq;    // server-monotonic
  uint32_t last_cmd_seq; // freshest command the servo has consumed (loss/lag)
  uint64_t t_mono_ns;    // server clock
  uint32_t flags;        // FLAG_*
  uint32_t fault_code;   // 0 = none
  double q[MAX_JOINTS];
  double dq[MAX_JOINTS];
  double tau[MAX_JOINTS];     // measured (fake: applied)
  double tau_cmd[MAX_JOINTS]; // servo's own output after clamp+slew (tau_J_d analogue)
  double q_cmd[MAX_JOINTS];   // servo's current position target (tracking plots)
  double wrench[6];           // reserved for an FT sensor; FLAG_WRENCH_VALID gates
  double reserved[2];
};

// Both directions, TCP, fixed 128-byte frames.
struct ControlPacket {
  uint32_t magic;     // MAGIC_CTL
  uint16_t version;   // VERSION
  uint16_t type;      // CtlType
  uint32_t seq;
  uint32_t arg;       // type-dependent (HELLO: n; STATUS: flags; FAULT: code)
  uint64_t t_mono_ns;
  char text[104];     // NUL-terminated where used
};

#pragma pack(pop)

static_assert(sizeof(CommandPacket) == 24 + 5 * 8 * MAX_JOINTS, "cmd layout");
static_assert(sizeof(CommandPacket) == 664, "cmd size");
static_assert(sizeof(StatePacket) == 736, "state size");
static_assert(sizeof(ControlPacket) == 128, "ctl size");

} // namespace arm_rt
