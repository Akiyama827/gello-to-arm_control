// Wire protocol between the PC's rt_backend.py and the RT server.
//
// Mirrored BY HAND in arm_control/plants/remote_rt/protocol.py — the two
// must byte-match. Parity is enforced, not hoped for: `protocol_selfcheck`
// prints canonical golden packets as hex and
// `python -m arm_control.plants.remote_rt.protocol --hex` prints the same;
// rt_backend's _demo diffs them automatically. Any field change versions its packet variant
// and edits both files in one commit.
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
#include <cstddef>
#include <cmath>
#include <cstring>

namespace arm_rt {

constexpr uint32_t MAGIC_CMD = 0x444D4341;   // bytes "ACMD" on the wire
constexpr uint32_t MAGIC_STATE = 0x41545341; // bytes "ASTA"
constexpr uint32_t MAGIC_CTL = 0x4C544341;   // bytes "ACTL"
constexpr uint32_t MAGIC_OBS = 0x53424F41;   // bytes "AOBS"
// v5 binds every command to a control SESSION and ARM epoch, adds the
// IDENTITY/CLAIM admission frames, the passive observer lease and the
// servo-confirmed stop epoch. v6 is its pose-hold command. v3/v4 (and older)
// peers are rejected: their commands carry no session and cannot be admitted.
constexpr uint16_t VERSION = 5;
constexpr uint16_t POSE_HOLD_VERSION = 6;
constexpr uint16_t REFERENCE_VERSION = 7;
constexpr uint16_t MAX_REFERENCE_KNOTS = 128;
constexpr uint32_t REFERENCE_CHUNK_BYTES = 88;
constexpr uint32_t MAX_REFERENCE_BYTES = 128 * (8 + 16 * 16);
constexpr int MAX_JOINTS = 16;
constexpr int ID_CHARS = 32;  // instance/model id fields, NUL-padded

// IDENTITY.capabilities bits
constexpr uint32_t CAP_POSE_HOLD = 1u << 0;     // v6 Cartesian pose hold
constexpr uint32_t CAP_REFERENCE_HORIZON = 1u << 3; // negotiated v7, not implicit
constexpr uint32_t CAP_OBSERVER = 1u << 1;      // UDP observer lease
// Task-space feedback uses a kinematic model the deployment has approved for
// THIS instance. No backend sets it today: libfranka's own model is not a
// calibrated model, and pose hold must not be sold as precision control.
constexpr uint32_t CAP_CALIBRATED_TASK_SPACE = 1u << 2;
// Armed-only bounded selection generation (CTL_MODE_SELECT). Advertised only
// where native reference is supported, so a bare (enable_reference: false)
// deployment neither offers it nor sets the mode flags below.
constexpr uint32_t CAP_MODE_SELECT = 1u << 4;

// StatePacket.flags bits
constexpr uint32_t FLAG_ARMED = 1u << 0;
constexpr uint32_t FLAG_FAULTED = 1u << 1;
constexpr uint32_t FLAG_HOLDING = 1u << 2;      // staleness hold active
constexpr uint32_t FLAG_WRENCH_VALID = 1u << 3; // reserved FT-sensor fields live
constexpr uint32_t FLAG_CLAIMED = 1u << 4;      // this control session's CLAIM matched
// Confirmed effective law for the current ARM epoch. Mutually exclusive, both
// zero = MODE_UNSELECTED. Set ONLY by servers advertising CAP_MODE_SELECT, and
// only once the selection is confirmed (never while a request is pending).
constexpr uint32_t FLAG_MODE_STREAMED = 1u << 5;
constexpr uint32_t FLAG_MODE_REFERENCE = 1u << 6;
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
  CTL_IDENTITY = 9,   // server -> client after HELLO: text = IdentityPayload
  CTL_REFERENCE_MODE = 11, // v5 opt-in after CLAIM, disarmed only
  CTL_CLAIM = 10,     // client -> server: text = expected IdentityPayload;
                      // STATUS answers with FLAG_CLAIMED set or a refusal
  CTL_MODE_SELECT = 12, // armed-only bounded selection generation; text =
                        // ModeSelectPayload
  CTL_MODE_STATUS = 13, // server -> client: text = ModeSelectReplyPayload
};
// ACK correlation (ControlPacket.seq of the STATUS reply):
//   ARM    -> the ARM generation every command of this epoch must carry
//   DISARM -> the DISARM generation StatePacket.stop_epoch must reach

// v7 control operations. Only a successful v5 REFERENCE_MODE exchange may
// enable these. Old clients/observers continue receiving v5 state.
enum ReferenceCtl : uint16_t {
  CTL_REFERENCE_BEGIN = 32, CTL_REFERENCE_CHUNK = 33,
  CTL_REFERENCE_SEAL = 34, CTL_REFERENCE_COMMIT = 35,
  CTL_REFERENCE_ABORT = 36, CTL_CANCEL_REFERENCE = 37,
  CTL_REFERENCE_STATUS = 38,
  // CTRL-040 group execution (additive; a server accepts these only when started with a group
  // record and advertises `grp=1` in HELLO). First-revision references only.
  CTL_GROUP_JOIN = 39, CTL_GROUP_COMMIT = 40, CTL_GROUP_CANCEL = 41,
  CTL_GROUP_STATUS = 42,  // server -> client: reply (operation = request opcode) or event (0)
};
enum ReferenceStatus : uint32_t {
  REFERENCE_NONE = 0, REFERENCE_RUNNING = 1, REFERENCE_END = 2,
  REFERENCE_UNDERFLOW = 3, REFERENCE_CANCELLED = 4,
};
enum ReferenceResult : uint32_t {
  REF_OK = 0, REF_MALFORMED = 1, REF_IDENTITY = 2, REF_BUSY = 3,
  REF_REVISION = 4, REF_INTEGRITY = 5, REF_PREFIX = 6, REF_LATE = 7,
  REF_RETIRED = 8, REF_NOT_ADMITTED = 9,
  REF_GROUP_REFUSED = 10,  // group precondition failed (latch, other group, identity)
  REF_GROUP_PEER = 11,     // a peer faulted/cancelled/went stale, or the group deadline passed
};
constexpr uint32_t REFERENCE_HERMITE = 1u << 0;
constexpr uint32_t REFERENCE_TERMINAL = 1u << 1;

// Armed-only mode selection. "Soft" (the operator alias) is MODE_STREAMED: the
// existing v5 joint-command / v6 pose-hold law. MODE_UNSELECTED is the
// pre-ARM/disarmed value and is never effective while armed: ARM seeds
// generation 0 from the disarmed CTL_REFERENCE_MODE opt-in.
enum ModeId : uint16_t {
  MODE_UNSELECTED = 0, MODE_STREAMED = 1, MODE_REFERENCE = 2,
};
enum ModeSelectResult : uint32_t {
  MODE_SEL_OK = 0,          // confirmed; this generation is authoritative
  MODE_SEL_PENDING = 1,     // no RT acknowledgement within the bounded budget
  MODE_SEL_MALFORMED = 2,   // body failed validation; no state change
  MODE_SEL_IDENTITY = 3,    // session / ARM epoch / model digest mismatch
  MODE_SEL_UNSUPPORTED = 4, // capability, opt-in or requested mode not offered
  MODE_SEL_ORDER = 5,       // generation/predecessor wrong; reply carries current
  MODE_SEL_BUSY = 6,        // a different request is still awaiting RT
  MODE_SEL_REFUSED = 7,     // faulted, latched or disarmed only
};

#pragma pack(push, 1)

// PC -> RT, UDP, latest-wins. The full bridge contract word — the same
// (q_des, qd_des, tau_ff, kp, kd) the Dora graph's motor_command carries.
// A session/epoch pair is flow BINDING, not authentication: anything on the
// direct link that learns both can still forge a packet.
struct CommandPacket {
  uint32_t magic;      // MAGIC_CMD
  uint16_t version;    // VERSION
  uint16_t n;          // valid joints
  uint32_t seq;        // sender-monotonic; server keeps the freshest only
  uint32_t arm_epoch;  // ARM ack seq this command belongs to; 0 while disarmed
  uint64_t t_mono_ns;  // sender clock, diagnostics only (clocks NOT synced)
  uint64_t session_id; // IDENTITY.session_id of the sender's control session
  double q_des[MAX_JOINTS];
  double qd_des[MAX_JOINTS];
  double tau_ff[MAX_JOINTS];
  double kp[MAX_JOINTS];
  double kd[MAX_JOINTS];
};

struct PoseHoldCommandPacket {
  CommandPacket command;
  double pose_hold[15];
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
  uint64_t session_id;   // control session holding the server (0 = none)
  uint64_t identity_hash;// identity_hash(instance id, joint order) — lets a
                         // passive observer check WHICH arm this is
  uint32_t arm_epoch;    // ARM generation the servo acted on this tick
  uint32_t stop_epoch;   // latest DISARM generation the servo has COMPLETED
                         // (disarmed at the sample, backend stop returned)
  double q[MAX_JOINTS];
  double dq[MAX_JOINTS];
  double tau[MAX_JOINTS];     // measured (fake: applied)
  double tau_cmd[MAX_JOINTS]; // servo's own output after clamp+slew (tau_J_d analogue)
  double q_cmd[MAX_JOINTS];   // servo's current position target (tracking plots)
  double qd_cmd[MAX_JOINTS];  // selected servo velocity target; zero in holds/disarm
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

// CTL_IDENTITY / CTL_CLAIM body, carried in ControlPacket.text. Fixed layout
// so identity never has to be parsed out of free text.
struct IdentityPayload {
  char instance_id[ID_CHARS];  // deployment-provisioned arm instance id
  char model_id[ID_CHARS];     // model the RT side uses for task space
  uint64_t session_id;         // nonzero, fresh per accepted TCP session
  uint64_t joint_order_hash;   // fnv1a64 of joint names joined by '\n'
  uint64_t identity_hash;      // identity_hash(instance_id, joint_order_hash)
  uint32_t capabilities;       // CAP_*
  uint32_t n;                  // joints
  uint8_t reserved[8];
};

// PC -> RT, UDP: ask for (or renew) a read-only copy of the state stream at
// this datagram's source address. Grants nothing, refreshes no deadman.
struct ObserveRequest {
  uint32_t magic;     // MAGIC_OBS
  uint16_t version;   // VERSION
  uint16_t reserved;
  uint32_t lease_ms;  // 100..5000
  uint32_t reserved2;
};

// All control bodies occupy exactly ControlPacket.text (104 bytes).
// Reserved bytes and unused array slots MUST be zero; receivers reject them.
// The model digest is SHA256 of the existing opaque execution model token,
// not a vendor model or a new calibration identity. Binding is per session.
struct ReferenceModePayload {
  uint64_t session_id;              // 0
  uint64_t joint_order_hash;        // 8
  uint8_t model_digest[32];         // 16
  uint32_t enable;                  // 48: exactly 1; reconnect to leave mode
  uint8_t reserved[52];             // 52
};
// CTL_MODE_SELECT body. Armed-only, one bounded selection generation
// subordinate to the session/ARM epoch. Structural validation is a pure
// predicate (valid_mode_select_body); the semantic checks (armed, opt-in,
// digest equality, ordering) happen where the authority lives.
struct ModeSelectPayload {
  uint64_t session_id;              // 0
  uint64_t generation;              // 8: nonzero; expected_predecessor + 1
  uint64_t expected_predecessor;    // 16: 0 for the first selection
  uint32_t arm_epoch;               // 24: nonzero, must equal the live epoch
  uint16_t desired_mode;            // 28: MODE_STREAMED | MODE_REFERENCE
  uint16_t reserved0;               // 30: zero (rejected otherwise)
  uint8_t model_digest[32];         // 32: nonzero iff MODE_REFERENCE
  uint32_t request_flags;           // 64: zero (reserved)
  uint8_t reserved[36];             // 68: zero
};
// CTL_MODE_STATUS body. `detail` is result-dependent. `predecessor` carries
// RT's current generation on MODE_SEL_ORDER, and the prior generation
// otherwise.
struct ModeSelectReplyPayload {
  uint64_t session_id;              // 0
  uint64_t generation;              // 8: echoes the request
  uint64_t predecessor;             // 16: RT's generation before this outcome
  uint32_t arm_epoch;               // 24
  uint32_t result;                  // 28: ModeSelectResult
  uint16_t selected_mode;           // 32: RT's effective mode at reply time
  uint16_t requested_mode;          // 34
  uint32_t detail;                  // 36: result-dependent
  uint8_t reserved[64];             // 40: zero
};
struct ReferenceBeginPayload {
  uint64_t session_id;              // 0
  uint64_t upload_id;               // 8: nonzero, monotonically increasing
  uint64_t reference_id;            // 16: nonzero execution identity
  uint64_t joint_order_hash;        // 24
  uint8_t model_digest[32];         // 32
  uint32_t arm_epoch;               // 64
  uint32_t expected_revision;       // 68: 0 for initial release
  uint32_t revision;                // 72: expected + 1, no wrapping
  uint32_t byte_count;              // 76: K * (8 + 8*N*(Hermite ? 2 : 1))
  uint16_t knots;                   // 80: 2..128
  uint16_t n;                       // 82: 1..16
  uint32_t flags;                   // 84: HERMITE / TERMINAL only
  uint8_t reserved[16];             // 88
};
struct ReferenceChunkPayload {
  uint64_t upload_id;               // 0 (session implicit on TCP)
  uint32_t offset;                  // 8: contiguous bytes or exact retry
  uint16_t length;                  // 12: 1..88
  uint16_t reserved;                // 14
  uint8_t data[REFERENCE_CHUNK_BYTES]; // 16; zero beyond length
};
struct ReferenceSealPayload {
  uint64_t upload_id;               // 0
  uint8_t sha256[32];               // 8: complete serialized knot body
  uint8_t reserved[64];             // 40
};
struct ReferenceCommitPayload {
  uint64_t session_id;              // 0
  uint64_t upload_id;               // 8
  uint64_t reference_id;            // 16
  uint32_t arm_epoch;               // 24
  uint32_t expected_revision;       // 28
  uint32_t revision;                // 32
  uint32_t reserved0;               // 36
  double protected_until;           // 40: exact shared cut knot, reference seconds
  double latest_phase;              // 48: refuse if RT phase reaches this cut
  uint64_t selection_budget_ns;     // 56: initial selection, since first successful SEAL; 0=none
  uint8_t reserved[40];             // 64
};
struct ReferenceRetirePayload {
  uint64_t session_id;              // 0
  uint64_t id;                      // 8: upload_id (ABORT), reference_id (CANCEL)
  uint32_t arm_epoch;               // 16
  uint32_t revision;                // 20: 0 for ABORT; selected/pending for CANCEL
  uint8_t reserved[80];             // 24
};
// CTRL-040. JOIN and CANCEL share one body.
struct GroupMemberPayload {
  uint64_t session_id;              // 0
  uint64_t group_id;                // 8: PC-assigned, never reused
  uint32_t arm_epoch;               // 16
  uint32_t member_index;            // 20
  uint32_t member_count;            // 24
  uint32_t reserved0;               // 28
  uint8_t reserved[72];             // 32
};
struct GroupCommitPayload {
  uint64_t session_id;              // 0
  uint64_t group_id;                // 8
  uint64_t upload_id;               // 16: the sealed first-revision upload
  uint64_t reference_id;            // 24
  uint64_t budget_ns;               // 32: group deadline = first committer's now + budget
  uint32_t arm_epoch;               // 40
  uint32_t member_index;            // 44
  uint8_t reserved[56];             // 48
};
struct GroupStatusPayload {
  uint64_t group_id;                // 0
  uint64_t deadline_ns;             // 8: group deadline on the RT host clock, 0 = none yet
  uint64_t selected_ns;             // 16: this member's selection stamp (host clock), 0 = none
  uint32_t result;                  // 24: ReferenceResult
  uint32_t operation;               // 28: request opcode, 0 for an unsolicited event
  uint32_t state;                   // 32: GroupState (group_record.hpp)
  uint32_t peer_verdict;            // 36: PeerVerdict of the peer that decided the outcome
  uint32_t reserved0;               // 40
  uint8_t reserved[60];             // 44
};
struct ReferenceReplyPayload {
  uint64_t upload_id;               // 0
  uint64_t reference_id;            // 8
  uint32_t revision;                // 16
  uint32_t result;                  // 20: ReferenceResult
  uint32_t operation;               // 24: request opcode; header seq echoes request
  uint32_t reserved0;               // 28
  uint8_t reserved[72];             // 32
};

// v7 replaces the streamed q/qd fields with a reference-bound PC dynamics
// envelope. Sender clock is diagnostic only. Age is PC elapsed compute/queue
// age at send; RT adds elapsed time since ingress. Network phase lag is bounded
// independently against RT's selected phase, never inferred by clock subtraction.
struct ReferenceCommandPacket {
  uint32_t magic; uint16_t version; uint16_t n; // 0
  uint32_t seq; uint32_t arm_epoch;            // 8
  uint64_t t_mono_ns; uint64_t session_id;     // 16
  uint64_t reference_id;                      // 32
  uint32_t revision; uint32_t age_us;          // 40
  double phase;                              // 48
  uint32_t valid_for_us; uint32_t phase_window_us; // 56, both >0
  double tau_ff[MAX_JOINTS];                  // 64
  double kp[MAX_JOINTS];                      // 192
  double kd[MAX_JOINTS];                      // 320
};
struct ReferenceStatePacket {
  StatePacket state;               // 0: version=7, existing output/acceptance intact
  uint64_t reference_id;            // 888
  uint32_t revision; uint32_t status; // 896
  double phase;                    // 904: RT selected phase, seconds from release
  double envelope_phase;           // 912: last accepted PC envelope's sample phase
  uint32_t envelope_age_us;         // 920: saturating; UINT32_MAX if absent
  uint32_t envelope_seq;            // 924: accepted envelope; not backend APPLIED
  double q_ref[MAX_JOINTS];         // 928: selected reference, distinct from q_cmd
  double qd_ref[MAX_JOINTS];        // 1056
  double qdd_ref[MAX_JOINTS];       // 1184
};

#pragma pack(pop)

static_assert(sizeof(CommandPacket) == 32 + 5 * 8 * MAX_JOINTS, "cmd layout");
static_assert(sizeof(CommandPacket) == 672, "cmd size");
static_assert(sizeof(PoseHoldCommandPacket) == 792, "pose hold cmd size");
static_assert(sizeof(StatePacket) == 888, "state size");
static_assert(sizeof(ControlPacket) == 128, "ctl size");
static_assert(sizeof(IdentityPayload) == 104, "identity size");
static_assert(sizeof(ObserveRequest) == 16, "observe size");

static_assert(sizeof(ReferenceModePayload) == 104, "reference mode layout");
static_assert(sizeof(ModeSelectPayload) == 104, "mode select layout");
static_assert(sizeof(ModeSelectReplyPayload) == 104, "mode status layout");
static_assert(sizeof(ReferenceBeginPayload) == 104, "reference begin layout");
static_assert(sizeof(ReferenceChunkPayload) == 104, "reference chunk layout");
static_assert(sizeof(ReferenceSealPayload) == 104, "reference seal layout");
static_assert(sizeof(ReferenceCommitPayload) == 104, "reference commit layout");
static_assert(sizeof(ReferenceRetirePayload) == 104, "reference retire layout");
static_assert(sizeof(ReferenceReplyPayload) == 104, "reference reply layout");
static_assert(sizeof(ReferenceCommandPacket) == 448, "reference command layout");
static_assert(sizeof(ReferenceStatePacket) == 1312, "reference state layout");
static_assert(sizeof(ReferenceStatePacket) + 8 + 40 <= 1500, "IPv6 UDP MTU");
static_assert(offsetof(ReferenceBeginPayload, arm_epoch) == 64, "begin offset");
static_assert(offsetof(ReferenceCommitPayload, protected_until) == 40, "commit offset");
static_assert(offsetof(ReferenceCommandPacket, tau_ff) == 64, "envelope offset");
static_assert(sizeof(GroupMemberPayload)==104 && sizeof(GroupCommitPayload)==104 &&
              sizeof(GroupStatusPayload)==104, "group payloads fill the control text");
static_assert(offsetof(ReferenceStatePacket, q_ref) == 928, "reference state offset");

inline bool zero_bytes(const void* data, size_t size) {
  const auto* b = static_cast<const uint8_t*>(data);
  for (size_t i=0; i<size; ++i) if (b[i]) return false;
  return true;
}
inline bool reference_opcode(uint16_t type) {
  return type >= CTL_REFERENCE_BEGIN && type <= CTL_GROUP_STATUS;
}
inline bool valid_reference_body(uint16_t type, const void* body) {
  // Copy packed payloads before inspecting them: no aliasing/lifetime assumptions.
  switch (type) {
    case CTL_REFERENCE_MODE: {
      ReferenceModePayload p; std::memcpy(&p,body,sizeof(p));
      return p.session_id && p.enable==1 && !zero_bytes(p.model_digest,32) &&
             zero_bytes(p.reserved,sizeof(p.reserved));
    }
    case CTL_REFERENCE_BEGIN: {
      ReferenceBeginPayload p; std::memcpy(&p,body,sizeof(p));
      return p.session_id && p.upload_id && p.reference_id && p.arm_epoch &&
        p.revision && p.revision==p.expected_revision+1 &&
        p.knots>=2 && p.knots<=MAX_REFERENCE_KNOTS && p.n>=1 && p.n<=MAX_JOINTS &&
        !(p.flags & ~(REFERENCE_HERMITE|REFERENCE_TERMINAL)) &&
        p.byte_count==uint32_t(p.knots)*(8+8*p.n*((p.flags & REFERENCE_HERMITE)?2:1)) &&
        !zero_bytes(p.model_digest,32) && zero_bytes(p.reserved,sizeof(p.reserved));
    }
    case CTL_REFERENCE_CHUNK: {
      ReferenceChunkPayload p; std::memcpy(&p,body,sizeof(p));
      return p.upload_id && !p.reserved && p.length && p.length<=REFERENCE_CHUNK_BYTES &&
        p.offset<=MAX_REFERENCE_BYTES-p.length &&
        zero_bytes(p.data+p.length,REFERENCE_CHUNK_BYTES-p.length);
    }
    case CTL_REFERENCE_SEAL: {
      ReferenceSealPayload p; std::memcpy(&p,body,sizeof(p));
      return p.upload_id && zero_bytes(p.reserved,sizeof(p.reserved));
    }
    case CTL_REFERENCE_COMMIT: {
      ReferenceCommitPayload p; std::memcpy(&p,body,sizeof(p));
      return p.session_id && p.upload_id && p.reference_id && p.arm_epoch &&
        p.revision && p.revision==p.expected_revision+1 && !p.reserved0 &&
        std::isfinite(p.protected_until) && std::isfinite(p.latest_phase) &&
        p.protected_until>=0 && p.latest_phase>=0 &&
        (p.expected_revision ? !p.selection_budget_ns && p.latest_phase<=p.protected_until && p.latest_phase>0
                             : p.latest_phase==0 && p.protected_until==0) &&
        zero_bytes(p.reserved,sizeof(p.reserved));
    }
    case CTL_REFERENCE_ABORT: case CTL_CANCEL_REFERENCE: {
      ReferenceRetirePayload p; std::memcpy(&p,body,sizeof(p));
      return p.session_id && p.id && p.arm_epoch &&
        (type==CTL_REFERENCE_ABORT ? p.revision==0 : p.revision>0) &&
        zero_bytes(p.reserved,sizeof(p.reserved));
    }
    case CTL_REFERENCE_STATUS: {
      ReferenceReplyPayload p; std::memcpy(&p,body,sizeof(p));
      return p.result<=REF_NOT_ADMITTED && p.operation>=CTL_REFERENCE_BEGIN &&
        p.operation<=CTL_CANCEL_REFERENCE && !p.reserved0 &&
        zero_bytes(p.reserved,sizeof(p.reserved));
    }
    case CTL_GROUP_JOIN: case CTL_GROUP_CANCEL: {
      GroupMemberPayload p; std::memcpy(&p,body,sizeof(p));
      return p.session_id && p.group_id && p.arm_epoch && p.member_count>=2 && p.member_count<=4 &&
        p.member_index<p.member_count && !p.reserved0 && zero_bytes(p.reserved,sizeof(p.reserved));
    }
    case CTL_GROUP_COMMIT: {
      GroupCommitPayload p; std::memcpy(&p,body,sizeof(p));
      return p.session_id && p.group_id && p.upload_id && p.reference_id && p.arm_epoch &&
        p.budget_ns && p.member_index<4 && zero_bytes(p.reserved,sizeof(p.reserved));
    }
    case CTL_GROUP_STATUS: {
      GroupStatusPayload p; std::memcpy(&p,body,sizeof(p));
      return p.group_id && p.result<=REF_GROUP_PEER &&
        (p.operation==0 || (p.operation>=CTL_GROUP_JOIN && p.operation<=CTL_GROUP_CANCEL)) &&
        p.state<=6 && p.peer_verdict<=4 && !p.reserved0 && zero_bytes(p.reserved,sizeof(p.reserved));
    }
    default: return false;
  }
}
inline bool decode_reference_control(const void* bytes, size_t size, bool negotiated,
                                     ControlPacket& out) {
  if (size!=sizeof(out)) return false;
  ControlPacket p; std::memcpy(&p,bytes,sizeof(p));
  if (p.magic!=MAGIC_CTL || p.arg ||
      (p.type==CTL_REFERENCE_MODE ? p.version!=VERSION
        : !negotiated || p.version!=REFERENCE_VERSION || !reference_opcode(p.type)) ||
      !valid_reference_body(p.type,p.text)) return false;
  out=p; return true;
}
inline bool mode_select_opcode(uint16_t type) {
  return type==CTL_MODE_SELECT || type==CTL_MODE_STATUS;
}
// Structural validation only: no state, no semantic authority. The semantic
// checks (armed, opt-in, digest equality, generation ordering) live where that
// authority lives.
inline bool valid_mode_select_body(const void* body) {
  ModeSelectPayload p; std::memcpy(&p,body,sizeof(p));
  const bool reference=p.desired_mode==MODE_REFERENCE;
  return p.session_id && p.generation && p.arm_epoch &&
    p.generation==p.expected_predecessor+1 &&
    (p.desired_mode==MODE_STREAMED || reference) &&
    !p.reserved0 && !p.request_flags &&
    (reference ? !zero_bytes(p.model_digest,32) : zero_bytes(p.model_digest,32)) &&
    zero_bytes(p.reserved,sizeof(p.reserved));
}
inline bool valid_mode_status_body(const void* body) {
  ModeSelectReplyPayload p; std::memcpy(&p,body,sizeof(p));
  return p.result<=MODE_SEL_REFUSED &&
    (p.selected_mode==MODE_UNSELECTED || p.selected_mode==MODE_STREAMED ||
     p.selected_mode==MODE_REFERENCE) &&
    (p.requested_mode==MODE_STREAMED || p.requested_mode==MODE_REFERENCE) &&
    zero_bytes(p.reserved,sizeof(p.reserved));
}
// v5-range control opcode; available whenever CAP_MODE_SELECT is advertised.
inline bool decode_mode_select(const void* bytes, size_t size, uint16_t want_type,
                               ControlPacket& out) {
  if (size!=sizeof(out)) return false;
  ControlPacket p; std::memcpy(&p,bytes,sizeof(p));
  if (p.magic!=MAGIC_CTL || p.arg || p.type!=want_type || p.version!=VERSION)
    return false;
  if (want_type==CTL_MODE_SELECT ? !valid_mode_select_body(p.text)
                                 : !valid_mode_status_body(p.text)) return false;
  out=p; return true;
}
inline bool decode_reference_command(const void* bytes, size_t size, bool negotiated,
                                     ReferenceCommandPacket& out) {
  if (!negotiated || size!=sizeof(out)) return false;
  ReferenceCommandPacket p; std::memcpy(&p,bytes,sizeof(p));
  if (p.magic!=MAGIC_CMD || p.version!=REFERENCE_VERSION || p.n<1 ||
      p.n>MAX_JOINTS || !p.session_id || !p.arm_epoch || !p.reference_id || !p.revision ||
      !std::isfinite(p.phase) || p.phase<0 || !p.valid_for_us || !p.phase_window_us ||
      p.age_us>p.valid_for_us) return false;
  for (int j=0; j<MAX_JOINTS; ++j) {
    if (!std::isfinite(p.tau_ff[j]) || !std::isfinite(p.kp[j]) ||
        !std::isfinite(p.kd[j]) || p.kp[j]<0 || p.kd[j]<0) return false;
    if (j>=p.n && (p.tau_ff[j]!=0 || p.kp[j]!=0 || p.kd[j]!=0)) return false;
  }
  out=p; return true;
}

inline bool decode_reference_state(const void* bytes, size_t size, bool negotiated,
                                   ReferenceStatePacket& out) {
  if (!negotiated || size!=sizeof(out)) return false;
  ReferenceStatePacket p; std::memcpy(&p,bytes,sizeof(p));
  if (p.state.magic!=MAGIC_STATE || p.state.version!=REFERENCE_VERSION ||
      p.state.n<1 || p.state.n>MAX_JOINTS || p.status>REFERENCE_CANCELLED ||
      !std::isfinite(p.phase) || p.phase<0 || !std::isfinite(p.envelope_phase) ||
      p.envelope_phase<0 || !zero_bytes(p.state.reserved,sizeof(p.state.reserved)))
    return false;
  for (int j=0; j<MAX_JOINTS; ++j) {
    if (!std::isfinite(p.q_ref[j]) || !std::isfinite(p.qd_ref[j]) ||
        !std::isfinite(p.qdd_ref[j])) return false;
    if (j>=p.state.n && (p.q_ref[j]!=0 || p.qd_ref[j]!=0 || p.qdd_ref[j]!=0)) return false;
  }
  out=p; return true;
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;
constexpr uint64_t FNV_PRIME = 0x100000001b3ull;

inline uint64_t fnv1a64(const void* data, size_t size, uint64_t h = FNV_OFFSET) {
  const auto* b = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) { h ^= b[i]; h *= FNV_PRIME; }
  return h;
}

// Over the NUL-padded 32-byte instance id, then the joint-order hash as 8
// little-endian bytes. Both ends compute it; neither parses the other's text.
inline uint64_t identity_hash(const char instance_id[ID_CHARS], uint64_t joint_order_hash) {
  uint64_t h = fnv1a64(instance_id, ID_CHARS);
  unsigned char le[8];
  for (int i = 0; i < 8; ++i) le[i] = uint8_t(joint_order_hash >> (8 * i));
  return fnv1a64(le, sizeof(le), h);
}

// Instance ids are provisioned names, not free text: 1..31 of [A-Za-z0-9_.-].
inline bool valid_id(const char* s) {
  size_t k = 0;
  for (; k < size_t(ID_CHARS) && s[k]; ++k) {
    const char c = s[k];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-'))
      return false;
  }
  return k >= 1 && k < size_t(ID_CHARS);
}

inline bool valid_pose_hold_spec(const double* s) {
  for (int i=0; i<15; ++i) if (!std::isfinite(s[i])) return false;
  if (s[0]<1 || s[0]>4294967295.0 || std::floor(s[0])!=s[0]) return false;
  for (int i=0; i<6; ++i)
    if (s[1+i]<0 || s[1+i]>(i<3 ? 1000 : 100) ||
        s[7+i]<0 || s[7+i]>(i<3 ? 200 : 50)) return false;
  return s[13]>=0 && s[13]<=20 && s[14]>=0 && s[14]<=10;
}

// Validate before granting flow ownership or refreshing the command deadman.
inline bool decode_command(const void* bytes, size_t size, bool supports_pose_hold,
                           PoseHoldCommandPacket& out) {
  if(size!=sizeof(CommandPacket) && size!=sizeof(PoseHoldCommandPacket)) return false;
  out={}; std::memcpy(&out,bytes,size);
  const auto& cmd=out.command;
  if(cmd.magic!=MAGIC_CMD || cmd.n<1 || cmd.n>MAX_JOINTS) return false;
  if(cmd.version==VERSION) { if(size!=sizeof(CommandPacket)) return false; }
  else if(cmd.version==POSE_HOLD_VERSION) {
    if(size!=sizeof(PoseHoldCommandPacket) || !supports_pose_hold ||
       !valid_pose_hold_spec(out.pose_hold)) return false;
  } else return false;
  for(int j=0;j<cmd.n;++j)
    if(!std::isfinite(cmd.q_des[j]) || !std::isfinite(cmd.qd_des[j]) ||
       !std::isfinite(cmd.tau_ff[j]) || !std::isfinite(cmd.kp[j]) ||
       !std::isfinite(cmd.kd[j]) || cmd.kp[j]<0 || cmd.kd[j]<0) return false;
  return true;
}

} // namespace arm_rt
