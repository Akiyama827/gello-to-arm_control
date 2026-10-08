// Mode-selection wire + authority unit check. No sockets, no threads: the
// cross-language golden is compared against tools/check/rt/mode_select.py, and
// the ArmMode gate is asserted directly so a regression in "which law may
// actuate" fails here rather than on a robot.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#include "arm_rt/mode_selection.hpp"
#include "arm_rt/protocol.hpp"

using namespace arm_rt;

static std::string hex(const void* data, size_t n) {
  static const char* d = "0123456789abcdef";
  const auto* b = static_cast<const unsigned char*>(data);
  std::string s;
  s.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) {
    s.push_back(d[b[i] >> 4]);
    s.push_back(d[b[i] & 0xF]);
  }
  return s;
}

static ControlPacket mode_frame(uint16_t type, uint32_t seq,
                                const void* body) {
  ControlPacket p{};
  p.magic = MAGIC_CTL;
  p.version = VERSION;
  p.type = type;
  p.seq = seq;
  p.arg = 0;
  p.t_mono_ns = 123456789;
  if (body) std::memcpy(p.text, body, sizeof(p.text));
  return p;
}

int main() {
  static_assert(sizeof(ModeSelectPayload) == 104, "mode select layout");
  static_assert(sizeof(ModeSelectReplyPayload) == 104, "mode status layout");

  // --- wire goldens (must equal tools/check/rt/mode_select.py exactly) ------
  ModeSelectPayload sel{};
  sel.session_id = 123;
  sel.generation = 6;
  sel.expected_predecessor = 5;
  sel.arm_epoch = 4;
  sel.desired_mode = MODE_REFERENCE;
  for (int i = 0; i < 32; ++i) sel.model_digest[i] = uint8_t(i + 1);
  const ControlPacket sf = mode_frame(CTL_MODE_SELECT, 42, &sel);
  const std::string sel_hex = hex(&sf, sizeof(sf));
  assert(sel_hex ==
         "4143544c05000c002a0000000000000015cd5b07000000007b00000000000000"
         "0600000000000000050000000000000004000000020000000102030405060708"
         "090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f200000000000000000"
         "0000000000000000000000000000000000000000000000000000000000000000");

  ModeSelectReplyPayload rep{};
  rep.session_id = 123;
  rep.generation = 6;
  rep.predecessor = 5;
  rep.arm_epoch = 4;
  rep.result = MODE_SEL_OK;
  rep.selected_mode = MODE_REFERENCE;
  rep.requested_mode = MODE_REFERENCE;
  const ControlPacket rf = mode_frame(CTL_MODE_STATUS, 43, &rep);
  const std::string rep_hex = hex(&rf, sizeof(rf));
  assert(rep_hex ==
         "4143544c05000d002b0000000000000015cd5b07000000007b00000000000000"
         "0600000000000000050000000000000004000000000000000200020000000000"
         "0000000000000000000000000000000000000000000000000000000000000000"
         "0000000000000000000000000000000000000000000000000000000000000000");

  // --- structural validation ------------------------------------------------
  assert(valid_mode_select_body(&sel));
  assert(valid_mode_status_body(&rep));

  {  // MODE_STREAMED carries no digest; a digest there is malformed.
    ModeSelectPayload p = sel;
    p.desired_mode = MODE_STREAMED;
    assert(!valid_mode_select_body(&p));
    std::memset(p.model_digest, 0, 32);
    assert(valid_mode_select_body(&p));
  }
  {  // MODE_REFERENCE requires a nonzero digest.
    ModeSelectPayload p = sel;
    std::memset(p.model_digest, 0, 32);
    assert(!valid_mode_select_body(&p));
  }
  {  // MODE_UNSELECTED is never requestable.
    ModeSelectPayload p = sel;
    p.desired_mode = MODE_UNSELECTED;
    assert(!valid_mode_select_body(&p));
  }
  {  // generation must be predecessor + 1, and identities nonzero.
    ModeSelectPayload p = sel;
    p.expected_predecessor = 6;
    assert(!valid_mode_select_body(&p));
    p = sel; p.generation = 0;
    assert(!valid_mode_select_body(&p));
    p = sel; p.session_id = 0;
    assert(!valid_mode_select_body(&p));
    p = sel; p.arm_epoch = 0;
    assert(!valid_mode_select_body(&p));
  }
  {  // reserved bytes are rejected, never treated as extension.
    ModeSelectPayload p = sel;
    p.reserved0 = 1;
    assert(!valid_mode_select_body(&p));
    p = sel; p.request_flags = 1;
    assert(!valid_mode_select_body(&p));
    p = sel; p.reserved[3] = 1;
    assert(!valid_mode_select_body(&p));
  }
  {  // header/opcode/version/arg checks.
    ControlPacket out{};
    assert(decode_mode_select(&sf, sizeof(sf), CTL_MODE_SELECT, out));
    assert(decode_mode_select(&rf, sizeof(rf), CTL_MODE_STATUS, out));
    // A select frame offered as STATUS (or vice versa) is refused.
    assert(!decode_mode_select(&sf, sizeof(sf), CTL_MODE_STATUS, out));
    ControlPacket bad = sf;
    bad.version = REFERENCE_VERSION;
    assert(!decode_mode_select(&bad, sizeof(bad), CTL_MODE_SELECT, out));
    bad = sf; bad.arg = 1;
    assert(!decode_mode_select(&bad, sizeof(bad), CTL_MODE_SELECT, out));
    bad = sf; bad.magic = 0;
    assert(!decode_mode_select(&bad, sizeof(bad), CTL_MODE_SELECT, out));
  }
  {  // reply results and modes are bounded.
    ModeSelectReplyPayload p = rep;
    p.result = MODE_SEL_REFUSED + 1;
    assert(!valid_mode_status_body(&p));
    p = rep; p.selected_mode = 9;
    assert(!valid_mode_status_body(&p));
    p = rep; p.requested_mode = MODE_UNSELECTED;
    assert(!valid_mode_status_body(&p));
    p = rep; p.reserved[0] = 1;
    assert(!valid_mode_status_body(&p));
  }

  // --- ArmMode: which law may actuate --------------------------------------
  const uint64_t S = 0xABCDEF, G = 7;
  ArmMode m;
  // Never armed: nothing is effective, either family.
  assert(!m.admits(false, 100, S, uint32_t(G)));
  assert(!m.admits(true, 100, S, uint32_t(G)));

  m.arm(S, uint32_t(G), MODE_STREAMED);
  assert(m.belongs(S, uint32_t(G)) && m.confirmed());
  assert(m.admits(false, 1, S, uint32_t(G)));       // streamed envelope, no floor yet
  assert(!m.admits(true, 1, S, uint32_t(G)));       // native is not this law
  assert(!m.admits(false, 1, S + 1, uint32_t(G)));  // other session
  assert(!m.admits(false, 1, S, uint32_t(G + 1)));  // other epoch

  // Another session/epoch never actuates, whatever the mode.
  assert(!m.admits(false, 1, S + 2, uint32_t(G)));
  assert(!m.admits(true, 1, S, uint32_t(G + 2)));

  // Confirmation moves the floor: the envelope already in the Mailbox is out.
  ModeSelectPayload to_ref = sel;
  to_ref.desired_mode = MODE_REFERENCE;
  m.confirm(to_ref, 500);
  assert(m.mode == MODE_REFERENCE && m.generation == 6 && m.floor_ns == 500);
  assert(m.admits(true, 501, S, uint32_t(G)));   // native, above the floor
  assert(m.admits(true, 499, S, uint32_t(G)));   // native has no ingress floor
  assert(!m.admits(false, 501, S, uint32_t(G))); // streamed belongs to the old law

  // Same-mode selection is allowed and allocates a new generation.
  ModeSelectPayload again = sel;
  again.generation = 7; again.expected_predecessor = 6;
  again.desired_mode = MODE_REFERENCE;
  m.confirm(again, 900);
  assert(m.mode == MODE_REFERENCE && m.generation == 7 && m.floor_ns == 900);

  // Back to streamed: the ingress floor now applies to v5/v6.
  ModeSelectPayload to_soft = sel;
  to_soft.generation = 8; to_soft.expected_predecessor = 7;
  to_soft.desired_mode = MODE_STREAMED;
  std::memset(to_soft.model_digest, 0, 32);
  m.confirm(to_soft, 1000);
  assert(!m.admits(false, 1000, S, uint32_t(G)));  // at the floor: not authority
  assert(m.admits(false, 1001, S, uint32_t(G)));   // strictly above it
  assert(!m.admits(true, 1001, S, uint32_t(G)));   // native retired

  // A new ARM epoch discards everything.
  m.arm(S, uint32_t(G + 1), MODE_STREAMED);
  assert(m.generation == 0 && m.floor_ns == 0);
  m.disarm();
  assert(m.mode == MODE_UNSELECTED && !m.confirmed());
  assert(!m.admits(false, 5000, S, uint32_t(G + 1)));

  std::printf("MODE_SELECT %s\n", sel_hex.c_str());
  std::printf("MODE_STATUS %s\n", rep_hex.c_str());
  std::printf("SIZES %zu %zu\n", sizeof(ModeSelectPayload),
              sizeof(ModeSelectReplyPayload));
  std::printf("mode_select: wire goldens + ArmMode gate PASS\n");
  return 0;
}
