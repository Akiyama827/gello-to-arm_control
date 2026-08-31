// Prints the three canonical golden packets as hex — one line each, prefixed
// CMD / STATE / CTL. `python -m arm_control.rt_protocol --hex` prints the
// identical lines from the hand-mirrored Python layout; rt_backend's _demo
// diffs the two automatically. If this ever disagrees, the wire is broken:
// fix protocol.hpp + rt_protocol.py in the same commit and bump VERSION.
#include <cstdio>
#include <cstring>

#include "arm_rt/protocol.hpp"

using namespace arm_rt;

namespace {

void print_hex(const char* tag, const void* p, size_t len) {
  std::printf("%s ", tag);
  const auto* b = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < len; ++i) std::printf("%02x", b[i]);
  std::printf("\n");
}

} // namespace

int main() {
  constexpr uint64_t T = 1234567890123456789ull;

  CommandPacket cmd = {};
  cmd.magic = MAGIC_CMD;
  cmd.version = VERSION;
  cmd.n = 7;
  cmd.seq = 42;
  cmd.flags = 0;
  cmd.t_mono_ns = T;
  for (int j = 0; j < 7; ++j) {
    cmd.q_des[j] = 0.1 * j;
    cmd.qd_des[j] = 0.01 * j;
    cmd.tau_ff[j] = 1.0 * j;
    cmd.kp[j] = 100.0 + j;
    cmd.kd[j] = 0.5 * j;
  }
  print_hex("CMD", &cmd, sizeof(cmd));

  StatePacket st = {};
  st.magic = MAGIC_STATE;
  st.version = VERSION;
  st.n = 7;
  st.state_seq = 1000;
  st.last_cmd_seq = 42;
  st.t_mono_ns = T;
  st.flags = with_online_mask(FLAG_ARMED, 0b101);
  st.fault_code = 0;
  for (int j = 0; j < 7; ++j) {
    st.q[j] = 0.1 * j + 0.01;
    st.dq[j] = 0.02 * j;
    st.tau[j] = 0.5 * j;
    st.tau_cmd[j] = 0.25 * j;
    st.q_cmd[j] = 0.1 * j;
  }
  print_hex("STATE", &st, sizeof(st));

  ControlPacket ctl = {};
  ctl.magic = MAGIC_CTL;
  ctl.version = VERSION;
  ctl.type = CTL_STATUS;
  ctl.seq = 7;
  ctl.arg = 1;
  ctl.t_mono_ns = T;
  std::strncpy(ctl.text, "ok", sizeof(ctl.text) - 1);
  print_hex("CTL", &ctl, sizeof(ctl));

  std::printf("SIZES %zu %zu %zu\n", sizeof(CommandPacket), sizeof(StatePacket),
              sizeof(ControlPacket));
  return 0;
}
