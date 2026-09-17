// Prints the three canonical golden packets as hex — one line each, prefixed
// CMD / STATE / CTL. `python -m arm_control.rt_protocol --hex` prints the
// identical lines from the hand-mirrored Python layout; rt_backend's _demo
// diffs the two automatically. If this ever disagrees, the wire is broken:
// fix protocol.hpp + rt_protocol.py in the same commit and bump VERSION.
#include <cstdio>
#include <cstring>
#include <initializer_list>

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
  constexpr uint64_t S = 0x0123456789abcdefull;
  const char joints[] = "fr3_joint1\nfr3_joint2\nfr3_joint3\nfr3_joint4\n"
                        "fr3_joint5\nfr3_joint6\nfr3_joint7";
  const uint64_t joint_hash = fnv1a64(joints, sizeof(joints) - 1);
  char instance[ID_CHARS] = {};
  std::strncpy(instance, "fr3_2", ID_CHARS - 1);
  const uint64_t ident = identity_hash(instance, joint_hash);

  CommandPacket cmd = {};
  cmd.magic = MAGIC_CMD;
  cmd.version = VERSION;
  cmd.n = 7;
  cmd.seq = 42;
  cmd.arm_epoch = 3;
  cmd.t_mono_ns = T;
  cmd.session_id = S;
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
  st.session_id = S;
  st.identity_hash = ident;
  st.arm_epoch = 3;
  st.stop_epoch = 2;
  for (int j = 0; j < 7; ++j) {
    st.q[j] = 0.1 * j + 0.01;
    st.dq[j] = 0.02 * j;
    st.tau[j] = 0.5 * j;
    st.tau_cmd[j] = 0.25 * j;
    st.q_cmd[j] = 0.1 * j;
    st.qd_cmd[j] = 0.01 * j;
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
  std::printf("HASH %016llx %016llx\n", (unsigned long long)joint_hash,
              (unsigned long long)ident);
  ControlPacket idf = {};
  idf.magic = MAGIC_CTL;
  idf.version = VERSION;
  idf.type = CTL_IDENTITY;
  idf.arg = 7;
  idf.t_mono_ns = T;
  IdentityPayload id{};
  std::memcpy(id.instance_id, instance, ID_CHARS);
  std::strncpy(id.model_id, "libfranka-vendor", ID_CHARS - 1);
  id.session_id = S;
  id.joint_order_hash = joint_hash;
  id.identity_hash = ident;
  id.capabilities = CAP_POSE_HOLD | CAP_OBSERVER;
  id.n = 7;
  std::memcpy(idf.text, &id, sizeof(id));
  print_hex("IDENTITY", &idf, sizeof(idf));
  ObserveRequest obs{MAGIC_OBS, VERSION, 0, 1000, 0};
  print_hex("OBSERVE", &obs, sizeof(obs));
  if (!valid_id("fr3_2") || valid_id("") || valid_id("bad id") ||
      valid_id("0123456789012345678901234567890123")) return 1;
  PoseHoldCommandPacket soft{};
  soft.command=cmd; soft.command.version=POSE_HOLD_VERSION;
  soft.pose_hold[0]=3;
  for(int i=0;i<6;++i) {
    soft.pose_hold[1+i]=i<3 ? 100 : 10;
    soft.pose_hold[7+i]=i<3 ? 20 : 2;
  }
  soft.pose_hold[13]=1; soft.pose_hold[14]=3;
  print_hex("POSE_HOLD", &soft, sizeof(soft));
  std::printf("POSE_HOLD_SIZE %zu\n",sizeof(soft));
  PoseHoldCommandPacket decoded{};
  for (uint16_t legacy : {uint16_t(1), uint16_t(2), uint16_t(3), uint16_t(4)}) {
    cmd.version=legacy;
    if (decode_command(&cmd,sizeof(cmd),true,decoded)) return 1;
    soft.command.version=legacy;
    if (decode_command(&soft,sizeof(soft),true,decoded)) return 1;
  }
  return 0;
}
