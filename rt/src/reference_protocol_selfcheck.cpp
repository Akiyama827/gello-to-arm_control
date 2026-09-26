// Golden schema and adversarial decoder seam; no server or device connection.
#include <cstdio>
#include <cstring>
#include <vector>
#include "arm_rt/protocol.hpp"
using namespace arm_rt;
namespace {
void hex(const char* name, const void* data, size_t n) {
  std::printf("%s ",name);
  const auto* p=static_cast<const unsigned char*>(data);
  for(size_t i=0;i<n;++i) std::printf("%02x",p[i]);
  std::puts("");
}
template<class T> void control(const char* name, uint16_t op, const T& body) {
  ControlPacket p{}; p.magic=MAGIC_CTL;
  p.version=op==CTL_REFERENCE_MODE?VERSION:REFERENCE_VERSION;
  p.type=op; p.seq=42; p.t_mono_ns=123456789;
  std::memcpy(p.text,&body,sizeof(body));
  hex(name,&p,sizeof(p));
}
}
int main(int argc,char**) {
  if(argc>1) {
    // kind, negotiated, uint32 size LE, packet. Refuse arbitrary large input.
    unsigned char h[6];
    while(std::fread(h,1,6,stdin)==6) {
      uint32_t n=0; for(int i=0;i<4;++i) n|=uint32_t(h[i+2])<<(8*i);
      if(n>2048) return 2;
      std::vector<unsigned char> b(n);
      if(std::fread(b.data(),1,n,stdin)!=n) return 2;
      ControlPacket c{}; ReferenceCommandPacket d{}; ReferenceStatePacket s{};
      const bool ok=h[0]=='C'?decode_reference_control(b.data(),n,h[1],c):
        h[0]=='D'?decode_reference_command(b.data(),n,h[1],d):
        h[0]=='S'?decode_reference_state(b.data(),n,h[1],s):false;
      std::putchar(ok?'1':'0');
    }
    return 0;
  }
  ReferenceModePayload mode{}; mode.session_id=123; mode.joint_order_hash=456;
  for(int i=0;i<32;++i) mode.model_digest[i]=uint8_t(i+1);
  mode.enable=1; control("MODE",CTL_REFERENCE_MODE,mode);
  ReferenceBeginPayload b{}; b.session_id=123; b.upload_id=2; b.reference_id=3;
  b.joint_order_hash=456; std::memcpy(b.model_digest,mode.model_digest,32);
  b.arm_epoch=4; b.expected_revision=5; b.revision=6; b.knots=128; b.n=16;
  b.flags=REFERENCE_HERMITE|REFERENCE_TERMINAL; b.byte_count=33792;
  control("BEGIN",CTL_REFERENCE_BEGIN,b);
  ReferenceChunkPayload ch{}; ch.upload_id=2; ch.offset=88; ch.length=3;
  ch.data[0]=1; ch.data[1]=2; ch.data[2]=3;
  control("CHUNK",CTL_REFERENCE_CHUNK,ch);
  ReferenceSealPayload seal{}; seal.upload_id=2;
  for(int i=0;i<32;++i) seal.sha256[i]=uint8_t(32-i);
  control("SEAL",CTL_REFERENCE_SEAL,seal);
  ReferenceCommitPayload cm{}; cm.session_id=123; cm.upload_id=2;
  cm.reference_id=3; cm.arm_epoch=4; cm.expected_revision=5; cm.revision=6;
  cm.protected_until=1.25; cm.latest_phase=1;
  control("COMMIT",CTL_REFERENCE_COMMIT,cm);
  cm.reference_id=4; cm.expected_revision=0; cm.revision=1;
  cm.protected_until=cm.latest_phase=0; cm.selection_budget_ns=123456789;
  control("DEADLINE",CTL_REFERENCE_COMMIT,cm);
  ReferenceRetirePayload re{}; re.session_id=123; re.id=2; re.arm_epoch=4;
  control("ABORT",CTL_REFERENCE_ABORT,re);
  re.id=3; re.revision=6; control("CANCEL",CTL_CANCEL_REFERENCE,re);
  ReferenceReplyPayload r{}; r.upload_id=2; r.reference_id=3; r.revision=6;
  r.result=REF_PREFIX; r.operation=CTL_REFERENCE_COMMIT;
  control("REPLY",CTL_REFERENCE_STATUS,r);
  ReferenceCommandPacket d{}; d.magic=MAGIC_CMD; d.version=REFERENCE_VERSION;
  d.n=7; d.seq=42; d.arm_epoch=4; d.t_mono_ns=123456789; d.session_id=123;
  d.reference_id=3; d.revision=6; d.age_us=100; d.phase=1.25;
  d.valid_for_us=20000; d.phase_window_us=20000;
  for(int j=0;j<7;++j) {d.tau_ff[j]=j*.5; d.kp[j]=100+j; d.kd[j]=10+j;}
  hex("DYNAMIC",&d,sizeof(d));
  ReferenceStatePacket s{}; s.state.magic=MAGIC_STATE; s.state.version=REFERENCE_VERSION;
  s.state.n=7; s.state.session_id=123; s.state.arm_epoch=4; s.state.last_cmd_seq=41;
  s.reference_id=3; s.revision=6; s.status=REFERENCE_RUNNING; s.phase=1.26;
  s.envelope_phase=1.25; s.envelope_age_us=1000; s.envelope_seq=42;
  for(int j=0;j<7;++j) {s.q_ref[j]=j*.25; s.qd_ref[j]=j*.5; s.qdd_ref[j]=j*.75;}
  hex("STATE",&s,sizeof(s));
}
