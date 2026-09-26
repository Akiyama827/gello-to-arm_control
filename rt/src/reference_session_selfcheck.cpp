// Non-RT control session on a test-provided socketpair; no plant thread/device.
// Enables the otherwise unadvertised capability solely to exercise B2 transport.
#include <cstdlib>
#include "control_session.cpp"
namespace arm_rt {
uint64_t mono_ns() {
  timespec t{}; clock_gettime(CLOCK_MONOTONIC,&t);
  return uint64_t(t.tv_sec)*1000000000ull+uint64_t(t.tv_nsec);
}
}
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  arm_rt::ServerCtx ctx;
  ctx.cfg.backend="fake"; ctx.backend_n.store(16); ctx.backend_ready.store(true);
  ctx.supports_reference.store(true); ctx.session_id.store(123);
  std::strcpy(ctx.instance_id,"fake"); std::strcpy(ctx.model_id,"fake");
  std::strcpy(ctx.backend_name,"fake"); ctx.joint_order_hash=456;
  ctx.identity_hash=arm_rt::identity_hash(ctx.instance_id,456);
  arm_rt::serve_client(ctx,std::atoi(argv[1]),123);
}
