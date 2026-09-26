// Software-only fake server: explicitly enables the gated native path for checks.
#include <csignal>
#include <cstdlib>
#include <thread>
#include "server.hpp"
namespace arm_rt {
std::unique_ptr<Backend> make_franka_backend(const std::string&,double,const double*) {return {};}
std::unique_ptr<Backend> make_dm_backend(const std::string&,uint32_t,double) {return {};}
}
static arm_rt::ServerCtx* running=nullptr;
static void stop(int) {if(running) running->shutdown.store(true);}
int main(int argc,char** argv) {
  if(argc!=3) return 2;
  arm_rt::ServerCtx ctx;
  ctx.cfg.backend="fake"; ctx.cfg.instance_id="fake"; ctx.cfg.n=3;
  ctx.cfg.udp_port=std::atoi(argv[1]); ctx.cfg.tcp_port=std::atoi(argv[2]);
  ctx.cfg.hold_ms=50; ctx.cfg.fault_ms=500; ctx.cfg.state_hz=200;
  ctx.supports_reference.store(true);
  running=&ctx; std::signal(SIGTERM,stop); std::signal(SIGINT,stop);
  std::thread udp(arm_rt::udp_rx_thread,std::ref(ctx));
  std::thread ctl(arm_rt::control_thread,std::ref(ctx));
  std::thread tx(arm_rt::state_tx_thread,std::ref(ctx));
  arm_rt::rt_loop(ctx);
  ctx.shutdown.store(true); udp.join(); ctl.join(); tx.join();
  arm_rt::close_udp_socket(ctx);
  return ctx.failed.load()?1:0;
}
