#define EIGEN_RUNTIME_NO_MALLOC
#include "arm_rt/pose_hold.hpp"
#include <cassert>
#include <cmath>
#include <limits>

int main() {
  using namespace arm_rt;
  PoseHold hold;
  double q[7]{}, dq[7]{}, J[42]{}, pose[7]{0,0,0,1,0,0,0}, bias[7]{}, tau[7]{};
  double spec[15]{1,100,100,100,10,10,10,20,20,20,2,2,2,0,3};
  for (int i=0; i<6; ++i) J[i*7+i]=1;
  Eigen::internal::set_is_malloc_allowed(false);
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.5,tau));
  pose[0]=.1; dq[6]=2;
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  assert(std::abs(tau[0]+10)<1e-10 && std::abs(tau[6]+6)<1e-10);
  pose[3]=std::cos(.05); pose[6]=std::sin(.05);
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  const double rz=tau[5]; assert(rz<0);
  for(int i=3;i<7;++i) pose[i]=-pose[i];
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  assert(std::abs(tau[5]-rz)<1e-10);
  // Nullspace spring/damping contributes nothing in the task's row space.
  spec[13]=10; q[0]=.2; q[6]=.3;
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  assert(std::abs(tau[0]+10)<1e-10 && std::abs(tau[6]+9)<1e-10);
  double zeros[7]{}, limits[7]{1,1,1,1,1,1,1}, limited[7]{};
  servo_torque(7,q,dq,q,zeros,tau,zeros,zeros,zeros,limits,.25,limited);
  for(double t:limited) assert(std::abs(t)<=.25);
  spec[0]=2; dq[6]=0;
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  for(double t:tau) assert(std::abs(t)<1e-10);
  spec[1]=-1; assert(!hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  spec[1]=100; spec[0]=1.5; assert(!valid_pose_hold_spec(spec));
  spec[0]=1; J[0]=std::numeric_limits<double>::quiet_NaN();
  assert(!hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  assert(!hold.torque(17,q,dq,J,pose,bias,spec,.001,tau));
  J[0]=1; hold.reset(); pose[0]=0;
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  pose[0]=.1;
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  assert(std::abs(tau[0]+.04)<1e-10);  // 2 ms into the 0.5 s gain ramp
  hold.reset();
  pose[3]=1; pose[4]=pose[5]=pose[6]=0;
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.5,tau));
  pose[3]=0; pose[4]=1;
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  const double pi_torque=tau[3];
  pose[4]=-1;
  assert(hold.torque(7,q,dq,J,pose,bias,spec,.001,tau));
  assert(std::abs(tau[3]-pi_torque)<1e-10);
  hold.reset();
  double large_q[16]{}, large_dq[16]{}, large_J[96]{}, large_bias[16]{}, large_tau[16]{};
  for(int r=0;r<6;++r) for(int j=0;j<16;++j)
    large_J[r*16+j]=std::sin(double(1+r*16+j));
  for(int j=0;j<16;++j) large_dq[j]=.1*j;
  double damping_spec[15]{1,0,0,0,0,0,0,0,0,0,0,0,0,0,3};
  assert(hold.torque(16,large_q,large_dq,large_J,pose,large_bias,damping_spec,.5,large_tau));
  for(int r=0;r<6;++r) {
    double residual=0;
    for(int j=0;j<16;++j) residual+=large_J[r*16+j]*large_tau[j];
    assert(std::abs(residual)<1e-9);  // rank-deficient, non-axis-aligned Jacobian
  }
  std::memset(large_J,0,sizeof(large_J)); hold.reset();
  assert(hold.torque(16,large_q,large_dq,large_J,pose,large_bias,damping_spec,.5,large_tau));
  for(int j=0;j<16;++j) assert(std::abs(large_tau[j]+3*large_dq[j])<1e-10);
  for(int r=0;r<6;++r) large_J[r*16+r]=r==5 ? 1e-10 : 1;
  assert(hold.torque(16,large_q,large_dq,large_J,pose,large_bias,damping_spec,.5,large_tau));
  for(int j=0;j<5;++j) assert(std::abs(large_tau[j])<1e-10);
  assert(std::abs(large_tau[5]+3*large_dq[5])<1e-10);  // relative rank cutoff
  PoseHoldCommandPacket packet{}, decoded{};
  packet.command.magic=MAGIC_CMD; packet.command.version=VERSION; packet.command.n=7;
  assert(decode_command(&packet,sizeof(CommandPacket),false,decoded));
  for(double s:decoded.pose_hold) assert(s==0);
  assert(!decode_command(&packet,sizeof(packet),true,decoded));
  packet.command.version=POSE_HOLD_VERSION;
  std::memcpy(packet.pose_hold,spec,sizeof(spec));
  assert(decode_command(&packet,sizeof(packet),true,decoded));
  assert(!decode_command(&packet,sizeof(packet),false,decoded));
  assert(!decode_command(&packet,sizeof(CommandPacket),true,decoded));
  assert(!decode_command(&packet,sizeof(packet)+1,true,decoded));
  packet.pose_hold[1]=-1; assert(!decode_command(&packet,sizeof(packet),true,decoded));
  packet.pose_hold[1]=100; packet.command.kp[0]=std::numeric_limits<double>::infinity();
  assert(!decode_command(&packet,sizeof(packet),true,decoded));
  Eigen::internal::set_is_malloc_allowed(true);
}
