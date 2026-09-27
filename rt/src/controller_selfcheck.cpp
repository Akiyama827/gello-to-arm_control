// Structural parity against the pre-extraction mode mapping, with heap guards.
#define EIGEN_RUNTIME_NO_MALLOC
#include "arm_rt/controller.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static bool forbid_alloc = false;
void* operator new(std::size_t n) {
  assert(!forbid_alloc);
  if (auto p = std::malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main() {
  using namespace arm_rt;
  unsigned compared = 0;
  for (int n : {2, 7, MAX_JOINTS}) {
    Controller controller;
    PoseHold previous_pose;
    PlantState state{};
    state.n = n;
    state.pose_valid = state.jacobian_valid = true;
    state.pose[3] = 1.;
    double q_des[MAX_JOINTS]{}, qd_des[MAX_JOINTS]{}, ff[MAX_JOINTS]{};
    double kp[MAX_JOINTS]{}, kd[MAX_JOINTS]{}, limits[MAX_JOINTS]{};
    double spec[15]{1,100,100,100,10,10,10,20,20,20,2,2,2,.5,2};
    Eigen::internal::set_is_malloc_allowed(false);
    forbid_alloc = true;
    for (int tick=0; tick<1200; ++tick) {
      const bool pose = (tick/200)%2;
      if(tick%173==0) { controller.reset(); previous_pose.reset(); }
      if(tick%137==0) spec[0] += 1;
      state.pose[0] = .0001*tick;
      for(int j=0;j<n;++j) {
        state.q[j] = .002*j + .00001*tick;
        state.dq[j] = -.01 + .0001*j;
        state.coriolis[j] = .1*j;
        state.tau_ref[j] = .05*j;
        q_des[j] = state.q[j] + .01;
        qd_des[j] = .02*j;
        ff[j] = .3*j;
        kp[j] = 100.; kd[j] = 10.; limits[j] = 4.;
      }
      for(int r=0;r<6;++r) for(int j=0;j<n;++j)
        state.jacobian[r*n+j] = r==j ? 1. : 0.;
      CommandPacket expected{};
      expected.n = uint16_t(n);
      if (pose) {
        assert(previous_pose.torque(n,state.q,state.dq,state.jacobian,state.pose,
                                    state.coriolis,spec,.001,expected.tau_ff));
        std::copy_n(state.q,n,expected.q_des);
        assert(controller.pose(n,state,spec,.001,true));
      } else {
        previous_pose.reset();
        std::copy_n(q_des,n,expected.q_des);
        std::copy_n(qd_des,n,expected.qd_des);
        std::copy_n(ff,n,expected.tau_ff);
        std::copy_n(kp,n,expected.kp);
        std::copy_n(kd,n,expected.kd);
        controller.joint(n,q_des,qd_des,ff,kp,kd);
      }
      const auto& actual=controller.command();
      assert(std::memcmp(&expected,&actual,sizeof(actual))==0);
      double old_tau[MAX_JOINTS]{}, new_tau[MAX_JOINTS]{};
      servo_torque(n,state.q,state.dq,expected.q_des,expected.qd_des,expected.tau_ff,
                   expected.kp,expected.kd,state.tau_ref,limits,.25,old_tau);
      servo_torque(n,state.q,state.dq,actual.q_des,actual.qd_des,actual.tau_ff,
                   actual.kp,actual.kd,state.tau_ref,limits,.25,new_tau);
      assert(std::memcmp(old_tau,new_tau,sizeof(old_tau))==0);
      ++compared;
    }
    // Refusal/reset stay explicit: the loop selects its existing measured hold.
    assert(!controller.pose(n,state,spec,.001,false));
    state.pose_valid=false;
    assert(!controller.pose(n,state,spec,.001,true));
    state.pose_valid=true; state.jacobian_valid=false;
    assert(!controller.pose(n,state,spec,.001,true));
    state.jacobian_valid=true; state.pose[0]=NAN;
    assert(!controller.pose(n,state,spec,.001,true));
    state.pose[0]=0.;
    controller.joint(n,q_des,qd_des,ff,kp,kd);
    PoseHold recaptured;
    double expected[MAX_JOINTS]{};
    assert(recaptured.torque(n,state.q,state.dq,state.jacobian,state.pose,
                            state.coriolis,spec,.001,expected));
    assert(controller.pose(n,state,spec,.001,true));
    assert(std::memcmp(expected,controller.command().tau_ff,sizeof(expected))==0);
    forbid_alloc=false;
    Eigen::internal::set_is_malloc_allowed(true);
  }
  std::printf("controller modes: %u byte-identical commands and limited torques; resets/refusals; no allocation\n",compared);
}
