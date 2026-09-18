#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "protocol.hpp"
#include "servo_law.hpp"

namespace arm_rt {

// Plant-owned captured target. All Eigen storage is fixed size, including
// the padded Jacobian and its orthonormal row-space basis; no RT heap use.
// The kinematic torque projector is I - J^T (J^T)^+, as in Franka's
// Cartesian impedance example. This is not a dynamically consistent projector.
class PoseHold {
public:
  void reset() { id_=0; n_=0; elapsed_=0; }

  bool torque(int n, const double* q, const double* dq, const double* J,
              const double* pose, const double* bias, const double* spec,
              double dt, double* out) {
    if (n<1 || n>MAX_JOINTS || !valid_pose_hold_spec(spec) ||
        !std::isfinite(dt) || dt<=0) return false;
    for(int i=0;i<n;++i)
      if (!std::isfinite(q[i]) || !std::isfinite(dq[i]) || !std::isfinite(bias[i])) return false;
    for(int i=0;i<6*n;++i) if(!std::isfinite(J[i])) return false;
    for(int i=0;i<7;++i) if(!std::isfinite(pose[i])) return false;
    double normalized[7]; std::memcpy(normalized,pose,sizeof(normalized));
    double norm=0;
    for(int i=3;i<7;++i) norm+=pose[i]*pose[i];
    if(!std::isfinite(norm) || norm<1e-12) return false;
    norm=std::sqrt(norm);
    for(int i=3;i<7;++i) normalized[i]/=norm;
    // Canonical sign also resolves the exactly-180-degree quaternion tie.
    for(int i=3;i<7;++i) {
      if(normalized[i]==0) continue;
      if(normalized[i]<0) for(int k=3;k<7;++k) normalized[k]=-normalized[k];
      break;
    }
    const uint32_t id=static_cast<uint32_t>(spec[0]);
    if(id_==id && n_!=n) return false;
    if(id_!=id) {
      id_=id; n_=n; elapsed_=0;
      std::memcpy(target_,normalized,sizeof(target_));
      std::memcpy(q_null_,q,sizeof(double)*n);
    }
    elapsed_=std::min(.5,elapsed_+dt);
    const double ramp=elapsed_/.5;
    double kc[6],dc[6],twist[6]{},zero[6]{};
    constexpr double R[9]{1,0,0,0,1,0,0,0,1};
    Eigen::Matrix<double,6,MAX_JOINTS> jac=Eigen::Matrix<double,6,MAX_JOINTS>::Zero();
    Eigen::Matrix<double,MAX_JOINTS,1> null_tau=Eigen::Matrix<double,MAX_JOINTS,1>::Zero();
    for(int r=0;r<6;++r) {
      kc[r]=ramp*spec[1+r]; dc[r]=ramp*spec[7+r];
      for(int j=0;j<n;++j) { jac(r,j)=J[r*n+j]; twist[r]+=J[r*n+j]*dq[j]; }
    }
    for(int j=0;j<n;++j) {
      out[j]=bias[j];
      null_tau[j]=ramp*(spec[13]*(q_null_[j]-q[j])-spec[14]*dq[j]);
    }
    cartesian_impedance(n,J,R,normalized,normalized+3,target_,target_+3,
                        twist,zero,kc,dc,out);
    for(int j=0;j<n;++j)
      if(!std::isfinite(out[j]) || !std::isfinite(null_tau[j])) return false;
    // Scaling leaves the row-space projector unchanged and keeps finite
    // extreme inputs from overflowing the factorization's arithmetic.
    const double jac_scale=jac.cwiseAbs().maxCoeff();
    if(jac_scale>0) jac/=jac_scale;
    // QR's loops are bounded by matrix dimensions; unlike iterative SVD,
    // no convergence loop is needed to obtain this row-space projector.
    Eigen::ColPivHouseholderQR<Eigen::Matrix<double,MAX_JOINTS,6>> qr(jac.transpose());
    const double absolute=jac_scale>0 ? 1e-12/jac_scale : 1e-12;
    qr.setThreshold(std::max(1e-8,qr.maxPivot()>0 ? absolute/qr.maxPivot() : 1e-12));
    const Eigen::Matrix<double,MAX_JOINTS,MAX_JOINTS> basis=qr.householderQ();
    // Remove the row-space components, retaining damping even when kp=0.
    auto projected=null_tau.eval();
    const int rank=qr.rank();
    for(int i=0;i<rank;++i)
      projected.noalias()-=basis.col(i)*basis.col(i).dot(null_tau);
    for(int j=0;j<n;++j) {
      out[j]+=projected[j];
      if(!std::isfinite(out[j])) return false;
    }
    return true;
  }

private:
  uint32_t id_=0;
  int n_=0;
  double elapsed_=0;
  double target_[7]{};
  double q_null_[MAX_JOINTS]{};
};
} // namespace arm_rt
