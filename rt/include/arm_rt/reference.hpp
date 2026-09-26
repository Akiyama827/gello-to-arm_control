// Pure bounded reference. Populated/validated off RT, then copied as an immutable
// value through the existing mailbox. This owns the native polynomial; probes
// and the servo must delegate here, never maintain separate evaluators.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include "arm_rt/protocol.hpp"
namespace arm_rt {
constexpr int NJ=MAX_JOINTS, NK=MAX_REFERENCE_KNOTS;
struct NativeReference {
  int n=0, k=0;
  bool hermite=false;
  std::array<double,NK> times{};
  std::array<double,NK*NJ> q{}, v{};
};
// No cursor, allocation, clock or mutation. Each call selects its own interval.
inline bool sample(const NativeReference& r, double t, double* out) {
  if(std::isnan(t)) return false;
  int endpoint=-1;
  if(t<=r.times[0]) endpoint=0;
  else if(t>=r.times[r.k-1]) endpoint=r.k-1;
  if(endpoint>=0) {
    out[0]=r.times[endpoint];
    for(int j=0;j<r.n;++j) {
      out[1+j]=r.q[endpoint*NJ+j];
      out[1+r.n+j]=r.hermite?r.v[endpoint*NJ+j]:0.;
      out[1+2*r.n+j]=0.;
    }
    return true;
  }
  const int upper=int(std::upper_bound(r.times.begin(),r.times.begin()+r.k,t)-r.times.begin());
  const int lower=upper-1;
  const double h=r.times[upper]-r.times[lower], s=(t-r.times[lower])/h;
  const double s2=s*s, s3=s*s*s;
  out[0]=t;
  for(int j=0;j<r.n;++j) {
    const double q0=r.q[lower*NJ+j], q1=r.q[upper*NJ+j];
    if(!r.hermite) {
      out[1+j]=(1.-s)*q0+s*q1;
      out[1+r.n+j]=(q1-q0)/h;
      out[1+2*r.n+j]=0.;
    } else {
      const double v0=r.v[lower*NJ+j], v1=r.v[upper*NJ+j];
      out[1+j]=(2*s3-3*s2+1)*q0+(s3-2*s2+s)*h*v0+(-2*s3+3*s2)*q1+(s3-s2)*h*v1;
      out[1+r.n+j]=(6*s2-6*s)*q0/h+(3*s2-4*s+1)*v0+(-6*s2+6*s)*q1/h+(3*s2-2*s)*v1;
      out[1+2*r.n+j]=(12*s-6)*q0/(h*h)+(6*s-4)*v0/h+(-12*s+6)*q1/(h*h)+(6*s-2)*v1/h;
    }
  }
  return true;
}
// Structural/numerical admission; no physical limit or scene policy is invented
// here. PC trajectory admission remains authoritative for model/scene bounds.
inline bool valid_reference(const NativeReference& r) {
  if(r.n<1||r.n>NJ||r.k<2||r.k>NK) return false;
  for(int i=0;i<r.k;++i) {
    if(!std::isfinite(r.times[i]) || (i && r.times[i]<=r.times[i-1])) return false;
    for(int j=0;j<r.n;++j)
      if(!std::isfinite(r.q[i*NJ+j]) || !std::isfinite(r.v[i*NJ+j])) return false;
    if(!i) continue;
    const double h=r.times[i]-r.times[i-1];
    if(!std::isfinite(h) || (r.hermite && (!std::isfinite(h*h) || h*h==0))) return false;
    for(int j=0;j<r.n;++j) {
      // Conservative magnitude bounds on every intermediate in the existing
      // polynomial, velocity and acceleration expressions, for s in [0,1].
      const long double q=std::abs((long double)r.q[(i-1)*NJ+j])+
                          std::abs((long double)r.q[i*NJ+j]);
      const long double v=std::abs((long double)r.v[(i-1)*NJ+j])+
                          std::abs((long double)r.v[i*NJ+j]);
      const long double H=h, max=std::numeric_limits<double>::max();
      if(r.hermite ? (32*(q+H*v+q/H+v+q/(H*H)+v/H)>max)
                   : (2*(q+q/H)>max)) return false;
    }
  }
  return true;
}
} // namespace arm_rt
