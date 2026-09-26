// C ABI test wrapper; all interpolation delegates to the production native reference.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include "arm_rt/mailbox.hpp"

#include "arm_rt/reference.hpp"
using arm_rt::NativeReference;
using arm_rt::NJ;
using arm_rt::NK;
using arm_rt::sample;
// Probe ABI only: construction/destruction is off-loop. Storage is privately copied.
extern "C" void* probe_create(int n,int k,const double* times,const double* q,const double* v) {
  if(n<1||n>NJ||k<2||k>NK||!times||!q) return nullptr;
  auto r=std::make_unique<NativeReference>();
  r->n=n; r->k=k; r->hermite=v!=nullptr;
  for(int i=0;i<k;++i) {
    if(!std::isfinite(times[i])||(i&&times[i]<=times[i-1])) return nullptr;
    r->times[i]=times[i];
    for(int j=0;j<n;++j) {
      if(!std::isfinite(q[i*n+j])||(v&&!std::isfinite(v[i*n+j]))) return nullptr;
      r->q[i*NJ+j]=q[i*n+j];
      r->v[i*NJ+j]=v?v[i*n+j]:0.;
    }
  }
  if(!arm_rt::valid_reference(*r)) return nullptr;
  return r.release();
}
extern "C" void probe_destroy(void* r) { delete static_cast<NativeReference*>(r); }
extern "C" bool probe_sample(const void* r,double t,double* out) {
  return r&&out&&sample(*static_cast<const NativeReference*>(r),t,out);
}
extern "C" std::size_t probe_bytes() { return sizeof(NativeReference); }
extern "C" std::size_t probe_mailbox_bytes() { return sizeof(arm_rt::Mailbox<NativeReference>); }
extern "C" double probe_benchmark(const void* p,int count,double* checksum) {
  const auto& r=*static_cast<const NativeReference*>(p);
  double out[1+3*NJ]; *checksum=0.;
  const auto start=std::chrono::steady_clock::now();
  for(int i=0;i<count;++i) {
    const double t=r.times[0]+(r.times[r.k-1]-r.times[0])*(i%997)/997.;
    sample(r,t,out);
    for(int j=0;j<1+3*r.n;++j) *checksum+=out[j];
  }
  return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/count;
}
