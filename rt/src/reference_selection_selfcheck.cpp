#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include "arm_rt/reference_selection.hpp"
#include "arm_rt/mailbox.hpp"
using namespace arm_rt;
static bool forbid_alloc=false;
void* operator new(std::size_t n) { assert(!forbid_alloc); if(auto p=std::malloc(n)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}

ReferenceCommit initial(uint64_t id=10) {
  ReferenceCommit c{};
  auto& b=c.admitted.identity;
  b.session_id=123; b.arm_epoch=4; b.reference_id=id; b.upload_id=id;
  b.revision=1; b.knots=128; b.n=16; b.flags=REFERENCE_HERMITE;
  b.model_digest[0]=7; b.joint_order_hash=456;
  auto& r=c.admitted.curve; r.n=16; r.k=128; r.hermite=true;
  for(int i=0;i<128;++i) {
    r.times[i]=i*.01;
    for(int j=0;j<16;++j) {r.q[i*NJ+j]=.001*(i+j); r.v[i*NJ+j]=.1;}
  }
  auto& p=c.request;
  p.session_id=123; p.arm_epoch=4; p.upload_id=id; p.reference_id=id; p.revision=1;
  c.ticket=1;
  return c;
}
int main() {
  ReferenceSelection s; s.reset(123,4);
  ReferenceCommit c=initial(); double point[49]{};
  Mailbox<ReferenceCommit> mailbox;
  static_assert(std::is_trivially_copyable<ReferenceCommit>::value,"bounded immutable publication");
  forbid_alloc=true;
  mailbox.write(c); ReferenceCommit received{}; assert(mailbox.read(received)==1);
  assert(s.commit(received,1000000000,0)==REF_OK);
  assert(s.evaluate(1000000000,point) && point[1]==0);
  assert(s.commit(received,1001000000,0)==REF_RETIRED);
  auto next=c; next.request.expected_revision=next.admitted.identity.expected_revision=1;
  next.request.revision=next.admitted.identity.revision=2;
  next.request.upload_id=next.admitted.identity.upload_id=11;
  next.request.protected_until=1.26; next.request.latest_phase=1.25;
  for(int i=0;i<127;++i) {
    next.admitted.curve.times[i]=c.admitted.curve.times[i+1];
    for(int j=0;j<16;++j) {
      next.admitted.curve.q[i*NJ+j]=c.admitted.curve.q[(i+1)*NJ+j];
      next.admitted.curve.v[i*NJ+j]=c.admitted.curve.v[(i+1)*NJ+j];
    }
  }
  next.admitted.curve.times[127]=1.28;
  for(int j=0;j<16;++j) next.admitted.curve.q[127*NJ+j]=.13;
  const auto before=s.info();
  auto refuse=[&](ReferenceCommit bad,ReferenceResult result) {
    assert(s.commit(bad,1250000000,0)==result);
    assert(s.info().reference_id==before.reference_id && s.info().revision==before.revision);
    assert(std::memcmp(&s.selected(),&c.admitted,sizeof(c.admitted))==0);
  };
  auto bad=next; bad.request.session_id++; refuse(bad,REF_IDENTITY);
  bad=next; bad.request.arm_epoch++; refuse(bad,REF_IDENTITY);
  bad=next; bad.cancel_generation=1; refuse(bad,REF_RETIRED);
  bad=next; bad.request.expected_revision=bad.admitted.identity.expected_revision=2;
  refuse(bad,REF_REVISION);
  bad=next; bad.admitted.identity.model_digest[0]++; refuse(bad,REF_PREFIX);
  bad=next; bad.admitted.curve.q[0]+=.001; refuse(bad,REF_PREFIX);
  bad=next; bad.admitted.curve.v[125*NJ]+=.001; refuse(bad,REF_PREFIX);
  bad=next; bad.request.latest_phase=.25; refuse(bad,REF_LATE);
  // Whole prefix, not just the cut endpoint, is frozen.
  for(int i=0;i<126;++i) {
    bad=next; bad.admitted.curve.q[i*NJ+15]+=.001; refuse(bad,REF_PREFIX);
  }
  assert(s.commit(next,1250000000,0)==REF_OK && s.info().revision==2);
  assert(s.evaluate(1500000000,point) && std::abs(point[1]-.05)<1e-15);
  assert(!s.evaluate(2280000000,point) && s.info().status==REFERENCE_UNDERFLOW);
  assert(!s.evaluate(2281000000,point)); // exhausted revision cannot heal
  assert(s.commit(next,2281000000,0)==REF_REVISION);
  auto terminal=initial(20); terminal.admitted.identity.flags|=REFERENCE_TERMINAL;
  assert(s.commit(terminal,3000000000,0)==REF_OK);
  assert(s.evaluate(4270000000,point) && s.info().status==REFERENCE_END);
  for(int j=0;j<16;++j) {
    assert(point[1+j]==terminal.admitted.curve.q[127*NJ+j]);
    assert(point[17+j]==0 && point[33+j]==0);
  }
  auto after_end=initial(21);
  assert(s.commit(after_end,4500000000,0)==REF_OK);
  assert(s.info().reference_id==21 && s.info().phase==0);
  s.retire(21); assert(!s.active() && s.info().status==REFERENCE_CANCELLED);
  auto pending=initial(21); assert(s.commit(pending,5000000000,0)==REF_RETIRED);
  auto fresh=initial(22); fresh.cancel_generation=1;
  assert(s.commit(fresh,5000000000,1)==REF_OK);
  s.reset(124,4); assert(!s.active());
  assert(s.commit(fresh,5000000000,1)==REF_IDENTITY);
  s.reset(123,5); assert(s.commit(fresh,5000000000,1)==REF_IDENTITY);
  // Largest immutable publication, validation and copy: fixed capacities,
  // one read and one write, no retry loop or allocation on selection.
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<10000;++i) {
    s.reset(123,uint32_t(100+i));
    c.request.arm_epoch=c.admitted.identity.arm_epoch=uint32_t(100+i);
    mailbox.write(c); mailbox.read(received);
    assert(s.commit(received,1000000000,0)==REF_OK);
    next.request.arm_epoch=next.admitted.identity.arm_epoch=uint32_t(100+i);
    mailbox.write(next); mailbox.read(received);
    assert(s.commit(received,1250000000,0)==REF_OK);
  }
  auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/10000;
  forbid_alloc=false;
  std::printf("PASS selection: immutable prefix, stale binding/epoch/revision, underflow, "
              "terminal endpoint, cancel/restart; commit %zu mailbox %zu bytes; "
              "host mean initial+replacement publish/read/select %.0f ns (not RT qualification)\n",
              sizeof(c),sizeof(mailbox),ns);
}
