#include <cassert>
#include <cstdio>
#include <limits>
#include <vector>
#include "reference_upload.hpp"
using namespace arm_rt;

int main() {
  uint8_t model[32]{}; model[0]=7;
  ReferenceUpload u(123,456,16,model); u.observe_epoch(4);
  ReferenceBeginPayload b{}; b.session_id=123; b.upload_id=1; b.reference_id=10;
  b.joint_order_hash=456; std::memcpy(b.model_digest,model,32);
  b.arm_epoch=4; b.revision=1; b.knots=128; b.n=16;
  b.flags=REFERENCE_HERMITE; b.byte_count=MAX_REFERENCE_BYTES;
  std::vector<double> values;
  for(int i=0;i<128;++i) {
    values.push_back(double(i)*.01);
    for(int j=0;j<16;++j) values.push_back(double(i+j)*.001);
    for(int j=0;j<16;++j) values.push_back(.1);
  }
  const auto* data=reinterpret_cast<const uint8_t*>(values.data());
  assert(values.size()*8==b.byte_count);
  ReferenceSealPayload seal{}; seal.upload_id=1;
  SHA256(data,b.byte_count,seal.sha256);
  const uint8_t golden[32]={0x20,0xfc,0xde,0x4c,0xc4,0xdd,0x8d,0x98,0x71,0x9c,0xc6,0xa0,0x1c,0xe2,0xb5,0x58,0xa8,0x62,0x7d,0xd,0x73,0xc5,0x83,0x58,0xc9,0x22,0x28,0x7b,0x1a,0xe0,0x7,0x7};
  assert(std::memcmp(golden,seal.sha256,32)==0);
  assert(u.begin(b)==REF_OK && u.busy() && !u.admitted());
  assert(u.begin(b)==REF_OK); // exact retry only
  auto bad=b; bad.reference_id++; assert(u.begin(bad)==REF_IDENTITY);
  bad=b; bad.upload_id++; assert(u.begin(bad)==REF_BUSY);
  for(int field=0;field<5;++field) {
    bad=b;
    if(field==0) ++bad.session_id;
    if(field==1) ++bad.arm_epoch;
    if(field==2) ++bad.joint_order_hash;
    if(field==3) ++bad.model_digest[0];
    if(field==4) {bad.n=7; bad.byte_count=128*(8+16*7);}
    assert(u.begin(bad)==REF_IDENTITY);
  }
  assert(u.seal(seal)==REF_NOT_ADMITTED);
  ReferenceChunkPayload c{}; c.upload_id=1; c.length=88; c.offset=88;
  std::memcpy(c.data,data+88,88); assert(u.chunk(c)==REF_MALFORMED);
  for(uint32_t offset=0;offset<b.byte_count;offset+=88) {
    c={}; c.upload_id=1; c.offset=offset;
    c.length=uint16_t(std::min(uint32_t(88),b.byte_count-offset));
    std::memcpy(c.data,data+offset,c.length);
    assert(u.chunk(c)==REF_OK && u.chunk(c)==REF_OK);
    c.data[0]^=1; assert(u.chunk(c)==REF_INTEGRITY); c.data[0]^=1;
  }
  auto wrong=seal; wrong.sha256[0]^=1; assert(u.seal(wrong)==REF_INTEGRITY);
  assert(u.seal(seal)==REF_OK && u.seal(seal)==REF_OK && u.admitted());
  assert(u.admitted()->curve.k==128 && u.admitted()->curve.n==16);
  double out[49]; assert(sample(u.admitted()->curve,.555,out));
  assert(std::abs(out[1]-.0555)<1e-14 && std::abs(out[17]-.1)<1e-14);
  assert(u.seal(wrong)==REF_INTEGRITY && u.admitted());
  for(uint32_t offset : {33791u,33792u,0xfffffff0u,0xffffffffu}) {
    c.offset=offset; c.length=88; assert(u.chunk(c)==REF_MALFORMED);
  }
  assert(u.abort(2)==REF_RETIRED && u.admitted());
  assert(u.abort(1)==REF_OK && u.abort(1)==REF_OK && !u.busy());
  assert(u.begin(b)==REF_RETIRED && u.seal(seal)==REF_RETIRED);
  b.upload_id=2; assert(u.begin(b)==REF_OK);
  u.observe_epoch(5); assert(!u.busy() && u.begin(b)==REF_IDENTITY);
  b.arm_epoch=5; assert(u.begin(b)==REF_RETIRED);
  b.upload_id=3; assert(u.begin(b)==REF_OK); u.discard();
  assert(u.begin(b)==REF_RETIRED);
  // Exact epoch/session checks survive producer flood without storage growth.
  for(int i=0;i<100000;++i) {bad=b; bad.session_id++; assert(u.begin(bad)==REF_IDENTITY);}
  // Hash-valid malformed curves never become admitted.
  for(int test=0;test<7;++test) {
    ++b.upload_id; b.knots=2; b.byte_count=2*(8+16*16);
    assert(u.begin(b)==REF_OK);
    std::vector<double> bad_values(values.begin(),values.begin()+66);
    if(test==0) bad_values[33]=0;       // repeated time
    if(test==1) bad_values[33]=-1;      // reversed
    if(test==2) bad_values[1]=std::numeric_limits<double>::quiet_NaN();
    if(test==3) bad_values[17]=std::numeric_limits<double>::infinity();
    if(test==4) bad_values[0]=.001;     // initial release not zero
    if(test==5) bad_values[33]=1e-200;  // h*h underflow
    if(test==6) bad_values[1]=1e308;    // overflowing evaluation intermediates
    const auto* bytes=reinterpret_cast<const uint8_t*>(bad_values.data());
    for(uint32_t at=0;at<b.byte_count;at+=88) {
      c={}; c.upload_id=b.upload_id; c.offset=at;
      c.length=uint16_t(std::min(uint32_t(88),b.byte_count-at));
      std::memcpy(c.data,bytes+at,c.length); assert(u.chunk(c)==REF_OK);
    }
    seal.upload_id=b.upload_id; SHA256(bytes,b.byte_count,seal.sha256);
    assert(u.seal(seal)==REF_MALFORMED && !u.admitted()); u.discard();
  }
  std::printf("PASS reference upload: capacity, identity, SHA256, retry, mutation, abort, "
              "epoch, flood, numeric rejection; bounded owner %zu bytes\n",sizeof(u));
}
