// Non-RT TCP-owned bounded upload/admission. No sockets, servo law or authority.
// SHA256 uses libcrypto off-loop; the servo only sees a validated POD copy.
#pragma once
#include <openssl/sha.h>
#include <array>
#include <cstring>
#include "arm_rt/reference_selection.hpp"

namespace arm_rt {
class ReferenceUpload {
 public:
  ReferenceUpload(uint64_t session, uint64_t joint_hash, uint16_t n,
                  const uint8_t model[32]) : session_(session), joints_(joint_hash), n_(n) {
    std::memcpy(model_.data(),model,32);
  }
  // The caller passes the current authority at each operation, not a cached
  // epoch from BEGIN. Epoch change discards only the unselected candidate.
  void observe_epoch(uint32_t epoch) {
    if(epoch!=epoch_) { epoch_=epoch; present_=sealed_=false; }
  }
  ReferenceResult begin(const ReferenceBeginPayload& b, uint64_t selected_id=0,
                        uint32_t selected_revision=0) {
    if(!valid_reference_body(CTL_REFERENCE_BEGIN,&b)) return REF_MALFORMED;
    if(b.session_id!=session_ || b.joint_order_hash!=joints_ || b.n!=n_ ||
       b.arm_epoch!=epoch_ || std::memcmp(b.model_digest,model_.data(),32)) return REF_IDENTITY;
    if(b.expected_revision!=selected_revision ||
       (selected_revision && b.reference_id!=selected_id)) return REF_REVISION;
    if(present_) {
      if(b.upload_id==candidate_.identity.upload_id)
        return std::memcmp(&b,&candidate_.identity,sizeof(b))?REF_IDENTITY:REF_OK;
      return REF_BUSY;
    }
    if(b.upload_id<=last_upload_) return REF_RETIRED;
    last_upload_=b.upload_id;
    candidate_.identity=b; candidate_.curve={};
    present_=true; sealed_=false; received_=0;
    return REF_OK;
  }
  ReferenceResult chunk(const ReferenceChunkPayload& c) {
    if(!valid_reference_body(CTL_REFERENCE_CHUNK,&c)) return REF_MALFORMED;
    if(!present_ || c.upload_id!=candidate_.identity.upload_id) return REF_RETIRED;
    const auto end=c.offset+uint32_t(c.length); // bounded by decoder, cannot wrap
    if(end>candidate_.identity.byte_count || c.offset%REFERENCE_CHUNK_BYTES ||
       c.length!=std::min(REFERENCE_CHUNK_BYTES,candidate_.identity.byte_count-c.offset))
      return REF_MALFORMED;
    // Exact retry of already stored bytes, never a patch or partial overlap.
    if(end<=received_)
      return std::memcmp(bytes_.data()+c.offset,c.data,c.length)?REF_INTEGRITY:REF_OK;
    if(sealed_ || c.offset!=received_) return REF_MALFORMED;
    std::memcpy(bytes_.data()+received_,c.data,c.length); received_=end;
    return REF_OK;
  }
  ReferenceResult seal(const ReferenceSealPayload& s) {
    if(!valid_reference_body(CTL_REFERENCE_SEAL,&s)) return REF_MALFORMED;
    if(!present_ || s.upload_id!=candidate_.identity.upload_id) return REF_RETIRED;
    if(received_!=candidate_.identity.byte_count) return REF_NOT_ADMITTED;
    uint8_t digest[32]; SHA256(bytes_.data(),received_,digest);
    if(std::memcmp(digest,s.sha256,32)) return REF_INTEGRITY;
    if(sealed_) return REF_OK;
    auto& r=candidate_.curve;
    const auto& b=candidate_.identity;
    r.n=b.n; r.k=b.knots; r.hermite=bool(b.flags & REFERENCE_HERMITE);
    size_t at=0;
    auto read=[&]() {double d; std::memcpy(&d,bytes_.data()+at,8); at+=8; return d;};
    for(int i=0;i<r.k;++i) {
      r.times[i]=read();
      for(int j=0;j<r.n;++j) r.q[i*NJ+j]=read();
      if(r.hermite) for(int j=0;j<r.n;++j) r.v[i*NJ+j]=read();
    }
    if(r.times[0]<0 || (!b.expected_revision && r.times[0]!=0) || !valid_reference(r))
      return REF_MALFORMED;
    sealed_=true;
    return REF_OK;
  }
  ReferenceResult abort(uint64_t id) {
    if(!id) return REF_MALFORMED;
    if(!present_) return id==last_aborted_?REF_OK:REF_RETIRED;
    if(id!=candidate_.identity.upload_id) return REF_RETIRED;
    last_aborted_=id; discard(); return REF_OK;
  }
  void discard() { present_=sealed_=false; }
  const AdmittedReference* admitted() const { return present_&&sealed_?&candidate_:nullptr; }
  const ReferenceBeginPayload* candidate_identity() const { return present_?&candidate_.identity:nullptr; }
  bool busy() const { return present_; }
  uint64_t upload_id() const { return present_?candidate_.identity.upload_id:0; }
 private:
  uint64_t session_, joints_;
  uint16_t n_;
  uint32_t epoch_=0, received_=0;
  uint64_t last_upload_=0, last_aborted_=0;
  bool present_=false, sealed_=false;
  std::array<uint8_t,32> model_{};
  std::array<uint8_t,MAX_REFERENCE_BYTES> bytes_{};
  AdmittedReference candidate_{};
};
} // namespace arm_rt
