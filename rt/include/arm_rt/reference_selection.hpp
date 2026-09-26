// RT-owned selection/phase, separate from pure evaluation and control law.
// All curve construction/validation/hashing stays on the TCP thread. Selection
// copies one bounded POD, checks bounded prefix bytes and never allocates.
#pragma once
#include <cstring>
#include "arm_rt/reference.hpp"
namespace arm_rt {
struct AdmittedReference {
  ReferenceBeginPayload identity{};
  NativeReference curve{};
  uint64_t sealed_at_ns=0;  // server clock, immutable across exact retries
};
struct ReferenceCommit {
  AdmittedReference admitted{};
  ReferenceCommitPayload request{};
  uint64_t ticket=0;
  uint64_t cancel_generation=0;
};
struct ReferenceSelectionInfo {
  uint64_t reference_id=0;
  uint32_t revision=0, status=REFERENCE_NONE;
  double phase=0;
  uint64_t session_id=0;
  uint32_t arm_epoch=0;
};
class ReferenceSelection {
 public:
  void reset(uint64_t session,uint32_t epoch) {
    if(session_==session && epoch_==epoch) return;
    session_=session; epoch_=epoch; highest_id_=0; info_={};
    info_.session_id=session; info_.arm_epoch=epoch;
  }
  bool active() const { return info_.status==REFERENCE_RUNNING || info_.status==REFERENCE_END; }
  const ReferenceSelectionInfo& info() const { return info_; }
  const AdmittedReference& selected() const { return active_; }
  double phase(uint64_t now) const {
    return now>=anchor_ns_?double(now-anchor_ns_)/1e9:0.;
  }
  ReferenceResult commit(const ReferenceCommit& c,uint64_t now,uint64_t cancel_generation) {
    const auto& b=c.admitted.identity;
    const auto& p=c.request;
    if(p.session_id!=session_ || b.session_id!=session_ || !session_ ||
       p.arm_epoch!=epoch_ || b.arm_epoch!=epoch_ || !epoch_ ||
       p.upload_id!=b.upload_id || p.reference_id!=b.reference_id ||
       p.revision!=b.revision || p.expected_revision!=b.expected_revision)
      return REF_IDENTITY;
    if(c.cancel_generation!=cancel_generation) return REF_RETIRED;
    if(p.selection_budget_ns && (!c.admitted.sealed_at_ns || now<c.admitted.sealed_at_ns ||
       now-c.admitted.sealed_at_ns>=p.selection_budget_ns)) return REF_LATE;
    if(!p.expected_revision) {
      // IDs come from a monotonically increasing per-epoch PC counter. One
      // high-water mark rejects replay without an unbounded retired-id cache.
      if(p.reference_id<=highest_id_) return REF_RETIRED;
      // A new explicitly admitted execution may replace a terminal endpoint;
      // until its COMMIT the old endpoint stays selected. A running reference
      // needs explicit cancellation or a prefix-preserving same-id revision.
      if(active() && info_.status!=REFERENCE_END) return REF_BUSY;
      if(c.admitted.curve.times[0]!=0) return REF_MALFORMED;
    } else {
      if(!active() || p.reference_id!=info_.reference_id || p.expected_revision!=info_.revision)
        return REF_REVISION;
      const double at=phase(now);
      if(at>=p.latest_phase) return REF_LATE;
      if(!prefix_matches(c.admitted,p.protected_until,at)) return REF_PREFIX;
    }
    if(!p.expected_revision) anchor_ns_=now;
    active_=c.admitted;
    highest_id_=p.reference_id;
    info_={p.reference_id,p.revision,REFERENCE_RUNNING,phase(now),session_,epoch_};
    return REF_OK;
  }
  // Cancellation also retires a pending id before it can be selected. The
  // caller validates the cancel target against selected/pending identity.
  void retire(uint64_t pending_id=0) {
    highest_id_=std::max(highest_id_,pending_id);
    if(active()) info_.status=REFERENCE_CANCELLED;
  }
  // false means no selected reference: existing RT lifecycle owns measured
  // hold. Terminal completion remains a higher-level settling/APPLIED decision.
  bool evaluate(uint64_t now,double* out) {
    if(!active()) return false;
    const auto& r=active_.curve;
    const double at=phase(now), end=r.times[r.k-1];
    info_.phase=std::min(at,end);
    if(at>=end) {
      if(!(active_.identity.flags & REFERENCE_TERMINAL)) {
        info_.status=REFERENCE_UNDERFLOW; return false;
      }
      info_.status=REFERENCE_END;
      sample(r,end,out);
      for(int j=0;j<r.n;++j) out[1+r.n+j]=out[1+2*r.n+j]=0;
      return true;
    }
    return sample(r,at,out);
  }
 private:
  bool prefix_matches(const AdmittedReference& candidate,double cut,double at) const {
    const auto& old=active_.curve; const auto& next=candidate.curve;
    if(old.n!=next.n || !old.hermite || !next.hermite ||
       std::memcmp(active_.identity.model_digest,candidate.identity.model_digest,32) ||
       active_.identity.joint_order_hash!=candidate.identity.joint_order_hash) return false;
    // Both ends of every retained segment must be existing exact knots. New
    // start can drop consumed segments, never the one covering current phase.
    if(next.times[0]>at || !(at<cut)) return false;
    int first=-1, stop=-1;
    for(int i=0;i<old.k;++i) {
      if(std::memcmp(&old.times[i],&next.times[0],8)==0) first=i;
      if(std::memcmp(&old.times[i],&cut,8)==0) stop=i;
    }
    if(first<0 || stop<=first || stop-first>=next.k) return false;
    for(int i=first;i<=stop;++i) {
      const int j=i-first;
      if(std::memcmp(&old.times[i],&next.times[j],8) ||
         std::memcmp(&old.q[i*NJ],&next.q[j*NJ],size_t(old.n)*8) ||
         std::memcmp(&old.v[i*NJ],&next.v[j*NJ],size_t(old.n)*8)) return false;
    }
    return true;
  }
  AdmittedReference active_{};
  ReferenceSelectionInfo info_{};
  uint64_t session_=0, anchor_ns_=0, highest_id_=0;
  uint32_t epoch_=0;
};
} // namespace arm_rt
