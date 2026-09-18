// The non-RT <-> RT seam: a single-slot seqlock over a POD value.
//
// Writer: seq goes odd, payload written, seq goes even (release). Reader:
// snapshot seq, copy, re-check seq — retry on a collision. The reader never
// blocks the writer and the writer never blocks at all, so it is safe in
// BOTH directions we use it: comms thread -> RT thread (commands, writer
// non-RT) and RT thread -> tx thread (state, writer IS the RT tick — two
// atomic stores and a memcpy, no branches that wait).
//
// Latest-wins by construction: a slow reader skips versions, exactly like
// the UDP link it sits behind. Single writer, any readers.
//
// (A plain double buffer was the first draft; it tears when the writer laps
// a reader onto the slot being copied. The seqlock's re-check closes that.)
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace arm_rt {

template <typename T>
class SeqLock {
  static_assert(std::is_trivially_copyable<T>::value, "POD only");

public:
  void write(const T& value) {
    const uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_release);          // odd: in progress
    std::atomic_thread_fence(std::memory_order_release);
    std::memcpy(&slot_, &value, sizeof(T));
    std::atomic_thread_fence(std::memory_order_release);
    seq_.store(s + 2, std::memory_order_release);          // even: published
  }

  // Copies the latest value; returns its even sequence (0 = never written,
  // or the writer was mid-write for the whole bounded retry — `out` is then
  // untouched and the caller keeps its previous copy, which is exactly the
  // right latest-wins fallback). Bounded so a preempted writer can never
  // spin a SCHED_FIFO reader forever (priority inversion on a shared core).
  uint64_t read(T& out) const {
    for (int attempt = 0; attempt < 16; ++attempt) {
      const uint64_t s1 = seq_.load(std::memory_order_acquire);
      if (s1 == 0) return 0;
      if (s1 & 1) continue;                                 // mid-write
      std::atomic_thread_fence(std::memory_order_acquire);
      std::memcpy(&out, &slot_, sizeof(T));
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint64_t s2 = seq_.load(std::memory_order_acquire);
      if (s1 == s2) return s2;
    }
    return 0;
  }

  uint64_t sequence() const { return seq_.load(std::memory_order_acquire); }

private:
  T slot_ = {};
  std::atomic<uint64_t> seq_{0};
};

} // namespace arm_rt
