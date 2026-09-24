// The non-RT <-> RT seam: a single-producer, single-consumer latest-value
// mailbox (triple buffer). Of three slots the writer owns one (back), the
// reader owns one (front) and the third (middle) is in transit. Ownership
// moves only by an atomic exchange of the middle index, so no slot is ever
// accessed by both threads: no overlapping ordinary copies, nothing to retry.
// (The SeqLock this replaces copied a slot the writer could be writing. The
// sequence re-check discarded torn results, but the overlapping non-atomic
// accesses are a data race under the C++ memory model regardless.)
//
// Happens-before: write() fills `back`, then publishes it as middle|FRESH with
// an acq_rel exchange (release: fill before publication). read() takes the
// middle slot with an acq_rel exchange only when FRESH (acquire: the fill is
// visible) and hands its previous front back in the same exchange; the
// writer's next exchange acquires that, so the reader's last copy of a slot
// happens-before the writer's next fill of it. Neither side loops, waits or
// allocates: O(sizeof T) per call, whatever the other side does.
//
// Latest-wins: a slow reader skips versions (like the UDP link behind it). A
// stalled reader never blocks the writer, and the writer never touches the
// reader's slot. Exactly ONE writer thread and ONE reader thread per mailbox;
// fan-out (observers) copies from the reader's value, never reads here.
#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

namespace arm_rt {

template <typename T>
class Mailbox {
  static_assert(std::is_trivially_copyable<T>::value, "POD only");
  static constexpr uint8_t INDEX = 3, FRESH = 4;
  struct Slot {
    T value{};
    uint64_t version = 0;
  };

public:
  void write(const T& value) {
    slots_[back_].value = value;
    slots_[back_].version = ++written_;
    back_ = middle_.exchange(uint8_t(back_ | FRESH), std::memory_order_acq_rel) & INDEX;
  }

  // Copies the newest published value into `out` and returns its version
  // (1, 2, ... in write order). 0 = nothing published yet (`out` = T{}). With
  // nothing new since the last read, the same value and version again.
  uint64_t read(T& out) {
    if (middle_.load(std::memory_order_relaxed) & FRESH)
      front_ = middle_.exchange(front_, std::memory_order_acq_rel) & INDEX;
    out = slots_[front_].value;
    return slots_[front_].version;
  }

private:
  Slot slots_[3];
  std::atomic<uint8_t> middle_{1};
  uint8_t back_ = 0;      // writer's
  uint8_t front_ = 2;     // reader's
  uint64_t written_ = 0;  // writer's
};

} // namespace arm_rt
