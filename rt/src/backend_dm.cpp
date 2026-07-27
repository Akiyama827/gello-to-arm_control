// DM-motor FDCAN backend — SKELETON.
//
// The plan: the RT box owns the FDCAN interface (SocketCAN, not the USB
// dongle), packs the same 8-byte MIT frames the Python dm_backend speaks
// today at 400 Hz, and lets the DM firmware close its own PD — the servo law
// then runs with kp/kd forwarded in the frame rather than torque-only.
// Structure lands with the FR3 loop proven; the pieces to port from
// arm_control/hardware/{dm_backend,gains,socketcan_transport}.py are the MIT
// pack/unpack, the enable/disable framing, and the per-motor-type gain
// encode ceilings (kp<=500, kd<=5).
#include <string>

#include "arm_rt/backend.hpp"

namespace arm_rt {

std::unique_ptr<Backend> make_dm_backend(const std::string& /*can_if*/) {
  return nullptr; // not implemented yet; main.cpp reports it
}

} // namespace arm_rt
