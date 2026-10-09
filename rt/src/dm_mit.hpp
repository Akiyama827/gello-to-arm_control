#pragma once
// DM-motor MIT codec of the base's backend (backend_dm.cpp). Bit-for-bit with
// the Python arm_control/plants/dm/backend.py.
#include <cstdint>
#include <string>

namespace arm_rt {

inline constexpr uint8_t DM_ENABLE_FRAME[8] = {0xFF, 0xFF, 0xFF, 0xFF,
                                        0xFF, 0xFF, 0xFF, 0xFC};
inline constexpr uint8_t DM_DISABLE_FRAME[8] = {0xFF, 0xFF, 0xFF, 0xFF,
                                         0xFF, 0xFF, 0xFF, 0xFD};

// Default per-type MIT limits, matching dm_backend.py DEFAULT_LIMITS_BY_TYPE.
// A per-motor spec override changes the host's encoding and feedback decoding.
struct DmLimits {
  double p_lo, p_hi, v_lo, v_hi, kp_lo, kp_hi, kd_lo, kd_hi, t_lo, t_hi;
};
inline constexpr DmLimits LIM_4310 = {-12.5, 12.5, -30, 30, 0, 500, 0, 5, -10, 10};
inline constexpr DmLimits LIM_4310P = {-12.5, 12.5, -50, 50, 0, 500, 0, 5, -10, 10};
inline constexpr DmLimits LIM_4340 = {-12.5, 12.5, -10, 10, 0, 500, 0, 5, -28, 28};
// 4340P: same electrical specs as 4340 (cross-roller bearing variant)

inline bool limits_for(const std::string& type, DmLimits& out) {
  if (type == "4310") out = LIM_4310;
  else if (type == "4310p") out = LIM_4310P;
  else if (type == "4340" || type == "4340p") out = LIM_4340;
  else return false;
  return true;
}

// float<->uint scaling, bit-for-bit with the Python _float_to_uint /
// _uint_to_float (same operand order, same truncation toward zero).
inline uint32_t f2u(double v, double lo, double hi, int bits) {
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return uint32_t((v - lo) * double((1u << bits) - 1) / (hi - lo));
}
inline double u2f(uint32_t raw, double lo, double hi, int bits) {
  return double(raw) * (hi - lo) / double((1u << bits) - 1) + lo;
}

// MIT control frame: pos16 | vel12 | kp12 | kd12 | tau12 (dm_backend.py
// pack_mit_control_frame). Authorized commands are validated by pack_native_mit
// BEFORE reaching this raw codec. Zero-gain lifecycle frames use it directly.
inline void pack_mit(double pos, double vel, double kp, double kd, double tau,
              const DmLimits& L, uint8_t out[8]) {
  const uint32_t p = f2u(pos, L.p_lo, L.p_hi, 16);
  const uint32_t v = f2u(vel, L.v_lo, L.v_hi, 12);
  const uint32_t s = f2u(kp, L.kp_lo, L.kp_hi, 12);
  const uint32_t d = f2u(kd, L.kd_lo, L.kd_hi, 12);
  const uint32_t t = f2u(tau, L.t_lo, L.t_hi, 12);
  out[0] = uint8_t((p >> 8) & 0xFF);
  out[1] = uint8_t(p & 0xFF);
  out[2] = uint8_t((v >> 4) & 0xFF);
  out[3] = uint8_t(((v & 0x0F) << 4) | ((s >> 8) & 0x0F));
  out[4] = uint8_t(s & 0xFF);
  out[5] = uint8_t((d >> 4) & 0xFF);
  out[6] = uint8_t(((d & 0x0F) << 4) | ((t >> 8) & 0x0F));
  out[7] = uint8_t(t & 0xFF);
}

// MIT reply: [err<<4|id, pos16, vel12, tau12, T_mos, T_rotor]
// (dm_backend.py decode_mit_reply).
struct MitReply {
  int id, err;
  double q, dq, tau, t_mos, t_rotor;
};
inline bool decode_mit(const uint8_t* d, int len, const DmLimits& L, MitReply& r) {
  if (len < 8) return false;
  r.id = d[0] & 0x0F;
  r.err = (d[0] >> 4) & 0x0F;
  const uint32_t pos_u = uint32_t(d[1] << 8) | d[2];
  const uint32_t vel_u = uint32_t(d[3] << 4) | uint32_t(d[4] >> 4);
  const uint32_t tau_u = (uint32_t(d[4] & 0x0F) << 8) | d[5];
  r.q = u2f(pos_u, L.p_lo, L.p_hi, 16);
  r.dq = u2f(vel_u, L.v_lo, L.v_hi, 12);
  r.tau = u2f(tau_u, L.t_lo, L.t_hi, 12);
  r.t_mos = double(d[6]);   // surfaced by the Python motor_health(); no
  r.t_rotor = double(d[7]); // PlantState slot — dropped here for now
  return true;
}

}  // namespace arm_rt
