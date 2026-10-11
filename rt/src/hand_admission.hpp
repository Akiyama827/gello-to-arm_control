#pragma once

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

struct HandAction {
  int kind = 0; // MOVE=1, HOME=2, GRASP=3
  double w = 0, s = 0, f = 0, ei = 0, eo = 0;
  uint64_t token = 0, generation = 0;
};

// A gripper's command envelope, stated to clients on the LIMITS line (a bridge
// that states none is the Franka Hand). force_max_n == 0: no force grasp.
struct HandLimits {
  double width_max_m, speed_max_mps, force_min_n, force_max_n;
};
// Franka Hand product manual 1.2 section 5.1: 80 mm travel, 0.10 m/s total
// opening speed, continuous adjustable force 30..70 N.
constexpr HandLimits franka_hand{.08, .10, 30, 70};

inline bool parse_hand_command(const std::string& line, HandAction& a,
                               const HandLimits& l = franka_hand) {
  a = {};
  std::istringstream in(line);
  std::string prefix, token, verb, extra;
  if (!(in >> prefix >> token >> verb) || prefix != "CMD" || token.empty() ||
      token.find_first_not_of("0123456789") != std::string::npos) return false;
  try { a.token = std::stoull(token); } catch (...) { return false; }
  if (verb == "HOME") a.kind = 2;
  else if (verb == "MOVE") {
    a.kind = 1;
    if (!(in >> a.w >> a.s)) return false;
  } else if (verb == "GRASP") {
    a.kind = 3;
    if (!(in >> a.w >> a.s >> a.f >> a.ei >> a.eo)) return false;
  } else return false;
  if (in >> extra) return false;
  if (a.kind == 2) return true;
  if (!std::isfinite(a.w) || a.w < 0 || a.w > l.width_max_m ||
      !std::isfinite(a.s) || a.s <= 0 || a.s > l.speed_max_mps) return false;
  return a.kind != 3 || (std::isfinite(a.f) && a.f >= l.force_min_n &&
      a.f <= l.force_max_n && l.force_max_n > 0 &&
      std::isfinite(a.ei) && a.ei >= 0 && a.ei <= l.width_max_m &&
      std::isfinite(a.eo) && a.eo >= 0 && a.eo <= l.width_max_m);
}

// All methods are called under the bridge mutex. One command, never a queue.
struct HandAdmission {
  HandAction pending;
  uint64_t epoch = 1;
  bool connected = false, client = false, acting = false;
  bool dispatch_committed = false;
  bool stopping = false, stop_fault = false;
  // A failed stop is cleared by a successful retried stop AND a measured
  // sample read after it -- never by a connection or a client existing.
  bool stop_retried_ok = false;
  uint64_t stop_serial = 0;
  void cancel() { pending = {}; ++epoch; }
  void request_stop() {
    cancel();
    stopping = true;
    stop_retried_ok = false;
    ++stop_serial;
  }
  void finish_stop(uint64_t serial, bool was_acting, bool ok) {
    // A stop sent before the SDK action starts can return before that action
    // enters the device. Keep stopping while acting, then send one final stop
    // after action exit. New actions are barred throughout this interval.
    if (serial != stop_serial || was_acting || acting) return;
    stopping = false;
    if (!ok) stop_fault = true;
    stop_retried_ok = ok && stop_fault;
  }
  // A new device session. A stop that failed on the old one is retried here,
  // once per session (a refused retry waits for the next session or GSTOP).
  void session_opened() {
    connected = true;
    if (stop_fault) request_stop();
  }
  // A measured sample whose read began after the final stop returned.
  void sampled() {
    if (stop_retried_ok && !stopping) stop_fault = stop_retried_ok = false;
  }
  const char* admit(HandAction action, bool fresh) {
    if (!connected || !client) return "offline";
    if (stopping) return "stopping";
    if (stop_fault) return "stop_failed";
    if (action.token != epoch) return "stale_epoch";
    if (acting || pending.kind) return "busy";
    if (!fresh) return "stale_state";
    action.generation = ++epoch;
    pending = action;
    return nullptr;
  }
  bool take(HandAction& action) {
    if (!connected || !client || !pending.kind || acting || stopping || stop_fault) return false;
    action = pending;
    pending = {};
    if (!current(action)) return false;
    acting = true;
    dispatch_committed = false;
    return true;
  }
  bool begin_action(const HandAction& action) {
    if (!acting || !current(action)) return false;
    dispatch_committed = true;
    return true;
  }
  bool current(const HandAction& action) const {
    return connected && client && action.generation == epoch;
  }
};
