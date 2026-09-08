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

inline bool parse_hand_command(const std::string& line, HandAction& a) {
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
  if (!std::isfinite(a.w) || a.w < 0 || a.w > .08 ||
      !std::isfinite(a.s) || a.s <= 0 || a.s > .10) return false;
  // Hand manual 1.2 section 5.1: continuous adjustable force 30..70 N.
  return a.kind != 3 || (std::isfinite(a.f) && a.f >= 30 && a.f <= 70 &&
      std::isfinite(a.ei) && a.ei >= 0 && a.ei <= .08 &&
      std::isfinite(a.eo) && a.eo >= 0 && a.eo <= .08);
}

// All methods are called under the bridge mutex. One command, never a queue.
struct HandAdmission {
  HandAction pending;
  uint64_t epoch = 1;
  bool connected = false, client = false, acting = false;
  void cancel() { pending = {}; ++epoch; }
  const char* admit(HandAction action, bool fresh) {
    if (!connected || !client) return "offline";
    if (action.token != epoch) return "stale_epoch";
    if (acting || pending.kind) return "busy";
    if (!fresh) return "stale_state";
    action.generation = ++epoch;
    pending = action;
    return nullptr;
  }
  bool take(HandAction& action) {
    if (!connected || !client || !pending.kind || acting) return false;
    action = pending;
    pending = {};
    if (!current(action)) return false;
    acting = true;
    return true;
  }
  bool current(const HandAction& action) const {
    return connected && client && action.generation == epoch;
  }
};
