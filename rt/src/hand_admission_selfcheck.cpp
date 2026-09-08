#include "hand_admission.hpp"
#include <cassert>
#include <iostream>

int main() {
  HandAdmission gate;
  HandAction action;
  assert(parse_hand_command("CMD 1 GRASP 0.04 0.05 40 0.02 0.02", action));
  for (const char* bad : {"CMD 1 MOVE nan 0.05", "CMD 1 MOVE 0.08 inf",
       "CMD 1 MOVE 0.081 0.05", "CMD 1 MOVE 0.04 0", "CMD 1 MOVE 0.04 0.101",
       "CMD 1 GRASP 0.04 0.05 29 0.02 0.02", "CMD 1 GRASP 0.04 0.05 71 0.02 0.02",
       "CMD 1 GRASP 0.04 0.05 40 -0.02 0.02", "CMD 1 HOME junk", "HOME",
       "CMD -1 HOME", "CMD 1 MOVE 0.04 0.05 junk"}) {
    assert(!parse_hand_command(bad, action));
  }
  assert(parse_hand_command("CMD 1 HOME", action));
  action.token = gate.epoch;
  assert(gate.admit(action, true) != nullptr); // offline
  gate.client = gate.connected = true;
  assert(gate.admit(action, false) != nullptr); // stale sample
  assert(gate.admit(action, true) == nullptr);
  assert(gate.admit(action, true) != nullptr); // repeated old token / busy
  const auto admitted_epoch = gate.epoch;
  gate.cancel(); // client or robot loss kills queued command
  assert(!gate.take(action));
  assert(gate.epoch > admitted_epoch);
  action.token = admitted_epoch;
  assert(gate.admit(action, true) != nullptr); // delayed after reconnect
  action.token = gate.epoch;
  assert(gate.admit(action, true) == nullptr);
  assert(gate.take(action));
  const auto carried = action;
  gate.cancel();
  assert(!gate.current(carried)); // canceled action cannot report into new client
  assert(gate.admit(action, true) != nullptr); // blocking call still busy
  gate.acting = false;
  action.token = gate.epoch;
  assert(gate.admit(action, true) == nullptr);
  assert(gate.take(action) && gate.current(action));
  std::cout << "PASS hand TCP parsing, admission, cancellation and stale generation\n";
}
