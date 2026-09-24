// FR3 backend + the real RT loop against a SCRIPTED franka:: surface: no
// libfranka, no network, no robot. Every device call consults a hook that may
// throw, so libfranka and non-libfranka (Poco-like) failures are injectable.
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

struct Script {
  std::function<void(const char*)> hook;  // may throw
  int writes = 0, finished = 0, recoveries = 0, sessions = 0, reads = 0;
} script;
inline void step(const char* op) { if (script.hook) script.hook(op); }

// What Poco::Net::NetException looks like to a catch site: a std::exception
// that is NOT a franka::Exception (verified against the real headers).
struct NetLike : std::exception {
  const char* what() const noexcept override { return "Net Exception"; }
};

namespace franka {
struct Exception : std::runtime_error { using std::runtime_error::runtime_error; };
enum class RealtimeConfig { kEnforce };
enum class Frame { kEndEffector };
struct Duration {
  uint64_t ms = 0;
  uint64_t toMSec() const { return ms; }
};
struct RobotState {
  Duration time;
  double control_command_success_rate = 1.0;
  std::array<double, 7> q{}, dq{}, tau_J{}, tau_J_d{};
  std::array<double, 6> O_F_ext_hat_K{};
  std::array<double, 16> O_T_EE{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};
struct Torques {
  Torques(const std::array<double, 7>& t) : tau_J(t) {}
  std::array<double, 7> tau_J;
  bool motion_finished = false;
};
struct Model {
  std::array<double, 42> zeroJacobian(Frame, const RobotState&) const { return {}; }
  std::array<double, 7> coriolis(const RobotState&) const { return {}; }
};
struct ActiveControlBase {
  ActiveControlBase() { ++script.sessions; }
  ~ActiveControlBase() { --script.sessions; }
  std::pair<RobotState, Duration> readOnce() {
    ++script.reads;
    step("control.readOnce");
    return {RobotState{}, Duration{1}};
  }
  void writeOnce(const Torques& t) {
    step("control.writeOnce");
    ++script.writes;
    script.finished += t.motion_finished;
  }
};
struct Robot {
  Robot(const std::string&, RealtimeConfig) { step("Robot"); }
  Model loadModel() { step("loadModel"); return {}; }
  void setLoad(double, const std::array<double, 3>&, const std::array<double, 9>&) {}
  void setCollisionBehavior(const std::array<double, 7>&, const std::array<double, 7>&,
                            const std::array<double, 6>&, const std::array<double, 6>&) {}
  RobotState readOnce() {
    ++script.reads;
    step("readOnce");
    return {};
  }
  void automaticErrorRecovery() {
    step("automaticErrorRecovery");
    ++script.recoveries;
  }
  std::unique_ptr<ActiveControlBase> startTorqueControl() {
    step("startTorqueControl");
    return std::make_unique<ActiveControlBase>();
  }
};
}  // namespace franka

#define ARM_RT_WITH_FRANKA
#define ARM_RT_FRANKA_SELFCHECK
#include "backend_franka.cpp"
#include "server.hpp"

namespace arm_rt {
uint64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000ull + uint64_t(ts.tv_nsec);
}
std::unique_ptr<Backend> make_fake_backend(int) { std::abort(); }
std::unique_ptr<Backend> make_dm_backend(const std::string&, uint32_t, double) {
  std::abort();
}
}  // namespace arm_rt

namespace {
using namespace arm_rt;
const double com[3] = {0, 0, 0};

template <class E>
std::function<void(const char*)> throw_at(const char* where, E exc) {
  return [where, exc](const char* op) {
    if (!std::strcmp(op, where)) throw exc;
  };
}

void check_backend() {
  script = {};
  script.hook = throw_at("Robot", NetLike());
  assert(!make_franka_backend("sim", 0, com));  // session open: no terminate
  script.hook = throw_at("loadModel", franka::Exception("refused"));
  assert(!make_franka_backend("sim", 0, com));
  script.hook = nullptr;
  auto b = make_franka_backend("sim", 0, com);
  assert(b);
  PlantState ps;
  double tau[7] = {};
  assert(b->read(ps));

  const std::string net = "non-libfranka exception: Net Exception";
  script.hook = throw_at("readOnce", NetLike());  // idle stream
  assert(!b->read(ps) && b->fault_text() == "read: " + net);

  script.hook = nullptr;  // ARM: recovery first (the failure set it), then open
  assert(b->write(tau, 7) && script.sessions == 1 && script.recoveries == 1);

  script.hook = throw_at("control.readOnce", NetLike());  // armed stream
  assert(!b->read(ps) && b->fault_text() == "read: " + net);
  assert(script.sessions == 0);  // session released; plant state unknown

  script.hook = nullptr;
  assert(b->write(tau, 7) && script.sessions == 1 && script.recoveries == 2);
  script.hook = throw_at("control.writeOnce", std::runtime_error("socket closed"));
  assert(!b->write(tau, 7) && script.sessions == 0);
  assert(b->fault_text() == "write: non-libfranka exception: socket closed");

  script.hook = throw_at("automaticErrorRecovery", NetLike());
  assert(!b->write(tau, 7) && script.sessions == 0 && b->fault_text() == "write: " + net);

  // libfranka's own exceptions keep their exact text (unchanged behavior).
  script.hook = nullptr;
  assert(b->write(tau, 7) && script.sessions == 1);
  script.hook = throw_at("control.readOnce",
                         franka::Exception("communication_constraints_violation"));
  assert(!b->read(ps) && b->fault_text() == "communication_constraints_violation");

  script.hook = nullptr;
  assert(b->write(tau, 7) && script.sessions == 1);
  const int finished = script.finished;
  script.hook = throw_at("control.writeOnce", NetLike());
  assert(!b->stop());  // controlled-stop write throws: contained, reported
  assert(script.sessions == 0 && script.finished == finished);
  assert(b->fault_text() == "stop: " + net);
  script.hook = nullptr;
  const int writes = script.writes;
  assert(b->stop());  // idle: no device call, nothing failed
  assert(script.writes == writes);
}

// The real rt_loop: an armed FR3 read throwing a non-libfranka exception
// latches FAULT_PLANT with the attributed cause and suppresses every torque
// write until the next ARM, instead of terminating the process.
ServerCtx* ctx;
int idle_reads, writes_at_fault = -1;

void check_rt_loop() {
  ServerCtx context;
  ctx = &context;
  context.cfg.backend = "franka";
  context.cfg.instance_id = "selfcheck";
  for (int j = 0; j < 7; ++j) context.cfg.joint_names.push_back("j" + std::to_string(j));
  context.cfg.rt_priority = 0;
  script = {};
  idle_reads = 0;
  script.hook = [](const char* op) {
    if (!std::strcmp(op, "readOnce")) {
      ++idle_reads;
      if (idle_reads == 1) {  // first sample: ARM
        ctx->last_cmd_rx_ns.store(mono_ns());
        ctx->arm_gen.fetch_add(1);
        ctx->armed.store(true);
      }
      if (writes_at_fault >= 0 && idle_reads >= 4) ctx->shutdown.store(true);
    }
    if (!std::strcmp(op, "control.readOnce") && script.reads >= 5) {
      writes_at_fault = script.writes;
      throw NetLike();
    }
  };
  rt_loop(context);
  assert(!context.failed);
  assert(writes_at_fault >= 3);                // it was servoing before
  assert(script.writes == writes_at_fault);    // and wrote nothing after
  assert(script.sessions == 0 && context.fault && context.armed);
  assert(context.fault_code == FAULT_PLANT);
  assert(std::strcmp(context.fault_text, "read: non-libfranka exception: Net Exception") == 0);
  const FaultRecord& first = context.fault_ring[0].rec;  // journaled, not only on the wire
  assert(context.fault_head == 1 && first.kind == FAULT_FIRST && first.armed);
  assert(first.code == FAULT_PLANT && first.arm_epoch == 1);
  assert(std::strcmp(first.text, context.fault_text) == 0);
}

// DISARM whose controlled stop fails: a separate stop-failed record, no latch.
void check_stop_failure_journal() {
  ServerCtx context;
  ctx = &context;
  context.cfg.backend = "franka";
  context.cfg.instance_id = "selfcheck";
  for (int j = 0; j < 7; ++j) context.cfg.joint_names.push_back("j" + std::to_string(j));
  context.cfg.rt_priority = 0;
  script = {};
  idle_reads = 0;
  script.hook = [](const char* op) {
    if (!std::strcmp(op, "readOnce") && ++idle_reads == 1) {
      ctx->last_cmd_rx_ns.store(mono_ns());
      ctx->arm_gen.fetch_add(1);
      ctx->armed.store(true);
    }
    if (!std::strcmp(op, "readOnce") && idle_reads >= 3) ctx->shutdown.store(true);
    if (!std::strcmp(op, "control.readOnce") && script.reads == 4)
      ctx->armed.store(false);  // DISARM sampled on this tick
    if (!std::strcmp(op, "control.writeOnce") && !ctx->armed.load()) throw NetLike();
  };
  rt_loop(context);
  assert(!context.failed && !context.fault && script.sessions == 0);
  assert(context.fault_head == 1);
  const FaultRecord& r = context.fault_ring[0].rec;
  assert(r.kind == FAULT_STOP_FAILED && r.code == FAULT_PLANT && !r.armed);
  assert(std::strcmp(r.text, "stop: non-libfranka exception: Net Exception") == 0);
}
}  // namespace

int main() {
  check_backend();
  check_rt_loop();
  check_stop_failure_journal();
  std::puts("PASS FR3 backend: non-libfranka exceptions in open/read/write/stop contained, "
            "cause attributed and journaled, armed read fault latches with writes "
            "suppressed, failed stop journaled separately");
}
