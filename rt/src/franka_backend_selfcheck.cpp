// FR3 backend + the real RT loop against a SCRIPTED franka:: surface: no
// libfranka, no network, no robot. Every device call consults a hook that may
// throw, so libfranka and non-libfranka (Poco-like) failures are injectable.
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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
  std::array<double, 16> F_T_EE{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::array<double, 16> EE_T_K{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};
struct Torques {
  Torques(const std::array<double, 7>& t) : tau_J(t) {}
  std::array<double, 7> tau_J;
  bool motion_finished = false;
};
struct Model {
  std::array<double, 42> zeroJacobian(Frame, const RobotState&) const { return {}; }
  std::array<double, 7> coriolis(const RobotState&) const { return {}; }
  // FK at q (column-major 4x4): a 90 deg turn about z and the translation (q0, q1, q2), so the backend's
  // column-major -> xyz/wxyz mapping is checkable.
  std::array<double, 16> pose(Frame, const std::array<double, 7>& q, const std::array<double, 16>&,
                              const std::array<double, 16>&) const {
    return {0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, q[0], q[1], q[2], 1};
  }
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
    step(t.motion_finished ? "control.stop" : "control.writeOnce");
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

namespace arm_rt {  // mono_ns, state_tx_thread, finish_fault_journal: udp_link.cpp
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
  // v8 target: FK at a COMMANDED q needs the flange/EE frames of a measured state first.
  const double q_des[7] = {.1, .2, .3, 0, 0, 0, 0};
  double pose[7] = {};
  assert(b->supports_cartesian() && !b->ee_pose_at(q_des, pose));
  assert(b->read(ps));
  assert(b->ee_pose_at(q_des, pose));
  const double s = std::sqrt(.5);  // 90 deg about z: w = cos 45, z = sin 45
  const double want[7] = {.1, .2, .3, s, 0, 0, s};
  for (int i = 0; i < 7; ++i) assert(std::fabs(pose[i] - want[i]) < 1e-12);

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
  script.hook = throw_at("control.stop", NetLike());
  assert(!b->stop());  // controlled-stop write throws: contained, reported
  assert(script.sessions == 0 && script.finished == finished);
  assert(b->fault_text() == "stop: " + net);
  script.hook = nullptr;
  const int writes = script.writes;
  assert(b->stop());  // idle: no device call, nothing failed
  assert(script.writes == writes);
}

// A kernel that grants the process-local RT setup and reports back what it
// was given: these harnesses run the real rt_loop under backend "franka",
// whose ARM is gated on that setup.
struct {
  int policy = SCHED_OTHER, prio = 0;
  cpu_set_t cpus;
} granted;
int grant_mlock(int) { return 0; }
int grant_set_sched(pthread_t, int policy, const sched_param* p) {
  granted.policy = policy;
  granted.prio = p->sched_priority;
  return 0;
}
int grant_get_sched(pthread_t, int* policy, sched_param* p) {
  *policy = granted.policy;
  p->sched_priority = granted.prio;
  return 0;
}
int grant_set_aff(pthread_t, size_t, const cpu_set_t* set) {
  granted.cpus = *set;
  return 0;
}
int grant_get_aff(pthread_t, size_t, cpu_set_t* set) {
  *set = granted.cpus;
  return 0;
}
int refuse_mlock(int) { return -1; }
RtSetupOps granting_kernel() {
  RtSetupOps ops;
  ops.lock_memory = grant_mlock;
  ops.set_sched = grant_set_sched;
  ops.get_sched = grant_get_sched;
  ops.set_affinity = grant_set_aff;
  ops.get_affinity = grant_get_aff;
  return ops;
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
  context.rt_setup_ops = granting_kernel();
  script = {};
  idle_reads = 0;
  script.hook = [](const char* op) {
    if (!std::strcmp(op, "readOnce")) {
      ++idle_reads;
      if (idle_reads == 1) {  // first sample: ARM
        begin_arm(*ctx);  // CTL_ARM's transition
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

// The real rt_loop under a kernel that refuses mlockall: the franka backend is
// up and keeps reading (observable), ARM is refused, nothing is written.
void check_rt_setup_refusal() {
  ServerCtx context;
  ctx = &context;
  context.cfg.backend = "franka";
  context.cfg.instance_id = "selfcheck";
  for (int j = 0; j < 7; ++j) context.cfg.joint_names.push_back("j" + std::to_string(j));
  context.cfg.rt_priority = 0;
  context.rt_setup_ops = granting_kernel();
  context.rt_setup_ops.lock_memory = refuse_mlock;
  script = {};
  idle_reads = 0;
  script.hook = [](const char* op) {
    if (!std::strcmp(op, "readOnce")) {
      ++idle_reads;
      if (idle_reads <= 3) assert(begin_arm(*ctx) == 0);  // refused, every time
      if (idle_reads >= 6) ctx->shutdown.store(true);
    }
  };
  rt_loop(context);
  assert(!context.failed && context.backend_ready && context.rt_setup_blocks_arm());
  assert(context.rt_setup_failed == RT_SETUP_MLOCK);
  assert(idle_reads >= 6 && script.writes == 0 && script.sessions == 0);
  assert(context.arm_gen == 0 && !context.armed && !context.fault);
}

// DISARM whose controlled stop fails: a separate stop-failed record, no latch.
void check_stop_failure_journal() {
  ServerCtx context;
  ctx = &context;
  context.cfg.backend = "franka";
  context.cfg.instance_id = "selfcheck";
  for (int j = 0; j < 7; ++j) context.cfg.joint_names.push_back("j" + std::to_string(j));
  context.cfg.rt_priority = 0;
  context.rt_setup_ops = granting_kernel();
  script = {};
  idle_reads = 0;
  script.hook = [](const char* op) {
    if (!std::strcmp(op, "readOnce") && ++idle_reads == 1) {
      begin_arm(*ctx);  // CTL_ARM's transition
    }
    if (!std::strcmp(op, "readOnce") && idle_reads >= 3) ctx->shutdown.store(true);
    if (!std::strcmp(op, "control.readOnce") && script.reads == 4)
      ctx->armed.store(false);  // DISARM sampled on this tick
    if (!std::strcmp(op, "control.stop")) throw NetLike();
  };
  rt_loop(context);
  assert(!context.failed && !context.fault && script.sessions == 0);
  assert(context.fault_head == 1);
  const FaultRecord& r = context.fault_ring[0].rec;
  assert(r.kind == FAULT_STOP_FAILED && r.code == FAULT_PLANT && !r.armed);
  assert(std::strcmp(r.text, "stop: non-libfranka exception: Net Exception") == 0);
}
}  // namespace

// Orderly shutdown while ARMED whose final controlled stop fails: that record
// is enqueued after rt_loop's loop exits. It must reach the sink exactly once,
// whether the running state_tx drains it or finish_fault_journal does after
// every thread is joined -- and it must reach it with no consumer running.
std::string shutdown_export(bool consumer_running) {
  ServerCtx context;
  ctx = &context;
  context.cfg.backend = "franka";
  context.cfg.instance_id = "selfcheck";
  for (int j = 0; j < 7; ++j) context.cfg.joint_names.push_back("j" + std::to_string(j));
  context.cfg.rt_priority = 0;
  context.rt_setup_ops = granting_kernel();
  char* buf = nullptr;
  size_t len = 0;
  context.journal_out = open_memstream(&buf, &len);
  script = {};
  idle_reads = 0;
  script.hook = [](const char* op) {
    if (!std::strcmp(op, "readOnce") && ++idle_reads == 1) {
      begin_arm(*ctx);  // CTL_ARM's transition
    }
    if (!std::strcmp(op, "control.readOnce") && script.reads == 6)
      ctx->shutdown.store(true);  // still armed: the exit path stops the session
    if (!std::strcmp(op, "control.stop")) throw NetLike();
  };
  std::thread tx;
  if (consumer_running) tx = std::thread(state_tx_thread, std::ref(context));
  rt_loop(context);
  if (tx.joinable()) tx.join();
  assert(context.fault_head == 1);  // enqueued after the loop, before this drain
  finish_fault_journal(context);    // as main does after its joins
  std::fclose(context.journal_out);
  std::string out(buf, len);
  free(buf);
  return out;
}

void check_shutdown_export() {
  const std::string line =
      "stop failed code=3 t=";
  const std::string text = "text=\"stop: non-libfranka exception: Net Exception\"\n";
  for (bool running : {false, true}) {
    const std::string out = shutdown_export(running);
    size_t count = 0;
    for (size_t at = out.find(line); at != std::string::npos; at = out.find(line, at + 1))
      ++count;
    assert(count == 1);  // exactly once
    assert(out.find(text) != std::string::npos);
    assert(out.find("lost") == std::string::npos);
  }
}

int main() {
  check_backend();
  check_rt_loop();
  check_rt_setup_refusal();
  check_stop_failure_journal();
  check_shutdown_export();
  std::puts("PASS FR3 backend: non-libfranka exceptions in open/read/write/stop contained, "
            "cause attributed and journaled, armed read fault latches with writes "
            "suppressed, failed stop journaled separately and exported once at shutdown, "
            "ARM refused under a failed process RT setup while reads continue");
}
