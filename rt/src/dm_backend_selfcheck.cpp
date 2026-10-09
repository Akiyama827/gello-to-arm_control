// Actual DM backend, with SocketCAN syscalls intercepted at link time.
// No sockets, interfaces, devices, privileges or kernel modules are needed.
#include <linux/can.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "arm_rt/backend.hpp"

namespace arm_rt {
std::unique_ptr<Backend> make_dm_backend(const std::string&, uint32_t, double);
}
namespace {
constexpr int fake_fd = 12345;
constexpr int probe_fd = 12346;  // the backend's second, unfiltered socket (the shared-bus checks)
int sockets = 0;                 // per backend: the first is the motor socket, the second the probe
std::vector<canfd_frame> sent, probe_queue;
bool is_fd(int fd) { return fd == fake_fd || fd == probe_fd; }
canfd_frame frame_on(uint32_t can_id, uint8_t first_byte) {
  canfd_frame f{};
  f.can_id = can_id;
  f.len = 8;
  f.data[0] = first_byte;
  return f;
}
canfd_frame feedback[2];
int pending;
bool fail_read, fail_second_command, fail_second_enable;

bool special(const canfd_frame& frame, uint8_t suffix) {
  for (int i = 0; i < 7; ++i) if (frame.data[i] != 0xff) return false;
  return frame.data[7] == suffix;
}
void queue_feedback() {
  pending = 2;
  for (int j = 0; j < 2; ++j) {
    feedback[j] = {};
    feedback[j].can_id = 0x11 + j;
    feedback[j].len = 8;
    const uint8_t bytes[] = {uint8_t(j + 1), 0x80, 0, 0x80, 8, 0, 25, 25};
    std::memcpy(feedback[j].data, bytes, 8);
  }
}
void assert_disabled() {
  bool first = false, second = false;
  for (const auto& frame : sent) {
    first |= frame.can_id == 1 && special(frame, 0xfd);
    second |= frame.can_id == 2 && special(frame, 0xfd);
  }
  assert(first && second);
}
}  // namespace

extern "C" {
int __wrap_socket(int domain, int, int) {
  assert(domain == PF_CAN);
  return sockets++ % 2 ? probe_fd : fake_fd;
}
int __wrap_setsockopt(int fd, int, int, const void*, socklen_t) {
  assert(is_fd(fd)); return 0;
}
int __wrap_ioctl(int fd, unsigned long request, ...) {
  assert(fd == fake_fd && request == SIOCGIFINDEX);
  va_list args;
  va_start(args, request);
  va_arg(args, ifreq*)->ifr_ifindex = 1;
  va_end(args);
  return 0;
}
int __wrap_bind(int fd, const sockaddr*, socklen_t) { assert(is_fd(fd)); return 0; }
int __wrap_close(int fd) { assert(is_fd(fd)); return 0; }
ssize_t __wrap_write(int fd, const void* data, size_t size) {
  assert(fd == fake_fd && size == CANFD_MTU);
  const auto& frame = *static_cast<const canfd_frame*>(data);
  assert(frame.len == 8 && frame.flags == (CANFD_FDF | CANFD_BRS));
  sent.push_back(frame);
  if (fail_second_enable && frame.can_id == 2 && special(frame, 0xfc)) {
    errno = ENETDOWN;
    return -1;
  }
  const unsigned kp_code = (unsigned(frame.data[3] & 15) << 8) | frame.data[4];
  if (fail_second_command && frame.can_id == 2 && kp_code &&
      !special(frame, 0xfc) && !special(frame, 0xfd)) {
    errno = ENETDOWN;
    return -1;
  }
  return ssize_t(size);
}
ssize_t __wrap_recv(int fd, void* out, size_t size, int flags) {
  assert(is_fd(fd) && size >= CANFD_MTU && flags == MSG_DONTWAIT);
  if (fd == probe_fd) {  // an empty bus unless a case puts frames on it
    if (probe_queue.empty()) { errno = EAGAIN; return -1; }
    std::memcpy(out, &probe_queue.back(), CANFD_MTU);
    probe_queue.pop_back();
    return CANFD_MTU;
  }
  if (fail_read) { errno = ENETDOWN; return -1; }
  if (!pending) { errno = EAGAIN; return -1; }
  std::memcpy(out, &feedback[--pending], CANFD_MTU);
  return CANFD_MTU;
}
int __wrap_ppoll(pollfd*, nfds_t, const timespec* timeout, const sigset_t*) {
  nanosleep(timeout, nullptr);
  return 0;
}
}  // extern C

int main() {
  using namespace arm_rt;
  auto backend = make_dm_backend(
      "mit_selfcheck;1:4340:0x11:12.5:20:28,2:4340p:0x12:12.5:20:28", 1, 27);
  assert(backend);
  queue_feedback();
  PlantState state;
  assert(backend->read(state));
  assert(backend->online_mask() == 3 && sent.size() == 2);
  for (const auto& frame : sent) assert(!special(frame, 0xfc));  // reads never ARM
  CommandPacket command{};
  command.n = 2;
  for (int j = 0; j < 2; ++j) {
    command.q_des[j] = .02;
    command.qd_des[j] = .1;
    command.kp[j] = 50;
    command.kd[j] = 2;
    command.tau_ff[j] = .5;
  }
  double torque[MAX_JOINTS]{};
  sent.clear();
  assert(backend->write_command(command, 100, torque));
  assert(sent.size() == 2 && special(sent[0], 0xfc));
  assert(sent[1].can_id == 1 && !special(sent[1], 0xfc));
  assert((sent[1].data[4] | (sent[1].data[3] & 15)) != 0);  // native stiffness
  assert(torque[0] > 1.5 && torque[0] < 2 && torque[1] == 0);

  sent.clear();
  assert(backend->set_active_mask(3));
  assert(sent.empty());  // activation alone must not enable the new motor
  command.kp[1] = 501;
  assert(!backend->write_command(command, 100, torque));
  for (const auto& frame : sent) assert(!special(frame, 0xfc));
  assert_disabled();
  command.kp[1] = 50;
  sent.clear();
  fail_second_enable = true;
  assert(!backend->write_command(command, 100, torque));
  assert_disabled();  // first enable succeeded, second failed
  fail_second_enable = false;
  sent.clear();
  assert(backend->write_command(command, 100, torque));
  assert(sent.size() == 4 && special(sent[0], 0xfc) && special(sent[1], 0xfc));
  sent.clear();
  assert(backend->write_command(command, 100, torque));
  assert(sent.size() == 2);  // repeated commands do not re-enable

  sent.clear();
  fail_second_command = true;
  assert(!backend->write_command(command, 100, torque));
  assert_disabled();  // partial batch failure removes prior native authority
  fail_second_command = false;
  assert(backend->write_command(command, 100, torque));
  sent.clear();
  fail_read = true;
  assert(!backend->read(state));
  assert_disabled();
  fail_read = false;
  sent.clear();
  command.n = 1;
  assert(!backend->write_command(command, 100, torque) && sent.empty());
  backend.reset();

  // Shared bus (operator 2026-10-09: one can0 for the base and this arm, ids colliding, one robot powered at a time).
  const char* spec = "mit_selfcheck;1:4340:0x11:12.5:20:28,2:4340p:0x12:12.5:20:28";
  // Another driver commands one of our ids before this server sent anything: no backend, and not one frame sent.
  sent.clear();
  sockets = 0;
  probe_queue = {frame_on(2, 0)};
  assert(!make_dm_backend(spec, 3, 27) && sent.empty());
  // Our own frames looped back to the probe after we started, and foreign ids, are fine; one of OUR ids answering on a
  // Master ID that is not ours (the other robot's motor, powered) latches a fault before anything is enabled.
  sockets = 0;
  probe_queue.clear();
  auto shared = make_dm_backend(spec, 3, 27);
  assert(shared);
  queue_feedback();
  assert(shared->read(state));
  probe_queue = {frame_on(1, 0), frame_on(0x11, 0x11), frame_on(0x07, 0x07), frame_on(0x00, 0x17)};
  queue_feedback();
  assert(shared->read(state));  // our echo, our own reply, another device's id 7, an id 7 reply on Master 0: fine
  probe_queue = {frame_on(0x00, 0x11)};  // motor id 1 answering on Master 0x00: the DM arm's motor 1, powered
  queue_feedback();
  assert(!shared->read(state));
  assert(std::string(shared->fault_text()).find("another robot is powered") != std::string::npos);
  for (const auto& frame : sent) assert(!special(frame, 0xfc));
  std::puts("PASS DM backend: native frames, batch admission, activation, write/read failure disable, shared-bus "
            "interlock (another driver before start, another robot's motor powered)");
}
