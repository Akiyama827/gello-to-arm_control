// READ-ONLY FCI probe: the robot's own calibrated flange pose, next to q.
//
// Answers exactly one question: does our NOMINAL URDF agree with the
// per-unit kinematics Franka measured on THIS arm at the factory? The
// controller keeps those corrections internally and never publishes them;
// the only way to see them is to ask the robot where it thinks its flange
// is, at a q we can run through our own FK.
//
// COMMANDS NOTHING. There is no control(), no startTorqueControl(), no
// writeOnce() and no setEE/setLoad in this file -- it constructs a Robot,
// calls readOnce(), prints, and exits. It cannot move the arm.
//
// Deliberately NO loadModel(): the flange pose is recoverable from the
// state alone as O_T_F = O_T_EE * inv(F_T_EE), which skips the seconds-long
// model-library download and one more thing to go wrong.
//
// PRECONDITION: the FCI allows a single client, and the client is the
// `franka::Robot` OBJECT, not the torque session. backend_franka.cpp holds
// one for the server's whole lifetime (`franka::Robot robot_;`), so
// DISARMING IS NOT ENOUGH -- the RT server must be STOPPED:
//
//   ssh ok@172.16.1.2 'sudo systemctl stop arm-rt-server.service'
//
// Runs fine from the PC: the default IP is the ROBOT on the FCI wire
// (172.16.0.3), which the PC reaches through the box's NAT, and readOnce is
// one non-realtime request. 172.16.1.2 is the RT BOX -- passing it here
// connects to nothing.
//
//   ./flange_probe [robot_ip] >> samples.jsonl
//
// One JSON object per run: hand-guide the arm, run it again, append. The
// comparator wants poses that differ a lot, not many poses that differ little.
#include <array>
#include <cstdio>
#include <string>

#include <Eigen/Dense>
#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/robot.h>

namespace {

// libfranka packs 4x4 homogeneous transforms COLUMN-major.
Eigen::Matrix4d mat(const std::array<double, 16>& a) {
  return Eigen::Map<const Eigen::Matrix4d>(a.data());
}

void emit(const char* key, const Eigen::Matrix4d& T, bool last) {
  const Eigen::Quaterniond q(T.block<3, 3>(0, 0));
  std::printf("\"%s\":{\"xyz\":[%.9f,%.9f,%.9f],\"wxyz\":[%.9f,%.9f,%.9f,%.9f]}%s",
              key, T(0, 3), T(1, 3), T(2, 3), q.w(), q.x(), q.y(), q.z(),
              last ? "" : ",");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string ip = argc > 1 ? argv[1] : "172.16.0.3";
  try {
    // kIgnore, deliberately: libfranka's constructor otherwise puts the
    // CALLING thread on SCHED_FIFO, which needs an rtprio allowance the
    // operator PC has no reason to grant (`ulimit -r` is 0 there) and which
    // buys this program nothing -- it makes one non-realtime request and
    // exits. Enforcing realtime is the SERVO's business, not the probe's.
    franka::Robot robot(ip, franka::RealtimeConfig::kIgnore);  // connect only
    const franka::RobotState s = robot.readOnce();   // the ONLY robot call
    const Eigen::Matrix4d O_T_EE = mat(s.O_T_EE);
    const Eigen::Matrix4d F_T_EE = mat(s.F_T_EE);
    // Flange pose in the robot base frame, free of any Desk end-effector
    // configuration: this is pure per-unit arm kinematics, which is the
    // only quantity our URDF can be held against.
    const Eigen::Matrix4d O_T_F = O_T_EE * F_T_EE.inverse();

    std::printf("{\"q\":[");
    for (int i = 0; i < 7; ++i) std::printf("%.9f%s", s.q[i], i == 6 ? "" : ",");
    std::printf("],");
    emit("O_T_F", O_T_F, false);
    emit("O_T_EE", O_T_EE, false);
    emit("F_T_EE", F_T_EE, false);
    emit("F_T_NE", mat(s.F_T_NE), true);
    std::printf("}\n");
    return 0;
  } catch (const franka::Exception& e) {
    std::fprintf(stderr,
                 "flange_probe: %s\n"
                 "(the FCI allows ONE client, held by the Robot object -- STOP the RT\n"
                 " server on the box, disarming it is not enough; and check FCI is\n"
                 " active in Desk. This probe needs NO realtime privileges.)\n",
                 e.what());
    return 1;
  }
}
