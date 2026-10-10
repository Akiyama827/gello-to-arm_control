// READ-ONLY FCI probe for the Cartesian law's hardware rungs (2) and (3): the box's FK against the robot's measured pose.
//
// The v8 law aims its spring at ee_pose_at(q_des) (backend_franka.cpp): the vendor model's
// model.pose(kEndEffector, q, F_T_EE, EE_T_K) with the frames of the last measured state. This makes that SAME call at the
// measured q and prints it next to O_T_EE (the robot's own pose at that q) and the difference -- the law's target and its
// measured pose must come from one model, so a nonzero difference is a frame or model mismatch, not noise. It also times
// model.pose (the per-tick cost the law adds inside the 1 kHz loop; rung 3's estimate on whatever CPU runs this).
//
// COMMANDS NOTHING: a Robot, one readOnce(), loadModel() and model.pose() -- no control(), no writeOnce(), no setEE/setLoad.
// PRECONDITION as flange_probe: the FCI allows ONE client, held by the RT server's Robot object for its whole lifetime, so
// the RT server must be STOPPED (disarming is not enough). Runs from the PC (the robot IP through the box's NAT) or the box.
//
//   ./cartesian_probe [robot_ip] [samples]
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

namespace {

Eigen::Matrix4d mat(const std::array<double, 16>& a) {  // libfranka: column-major 4x4
  return Eigen::Map<const Eigen::Matrix4d>(a.data());
}

void emit(const char* key, const Eigen::Matrix4d& T) {
  const Eigen::Quaterniond q(T.block<3, 3>(0, 0));
  std::printf("\"%s\":{\"xyz\":[%.9f,%.9f,%.9f],\"wxyz\":[%.9f,%.9f,%.9f,%.9f]},", key, T(0, 3), T(1, 3), T(2, 3), q.w(),
              q.x(), q.y(), q.z());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string ip = argc > 1 ? argv[1] : "172.16.0.3";
  const int samples = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1000;
  try {
    franka::Robot robot(ip, franka::RealtimeConfig::kIgnore);  // connect only; see flange_probe for kIgnore
    const franka::RobotState s = robot.readOnce();
    franka::Model model = robot.loadModel();
    // ponytail: the same call as backend_franka.cpp's ee_pose_at, restated (one libfranka call) rather than linking the
    // backend into a probe; a change there must be mirrored here.
    const Eigen::Matrix4d fk = mat(model.pose(franka::Frame::kEndEffector, s.q, s.F_T_EE, s.EE_T_K));
    const Eigen::Matrix4d measured = mat(s.O_T_EE);
    const double pos_mm = (fk.block<3, 1>(0, 3) - measured.block<3, 1>(0, 3)).norm() * 1e3;
    const double rot_deg =
        Eigen::AngleAxisd(fk.block<3, 3>(0, 0).transpose() * measured.block<3, 3>(0, 0)).angle() * 180.0 / M_PI;

    std::vector<double> us;
    us.reserve(samples);
    for (int i = 0; i < samples; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto T = model.pose(franka::Frame::kEndEffector, s.q, s.F_T_EE, s.EE_T_K);
      const auto t1 = std::chrono::steady_clock::now();
      if (!std::isfinite(T[12])) {
        std::fprintf(stderr, "cartesian_probe: model.pose returned a non-finite pose\n");
        return 1;
      }
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());

    std::printf("{\"q\":[");
    for (int i = 0; i < 7; ++i) std::printf("%.9f%s", s.q[i], i == 6 ? "" : ",");
    std::printf("],");
    emit("O_T_EE", measured);
    emit("ee_pose_at", fk);
    emit("F_T_EE", mat(s.F_T_EE));
    emit("EE_T_K", mat(s.EE_T_K));
    std::printf("\"pos_err_mm\":%.6f,\"rot_err_deg\":%.6f,", pos_mm, rot_deg);
    std::printf("\"pose_cost_us\":{\"n\":%d,\"p50\":%.2f,\"p99\":%.2f,\"max\":%.2f}}\n", samples, us[us.size() / 2],
                us[std::min(us.size() - 1, us.size() * 99 / 100)], us.back());
    return 0;
  } catch (const franka::Exception& e) {
    std::fprintf(stderr,
                 "cartesian_probe: %s\n"
                 "(the FCI allows ONE client -- STOP the RT server on the box, disarming is not enough;\n"
                 " and check FCI is active in Desk. No realtime privileges needed.)\n",
                 e.what());
    return 1;
  }
}
