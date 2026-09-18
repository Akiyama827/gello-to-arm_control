// pybind11 over the ONE servo law (servo_law.hpp) so the MuJoCo plant can
// close the exact compiled code the RT thread runs. Built by rt/bindings/
// (`pip install -e rt/bindings`) — optional: without it the sim falls back
// to its Python PD and must WARN, never silently.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "arm_rt/servo_law.hpp"
#include "arm_rt/pose_hold.hpp"

namespace py = pybind11;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

Arr servo_torque_py(Arr q, Arr dq, Arr q_des, Arr qd_des, Arr tau_ff, Arr kp,
                    Arr kd, Arr tau_ref, Arr tau_limit, double slew_per_tick) {
  const int n = int(q.size());
  for (const Arr* a : {&dq, &q_des, &qd_des, &tau_ff, &kp, &kd, &tau_ref, &tau_limit}) {
    if (int(a->size()) != n) throw std::invalid_argument("all arrays must share length");
  }
  Arr out(n);
  arm_rt::servo_torque(n, q.data(), dq.data(), q_des.data(), qd_des.data(),
                       tau_ff.data(), kp.data(), kd.data(), tau_ref.data(),
                       tau_limit.data(), slew_per_tick, out.mutable_data());
  return out;
}

void want(const Arr& a, int n, const char* what) {
  if (int(a.size()) != n) throw std::invalid_argument(std::string(what) + ": wrong length");
}

// Returns tau_ff PLUS the Cartesian impedance contribution — the caller feeds
// the result straight back into servo_torque's tau_ff, so the clamp and slew
// still govern. Not in-place: forcecast may hand us a copy of the input.
Arr cartesian_torque_py(Arr tau_ff, Arr J, Arr R_task, Arr x, Arr quat, Arr x_des,
                        Arr quat_des, Arr twist, Arr twist_des, Arr kc, Arr dc) {
  const int n = int(tau_ff.size());
  want(J, 6 * n, "J (6 x n row-major)");
  want(R_task, 9, "R_task (3x3 row-major)");
  want(x, 3, "x");
  want(x_des, 3, "x_des");
  want(quat, 4, "quat [w,x,y,z]");
  want(quat_des, 4, "quat_des [w,x,y,z]");
  want(twist, 6, "twist");
  want(twist_des, 6, "twist_des");
  want(kc, 6, "kc");
  want(dc, 6, "dc");
  Arr out(n);
  std::memcpy(out.mutable_data(), tau_ff.data(), sizeof(double) * size_t(n));
  arm_rt::cartesian_impedance(n, J.data(), R_task.data(), x.data(), quat.data(),
                              x_des.data(), quat_des.data(), twist.data(),
                              twist_des.data(), kc.data(), dc.data(),
                              out.mutable_data());
  return out;
}

} // namespace

PYBIND11_MODULE(arm_rt_servo, m) {
  m.doc() = "The RT servo law, compiled once, shared by sim and the RT loop";
  py::class_<arm_rt::PoseHold>(m, "PoseHold")
      .def(py::init<>())
      .def("reset", &arm_rt::PoseHold::reset)
      .def("torque", [](arm_rt::PoseHold& hold, Arr q, Arr dq, Arr J,
                         Arr pose, Arr bias, Arr spec, double dt) {
        const int n=int(q.size());
        if(q.ndim()!=1 || dq.ndim()!=1 || bias.ndim()!=1 || pose.ndim()!=1 ||
           spec.ndim()!=1 || J.ndim()!=2 || J.shape(0)!=6 || J.shape(1)!=n)
          throw std::invalid_argument("PoseHold: expected vectors and J shape (6,n)");
        want(dq,n,"dq"); want(bias,n,"bias"); want(pose,7,"pose"); want(spec,15,"spec");
        Arr out(n);
        if(!hold.torque(n,q.data(),dq.data(),J.data(),pose.data(),bias.data(),spec.data(),dt,out.mutable_data()))
          throw std::invalid_argument("PoseHold: invalid state, id, gains, joint count or dt");
        return out;
      },py::arg("q"),py::arg("dq"),py::arg("J"),py::arg("pose"),py::arg("bias"),py::arg("spec"),py::arg("dt"));
  m.def("servo_torque", &servo_torque_py, py::arg("q"), py::arg("dq"),
        py::arg("q_des"), py::arg("qd_des"), py::arg("tau_ff"), py::arg("kp"),
        py::arg("kd"), py::arg("tau_ref"), py::arg("tau_limit"),
        py::arg("slew_per_tick"));
  m.def("cartesian_torque", &cartesian_torque_py, py::arg("tau_ff"), py::arg("J"),
        py::arg("R_task"), py::arg("x"), py::arg("quat"), py::arg("x_des"),
        py::arg("quat_des"), py::arg("twist"), py::arg("twist_des"),
        py::arg("kc"), py::arg("dc"),
        "tau_ff + J^T (R Kc R^T e_x + R Dc R^T edot_x); see servo_law.hpp for "
        "the frame/layout conventions (J is 6 x n ROW-major, base frame; "
        "Kc/Dc are TASK-frame diagonals; quats are [w,x,y,z]).");
}
