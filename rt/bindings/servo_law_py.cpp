// pybind11 over the ONE servo law (servo_law.hpp) so the MuJoCo plant can
// close the exact compiled code the RT thread runs. Built by rt/bindings/
// (`pip install -e rt/bindings`) — optional: without it the sim falls back
// to its Python PD and must WARN, never silently.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "arm_rt/servo_law.hpp"

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

} // namespace

PYBIND11_MODULE(arm_rt_servo, m) {
  m.doc() = "The RT servo law, compiled once, shared by sim and the RT loop";
  m.def("servo_torque", &servo_torque_py, py::arg("q"), py::arg("dq"),
        py::arg("q_des"), py::arg("qd_des"), py::arg("tau_ff"), py::arg("kp"),
        py::arg("kd"), py::arg("tau_ref"), py::arg("tau_limit"),
        py::arg("slew_per_tick"));
}
