#!/usr/bin/env python3
"""load_model/1: a revisioned rigid-body load model as the controller's RNEA (operator 2026-10-09, B1).

    PYTHONPATH=. python tools/check/check_load_model.py

In process: ``BodyListDynamics`` over a toy chain gives the gravity a hand calculation gives, and ``set_dynamics`` puts it into
the executor's hold (and refuses another joint order, or a declared payload). Over a simulated FR3 supervisor (as
`examples/run_console.py` starts it): the op is advertised; refused without the claim, for another joint order, for a
revision not above the installed one, for gravity above the policy's torque limits and while an execution is live; accepted
otherwise; a payload is refused once a load model carries the load.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arm_control.dynamics import BodyListDynamics  # noqa: E402


def chain(joints, mass=0.5, reach=0.1):
    """Each joint a hinge about world y at the parent's tip, carrying one point-ish mass ``reach`` out along x."""
    bodies = []
    for i, name in enumerate(joints):
        bodies.append(dict(parent=i, pos=[reach if i else 0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0], mass=mass,
                           com=[reach, 0.0, 0.0], inertia=[1e-6, 1e-6, 1e-6], inertia_quat=[1.0, 0.0, 0.0, 0.0],
                           joint=dict(name=name, kind=3, axis=[0.0, 1.0, 0.0], pos=[0.0, 0.0, 0.0])))
    return dict(joints=list(joints), inactive_joints=[], bodies=bodies, gravity=[0.0, 0.0, -9.81])


def in_process() -> None:
    from arm_control.control.trajectory_executor import JointTrajectoryExecutor
    from arm_control.motion import JointState

    # One hinge about y, 2 kg at 0.3 m along x: gravity's moment about +y is +m g x ((x,0,0) x (0,0,-m g)), so the torque that
    # holds it (RNEA's gravity term) is -m g x; zero once the arm hangs straight down (q = pi/2 turns +x into -z).
    one = chain(["j"], mass=2.0, reach=0.3)
    dyn = BodyListDynamics(one)
    assert abs(dyn.gravity(np.zeros(1))[0] + 2 * 9.81 * 0.3) < 1e-6, dyn.gravity(np.zeros(1))
    assert abs(dyn.gravity(np.array([np.pi / 2]))[0]) < 1e-9

    class Construction:  # the construction model (a URDF's), with limits the load model does not carry
        joints = ["a", "b"]
        joint_limits = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

        def gravity(self, q):
            return np.zeros(2)

        def rnea(self, q, qd, qdd):
            return np.zeros(2)

    ex = JointTrajectoryExecutor("arm", ["a", "b"], Construction(), np.full(2, 10.0), np.full(2, 1.0), np.full(2, 50.0),
                                 payload_frame="tool")
    state = JointState(np.zeros(2), np.zeros(2))
    assert np.allclose(ex.hold_command(state).tau_ff, 0.0)
    for bad in (chain(["b", "a"]), chain(["a"])):
        try:
            ex.set_dynamics(BodyListDynamics(bad))
        except ValueError:
            pass
        else:
            raise AssertionError("another joint order installed")
    ex.set_payload(1.0, com_offset=[0, 0, 0])
    try:
        ex.set_dynamics(BodyListDynamics(chain(["a", "b"])))
    except ValueError:
        pass
    else:
        raise AssertionError("a load model installed over a declared payload")
    ex.set_payload(0.0)
    model = BodyListDynamics(chain(["a", "b"]))
    ex.set_dynamics(model)
    assert np.allclose(ex.hold_command(state).tau_ff, model.gravity(np.zeros(2))), "the hold does not use the load model"
    assert ex.joint_limits[0][0] == -1.0, "joint limits must stay the construction model's"
    try:
        ex.set_payload(1.0, com_offset=[0, 0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("a payload declared on top of a load model")


def supervised() -> None:
    from arm_control.execution.contract import TrajectoryRequest
    from arm_control.execution.controller import PayloadValue
    from arm_control.motion import JointTrajectory
    from arm_control.sdk import ArmClient, Refused
    from arm_control.supervisor.launch import start_supervisor, stop_supervisor
    from examples.run_console import DEFAULT_CONFIG, MODE_PROFILE, SUPERVISED_PROFILE

    env = dict(os.environ, ARM_CONTROL_ROOT=str(ROOT), PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    supervisor, graph_env = start_supervisor(DEFAULT_CONFIG, env, overlays=(SUPERVISED_PROFILE,), control_mode=MODE_PROFILE,
                                             cwd=ROOT, timeout_s=300.0)
    from arm_control.config import load_robot_config

    instance = load_robot_config(graph_env["ARM_CONTROL_CONFIG"]).get("rt")["instance_id"]
    arm = other = None
    try:
        arm = ArmClient(instance, socket_path=graph_env["ARM_CONTROL_SOCKET"])
        other = ArmClient(instance, socket_path=graph_env["ARM_CONTROL_SOCKET"])
        assert arm.extensions.get("load_model") == 1, arm.extensions
        joints = list(arm.identity.joint_names)

        def refused(client, revision, model, words):
            try:
                client.set_load_model(revision, model)
            except Refused as exc:
                assert words in str(exc), (words, exc)
            else:
                raise AssertionError(f"accepted: {words}")

        arm.claim()
        refused(other, 1, chain(joints), "")  # not the owner
        refused(arm, 1, chain(list(reversed(joints))), "joints")
        refused(arm, 1, chain(joints, mass=1e4), "torque limits")
        assert arm.set_load_model(1, chain(joints)) == 1
        refused(arm, 1, chain(joints), "revision")
        try:
            arm.set_payload(PayloadValue(1.0))
        except Refused as exc:
            assert "load model" in str(exc), exc
        else:
            raise AssertionError("a payload on top of a load model")
        arm.arm()
        q0 = np.asarray(arm.state().q)
        traj = JointTrajectory([0.0, 3.0], np.stack([q0, q0 + np.eye(len(q0))[0] * 0.05]), np.zeros((2, len(q0))))
        status = arm.execute(TrajectoryRequest("load_model_live", instance, tuple(joints), traj, model_token=None))
        refused(arm, 2, chain(joints), "execution")
        arm.cancel(status.execution_id, "check done")
        arm.disarm()
    finally:
        for client in (arm, other):
            if client is not None:
                client.close()
        stop_supervisor(supervisor)


def main() -> int:
    in_process()
    supervised()
    print("load model: body-list gravity matches a hand calculation; set_dynamics drives the hold, keeps the construction "
          "limits, refuses another joint order and a payload; load_model/1 refused unclaimed, out of order, over the torque "
          "limits, at a stale revision and during an execution, accepted otherwise, and blocks a payload: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
