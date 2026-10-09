"""Build the EXECUTION half of an arm's stack — servo only, no planning.

Split out of ``planning.stack`` so a controller host can import a servo
without importing a planner. That was already the intent: ``arm_controller``
reaches for ``build_executor`` precisely to avoid loading IK, OMPL, a
collision world or a scene. But it reached for it THROUGH ``planning.stack``,
which imports ``viz.preview``, which imports ``rerun`` at module level --
and ``rerun-sdk`` is an optional ``[viz]`` extra. A headless install without
that extra therefore died at import, in the one node that most needs to run
without a display. (The comment on that import in ``stack.py`` claimed the
opposite; it was wrong, and it is corrected there.)

Nothing here may pull in ``rerun`` or the planning stack.
The executor consumes neutral ``arm_control.motion`` values; retiming belongs
to the planning side and is not imported by this factory.
``_self_check`` asserts the boundary rather than trusting it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from arm_control.config import CONTROL_ROOT, _arm_block, arm_joints, ee_frame
from arm_control.dynamics import PinocchioDynamics
from arm_control.control.arm_controller import ArmController
from arm_control.control.execution_policy import build_execution_policy
from arm_control.control.trajectory_executor import JointTrajectoryExecutor
from arm_control.control.gains import validate_integral_vector
from arm_control.control.integral import IntegralState
from arm_control.model_bundle import execution_gravity, model_token, pose_provenance
from arm_control.node_utils import planner_speed_limits, resolve_gains

__all__ = [
    "arm_urdf", "gain_vector", "build_executor",
    "build_arm_controller",
]


def arm_urdf(cfg) -> str:
    """The arm's URDF, resolved against the DEPLOYMENT root.

    Relative paths resolve against ``CONTROL_ROOT``, which is this package's
    own root standalone and ``$ARM_CONTROL_ROOT`` when embedded. That seam is
    deliberate -- this repo owns the CODE and the arm identity configs, the
    consuming project owns the CAD -- but it means a config shipped HERE can
    name a URDF that only exists THERE, and the failure then surfaces as an
    unreadable urdfdom parse error several frames down. Say it plainly
    instead: this is the first thing a new consumer of this package hits.
    """
    raw = _arm_block(cfg).get("urdf") or cfg.get("urdf_path")
    if not raw:
        raise ValueError("config missing an arm URDF (arm.urdf / urdf_path)")
    path = Path(str(raw))
    resolved = path if path.is_absolute() else CONTROL_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(
            f"arm URDF {str(raw)!r} not found at {resolved}.\n"
            f"Relative URDF paths resolve against the deployment root, which "
            f"is currently {CONTROL_ROOT}.\n"
            f"If this package is embedded in a project that owns the CAD, "
            f"export ARM_CONTROL_ROOT=<that project's root>; if you are using "
            f"it standalone, point arm.urdf / urdf_path at your own robot "
            f"description (absolute paths are honoured as-is)."
        )
    return str(resolved)


def gain_vector(cfg, key: str, n: int) -> np.ndarray:
    """Per-joint vector from ``arm.<key>``; a scalar broadcasts to ``n``."""
    value = _arm_block(cfg).get(key)
    if value is None:
        raise ValueError(
            f"config missing arm.{key}: every arm declares its own per-joint "
            f"values (there is no robot-shaped default to inherit)"
        )
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    if arr.shape != (n,):
        raise ValueError(f"arm.{key} must have {n} values for this arm, got {arr.size}")
    return arr


def arm_speed_limits(cfg, n: int, *, required: bool = True):
    """``(max_vel, max_acc)`` a planner times plans with from ``arm.max_vel``/``arm.max_acc``, under the console's rule
    (``planner_speed_limits``): with an execution policy the arm's statement defaults to it and may only be slower (refused at load
    otherwise -- the controller would refuse every plan at admission); without one the arm states its own (``required``), or the
    planner is unbounded here (``required=False``, a caller that bounds speed itself)."""
    arm = _arm_block(cfg)
    policy = cfg.get("execution_policy") or {}
    joints = list(arm.get("joints") or [])
    names = joints if len(joints) == n else [str(i) for i in range(n)]
    limits = []
    for index, (key, plan_key, policy_key) in enumerate((("max_vel", "vel_limits", "velocity_limits"),
                                                         ("max_acc", "acc_limits", "acceleration_limits"))):
        if policy.get(policy_key) is None:   # the policy leaves this one unbounded: the arm states its own, as before
            limits.append(gain_vector(cfg, key, n) if required or arm.get(key) is not None else np.full(n, np.inf))
            continue
        stated = {} if arm.get(key) is None else {plan_key: arm[key]}
        try:
            limits.append(planner_speed_limits(stated, {policy_key: policy[policy_key]}, names)[index])
        except ValueError as exc:
            raise ValueError(f"arm.{key}: {exc}".replace(f"planner.{plan_key} ", "")) from None
    return limits[0], limits[1]


def build_executor(
    cfg, *, arm_id: str = "arm", gains: dict | None = None, simulation_gravity=None
) -> JointTrajectoryExecutor:
    """The EXECUTION half of an arm's stack, with no planning half at all.

    Bound hardware consumes staged model gravity. Only simulation may supply
    ``simulation_gravity`` from its own physics/actor placement; analytical
    callers can construct PinocchioDynamics directly.

    Split out of build_planning_stack so ``arm_controller`` can have a servo
    without an IK, an OMPL instance, a collision world or a scene -- the whole
    point of the planner/controller split is that the controller loads none of
    those, and calling build_planning_stack just to reach ``.executor`` would
    load every one of them.

    ``gains`` (per-MOTOR vectors, as ``node_utils.resolve_gains`` returns them)
    overrides the arm table. Two config styles have to meet here: an assembly
    config states ``arm.kp``, while a motion mode config states
    ``controller.kp`` and its robot config states no gains at all. Reading only
    the arm table meant this executor could not be built for a motion graph --
    it raised "config missing arm.kp" and the node died on startup. The caller
    resolves across both and passes the answer in; absent, the arm table is
    still the source, so every existing caller is unchanged.
    """
    # Simulation supplies its own physics vector; hardware consumes staged facts.
    joints = arm_joints(cfg)
    n = len(joints)
    # done() tolerances are per-arm feedback facts (DM quantization, payload
    # sag); absent keys keep the executor's sim-tuned defaults.
    arm_blk = _arm_block(cfg)
    tolerances = {
        k: float(arm_blk[k]) for k in ("done_pos_tol", "done_vel_tol") if k in arm_blk
    }
    def _gain(key: str, arm_key: str) -> np.ndarray:
        if gains is None:
            return gain_vector(cfg, arm_key, n)
        return np.asarray(gains[key], dtype=float).ravel()[:n]

    executor = JointTrajectoryExecutor(
        arm_id=arm_id,
        joint_names=joints,
        dynamics=PinocchioDynamics(
            arm_urdf(cfg), joints,
            gravity=(execution_gravity(cfg, arm_urdf(cfg)) if simulation_gravity is None
                     else simulation_gravity),
        ),
        kp_default=_gain("kp", "kp"),
        kd_default=_gain("kd", "kd"),
        max_torque=_gain("torque_limits", "max_tau"),
        # The plant compensates gravity (the FR3 control box on the bench, and
        # now the twin too), so ship RNEA MINUS gravity or the arm gets it
        # twice. The legacy standalone executor already honoured this flag; this
        # path silently ignored it.
        gravity_comp=bool(arm_blk.get("plant_gravity_comp", False)),
        payload_frame=ee_frame(cfg),
        **tolerances,
    )
    return executor


def build_arm_controller(
    cfg, node, *, arm_id: str, mode_cfg: dict | None = None,
    require_policy: bool = False, **options,
) -> ArmController:
    """The one recipe for an arm's ``ArmController`` from its config.

    Gains (mode config first, arm table second; ``mode_cfg`` defaults to the
    config's ``control_mode``), executor, execution policy, model binding,
    pose provenance and settling are decided here and nowhere else. Callers
    pass what their process owns -- the node, clocks, health reporting -- as
    ``options``; ``require_policy`` refuses a config without the 100 Hz
    deadline policy.
    """
    n = len(arm_joints(cfg))
    names = list(cfg.get("joint_names") or cfg.get("motor_names") or [])
    if mode_cfg is None:
        mode_cfg = cfg.get("control_mode") or {}
    gains = resolve_gains(cfg, mode_cfg, names, n)
    executor = build_executor(cfg, arm_id=arm_id, gains=gains)
    # Validate the complete motor vector before projecting arm slots. The
    # state owner consumes these gains; the pure executor does not.
    integral = IntegralState(n, **{
        key: validate_integral_vector(None if gains is None else gains.get(key),
                                      len(names) if names else n, where=key)[:n]
        for key in ("ki", "i_limit")
    })
    policy_cfg = cfg.get("execution_policy")
    if policy_cfg is not None and gains is None:
        raise ValueError("execution_policy requires resolved per-joint torque limits")
    # The position envelope comes from the robot description, not from YAML --
    # see build_execution_policy.
    policy = build_execution_policy(
        policy_cfg,
        torque_limits=None if gains is None else gains["torque_limits"][:n],
        joint_limits=executor.joint_limits,
    )
    if require_policy and policy is None:
        raise ValueError(f"{arm_id}: needs an execution_policy (100 Hz deadline mode)")
    arm = cfg.get("arm") or {}
    # How long a jog/setpoint stays live: read from the mode config the console reads, so both deadmen agree (too tight and a
    # scheduling hiccup stutters the arm, too loose and it coasts after the button is released).
    options.setdefault("jog_timeout_s", float((mode_cfg.get("jog") or {}).get("timeout_s", 0.2)))
    return ArmController(
        node, executor, arm_id=arm_id,
        integral_state=integral, execution_policy=policy, model_token=model_token(cfg),
        pose_provenance=pose_provenance(cfg),
        settle_timeout_s=float(arm.get("settle_timeout_s", 5.0)),
        settle_dwell_s=float(arm.get("settle_dwell_s", 0.2)),
        **options,
    )



def _self_check() -> None:
    """The import boundary IS the feature: assert it, do not trust it."""
    import subprocess
    import sys

    # A fresh interpreter, so an already-imported rerun in this process cannot
    # mask the very thing under test.
    # Poisoning sys.modules makes any `import rerun` raise, so the import
    # SUCCEEDING is itself the proof that none was needed. The banned list then
    # catches the heavy planning modules, which are what drag rerun in.
    banned = (
        "arm_control.planning",
        "arm_control.scene",
        "arm_control.data",
        "arm_control.planning.stack",
        "arm_control.planning.planner",
        "arm_control.viz.preview",
        "arm_control.planning.ik",
        "arm_control.planning.mujoco_collision",
    )
    probe = (
        "import sys, importlib;"
        "sys.modules['rerun'] = None;"
        "importlib.import_module('arm_control.control.factory');"
        f"leaked = [m for m in {banned!r} if m in sys.modules];"
        "assert not leaked, leaked;"
        "print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert out.returncode == 0 and "clean" in out.stdout, (
        "the executor factory must import with no rerun and no planning module:\n"
        + out.stdout + out.stderr
    )
    assert gain_vector({"arm": {"kp": 3.0}}, "kp", 4).tolist() == [3.0] * 4
    assert gain_vector({"arm": {"kp": [1, 2]}}, "kp", 2).tolist() == [1.0, 2.0]
    for bad in ({"arm": {}}, {"arm": {"kp": [1, 2, 3]}}):
        try:
            gain_vector(bad, "kp", 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must not yield a gain vector")
    print("control.factory: OK")


if __name__ == "__main__":
    _self_check()
