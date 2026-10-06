"""The single-model MuJoCo plant a robot config describes (one MJCF/URDF, no scene, no welds, no ground).

Shared by the Dora ``mujoco_interface`` node and the supervisor's simulated link, so both build the same plant from the same
keys: ``sim_model_path``, ``sim_timestep``, ``sim_launch_viewer``, ``sim_default_joint_positions``, ``sim_gravcomp`` and the
arm block (``arm.gripper_joints``, the EE frame).
"""
from arm_control.config import ee_frame, gripper_joints
from arm_control.plants.mujoco.backend import MuJoCoBackend
from arm_control.plants.mujoco.composer import _resolve


def build_single_model_backend(cfg, *, control_period: float) -> MuJoCoBackend:
    joint_names = list(cfg.joint_names)
    return MuJoCoBackend(
        joint_names=joint_names,
        single_model_path=_resolve(str(cfg.get("sim_model_path", ""))),
        timestep=float(cfg.get("sim_timestep", 0.001)),
        control_period=control_period,
        ground_z=None,
        launch_viewer=bool(cfg.get("sim_launch_viewer", False)),
        # Single-model joints carry no scene prefix; servo the fingers so
        # they hold instead of free-sliding (and take gripper commands).
        # From the arm's OWN config (`arm.gripper_joints`) — a new robot is a
        # new YAML, never an edit here.
        gripper_joints=tuple(gripper_joints(cfg)),
        ee_body=ee_frame(cfg),
        # Spawn pose. The block below argues the plant must DECLARE its
        # starting state and then only did so for the jaws: the arm itself
        # was left at MuJoCo's implicit zero, which for the FR3 is not even
        # inside the joint limits (joint4's range is [-3.077, -0.117], so
        # zero is 0.117 rad outside it). Measured: the demo spawned there,
        # sagged onto the joint4/joint6 stops within seconds, and the first
        # Cartesian jog was refused with "no IK solution" — from a pose no
        # operator chose. Same key and same values the scenario configs
        # already use.
        default_joint_positions={
            str(k): float(v)
            for k, v in (cfg.get("sim_default_joint_positions") or {}).items()
        },
        # A twin that lets the arm fall is the one telling the lie: the
        # FR3's control box runs its own gravity compensation whether or
        # not a host is streaming, so a DISARMED arm holds its pose on the
        # bench. Scene mode has always modelled that (gravcomp_prefixes);
        # single-model mode did not, which is why an unarmed demo slumped.
        # The empty prefix matches every body — in this mode the model IS
        # the arm. Opt-in, because a single model that is NOT a
        # self-supporting robot should still fall.
        gravcomp_prefixes=("",) if cfg.get("sim_gravcomp", False) else (),
    )


def open_jaws(backend, cfg) -> None:
    """The plant declares its starting state: the Hand spawns OPEN at the arm's configured width (MuJoCo's zero is a SHUT hand).

    Shared by the Dora ``mujoco_interface`` node and the supervisor's simulated link, so both spawn the same twin (an open hand, not
    MuJoCo's zero). It does not change the wrist behaviour of the simulated FR3, which is the same with the jaws shut.
    """
    backend.load()
    fingers = int(backend.gripper_positions().size)
    # Same accessor sim_bridge uses for this block (the gripper config sits
    # under the robot key, not the arm block).
    gripper_cfg = dict((cfg.get("franka") or {}).get("gripper") or {})
    open_w = float(gripper_cfg.get("open_width_m", 0.0))
    if fingers and open_w > 0:
        backend.apply_gripper_command([open_w / fingers] * fingers)
        print(
            f"[mujoco_interface] jaws spawned OPEN at {open_w * 1000:.0f} mm",
            flush=True,
        )


