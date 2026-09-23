"""Rigid two-grasp motion: one object path, one clock, checked closure.

This module knows kinematics and collision geometry, not pickup policy. A caller
must supply established object-to-TCP transforms. It never infers a grasp from
a closed gripper or two coincident endpoint targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from arm_control import frames
from arm_control.config import ee_frame
from arm_control.motion import JointTrajectory
from arm_control.planning.ik import PinocchioIK
from arm_control.planning.refinement import finish_path

# Empty-Hand virtual preparation may retain a small measured endpoint residual.
# These limits admit capture only; they do not relax path or live closure checks.
VIRTUAL_CAPTURE_POSITION_M = 0.010
VIRTUAL_CAPTURE_ANGLE_RAD = np.deg2rad(2.0)


def virtual_capture_reason(errors):
    """Explain a refused capture, naming the measured per-arm residuals."""
    if any(
        not np.isfinite([distance, angle]).all()
        or distance > VIRTUAL_CAPTURE_POSITION_M
        or angle > VIRTUAL_CAPTURE_ANGLE_RAD
        for distance, angle in errors.values()
    ):
        measured = "; ".join(
            f"{a}: {distance * 1000:.2f} mm / {np.rad2deg(angle):.2f} degrees"
            for a, (distance, angle) in errors.items()
        )
        return (
            "Move both TCPs near the nominal virtual grasps "
            f"(within 10 mm / 2 degrees). Measured: {measured}."
        )
    return ""


def rigid_transform(value):
    T = frames.T_from_spec(value)
    if (
        not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1])
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-7)
        or np.linalg.det(T[:3, :3]) < 0
    ):
        raise ValueError("expected a finite rigid transform")
    return T


def pose_error(A, B):
    return (
        float(np.linalg.norm(A[:3, 3] - B[:3, 3])),
        float(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude()),
    )


@dataclass
class RigidGrasps:
    """Fixed object-to-TCP transforms, keyed by arbitrary arm identities."""

    object_T_tcp: dict[str, np.ndarray]

    def __post_init__(self):
        if len(self.object_T_tcp) != 2:
            raise ValueError("a shared object requires exactly two grasps")
        self.object_T_tcp = {
            a: rigid_transform(T) for a, T in self.object_T_tcp.items()
        }

    def targets(self, world_T_object):
        return {a: world_T_object @ T for a, T in self.object_T_tcp.items()}

    def infer(self, world_T_tcp):
        if set(world_T_tcp) != set(self.object_T_tcp):
            raise ValueError("TCP poses must name both grasping arms")
        return {
            a: T @ np.linalg.inv(self.object_T_tcp[a]) for a, T in world_T_tcp.items()
        }


class CooperativeKinematics:
    def __init__(self, config):
        self.grasps = config.shared_object.grasps
        self.ik = {
            a.name: PinocchioIK(
                a.path,
                ee_frame(config.robots[a.name]),
                a.joints,
                tol_pos=1e-6,
                tol_rot=1e-6,
            )
            for a in config.scene.actors
        }
        self.bases = {a: frames.world_T_arm(config.robots[a]) for a in self.ik}
        self.frames = {a: f"{a}__{ee_frame(config.robots[a])}" for a in self.ik}

    def tcp(self, joints):
        return {a: self.bases[a] @ ik.fk(joints[a]) for a, ik in self.ik.items()}

    def objects(self, joints):
        return self.grasps.infer(self.tcp(joints))

    def solve(self, world_T_object, seeds):
        result = {}
        for a, target in self.grasps.targets(world_T_object).items():
            q = self.ik[a].solve(np.linalg.inv(self.bases[a]) @ target, seeds[a])
            if q is None:
                raise ValueError(f"{a}: shared-object path has no IK solution")
            result[a] = q
        return result

    def nearer_posture(self, world_T_object, starts, solutions):
        """Reduce joint travel while retaining each prescribed six-axis grasp.

        Used for the free-arm approach only. Never alter a held grasp or
        independently optimize successive points of a closed-chain move.
        The caller must still check the combined endpoint and complete path.
        """
        from scipy.optimize import minimize

        result = dict(solutions)
        for a, target in self.grasps.targets(world_T_object).items():
            ik, start = self.ik[a], np.asarray(starts[a])
            local = np.linalg.inv(self.bases[a]) @ target

            def residual(q):
                pose = ik.fk(q)
                return np.r_[
                    pose[:3, 3] - local[:3, 3],
                    Rotation.from_matrix(local[:3, :3].T @ pose[:3, :3]).as_rotvec(),
                ]

            candidate = minimize(
                lambda q: float(np.sum((q - start) ** 2)), solutions[a],
                jac=lambda q: 2 * (q - start), method="SLSQP",
                bounds=list(zip(*ik.hard_limits)),
                constraints=[{"type": "eq", "fun": residual}],
                options={"ftol": 1e-10, "maxiter": 150},
            )
            q = candidate.x
            if (
                candidate.success and np.isfinite(q).all()
                and np.max(np.abs(residual(q))) <= 1e-6
                and np.linalg.norm(q - start) < np.linalg.norm(solutions[a] - start)
            ):
                result[a] = q
        return result


def plan_cooperative(planner, kinematics, body, starts, object_start, object_goal):
    """Straight SE(3) path with continuation IK; no unconstrained joint detours.

    Every executed Hermite segment is sampled at <= 0.001 rad joint travel.
    The two inferred object poses must agree within 1 mm / 0.5 degrees, and
    the inferred object must stay within those bounds of the SE(3) path.
    Collision checks include the moving object, both arms and the environment.
    This is a sampled validation, not a continuous collision certificate.
    """
    start, goal = rigid_transform(object_start), rigid_transform(object_goal)
    translation, rotation = pose_error(start, goal)
    if translation > 0.5 or rotation > np.pi / 2:
        raise ValueError(
            "shared-object preview limited to 500 mm / 90 degrees per move"
        )
    count = max(
        4, int(np.ceil(max(translation / 0.002, rotation / np.deg2rad(0.5)))) + 1
    )
    delta = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).as_rotvec()

    def object_at(fraction):
        T = start.copy()
        T[:3, 3] += fraction * (goal[:3, 3] - start[:3, 3])
        T[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(delta * fraction).as_matrix()
        return T

    maxima = np.zeros(4)

    def valid(q):
        if np.any(q < planner.world.lower) or np.any(q > planner.world.upper):
            raise ValueError("shared-object path exceeds joint limits")
        poses = kinematics.objects({a: q[sl] for a, sl in planner.slices.items()})
        A, B = poses.values()
        closure = pose_error(A, B)
        # Project to the commanded segment using its dominant motion component.
        if translation > 1e-9:
            fraction = (
                np.dot(A[:3, 3] - start[:3, 3], goal[:3, 3] - start[:3, 3])
                / translation**2
            )
        elif rotation > 1e-9:
            fraction = (
                np.dot(
                    Rotation.from_matrix(start[:3, :3].T @ A[:3, :3]).as_rotvec(), delta
                )
                / rotation**2
            )
        else:
            fraction = 0.0
        path_error = pose_error(A, object_at(np.clip(fraction, 0, 1)))
        maxima[:] = np.maximum(maxima, [*closure, *path_error])
        if max(closure[0], path_error[0]) > 0.001 or max(
            closure[1], path_error[1]
        ) > np.deg2rad(0.5):
            raise ValueError(
                "shared-object interpolation breaks grasp closure or leaves object path (>1 mm / 0.5 deg)"
            )
        planner.world.set_free_body_pose(body, A)
        return not planner.world.in_collision(q)

    q = planner.join(starts)
    if not valid(q):
        raise ValueError(
            f"shared-object start collision: {planner.world.collision_contacts(q)[:3]}"
        )
    path, seeds = [q], starts
    for fraction in np.linspace(0, 1, count)[1:]:
        seeds = kinematics.solve(object_at(fraction), seeds)
        q = planner.join(seeds)
        if np.max(np.abs(q - path[-1])) > 0.08:
            raise ValueError("shared-object IK branch jumped; choose a smaller move")
        if not valid(q):
            raise ValueError(
                f"shared-object path collision: {planner.world.collision_contacts(q)[:3]}"
            )
        path.append(q)
    if np.max(np.abs(path[-1] - path[0])) < 1e-10:
        combined = JointTrajectory(
            [0.0, 0.1], np.asarray([path[0], path[0]]), np.zeros((2, len(q)))
        )
    else:
        targets = None
        if planner.refinement == "trajopt":
            first = planner.ids[0]
            planner.world.bind_free_body(
                body,
                kinematics.frames[first],
                np.linalg.inv(kinematics.grasps.object_T_tcp[first]),
            )
            targets = {
                kinematics.frames[a]: [
                    object_at(f) @ grasp for f in np.linspace(0, 1, count)
                ]
                for a, grasp in kinematics.grasps.object_T_tcp.items()
            }
        combined = finish_path(
            planner.world,
            path,
            planner.vmax,
            planner.amax,
            refinement=planner.refinement,
            collides=lambda q: not valid(q),
            cartesian_targets=targets,
            label="shared-object",
        )
    motion = planner.motion(combined)
    motion.cooperative_metrics = dict(
        closure_m=float(maxima[0]),
        closure_rad=float(maxima[1]),
        path_m=float(maxima[2]),
        path_rad=float(maxima[3]),
    )
    return motion
