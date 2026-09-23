"""Common path finishing for manual and coordinated motion planners."""

import copy

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.planning.retiming import collision_checked_retiming


def refinement_choice(value):
    if value not in (None, "none", "trajopt"):
        raise ValueError("refinement must be none or trajopt")
    return "none" if value is None else value


def finish_path(
    world,
    path,
    vmax,
    amax,
    *,
    refinement="none",
    active_indices=None,
    collides=None,
    cartesian_targets=None,
    label="arm",
):
    """Refine and time a path, retaining the caller's canonical validation.

    Frozen joints are removed from optimization and restored exactly, including
    zero velocity. A scene adapter supplies its current raw world via .world.
    Cartesian targets map body names to one world pose per seed knot.
    """
    refinement = refinement_choice(refinement)
    collides = world.in_collision if collides is None else collides
    if refinement == "none":
        return collision_checked_retiming(path, vmax, amax, collides)
    from arm_control.planning.trajopt import refine_trajectory

    raw = getattr(world, "world", world)
    path = np.asarray(path, dtype=float)
    indices = (
        np.arange(path.shape[1])
        if active_indices is None
        else np.asarray(active_indices)
    )
    frozen = np.setdiff1d(np.arange(path.shape[1]), indices)
    if not np.all(path[:, frozen] == path[0, frozen]):
        raise ValueError("held joints moved in refinement seed")
    view = copy.copy(raw)
    view._held_qpos = raw._full_q(path[0])
    view.joint_names = [raw.joint_names[i] for i in indices]
    view._qadr = raw._qadr[indices]
    view.lower, view.upper = raw.lower[indices], raw.upper[indices]

    def expand(q):
        full = path[0].copy()
        full[indices] = q
        return full

    trajectory = refine_trajectory(
        view,
        path[:, indices],
        np.asarray(vmax)[indices],
        np.asarray(amax)[indices],
        arm_id=label,
        cartesian_targets=cartesian_targets,
        collision_fn=lambda q: collides(expand(q)),
    )
    positions = np.tile(path[0], (len(trajectory.times), 1))
    velocities = np.zeros_like(positions)
    positions[:, indices], velocities[:, indices] = (
        trajectory.positions,
        trajectory.velocities,
    )
    return JointTrajectory(trajectory.times, positions, velocities)
