"""The simulated pair's world, hosted by the first arm's supervisor (the second is its co-host): a ``WORKCELL_BACKEND_FACTORY``.

``WORKCELL_SCENE`` names the pair config (``load_pair_config``); this builds the arms and the scenery of its scene with each
arm's gravity compensated (an FR3 twin, as the control box: the controllers ship RNEA minus gravity) and its Hand's two
fingers, arm-arm contact on (the pair must not pass through itself). The shared object is NOT in the physics: coupling is
virtual, as on the bench (operator 2026-10-09; empty Hands, the plate inferred from both TCPs by the console), and the plate
stays in the console's planning and display world only.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from arm_control.config import gripper_joints
from arm_control.multi_arm.config import load_pair_config
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.plants.mujoco.backend import MuJoCoBackend


def refuse_colliding_start(config) -> None:
    """ValueError when the configured start collides in the planning world (arms, scenery, the virtual plate at its pose)."""
    scene = config.scene
    world = MuJoCoCollisionWorld.from_scene(scene, scene.state(), scene.actor_names, ground_z=config.ground_z,
                                            workspace_xy=config.workspace_xy)
    if config.shared_object:
        world.set_free_body_pose(config.shared_object.body, config.shared_object.world_T_object)
    q = np.concatenate([a.q for a in scene.actors])
    if world.in_collision(q):
        raise ValueError(f"initial scene collides: {world.collision_contacts(q)[:3]}")


def build_pair_world(scene_path, control_period, *, timestep=0.001, launch_viewer=False, enable_self_collision=True,
                     loader=None, chain_factory=None):
    """``(backend, slices, joint_names, n)`` for ``build_world`` (the library's factory contract)."""
    del loader, chain_factory, enable_self_collision  # the pair has no project loader or chain; contact is always on
    config = load_pair_config(scene_path)
    scene = config.scene
    names = scene.actor_names
    refuse_colliding_start(config)
    objects = tuple(f"{o.name}__" for o in scene.objects)  # not in the physics: neither are their exclusions
    exclusions = tuple(e for e in scene.contact_exclusions if not any(b.startswith(objects) for b in e))
    backend = MuJoCoBackend.from_workcell_scene(
        replace(scene, objects=(), welds=(), contact_exclusions=exclusions), scene.state(),
        control_period=control_period, timestep=timestep, ground_z=config.ground_z, enable_self_collision=True,
        launch_viewer=launch_viewer, gravcomp_prefixes=tuple(f"{a}__" for a in names),
        gripper_joints=tuple(f"{a}__{j}" for a in names for j in gripper_joints(config.robots[a])))
    backend.load()
    slices, start = {}, 0
    for actor in scene.actors:
        slices[actor.name] = {"start": start, "n": len(actor.joints)}
        start += len(actor.joints)
    if start != backend.num_motors:
        raise ValueError("the simulated pair supports actor motors only")
    return backend, slices, list(backend.joint_names), len(backend.joint_names)
