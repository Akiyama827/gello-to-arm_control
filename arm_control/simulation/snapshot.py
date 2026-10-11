"""Freeze declared scene state into a cached, zero-DoF fixture model."""

from pathlib import Path
import hashlib
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from arm_control.plants.mujoco.backend import compose_workcell_scene


def freeze_scene(scene, state, cache_dir):
    """Preserve visual/collision geometry and body poses; remove all motion.

    This is a declared snapshot, never a claim about measured hardware state.
    Attachments are resolved by the canonical composer before freezing.
    """
    spec = compose_workcell_scene(scene, state, ground_z=None)
    # Attached specs retain their own compiler paths in memory. XML export
    # has one compiler, so retain those asset origins explicitly.
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str((Path(mesh.compiler.meshdir) / mesh.file).resolve())
    for index, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"snapshot_geom_{index}"
    model = spec.compile()
    data = mujoco.MjData(model)
    for actor in scene.actors:
        for joint, value in zip(actor.joints, state.actor_q[actor.name]):
            data.qpos[model.joint(f"{actor.name}__{joint}").qposadr[0]] = value
    mujoco.mj_forward(model, data)
    for body in spec.bodies:
        if body.name == "world":
            continue
        index = model.body(body.name).id
        parent = model.body_parentid[index]
        R = data.xmat[parent].reshape(3, 3)
        position = R.T @ (data.xpos[index] - data.xpos[parent])
        rotation = R.T @ data.xmat[index].reshape(3, 3)
        # Frames can also own geoms/sites, so retain them. Express the frozen
        # body pose in its original frame rather than applying that frame twice.
        frames = []
        frame = body.frame
        while frame is not None:
            frames.append(frame)
            frame = frame.frame
        for frame in reversed(frames):
            q = np.asarray(frame.quat)
            frame_rotation = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
            position = frame_rotation.T @ (position - frame.pos)
            rotation = frame_rotation.T @ rotation
        body.pos = position
        q = Rotation.from_matrix(rotation).as_quat()
        body.quat = np.r_[q[3], q[:3]]
    for collection in (
        spec.actuators,
        spec.sensors,
        spec.equalities,
        spec.tendons,
        spec.keys,
        spec.joints,
    ):
        for item in list(collection):
            spec.delete(item)
    xml = spec.to_xml()
    frozen = mujoco.MjSpec.from_string(xml).compile()
    check = mujoco.MjData(frozen)
    mujoco.mj_forward(frozen, check)
    if frozen.nq or frozen.nv:
        raise ValueError("static snapshot retained joints")
    for index in range(model.ngeom):
        name = model.geom(index).name
        gid = frozen.geom(name).id
        # MuJoCo's XML writer rounds numeric fields; allow one micrometre
        # and 1e-6 per rotation component across the serialization round trip.
        if not (
            np.allclose(data.geom_xpos[index], check.geom_xpos[gid], atol=1e-6, rtol=0)
            and np.allclose(
                data.geom_xmat[index], check.geom_xmat[gid], atol=1e-6, rtol=0
            )
        ):
            raise ValueError(f"static snapshot moved geometry {name}")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (hashlib.sha256(xml.encode()).hexdigest() + ".xml")
    if not path.exists():
        with tempfile.NamedTemporaryFile(
            mode="w", dir=cache, suffix=".tmp", delete=False
        ) as handle:
            handle.write(xml)
            temporary = Path(handle.name)
        temporary.replace(path)
    return path
