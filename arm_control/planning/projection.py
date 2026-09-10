"""Project one current MuJoCo collision snapshot into a Tesseract environment.

Only limited scalar hinges at body origins are supported. Other joints are
frozen at the canonical world's held positions. Meshes use MuJoCo's convex
envelope semantics. The ground replacement bounds all reachable robot geometry.
"""

import numpy as np


def _transform(position, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation.reshape(3, 3)
    result[:3, 3] = position
    return result


def _workspace_bound(world):
    import mujoco

    model, data = world.model, world.data
    offsets = np.zeros(model.nbody)
    for body in range(1, model.nbody):
        parent = model.body_parentid[body]
        offsets[body] = offsets[parent] + np.linalg.norm(data.xpos[body] - data.xpos[parent])
    bound = 0.0
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        kind, size = model.geom_type[geom], model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        radius = np.linalg.norm(size)
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            radius = np.max(np.linalg.norm(model.mesh_vert[start:start + count], axis=1))
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = sum(size[:2])
        bound = max(bound, offsets[model.geom_bodyid[geom]] + np.linalg.norm(model.geom_pos[geom]) + radius)
    if not np.isfinite(bound):
        raise ValueError('nonfinite projected workspace bound')
    return float(bound)


def project_world(world):
    """Rebuild geometry and filters from the supplied snapshot; fail closed."""
    import mujoco
    from scipy.spatial import ConvexHull
    from tesseract_robotics import ensure_configured
    from tesseract_robotics import tesseract_geometry as geometry
    from tesseract_robotics.tesseract_common import Isometry3d, GeneralResourceLocator
    from tesseract_robotics.tesseract_environment import Environment
    from tesseract_robotics.tesseract_scene_graph import SceneGraph, Link, Joint, JointType, JointLimits, Collision
    from tesseract_robotics.tesseract_srdf import SRDFModel, KinematicsInformation

    ensure_configured()
    model, data = world.model, world.data
    if (model.npair or model.opt.disableflags or model.nflex
            or np.any(model.geom_margin != 0) or np.any(model.geom_gap != 0)):
        raise ValueError('unsupported explicit pairs, collision flags, flex, margins or gaps')
    if not np.isfinite(world._pad) or world._pad > 0:
        raise ValueError('self collision padding must be finite and nonpositive')
    if np.shape(world._env_geom) != (model.ngeom,):
        raise ValueError('invalid environment geometry mask')
    planned_body = getattr(world, '_planned_body', np.ones(model.nbody, dtype=bool))
    if np.shape(planned_body) != (model.nbody,):
        raise ValueError('invalid planned body mask')
    active_ids = {model.joint(name).id for name in world.joint_names}
    if len(active_ids) != len(world.joint_names):
        raise ValueError('duplicate planned joints')
    for joint in active_ids:
        body = model.jnt_bodyid[joint]
        if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                or model.body_jntnum[body] != 1 or not model.jnt_limited[joint]
                or np.max(np.abs(model.jnt_pos[joint])) > 1e-12):
            raise ValueError('projection requires limited single hinges at body origins')
    world._fk(np.zeros(len(world.joint_names)))
    if not np.isfinite(data.xpos).all() or not np.isfinite(data.xmat).all():
        raise ValueError('nonfinite canonical forward kinematics')
    graph = SceneGraph('planning_snapshot')
    # Generated names prevent collisions with user bodies named geom_0 etc.
    body_names = [f'body_{body}' for body in range(model.nbody)]
    body_transforms = [_transform(data.xpos[body], data.xmat[body]) for body in range(model.nbody)]
    radius = _workspace_bound(world) + 1.0
    for body, name in enumerate(body_names):
        link = Link(name)
        if body == 0:
            added = graph.addLink(link)
            graph.setRoot(name)
        else:
            joint_id = int(model.body_jntadr[body])
            active = joint_id in active_ids
            joint = Joint(model.joint(joint_id).name if active else f'fixed_body_{body}')
            joint.type = JointType.REVOLUTE if active else JointType.FIXED
            parent = model.body_parentid[body]
            joint.parent_link_name, joint.child_link_name = body_names[parent], name
            joint.parent_to_joint_origin_transform = Isometry3d(np.linalg.inv(body_transforms[parent]) @ body_transforms[body])
            if active:
                joint.axis = model.jnt_axis[joint_id].copy()
                lo, hi = model.jnt_range[joint_id]
                joint.limits = JointLimits(float(lo), float(hi), 100., 3., 10., 100.)
            added = graph.addLink(link, joint)
        if not added:
            raise ValueError(f'cannot project body {body}')
    live = [geom for geom in range(model.ngeom) if model.geom_contype[geom] or model.geom_conaffinity[geom]]
    for geom in live:
        kind, size = int(model.geom_type[geom]), model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count].astype(float)
            hull = ConvexHull(vertices)
            vertices = vertices[hull.vertices]
            hull = ConvexHull(vertices)
            faces = np.column_stack((np.full(len(hull.simplices), 3), hull.simplices)).astype(np.int32).ravel()
            shape = geometry.ConvexMesh(list(vertices), faces)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            shape = geometry.Box(*map(float, 2 * size))
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            if (model.geom_bodyid[geom] != 0 or not np.array_equal(model.geom_pos[geom], [0, 0, 0])
                    or not np.array_equal(model.geom_quat[geom], [1, 0, 0, 0])):
                raise ValueError('only a world z=0 ground plane is supported')
            shape = geometry.Box(8 * radius, 8 * radius, 4 * radius)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            shape = geometry.Sphere(float(size[0]))
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            constructor = geometry.Capsule if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else geometry.Cylinder
            shape = constructor(float(size[0]), float(size[1] * 2))
        else:
            raise ValueError(f'unsupported collision geometry {geom}: type {kind}')
        link = Link(f'geom_{geom}')
        collision = Collision()
        collision.name, collision.geometry = f'geom_{geom}', shape
        collision.origin = Isometry3d.Identity()
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            collision.origin = Isometry3d(_transform(np.array([0., 0., -2 * radius]), np.eye(3)))
        link.addCollision(collision)
        joint = Joint(f'fixed_geom_{geom}')
        joint.type = JointType.FIXED
        joint.parent_link_name = body_names[model.geom_bodyid[geom]]
        joint.child_link_name = link.getName()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        joint.parent_to_joint_origin_transform = Isometry3d(_transform(model.geom_pos[geom], rotation))
        if not graph.addLink(link, joint):
            raise ValueError(f'cannot project geometry {geom}')
    excluded = set(map(int, model.exclude_signature))
    for index, a in enumerate(live):
        for b in live[index + 1:]:
            ba, bb = map(int, model.geom_bodyid[[a, b]])
            wa, wb = map(int, model.body_weldid[[ba, bb]])
            parent = wa != 0 and wb != 0 and (model.body_weldid[model.body_parentid[wa]] == wb or model.body_weldid[model.body_parentid[wb]] == wa)
            masks = (model.geom_contype[a] & model.geom_conaffinity[b]) or (model.geom_contype[b] & model.geom_conaffinity[a])
            signature = (min(ba, bb) << 16) + max(ba, bb)
            allowed_contact = tuple(sorted((ba, bb))) in getattr(world, '_allowed_body_pairs', ())
            if wa == wb or parent or not masks or signature in excluded or allowed_contact or not (planned_body[ba] or planned_body[bb]):
                graph.addAllowedCollision(f'geom_{a}', f'geom_{b}', 'canonical MuJoCo filter')
    srdf = SRDFModel()
    srdf.initString(graph, '<robot name="planning_snapshot"><contact_managers_plugin_config filename="package://tesseract/support/urdf/contact_manager_plugins.yaml"/></robot>', GeneralResourceLocator())
    kin = KinematicsInformation()
    kin.addJointGroup('manipulator', world.joint_names)
    srdf.kinematics_information = kin
    env = Environment()
    if not env.init(graph, srdf):
        raise ValueError('Tesseract environment initialization failed')
    # Verify every projected body at held state and two nonzero planned poses.
    for q in (np.zeros(len(world.joint_names)), world.lower * .25 + world.upper * .75,
              world.lower * .75 + world.upper * .25):
        world._fk(q)
        env.setState(world.joint_names, q)
        transforms = env.getState().link_transforms
        for body, name in enumerate(body_names):
            if not np.allclose(transforms[name].matrix, _transform(data.xpos[body], data.xmat[body]), atol=1e-9, rtol=0):
                raise ValueError(f'projected kinematics disagree for body {body}')
    return env
