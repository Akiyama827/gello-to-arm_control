"""MuJoCo collision oracle (self + environment) for planning validity.

Provides the ``collision_fn`` that ``OMPLPlanner`` expects, backed by MuJoCo
contact queries over the URDF's collision geometry plus optional static
environment boxes on the world body. MuJoCo parses the URDF directly (STL
meshes included) and convex-hulls every collision mesh — the same conservative
envelope the old Drake OBJ-conversion path produced, without the conversion.

The planning model is the sim engine's own geometry: the plan world and the
MuJoCo twin can never disagree about what a mesh hulls to.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF-convention rpy (extrinsic xyz, R = Rz·Ry·Rx) -> wxyz quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ]
    )

def build_planning_model(
    urdf_path: str | Path, cache_dir: str | Path, *, keep_visual: bool = False
) -> Path:
    """Write a URDF copy carrying the ``<mujoco>`` compiler extension, cached.

    MuJoCo's URDF loader strips mesh paths to basenames by default, so it needs a
    meshdir. Two cases:

    - all refs resolve into ONE directory -> ``meshdir`` + ``strippath="true"``
      (the DM assembler arm, and how this always worked)
    - refs span several directories AND are all relative -> keep the paths
      (``strippath="false"``) with ``meshdir`` at the URDF's own directory, so
      ``../meshes/{collision,visual}/...`` resolves as written. The FR3 needs
      this: Franka splits collision STLs from visual meshes.

    ``package://`` refs cannot use the second form (MuJoCo does not resolve ROS
    package URIs), so a multi-directory package URDF is still an error.
    The default ``discardvisual`` keeps the plan model collision-only;
    ``keep_visual`` keeps visual geometry for sim/viewer use of the same loader.
    """
    urdf_path = Path(urdf_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "sim" if keep_visual else "planning"
    out = cache_dir / f"{urdf_path.stem}_mj_{suffix}.urdf"
    if out.exists() and out.stat().st_mtime >= urdf_path.stat().st_mtime:
        return out
    text = urdf_path.read_text()
    package_root = urdf_path.parent.parent
    mesh_refs = set(re.findall(r'filename="([^"]+\.(?:STL|stl|obj|OBJ))"', text))
    mesh_dirs = set()
    for ref in mesh_refs:
        rel = re.sub(r"^package://[^/]+/", "", ref)
        path = (
            (package_root / rel) if ref.startswith("package://") else (urdf_path.parent / rel)
        )
        # Keep the package path here instead of resolving symlinks. Modular
        # packages intentionally symlink common dock plates from one shared
        # directory; MuJoCo can open those links from the package meshdir.
        mesh_dirs.add(path.absolute().parent)
    keep_paths = False
    if len(mesh_dirs) > 1:
        if any(ref.startswith("package://") for ref in mesh_refs):
            raise ValueError(
                f"{urdf_path}: meshes span {sorted(map(str, mesh_dirs))} and use "
                "package:// refs — MuJoCo resolves neither. Stage them into one "
                "directory or rewrite the refs relative to the URDF."
            )
        # Relative refs across several dirs: keep them and anchor meshdir at the
        # URDF, rather than flattening the tree just to satisfy strippath.
        keep_paths = True
    discard = "false" if keep_visual else "true"
    # MuJoCo decodes STL/OBJ/MSH only, so a URDF whose visuals are e.g. COLLADA
    # would fail the compile with "no decoder found for mesh file". Fall back to
    # collision-only geometry — the engine convex-hulls it, which is what the DM
    # arm already relies on. (For the FR3 this no longer triggers:
    # tools/assets/setup_fr3.py converts Franka's .dae visuals to .stl at
    # staging time, because Rerun cannot read COLLADA either.)
    if keep_visual:
        undecodable = sorted(
            {
                Path(ref).suffix.lower()
                for ref in re.findall(r'filename="([^"]+)"', text)
                if Path(ref).suffix and Path(ref).suffix.lower() not in
                (".stl", ".obj", ".msh")
            }
        )
        if undecodable:
            discard = "true"
            print(
                f"[mujoco] {urdf_path.name}: visual meshes are "
                f"{', '.join(undecodable)} which MuJoCo cannot decode — loading "
                "collision geometry only",
                flush=True,
            )
    ext = f'<mujoco><compiler balanceinertia="true" discardvisual="{discard}"'
    if keep_paths:
        ext += f' meshdir="{urdf_path.parent}" strippath="false"'
    elif mesh_dirs:
        ext += f' meshdir="{mesh_dirs.pop()}" strippath="true"'
    ext += "/></mujoco>"
    m = re.search(r"<robot[^>]*>", text)
    if m is None:
        raise ValueError(f"{urdf_path}: no <robot> element")
    out.write_text(text[: m.end()] + "\n  " + ext + text[m.end() :])
    return out


class MuJoCoCollisionWorld:
    """Collision queries (self + environment) over a URDF's single-dof joints.

    ``planned_joints`` are the actively planned dofs; every other dof is held
    at ``held_positions`` (default zero — e.g. the gripper fingers OPEN, the
    conservative envelope). ``environment`` adds static boxes (table, walls)
    in the arm-base frame checked against every robot body at zero padding.
    ``cloud_obstacles`` pre-allocates a pose-settable pool of voxel
    boxes (``set_cloud_obstacles`` / ``enable_cloud_obstacles``) so a
    perceived module can become a planning obstacle without rebuilding the
    model. ``in_collision`` matches the ``OMPLPlanner`` collision_fn contract.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        planned_joints: list[str],
        *,
        cache_dir: str | Path,
        self_collision_padding_m: float = -0.002,
        environment: list[dict] | None = None,
        cloud_obstacles: dict | None = None,
        held_positions: dict[str, float] | None = None,
        ignore_bodies: list[str] | None = None,
    ) -> None:
        if self_collision_padding_m > 0.0:
            raise ValueError(
                "positive self_collision_padding_m needs geom margins — only "
                "zero/negative (tolerated hull overlap) is supported"
            )
        self.plan_urdf = build_planning_model(urdf_path, cache_dir)
        # from_string, not from_file: MjSpec's file decoder keys on extension
        # and refuses .urdf, while the parser itself handles URDF content fine
        # (meshdir in the injected <mujoco> extension is absolute).
        urdf_text = self.plan_urdf.read_text()

        def _build():
          spec = mujoco.MjSpec.from_string(urdf_text)
          toggleable_geoms: list[str] = []
          for box in environment or []:
              name = str(box["name"])
              pose = [float(v) for v in box["pose"]]
              if len(pose) != 6:
                  raise ValueError(f"environment {name!r}: pose needs 6 values")
              if box.get("mesh"):
                  # MESH obstacle (a scene body's real geometry). MuJoCo collides
                  # the mesh's CONVEX HULL, so per-part meshes are far tighter
                  # than one box per part without being optimistic: a hull always
                  # contains its mesh.
                  #
                  # `pose` places the RAW FILE — the same transform the viewers
                  # draw with, so the obstacle and the picture cannot drift
                  # apart. MuJoCo recenters mesh assets (mesh_pos/mesh_quat) but
                  # BAKES that into geom_pos at compile, so the raw transform is
                  # exactly what belongs here. Composing it by hand first
                  # double-applies the recentering and lands the hull ~28 mm off
                  # with no error raised anywhere (2026-08-06 — caught only by
                  # comparing mesh centroids against the source scene model).
                  spec.add_mesh(name=f"envmesh_{name}", file=str(box["mesh"]))
                  spec.worldbody.add_geom(
                      name=f"env_{name}",
                      type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f"envmesh_{name}",
                      pos=pose[:3],
                      quat=_rpy_to_quat(*pose[3:]),
                  )
              else:
                  size = [float(v) for v in box["size"]]
                  if len(size) != 3:
                      raise ValueError(f"environment box {name!r}: size needs 3 values")
                  spec.worldbody.add_geom(
                      name=f"env_{name}",
                      type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[s / 2.0 for s in size],  # MuJoCo sizes are half-extents
                      pos=pose[:3],
                      quat=_rpy_to_quat(*pose[3:]),
                  )
              # Phase-scoped obstacles (the dock the arm must eventually enter).
              # The table is NOT toggleable and never should be.
              if box.get("toggleable"):
                  toggleable_geoms.append(f"env_{name}")
          robot_bodies = [b.name for b in spec.bodies if b.name not in ("world", "")]
          for body_name in ignore_bodies or []:
              # Bodies whose fat hulls phantom-collide at legitimate postures
              # (the open gripper fingers graze the wrist hulls permanently).
              # Robot-robot pairs are excluded; environment pairs stay checked —
              # a finger really can hit the table.
              if str(body_name) not in robot_bodies:
                  raise ValueError(f"ignore_bodies: unknown body {body_name!r}")
              for other in robot_bodies:
                  if other != str(body_name):
                      spec.add_exclude(
                          bodyname1=str(body_name), bodyname2=other
                      )
          # Perception-cloud obstacle pool (P1): pre-allocated MOCAP bodies, not
          # plain worldbody geoms. MuJoCo caches a worldbody-attached geom's
          # broadphase bounding volume at COMPILE time — mutating geom_pos
          # after compiling moves the geom for FK/rendering but mj_collision
          # never sees it move, so it can never be repositioned live. Mocap
          # bodies are MuJoCo's zero-DOF "kinematic prop" mechanism (position
          # lives in data.mocap_pos, re-read every mj_kinematics) and collide
          # correctly. Added AFTER robot_bodies/ignore_bodies above so the pool
          # is never a candidate for an ignore_bodies exclude — it must behave
          # like an environment box (collides with everything, ignored bodies
          # included). Geoms compile at the contype/conaffinity DEFAULT (1, 1):
          # this planning URDF sets discardvisual="true" and MuJoCo prunes any
          # (0, 0) geom at compile time, so a pool created pre-disabled would
          # compile to nothing — forced to (0, 0) AFTER compiling instead
          # (below). Named ``env_cloud_NNN`` (not the "cloud_NNN" public name)
          # so the env_-prefix contact scan below catches them for free.
          cloud_geom_names: list[str] = []
          cloud_body_names: list[str] = []
          if cloud_obstacles is not None:
              self._cloud_max = int(cloud_obstacles.get("max_voxels", 200))
              self._cloud_voxel_m = float(cloud_obstacles.get("voxel_m", 0.03))
              self._cloud_z_min = float(cloud_obstacles.get("z_min", 0.02))
              margin_m = float(cloud_obstacles.get("margin_m", 0.005))
              half = self._cloud_voxel_m / 2.0 + margin_m
              width = max(3, len(str(self._cloud_max)))
              for i in range(self._cloud_max):
                  body_name = f"mocap_cloud_{i:0{width}d}"
                  geom_name = f"env_cloud_{i:0{width}d}"
                  body = spec.worldbody.add_body(
                      name=body_name, mocap=True, pos=[0.0, 0.0, -1.0]
                  )
                  body.add_geom(
                      name=geom_name,
                      type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[half, half, half],
                  )
                  cloud_body_names.append(body_name)
                  cloud_geom_names.append(geom_name)

          return spec, toggleable_geoms, cloud_geom_names, cloud_body_names

        spec, toggleable_geoms, cloud_geom_names, cloud_body_names = _build()
        self._scene_names = list(toggleable_geoms)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self._scene_gids = (
            np.array([self.model.geom(n).id for n in toggleable_geoms], dtype=int)
            if toggleable_geoms
            else None
        )
        self._cloud_gids: np.ndarray | None = None
        self._cloud_mocap_ids: np.ndarray | None = None
        self._cloud_live_k = 0
        if cloud_obstacles is not None:
            self._cloud_gids = np.array(
                [self.model.geom(n).id for n in cloud_geom_names], dtype=int
            )
            self._cloud_mocap_ids = np.array(
                [self.model.body(n).mocapid[0] for n in cloud_body_names], dtype=int
            )
            self.model.geom_contype[self._cloud_gids] = 0
            self.model.geom_conaffinity[self._cloud_gids] = 0

        self._qadr = []
        lower, upper = [], []
        for name in planned_joints:
            joint = self.model.joint(str(name))
            if joint.qposadr.size != 1 or int(self.model.jnt_type[joint.id]) not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(f"joint {name!r} is not single-dof")
            self._qadr.append(int(joint.qposadr[0]))
            lo, hi = self.model.jnt_range[joint.id]
            lower.append(float(lo))
            upper.append(float(hi))
        self._qadr = np.asarray(self._qadr, dtype=int)
        self.joint_names = list(planned_joints)
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self._held_qpos = self.model.qpos0.copy()
        self.set_held_positions(held_positions or {})
        self._pad = float(self_collision_padding_m)
        self._env_geom = np.array(
            [
                (self.model.geom(i).name or "").startswith("env_")
                for i in range(self.model.ngeom)
            ],
            dtype=bool,
        )

    @classmethod
    def from_scene(
        cls,
        scene,
        state,
        planned_actor: str,
        *,
        self_collision_padding_m: float = -0.002,
        held_positions: dict[str, float] | None = None,
        ground_z: float | None = 0.0,
    ) -> "MuJoCoCollisionWorld":
        """Build collision truth from the same generic composed workcell."""
        if self_collision_padding_m > 0.0:
            raise ValueError("positive self_collision_padding_m is unsupported")
        actor = next((item for item in scene.actors if item.name == planned_actor), None)
        if actor is None:
            raise KeyError(f"unknown planned actor: {planned_actor}")
        # Local import avoids the composer/collision helper import cycle.
        from arm_control.plants.mujoco.backend import compose_workcell_scene

        self = cls.__new__(cls)
        self.model = compose_workcell_scene(
            scene,
            state,
            ground_z=ground_z,
        ).compile()
        self.data = mujoco.MjData(self.model)
        for item in scene.actors:
            for joint, value in zip(item.joints, state.actor_q[item.name]):
                self.data.qpos[self.model.joint(f"{item.name}__{joint}").qposadr[0]] = value
        self.joint_names = [f"{actor.name}__{name}" for name in actor.joints]
        self._qadr = np.asarray(
            [int(self.model.joint(name).qposadr[0]) for name in self.joint_names], dtype=int
        )
        self.lower = np.asarray([self.model.jnt_range[self.model.joint(name).id][0] for name in self.joint_names])
        self.upper = np.asarray([self.model.jnt_range[self.model.joint(name).id][1] for name in self.joint_names])
        self._held_qpos = self.data.qpos.copy()
        self.set_held_positions(held_positions or {})
        mujoco.mj_forward(self.model, self.data)
        self._pad = float(self_collision_padding_m)
        fixture_names = tuple(fixture.name for fixture in scene.fixtures)
        fixture_prefixes = tuple(f"{name}__" for name in fixture_names)
        self._env_geom = np.asarray(
            [
                (self.model.geom(i).name or "").startswith(("ground", "obstacle__"))
                or (
                    bool(fixture_prefixes)
                    and (
                        body_name := self.model.body(self.model.geom_bodyid[i]).name or ""
                    )
                    and (
                        body_name in fixture_names
                        or body_name.startswith(fixture_prefixes)
                    )
                )
                for i in range(self.model.ngeom)
            ],
            dtype=bool,
        )
        actor_prefix = f"{actor.name}__"
        self._planned_body = np.asarray(
            [
                (self.model.body(i).name or "").startswith(actor_prefix)
                for i in range(self.model.nbody)
            ],
            dtype=bool,
        )
        self._scene_gids = np.asarray(
            [
                i
                for i in range(self.model.ngeom)
                if self.model.geom_bodyid[i] > 0
                and not self._planned_body[self.model.geom_bodyid[i]]
            ],
            dtype=int,
        )
        self._scene_names = [
            self.model.body(self.model.geom_bodyid[i]).name or ""
            for i in self._scene_gids
        ]
        self._cloud_gids = self._cloud_mocap_ids = None
        self._cloud_live_k = 0
        return self

    def set_held_positions(self, positions: dict[str, float]) -> None:
        for name, value in positions.items():
            joint = self.model.joint(str(name))
            value = float(value)
            if joint.qposadr.size != 1 or int(self.model.jnt_type[joint.id]) not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ) or not np.isfinite(value):
                raise ValueError(f"held joint {name!r} needs one finite position")
            address = int(joint.qposadr[0])
            if address in self._qadr:
                raise ValueError(f"planned joint {name!r} cannot be held")
            self._held_qpos[address] = value

    def _full_q(self, q: np.ndarray) -> np.ndarray:
        out = self._held_qpos.copy()
        out[self._qadr] = np.asarray(q, dtype=float)
        return out

    def _fk(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self._full_q(q)
        mujoco.mj_kinematics(self.model, self.data)

    def in_collision(self, q: np.ndarray) -> bool:
        """True when the configuration collides — self OR environment (OMPL fn)."""
        self._fk(q)
        mujoco.mj_collision(self.model, self.data)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if hasattr(self, "_planned_body") and not (
                self._planned_body[body1] or self._planned_body[body2]
            ):
                continue
            if self._env_geom[contact.geom1] or self._env_geom[contact.geom2]:
                return True  # environment: zero padding, any contact counts
            if contact.dist <= self._pad:
                return True  # self: deeper than the tolerated hull overlap
        return False

    def set_cloud_obstacles(self, points_base: np.ndarray) -> None:
        """Voxelize ``points_base`` (Nx3, arm-base frame — e.g. a perception
        scene cloud) into the pre-allocated ``env_cloud_*`` pool and record
        the live voxel count. Position only: this does NOT enable collision
        for them (``enable_cloud_obstacles`` does — enabling is phase
        policy, not this call's job). No-op if the pool wasn't allocated
        (``cloud_obstacles=None`` at construction).
        """
        if self._cloud_gids is None:
            return
        pts = np.asarray(points_base, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_base must be (N, 3), got shape {pts.shape}")
        pts = pts[pts[:, 2] >= self._cloud_z_min]  # floor cutoff
        keys = (
            np.unique(np.floor(pts / self._cloud_voxel_m).astype(int), axis=0)
            if pts.shape[0]
            else np.empty((0, 3), dtype=int)
        )
        pool = self._cloud_gids.size
        if keys.shape[0] > pool:
            n_before = keys.shape[0]
            # Lexicographic order on the integer voxel key -> which voxels
            # survive a truncation is deterministic, not input-order-dependent.
            order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
            keys = keys[order][:pool]
            print(
                f"[mujoco_collision] cloud_obstacles: {n_before} voxels > "
                f"{pool}-slot pool, truncated",
                flush=True,
            )
        k = int(keys.shape[0])
        if k:
            centers = (keys.astype(float) + 0.5) * self._cloud_voxel_m
            self.data.mocap_pos[self._cloud_mocap_ids[:k]] = centers
        self._cloud_live_k = k
        # ponytail: a shrinking cloud (new k < old k) leaves the now-unused
        # tail at its stale pose; harmless since enable_cloud_obstacles only
        # ever touches [0:k) — the tail stays disabled until re-enabled.

    def enable_cloud_obstacles(self, on: bool) -> None:
        """Toggle collision for the live voxels from the last
        ``set_cloud_obstacles`` call only — the rest of the pool stays
        parked and disabled either way. No-op if the pool wasn't allocated;
        safe to call before any cloud has arrived (zero live voxels)."""
        if self._cloud_gids is None:
            return
        val = 1 if on else 0
        live = self._cloud_gids[: self._cloud_live_k]
        self.model.geom_contype[live] = val
        self.model.geom_conaffinity[live] = val

    def enable_scene_obstacles(self, on: bool, exclude: tuple = ()) -> None:
        """Toggle the static scene bodies (dock, nests, parked modules).

        Phase-scoped for the same reason the cloud is: the dock is an obstacle
        for every transit leg and a TARGET for the insertion leg. Left on, the
        final approach can never be planned — the goal pose is inside the
        obstacle. No-op when the config declared no scene bodies.

        ``exclude`` names obstacles that stay OFF even when enabling — geoms
        whose name contains any of the given substrings. That is how the module
        being picked stops blocking its own approach while its NEIGHBOURS keep
        blocking: with no inventory module an obstacle at all, a reach for one
        module ploughs the hand through the next one along (measured — the
        fingers ended up 11 mm inside s2 while reaching for s0).
        """
        if self._scene_gids is None:
            return
        names = self._scene_names or []
        for gid, name in zip(self._scene_gids, names):
            val = 1 if (on and not any(x and x in name for x in exclude)) else 0
            self.model.geom_contype[gid] = val
            self.model.geom_conaffinity[gid] = val

    def preview(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Transition stub (the old meshcat animation seam) — Rerun PreviewScene
        owns trajectory animation now; nothing to do at the world layer."""


def _self_check() -> None:
    """Accept scalar joints without weakening held-joint validation."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        urdf = root / 'two_joints.urdf'
        link = '<inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
        urdf.write_text(
            '<robot name="scalar"><link name="base"/>'
            f'<link name="a">{link}</link><link name="b">{link}</link>'
            '<joint name="hinge" type="revolute"><parent link="base"/><child link="a"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            '<joint name="slide" type="prismatic"><parent link="a"/><child link="b"/>'
            '<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>'
        )
        for planned, held in (('hinge', 'slide'), ('slide', 'hinge')):
            world = MuJoCoCollisionWorld(urdf, [planned], cache_dir=root / 'cache', held_positions={held: 0.25})
            address = int(world.model.joint(held).qposadr[0])
            assert world._held_qpos[address] == 0.25
            for bad in ({planned: 0.0}, {held: float('nan')}, {held: float('inf')}):
                try:
                    world.set_held_positions(bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'accepted invalid held joints: {bad}')
    print('mujoco_collision: scalar joint guards OK')


if __name__ == '__main__':
    _self_check()
