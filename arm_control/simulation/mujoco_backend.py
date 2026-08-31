"""MuJoCo backend that mirrors the hardware command/state contract.

The backend owns an mjSpec-composed scene (assembler + 2-DOF base + free
module over a ground plane), exposes a PD-with-feedforward actuator interface
keyed on joint names, and implements grasp/dock as **weld equalities** that
are pre-declared inactive and toggled at runtime — the module physically rides
the gripper after a grasp, and the keyed dock mate engages on a release with
the connector actually AT the seat (mm lead-in gate; no magnets). (The Drake
predecessor could only bookkeep: no post-Finalize welds.)

Physical torque truth lives in the MJCF joints' ``actuatorfrcrange``
(±10/±4 N·m arm, ±22 N·m base) — the programmatically added ``motor``
actuators are wide-open and the joint-level clamp governs, mirroring how the
SDF effort limits governed the Drake scene.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from arm_control.planning.mujoco_collision import _rpy_to_quat
from arm_control.scene import SceneSpec, SceneState
from arm_control.simulation.convex_decomp import replace_with_decomposition
from arm_control.topology import AssemblyChain

# THE law, compiled once (rt/include/arm_rt/servo_law.hpp via rt/bindings).
# Optional by construction — a plain `pip install -e .` of arm_control does not
# build it — but its absence is a real divergence between the twin and the RT
# thread (the Python fallback below has NO torque clamp and NO slew limiter),
# so it warns once and loudly rather than silently.
try:  # pragma: no cover - presence depends on whether rt/bindings was built
    import arm_rt_servo as _servo
except ImportError:  # pragma: no cover
    _servo = None

_pd_fallback_warned = [False]


def _warn_python_pd_once() -> None:
    if _pd_fallback_warned[0]:
        return
    _pd_fallback_warned[0] = True
    print(
        "[mujoco_backend] WARNING: arm_rt_servo is not importable — the twin is "
        "closing a PYTHON PD instead of the compiled RT law. It has no torque "
        "clamp, no slew limiter and no Cartesian impedance, so sim results do "
        "NOT predict the RT loop. Build it: pip install -e "
        "libs/arm_control/rt/bindings",
        flush=True,
    )


class MuJoCoUnavailableError(RuntimeError):
    """Raised when the scene cannot be built (missing assets, bad config)."""


@dataclass(frozen=True)
class SceneModelSpec:
    model_path: str
    name: str
    prefix: str
    world_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ModuleSlot:
    """One inventory module: a model, a nest to sit in, and its port names.

    ``slot`` is the instance id everywhere — the composed-model prefix, the
    topology chain's instance, and the stem of the joint names the plant
    publishes. Slots are unique and never reused, so every name derived from
    one is stable across a dock (which is what lets ``MjSpec.recompile`` carry
    state over by name).
    """

    slot: str
    module_id: str                       # TYPE id; indexes module_grasps
    spec: SceneModelSpec                 # model + nest pose
    body: str                            # root body name INSIDE the model
    joints: tuple[str, ...] = ()         # driven once docked, unprefixed
    passive_site: str = "passive_connector"
    active_site: str = "active_connector"
    grasp_site: str = "grasp_frame"

    @property
    def prefix(self) -> str:
        return self.spec.prefix

    def _n(self, name: str) -> str:
        return f"{self.prefix}{name}"

    @property
    def body_name(self) -> str:
        return self._n(self.body)

    @property
    def passive_name(self) -> str:
        return self._n(self.passive_site)

    @property
    def active_name(self) -> str:
        return self._n(self.active_site)

    @property
    def grasp_name(self) -> str:
        return self._n(self.grasp_site)

    @property
    def joint_names(self) -> list[str]:
        return [self._n(j) for j in self.joints]

    @property
    def fixture_eq(self) -> str:
        return f"fixture_weld_{self.slot}"


@dataclass(frozen=True)
class MuJoCoSceneSpec:
    arm: SceneModelSpec
    base: SceneModelSpec
    modules: tuple[ModuleSlot, ...]
    timestep: float = 0.001


def _mat(quat) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


def _quat(rot: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.ascontiguousarray(rot, dtype=float).ravel())
    return out


def _site_body(spec: mujoco.MjSpec, site_name: str) -> str:
    """Name of the body a site hangs off — mjSpec has no back-pointer."""
    for body in spec.bodies:
        if any(site.name == site_name for site in body.sites):
            return body.name
    raise MuJoCoUnavailableError(f"no body carries site {site_name!r}")


def _clocking_frame(site, clocking: int) -> tuple[list, list]:
    """Mate frame in the port's parent body: the port, rolled by the key.

    ``clocking`` counts quarter turns about the port's own z (the mating
    axis), so the roll multiplies on the RIGHT of the site's orientation.
    This is the only place the discrete key becomes geometry — everywhere
    else it travels as the integer the dock's one-hot sensor reports.
    """
    angle = float(clocking) * np.pi / 2.0
    cos, sin = np.cos(angle), np.sin(angle)
    roll = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return [float(v) for v in site.pos], [
        float(v) for v in _quat(_mat(site.quat) @ roll)
    ]


def _graft_module(
    spec: mujoco.MjSpec,
    slot: "ModuleSlot",
    parent_body: str,
    port_site: str,
    clocking: int,
) -> None:
    """Attach ``slot``'s model into the chain, mated at ``port_site``.

    The module's PASSIVE port is placed on the mate frame, which is what the
    keyed connector does mechanically. Its freejoint is dropped: a docked
    module is a LINK, not a loose body held by a constraint, and that is the
    whole difference between a soft weld chain (which sags and drifts — the
    failure mode BrickSim demonstrates) and a real kinematic chain.
    """
    pos, quat = _clocking_frame(spec.site(port_site), clocking)
    frame = spec.body(parent_body).add_frame(pos=pos, quat=quat)
    child = _load_model_spec(slot.spec.model_path)
    root = child.body(slot.body)
    for joint in list(root.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            child.delete(joint)
    # Root pose = inverse of the passive site's pose in the root body, so the
    # site lands exactly on the mate frame.
    site = child.site(slot.passive_site)
    inv_q = np.zeros(4)
    mujoco.mju_negQuat(inv_q, np.asarray(site.quat, dtype=float))
    inv_p = np.zeros(3)
    mujoco.mju_rotVecQuat(inv_p, -np.asarray(site.pos, dtype=float), inv_q)
    root.pos = [float(v) for v in inv_p]
    root.quat = [float(v) for v in inv_q]
    spec.attach(child, prefix=slot.prefix, frame=frame)


def _load_model_spec(
    path: str | Path, *, cache_dir: str | Path | None = None
) -> mujoco.MjSpec:
    path = Path(path)
    if path.suffix.lower() == ".urdf":
        # Same shims as single-model mode: URDF needs the <mujoco> compiler
        # extension (meshdir, balanceinertia, undecodable-visual fallback) —
        # the FR3's raw URDF fails on all three. keep_visual: this model feeds
        # the viewer/Rerun mirror too.
        from arm_control.planning.mujoco_collision import build_planning_model

        path = build_planning_model(
            path,
            path.parent / ".mj_cache" if cache_dir is None else cache_dir,
            keep_visual=True,
        )
        return mujoco.MjSpec.from_string(path.read_text())
    if path.suffix.lower() in (".xml", ".mjcf"):
        return mujoco.MjSpec.from_file(str(path))
    raise MuJoCoUnavailableError(
        f"{path}: MuJoCo scene models must be MJCF (.xml) or URDF — SDF-era "
        "paths belong to the removed Drake stack (use models/mjcf/…)"
    )


def _inverse_pose(pos, quat) -> tuple[np.ndarray, np.ndarray]:
    inv_quat = np.empty(4)
    mujoco.mju_negQuat(inv_quat, np.asarray(quat, dtype=float))
    inv_pos = np.empty(3)
    mujoco.mju_rotVecQuat(inv_pos, -np.asarray(pos, dtype=float), inv_quat)
    return inv_pos, inv_quat


def _prefixed(name: str, local: str) -> str:
    return f"{name}__{local}"


def compose_workcell_scene(
    scene: SceneSpec,
    state: SceneState,
    *,
    timestep: float = 0.001,
    ground_z: float | None = 0.0,
) -> mujoco.MjSpec:
    """Compose arbitrary actors, separable objects, and static obstacles.

    The caller supplies every attachment frame and mating pose.  This function
    deliberately has no knowledge of application roles or attachment policy.
    """
    spec = mujoco.MjSpec()
    spec.option.timestep = float(timestep)
    if ground_z is not None:
        spec.worldbody.add_geom(
            name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[4.0, 4.0, 0.1], pos=[0.0, 0.0, float(ground_z)],
        )
    for obstacle in scene.obstacles:
        kwargs = {
            "name": f"obstacle__{obstacle.name}",
            "pos": list(obstacle.pos),
            "quat": list(_rpy_to_quat(*obstacle.rpy)),
        }
        if obstacle.shape == "box":
            kwargs.update(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[value / 2.0 for value in obstacle.size],
            )
        else:
            mesh_name = f"obstacle__{obstacle.name}__mesh"
            spec.add_mesh(name=mesh_name, file=str(obstacle.path))
            kwargs.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name)
        spec.worldbody.add_geom(**kwargs)
    for actor in scene.actors:
        child = _load_model_spec(actor.path)
        spec.attach(
            child, prefix=f"{actor.name}__",
            frame=spec.worldbody.add_frame(pos=list(actor.pos), quat=list(_rpy_to_quat(*actor.rpy))),
        )
    for obj in scene.objects:
        child = _load_model_spec(obj.path)
        attachment = state.attachments.get(obj.name)
        if attachment is not None:
            parent_site = spec.site(attachment.parent_frame)
            parent_body = spec.body(_site_body(spec, attachment.parent_frame))
            mate = parent_body.add_frame(
                pos=list(parent_site.pos), quat=list(parent_site.quat)
            ).add_frame(pos=list(attachment.mate_pose[:3]), quat=list(attachment.mate_pose[3:]))
            body = child.body(attachment.body)
            child_site = child.site(attachment.child_frame)
            for joint in list(body.joints):
                if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                    child.delete(joint)
            body.pos, body.quat = _inverse_pose(child_site.pos, child_site.quat)
            # ``child`` is attached below with this prefix.  Applying it here
            # as well double-prefixes the detached body's sites.
            mate.attach_body(body)
        storage = spec.worldbody.add_frame(
            pos=list(obj.pos), quat=list(_rpy_to_quat(*obj.rpy))
        )
        spec.attach(child, prefix=f"{obj.name}__", frame=storage)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def compose_scene(
    spec_cfg: MuJoCoSceneSpec,
    ground_z: float | None = 0.0,
    static_boxes: list[dict] | None = None,
    chain: AssemblyChain | None = None,
) -> mujoco.MjSpec:
    """mjSpec world: ground plane + arm + base + every inventory module.

    The scene is a PURE FUNCTION of the config and ``chain``: a module the
    chain lists is grafted into the base's kinematic chain at its keyed mate;
    every other module free-floats at its nest. Re-docking therefore REBUILDS
    rather than mutates — there is no incremental path that could disagree
    with the topology, and MuJoCo forbids re-attaching a prefix anyway
    (``repeated name '<prefix>_row_active' in mesh``, measured).

    Attached actuators are dropped (the backend adds plain ``motor`` actuators
    for the actively driven joints); two weld equalities are pre-declared
    inactive: (the GRASP is contact-physical — blade pads squeeze, friction
    carries; no gripper weld by user decision 2026-07-21)
    and ``dock_weld`` (dock site ↔ module passive-connector site — activating
    it IS the mechanical seat: the constraint pulls the sites coincident).
    """
    spec = mujoco.MjSpec()
    spec.option.timestep = float(spec_cfg.timestep)
    # The attached child models author elliptic cone / impratio 10 /
    # implicitfast — and mjSpec.attach DISCARDS child options (parent wins).
    # Without these the friction cone is pyramidal at impratio 1 and the
    # contact GRASP measurably cannot hold: the gripped module slid ~95 mm
    # along the pads under a gentle ramped swing at 9 N of grip force.
    spec.option.impratio = 10.0
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Regularized friction CREEPS under sustained load (measured: the gripped
    # module drifted ~2.5 mm/s along its axis under plain gravity — visible
    # as "slip" at every held gate and at lift end; friction coefficients
    # cannot stop it because the cone never saturates). The noslip post-pass
    # exists precisely to remove this drift.
    spec.option.noslip_iterations = 5
    if ground_z is not None:
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[4.0, 4.0, 0.1],
            pos=[0.0, 0.0, float(ground_z)],
            rgba=[0.55, 0.55, 0.55, 1.0],
        )
    for box in static_boxes or []:
        # Static workcell fixtures (module pedestal, table) — collide like the
        # ground under the ground-only contact policy.
        size = [float(v) / 2.0 for v in box["size"]]  # MuJoCo half-extents
        spec.worldbody.add_geom(
            name=f"static_{box['name']}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=[float(v) for v in box["pos"]],
            rgba=[0.5, 0.42, 0.35, 1.0],
        )
    docked = {m.slot: m.clocking for m in (chain.modules if chain else ())}
    for model in (spec_cfg.arm, spec_cfg.base):
        child = _load_model_spec(model.model_path)
        frame = spec.worldbody.add_frame(
            pos=list(model.world_pos), quat=list(_rpy_to_quat(*model.world_rpy))
        )
        spec.attach(child, prefix=f"{model.prefix}", frame=frame)
    for slot in spec_cfg.modules:
        if slot.slot in docked:
            continue
        child = _load_model_spec(slot.spec.model_path)
        frame = spec.worldbody.add_frame(
            pos=list(slot.spec.world_pos),
            quat=list(_rpy_to_quat(*slot.spec.world_rpy)),
        )
        spec.attach(child, prefix=slot.prefix, frame=frame)
    # Docked modules go on IN CHAIN ORDER: each mates onto a port that only
    # exists once the module before it has been attached.
    if chain is not None and chain.modules:
        by_slot = {s.slot: s for s in spec_cfg.modules}
        port = chain.root_port
        for module in chain.modules:
            slot = by_slot[module.slot]
            _graft_module(spec, slot, _site_body(spec, port), port, module.clocking)
            port = slot.active_name
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec

_QPOS_WIDTH = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
_DOF_WIDTH = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}


def _transfer_state(old_m, old_d, new_m, new_d) -> None:
    """Carry live state across a recompile, BY NAME and explicitly.

    ``MjSpec.recompile`` only maps state when handed the very spec object the
    old model came from; given a freshly built spec it silently zeroes
    everything (measured — the base went from 0.37 rad to 0.0 with no error).
    Since the whole design rebuilds the spec from the topology, the transfer is
    done here instead, which also makes what survives a dock a visible
    decision rather than undocumented behaviour:

    - every joint present in BOTH models keeps its qpos/qvel;
    - a joint that vanished (the freejoint of a module that just docked) is
      dropped — its 6 dofs have no meaning once the module is a link;
    - a joint that appeared starts at its model rest;
    - actuator ctrl carries by name, so the gripper does not fling open when
      the fingers are mid-grip on the module being docked;
    - equality activation carries by name (which fixtures have yielded).
    """
    for jid in range(new_m.njnt):
        name = mujoco.mj_id2name(new_m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        old_id = mujoco.mj_name2id(old_m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if old_id < 0 or old_m.jnt_type[old_id] != new_m.jnt_type[jid]:
            continue
        jtype = new_m.jnt_type[jid]
        nq = _QPOS_WIDTH.get(jtype, 1)
        nv = _DOF_WIDTH.get(jtype, 1)
        src, dst = old_m.jnt_qposadr[old_id], new_m.jnt_qposadr[jid]
        new_d.qpos[dst : dst + nq] = old_d.qpos[src : src + nq]
        src, dst = old_m.jnt_dofadr[old_id], new_m.jnt_dofadr[jid]
        new_d.qvel[dst : dst + nv] = old_d.qvel[src : src + nv]
    for aid in range(new_m.nu):
        name = mujoco.mj_id2name(new_m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        old_id = mujoco.mj_name2id(old_m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if old_id >= 0:
            new_d.ctrl[aid] = old_d.ctrl[old_id]
    for eid in range(new_m.neq):
        name = mujoco.mj_id2name(new_m, mujoco.mjtObj.mjOBJ_EQUALITY, eid)
        old_id = mujoco.mj_name2id(old_m, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if old_id >= 0:
            new_d.eq_active[eid] = old_d.eq_active[old_id]
    new_d.time = old_d.time


@dataclass
class MuJoCoBackend:
    """Sim plant with the old DrakeBackend's public surface.

    Two modes: a composed ``scene`` (assembler + base + free module, welds,
    ground) or a ``single_model_path`` (one MJCF/URDF, no welds/ground —
    the view/motion graphs' plant).
    """

    joint_names: list[str]  # actuated; PREFIXED in scene mode (<arm prefix>Joint1 …)
    scene: MuJoCoSceneSpec | None = None
    workcell_scene: SceneSpec | None = None
    scene_state: SceneState | None = None
    single_model_path: str | Path | None = None
    timestep: float = 0.001
    control_period: float | None = None
    ground_z: float | None = 0.0
    launch_viewer: bool = False
    enable_self_collision: bool = False
    default_joint_positions: dict[str, float] = field(default_factory=dict)
    # Seat gate for the dock weld: NO magnets — the weld models the keyed
    # mechanical mate, which only engages within its lead-in chamfer
    # (bench rung 14a measures the real lead-in).
    static_boxes: list = field(default_factory=list)
    dock_capture_m: float = 0.004
    dock_capture_deg: float = 5.0
    # REQUIRED in scene mode (validated in load()); irrelevant in single-model
    # mode. No robot-shaped defaults: these are model facts the scene config
    # states (scene.welds.*).
    ee_body: str | None = None
    # The BASE's own port — the root of the assembly chain. The mate target
    # for the next module is ``chain.tip_port``, which walks out to the tip as
    # modules are added; this never moves.
    dock_site: str | None = None
    # Scene-mode gripper: the fingers are position-servoed (the bridge maps
    # the 7th motor slot onto them). Grip contact runs on the finger MESHES
    # as SDF geoms (2026-07-22 experiment): the true printed geometry —
    # slots, ribs — carries the pinch. (History: convex finger hulls were
    # fiction that stalled approaches; flat-pad stand-ins fixed that but
    # interpenetrated visually and idealized the jam. SDF meshes are the
    # honest third iteration.)
    gripper_joints: tuple = ()
    # Grip force = kp x (ctrl - q), and ctrl clamps at full-close travel
    # (0.0439) — the squeeze depth is capped at ~0.031 m on the 62 mm module,
    # so KP is the grip-force knob, not the squeeze setting. kp 300 gave
    # ~16 N total pad normal and the module CREPT along the pad faces during
    # the 90-deg carry (user-observed sliding; MuJoCo contacts creep under
    # sustained tangential load). kp 600 -> ~31 N grip held a STATIC carry to
    # 0.1 mm/60 s (measured), but the motion legs RATCHET the module ~8 mm
    # down the blades per pick-to-dock run (acceleration transients beat the
    # margin in micro-slips; the mm dock seat then refuses the release).
    # kp 1200 -> ~56 N doubles the transient margin, still under the 45 N
    # per-finger servo cap. Rung 11 measures the real gripper.
    gripper_kp: float = 1200.0
    gripper_kd: float = 5.0
    gripper_force_n: float = 45.0  # per-finger servo cap (headroom over kp x 0.031)
    # Substrings naming this scene's grip-finger BODIES (scene.finger_body_match):
    # they get fine CoACD decomposition, module-only contact bits, and pad-force
    # readout. Empty = a scene with no grip fingers.
    finger_body_match: tuple = ()
    # Servo slew budget, N·m per plant tick. Same default as the RT server's
    # ServerConfig::slew (rt/src/server.hpp) — the sim has no better source,
    # and a twin that ramps faster than the robot flatters every transient.
    slew_per_tick: float = 1.0
    model_revision: int = 0

    model: Any = field(default=None, init=False, repr=False)
    data: Any = field(default=None, init=False, repr=False)
    viewer: Any = field(default=None, init=False, repr=False)
    _viewer_next_sync: float = field(default=0.0, init=False, repr=False)
    _qadr: np.ndarray | None = field(default=None, init=False, repr=False)
    _vadr: np.ndarray | None = field(default=None, init=False, repr=False)
    _act_id: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_command: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    chain: Any = field(default=None, init=False, repr=False)
    _slots: dict = field(default_factory=dict, init=False, repr=False)
    _joint_slot: dict = field(default_factory=dict, init=False, repr=False)
    _held: Any = field(default=None, init=False, repr=False)
    _driven: np.ndarray | None = field(default=None, init=False, repr=False)
    _applied_attachments: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_workcell_scene(
        cls,
        scene: SceneSpec,
        state: SceneState | None = None,
        **kwargs,
    ) -> "MuJoCoBackend":
        """Build a plant from the policy-free workcell representation."""
        state = state or scene.state()
        return cls(
            joint_names=[_prefixed(actor.name, joint) for actor in scene.actors for joint in actor.joints],
            workcell_scene=scene,
            scene_state=state,
            default_joint_positions={
                _prefixed(actor.name, joint): value
                for actor in scene.actors
                for joint, value in zip(actor.joints, state.actor_q[actor.name])
            },
            **kwargs,
        )

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def num_motors(self) -> int:
        return len(self.joint_names)

    @property
    def staged_slots(self) -> list:
        """Inventory modules not yet docked, in plan order."""
        if self.scene is None:
            return []
        docked = set(self.chain.slots) if self.chain is not None else set()
        return [s for s in self.scene.modules if s.slot not in docked]

    @property
    def active_slot(self):
        """The module the telemetry is ABOUT: the held one, else the next one.

        Replaces the old single ``module_body``/``module_site`` config fields.
        Falling back to the next staged module keeps the pre-grasp prints and
        the in-hand probe pointed at something real before the first pick.
        """
        if self._held is not None:
            return self._held
        staged = self.staged_slots
        return staged[0] if staged else None

    @property
    def module_body(self) -> str | None:
        slot = self.active_slot
        return slot.body_name if slot is not None else None

    @property
    def module_site(self) -> str | None:
        slot = self.active_slot
        return slot.passive_name if slot is not None else None

    @property
    def dock_target_site(self) -> str:
        """Where the NEXT module mates — the base port, or the chain's tip."""
        return self.chain.tip_port if self.chain is not None else str(self.dock_site)

    @property
    def model_path(self) -> str:
        if self.scene is not None:
            return str(self.scene.arm.model_path)
        return str(self.single_model_path)

    # -- lifecycle --------------------------------------------------------
    def _build_spec(self) -> mujoco.MjSpec:
        if self.workcell_scene is not None:
            if self.scene_state is None:
                raise MuJoCoUnavailableError("workcell scene requires scene state")
            return compose_workcell_scene(
                self.workcell_scene, self.scene_state, timestep=self.timestep,
                ground_z=self.ground_z,
            )
        if self.scene is not None:
            return compose_scene(
                self.scene, self.ground_z, self.static_boxes, chain=self.chain
            )
        if self.single_model_path is None:
            raise MuJoCoUnavailableError("need either a scene or single_model_path")
        path = Path(self.single_model_path)
        if path.suffix.lower() == ".urdf":
            # URDF needs the <mujoco> compiler extension (meshdir/strippath);
            # keep visuals — this model feeds the viewer too.
            from arm_control.planning.mujoco_collision import build_planning_model

            path = build_planning_model(
                path, path.parent / ".mj_cache", keep_visual=True
            )
            spec = mujoco.MjSpec.from_string(path.read_text())
        else:
            spec = _load_model_spec(path)
        spec.option.timestep = float(self.timestep)
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        return spec

    def load(self) -> None:
        if self.loaded:
            return
        if self.scene is not None:
            missing = [k for k in ("ee_body", "dock_site") if not getattr(self, k)]
            if missing:
                raise ValueError(
                    f"scene mode requires {missing} (scene.welds.* in the "
                    "scenario config — model facts, no default robot)"
                )
            if not self.scene.modules:
                raise ValueError(
                    "scene mode needs at least one inventory module "
                    "(scene.module or scene.inventory)"
                )
            # The topology graph is the ground truth this plant derives its
            # model from; ``dock_site`` names the base's own port, the root.
            self.chain = AssemblyChain(root_port=str(self.dock_site))
            self._slots = {s.slot: s for s in self.scene.modules}
            self._joint_slot = {
                name: slot for slot in self.scene.modules for name in slot.joint_names
            }
        self._compile()
        if self.scene is not None:
            # Every un-picked module is HELD at its nest. Bench-fixture
            # equivalent: contact-only emulation (pedestal, pocket walls) was
            # tried and always drifts/leans — the module's collision hull has a
            # rounded bottom; a real fixture's whole job is that the module does
            # not move.
            for slot in self.scene.modules:
                self._activate_body_weld(slot.fixture_eq, "world", slot.body_name)
        if self.launch_viewer:
            try:
                from mujoco import viewer as mj_viewer

                self.viewer = mj_viewer.launch_passive(self.model, self.data)
                print("[mujoco_backend] viewer launched", flush=True)
            except Exception as exc:  # headless host — sim runs fine without it
                print(f"[mujoco_backend] viewer unavailable: {exc}", flush=True)

    def _compile(self) -> None:
        """Build, compile and index the model for the CURRENT topology.

        Runs at load AND after every dock, so the initial model and every
        re-docked model come out of ONE code path — a post-dock model can
        never quietly miss a contact-bit or friction policy the first one had.
        Live state is carried across by ``_transfer_state``.
        """
        spec = self._build_spec()
        for name in self._actuated_joints(spec):
            actuator = spec.add_actuator(
                name=f"{name}_motor", target=str(name), trntype=mujoco.mjtTrn.mjTRN_JOINT
            )
            actuator.gainprm[0] = 1.0  # ctrl IS torque; joint actuatorfrcrange clamps
        # Finger servos in BOTH modes: a single-model gripper (sim motion) is
        # otherwise a free prismatic pair — gravity slides a finger to its
        # stop, and gripper commands have no actuator to land on.
        gripper_present = []
        spec_joints = {spec_joint.name for spec_joint in spec.joints}
        for name in self.gripper_joints:
            if name not in spec_joints:
                continue
            servo = spec.add_actuator(
                name=f"{name}_servo",
                target=str(name),
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
            )
            # Position servo: gain kp, bias [0, -kp, -kd].
            servo.gainprm[0] = float(self.gripper_kp)
            servo.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            servo.biasprm[0] = 0.0
            servo.biasprm[1] = -float(self.gripper_kp)
            servo.biasprm[2] = -float(self.gripper_kd)
            servo.forcerange[0] = -float(self.gripper_force_n)
            servo.forcerange[1] = float(self.gripper_force_n)
            gripper_present.append(name)
        if self.scene is not None:
            # One fixture weld per UNDOCKED module. No dock weld any more: a
            # seated module is re-grafted as a real link (see _graft), so there
            # is no constraint left to hold the mate — which is the point, an
            # equality-held chain sags and drifts under its own weight.
            for slot in self.scene.modules:
                if slot.slot in set(self.chain.slots):
                    continue
                spec.add_equality(
                    name=slot.fixture_eq,
                    type=mujoco.mjtEq.mjEQ_WELD,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,
                    name1="world",
                    name2=slot.body_name,
                    active=False,
                )
        if self.scene is not None and gripper_present:
            # Finger + module collide on their TRUE printed geometry via
            # cached convex decomposition (CoACD). Not hulls (bloat ~2 cm,
            # stalled approaches), not mesh-SDFs (need watertight input; our
            # open-shell STLs gave a corrupted field that reported -2.3 mm
            # "contact" at 19 mm true clearance — measured). Must run before
            # compile.
            cache = Path(__file__).resolve().parents[2] / ".cache" / "decomp"
            dirs = [
                Path(mdl.model_path).parent / "meshes"
                for mdl in (
                    self.scene.arm,
                    self.scene.base,
                    *(s.spec for s in self.scene.modules),
                )
            ]
            # Fingers fine (0.02 — the grip slots ARE the contact feature);
            # module coarse (0.06 — at 0.02 CoACD chased rib fillets into
            # ~1500 pieces and slivers the compiler rejects).
            if self.finger_body_match:
                replace_with_decomposition(
                    spec, bodies_containing=list(self.finger_body_match),
                    cache_dir=cache, mesh_search_dirs=dirs, threshold=0.02,
                )
            replace_with_decomposition(
                spec, bodies_containing=[s.prefix for s in self.scene.modules],
                cache_dir=cache, mesh_search_dirs=dirs, threshold=0.06,
            )
        model = spec.compile()
        data = mujoco.MjData(model)
        previous = (self.model, self.data)
        self.model, self.data = model, data
        if self.scene is not None:
            # A module's INTERNAL joint (passive-to-motor) is unactuated while
            # the module is loose, but 0.42 of its 0.49 kg hangs on it — free,
            # it turns the standing module into a pendulum that topples itself.
            # Real modules hold that joint by gearing when unpowered; emulate
            # with static friction. 5.0, not 1.0: carry-swing transients
            # back-drove the joint ~55° at 1.0 (measured), visually dangling
            # the motor half of the carried module.
            #
            # DOCKED modules are exempt: their joint is now driven by the
            # modular arm's own servo, and 5 N·m of stiction there would fight
            # every commanded motion of the assembled robot.
            driven = set(self._actuated_joints(spec))
            prefixes = tuple(s.prefix for s in self.scene.modules)
            for jid in range(self.model.njnt):
                joint = self.model.joint(jid)
                if joint.name in driven:
                    continue
                if joint.name.startswith(prefixes) and self.model.jnt_type[jid] in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ):
                    self.model.dof_frictionloss[joint.dofadr[0]] = max(
                        5.0, float(self.model.dof_frictionloss[joint.dofadr[0]])
                    )
        if self.finger_body_match:
            # condim 3 (the URDF-import default) has NO torsional term, so spin
            # about the contact normal — exactly the in-hand roll axis the dock
            # aim fights — was frictionless regardless of grip force, while the
            # bench holds roll firmly at 40 N. condim 4 turns the term on for
            # every contact involving a pad (contact condim/friction take the
            # per-pair max).
            # ponytail: 0.02 m is a POC-firm guess, not a bench measurement
            # (2026-08-04 user call: sim is POC-only) — retune if a rung needs
            # sim/bench roll parity.
            for gid in range(self.model.ngeom):
                body = self.model.body(self.model.geom_bodyid[gid]).name
                if any(tok in body for tok in self.finger_body_match):
                    self.model.geom_condim[gid] = max(
                        4, int(self.model.geom_condim[gid])
                    )
                    self.model.geom_friction[gid, 1] = max(
                        0.02, float(self.model.geom_friction[gid, 1])
                    )
        if not self.enable_self_collision:
            self._contacts_ground_only()
        for name in gripper_present:
            # The open stop is physical: without the limit, module contact
            # shoves the fingers to negative travel (outside the mechanism).
            self.model.jnt_limited[self.model.joint(name).id] = 1
        if previous[0] is None:
            mujoco.mj_resetData(self.model, self.data)
            # Spawn pose via data.qpos, NEVER model.qpos0: MuJoCo poses a hinge
            # at (qpos - qpos0), so writing spawn angles into qpos0 silently
            # shifts those joints' zero — the plant then lives in a coordinate
            # frame offset by the spawn pose while IK/RNEA speak URDF
            # coordinates.
            for name, value in (self.default_joint_positions or {}).items():
                joint = self.model.joint(str(name))
                self.data.qpos[joint.qposadr[0]] = float(value)
        else:
            _transfer_state(*previous, self.model, self.data)
        if self.workcell_scene is not None and self.scene_state is not None:
            for actor in self.workcell_scene.actors:
                for joint, value in zip(actor.joints, self.scene_state.actor_q[actor.name]):
                    self.data.qpos[self.model.joint(_prefixed(actor.name, joint)).qposadr[0]] = value
            for name, active in self.scene_state.constraints.items():
                equality = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
                if equality < 0:
                    raise ValueError(f"unknown scene constraint: {name}")
                self.data.eq_active[equality] = int(active)
            self._applied_attachments = dict(self.scene_state.attachments)
        self._cache_indices(gripper_present, first=previous[0] is None)
        mujoco.mj_forward(self.model, self.data)

    def _actuated_joints(self, spec: mujoco.MjSpec) -> list[str]:
        """Planned joints this model can actually drive, in command order.

        A module's joint is driven only once the module is DOCKED: while it
        sits in its nest it is a loose part that must be free to spin. The
        command vector keeps its full planned width regardless — see
        ``_cache_indices``.
        """
        present = {joint.name for joint in spec.joints}
        docked = set(self.chain.slots) if self.chain is not None else set()
        out = []
        for name in self.joint_names:
            slot = self._joint_slot.get(name)
            if slot is not None and slot.slot not in docked:
                continue
            if name in present:
                out.append(name)
        return out

    def _cache_indices(self, gripper_present: list[str], *, first: bool) -> None:
        """Re-derive every model-index cache. Invalid after any recompile."""
        self._qadr = np.array(
            [self.model.joint(n).qposadr[0] for n in self.joint_names], dtype=int
        )
        self._vadr = np.array(
            [self.model.joint(n).dofadr[0] for n in self.joint_names], dtype=int
        )
        # Fixed-width command/state vectors, growing actuator set: a planned
        # joint with no actuator yet (an undocked module's) is READ like any
        # other but commands to it land nowhere. Keeping the width constant
        # means the Dora contract — num_motors, arm_slices, message shapes —
        # never has to be renegotiated mid-run.
        driven = [
            i
            for i, n in enumerate(self.joint_names)
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
            >= 0
        ]
        self._driven = np.array(driven, dtype=int)
        self._act_id = np.array(
            [self.model.actuator(f"{self.joint_names[i]}_motor").id for i in driven],
            dtype=int,
        )
        # Per-joint torque limit for the servo law. The MJCF's
        # ``actuatorfrcrange`` is the authority (module docstring): MuJoCo
        # already clamps the applied force there, so mirroring it into the law
        # keeps the twin's torque identical to the RT loop's instead of merely
        # similar. A joint with no range declared gets inf — i.e. exactly the
        # unclamped behaviour it has today, not a guessed number.
        self._tau_limit = np.array(
            [
                float(self.model.jnt_actfrcrange[self.model.joint(n).id][1])
                if self.model.jnt_actfrclimited[self.model.joint(n).id]
                else np.inf
                for n in self.joint_names
            ]
        )
        self._ee_bid = int(self.model.body(self.ee_body).id) if self.ee_body else -1
        # Bodies whose external forces count as "acting on the arm" — every
        # body sharing the EE's kinematic-tree root. NOT the EE body's own
        # subtree: a TCP frame is typically a childless leaf hung off the hand
        # (the FR3's asm_fr3_hand_tcp is a SIBLING of the fingers), so its own
        # cfrc_ext is zero forever and the grip reaction would never show up.
        self._ee_tree = (
            np.flatnonzero(self.model.body_rootid == self.model.body_rootid[self._ee_bid])
            if self._ee_bid > 0
            else np.empty(0, dtype=int)
        )
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._gripper_act = [
            self.model.actuator(f"{n}_servo").id for n in gripper_present
        ]
        # The joints that ACTUALLY exist in this model, not the ones configured:
        # ``gripper_joints`` is a superset covering every arm (the DM assembler's
        # Gripper_1/2 and the FR3's fr3_finger_joint1/2), and only one pair is
        # present in any given model.
        self._gripper_names = list(gripper_present)
        # Finger travel from the MODEL, not a constant: the clamp in
        # _set_gripper_targets must match whatever gripper this scene has.
        self._gripper_range = [
            (
                float(self.model.jnt_range[self.model.joint(n).id][0]),
                float(self.model.jnt_range[self.model.joint(n).id][1]),
            )
            for n in gripper_present
        ]
        if not first:
            return
        # Servo targets start at the spawn rest (ctrl defaults to 0, which for
        # an open-at-nonzero gripper like the FR3 would slam the fingers shut).
        for act, name in zip(self._gripper_act, gripper_present):
            self.data.ctrl[act] = self.data.qpos[self.model.joint(name).qposadr[0]]
        # The spawn rest IS this gripper's OPEN: the release trigger compares
        # commanded ctrl against it (a literal `<= 0.004` assumed the DM's
        # open-at-zero and never fired for the FR3's open-at-0.04). Latched
        # ONCE, at spawn — re-deriving it after a dock would record whatever
        # the fingers happen to be doing (mid-grip!) as "open".
        self._gripper_open_ctrl = [float(self.data.ctrl[a]) for a in self._gripper_act]


    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        self.viewer = None
        self.data = None
        self.model = None

    def _contacts_ground_only(self) -> None:
        """Shadow-run contact policy: ground contact for everything, plus the
        FINGER<->module grip pairs and nothing else between bodies.

        Hull self-contact stalls execution at postures the real links clear —
        the (padded) planner owns self-collision avoidance, welds own
        grasp/dock. Grip contact runs on the finger SDF meshes (true printed
        geometry, 2026-07-22 experiment replacing the flat-pad stand-ins).
        Bit layout: world 1/2, robot 2/1, fingers 2|4 / 1|8, module 2|8 /
        1|4 — finger&module intersect (4 and 8); finger&finger,
        module&module and arm&module all stay OFF.
        """
        module_prefixes = (
            tuple(s.prefix for s in self.scene.modules)
            if self.scene is not None
            else ()
        )
        for i in range(self.model.ngeom):
            geom = self.model.geom(i)
            if geom.contype[0] == 0 and geom.conaffinity[0] == 0:
                continue  # visual-class geom — must never gain contact
            body = self.model.body(self.model.geom_bodyid[i]).name or ""
            if any(m in body for m in self.finger_body_match):
                # Grip bits: pair with the module (4/8) but not with the
                # other finger, the arm, or the module-free rest. Without
                # this the fingers inherit the generic robot bits and can
                # NEVER touch the module (2&5 = 10&1 = 0) — every grasp
                # would close on air and MISS.
                geom.contype[:] = 2 | 4
                geom.conaffinity[:] = 1 | 8
                continue
            if body.endswith(("_Base", "_Link1")) or body in ("Base", "Link1"):
                # The arm is BOLTED to the bench: its base plate resting on
                # the ground must not RUB. The plate's -0.8 mm ground contact
                # fought J1 with ~10 N.m of friction — measured: a 1.5 rad
                # J1 sweep never completes (30 s budget) with the contact,
                # 0.26 s without. This was the "super slow arm".
                geom.contype[:] = 0
                geom.conaffinity[:] = 0
                continue
            if self.model.geom_bodyid[i] == 0:  # world body: ground + fixtures
                geom.contype[:] = 1
                geom.conaffinity[:] = 2
            elif module_prefixes and body.startswith(module_prefixes):
                geom.contype[:] = 2 | 8
                geom.conaffinity[:] = 1 | 4
            else:
                geom.contype[:] = 2
                geom.conaffinity[:] = 1

    def apply_gripper_command(self, positions) -> None:
        """Servo the finger joints toward ``positions`` (clamped to travel).

        The plant takes every command verbatim: gripper OWNERSHIP (grasp gate
        vs orchestrator open/hold) is bridge policy, not plant policy."""
        self._set_gripper_targets(positions)

    def _set_gripper_targets(self, positions) -> None:
        if not getattr(self, "_gripper_act", None):
            return
        p = np.asarray(positions, dtype=float).reshape(-1)
        for act, value, (lo, hi) in zip(self._gripper_act, p, self._gripper_range):
            self.data.ctrl[act] = float(np.clip(value, lo, hi))

    def gripper_positions(self) -> np.ndarray:
        """TRUE finger joint positions (empty when the scene has no gripper).

        The bridge echoes these into the 7th motor slot — a jammed finger
        (contact holding it off its servo target) must reach the operator
        mirror, not a synthesized 'open'."""
        if not getattr(self, "_gripper_act", None):
            return np.empty(0)
        self.load()
        return np.array(
            [float(self.data.joint(str(n)).qpos[0]) for n in self._gripper_names]
        )

    def gripper_efforts(self) -> np.ndarray:
        """Per-finger servo forces in N (plant SENSING, the sim's analog of the
        DM gripper motor's torque estimate). The bridge's GraspGate thresholds
        these — grasp success is policy and does not live in the plant."""
        if not getattr(self, "_gripper_act", None):
            return np.empty(0)
        self.load()
        return np.array(
            [float(self.data.actuator_force[a]) for a in self._gripper_act]
        )

    # -- motor I/O --------------------------------------------------------
    def apply_motor_command(self, command: dict[str, np.ndarray]) -> None:
        self.load()
        n = self.num_motors
        self._last_command = {
            key: np.asarray(command.get(key, np.zeros(n)), dtype=np.float64)
            .reshape(n)
            .copy()
            for key in ("position", "velocity", "torque", "kp", "kd")
        }
        # Optional Cartesian-impedance block (messages.unpack_motor_command).
        # None = every graph that does not send one behaves exactly as before.
        self._last_command["cartesian"] = command.get("cartesian")

    def _cartesian_tau_ff(self, tau_ff: np.ndarray, cart: dict) -> np.ndarray:
        """Fold the task-frame Cartesian impedance into tau_ff.

        The twin computes J and the EE pose ITSELF, from MuJoCo, exactly as
        the RT loop computes them from libfranka — the PC only ever sends the
        target and the stiffness. Keeping that split honest here is the point:
        a sim that took a PC-computed Jacobian would not be testing the thing
        the robot actually runs.
        """
        if self._ee_bid < 0 or _servo is None:
            return tau_ff  # no EE configured, or no compiled law to call
        mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self._ee_bid)
        # 6 x n row-major, ACTUATED columns only — the layout servo_law.hpp
        # documents. Rows the arm cannot drive would just be noise in J^T f.
        jac = np.vstack((self._jacp[:, self._vadr], self._jacr[:, self._vadr]))
        body = self.data.body(self._ee_bid)
        return _servo.cartesian_torque(
            tau_ff=tau_ff,
            J=jac,
            R_task=cart["task_R"],
            x=body.xpos,
            quat=body.xquat,  # MuJoCo quats are [w,x,y,z], like the law's
            x_des=cart["pose"][:3],
            quat_des=cart["pose"][3:7],
            # Twist from the SAME J, so the damping term can never disagree
            # with the stiffness term about what the EE is doing.
            twist=jac @ self.data.qvel[self._vadr],
            # ponytail: desired twist is zero — the dock target crawls
            # (speed_scale 0.3 over 65 mm), so D_c * v_des is well under a
            # newton. Send a real one if fast Cartesian legs ever appear.
            twist_des=np.zeros(6),
            kc=cart["kc"],
            dc=cart["dc"],
        )

    def _apply_pd(self) -> None:
        """One servo tick: THE compiled RT law, or a warned Python fallback.

        Calling ``arm_rt_servo.servo_torque`` is the whole point of the pybind
        module — the twin then closes the byte-identical law the RT thread
        runs, INCLUDING the per-joint torque clamp and the slew limiter that
        the old Python PD here silently omitted.
        """
        if self._last_command is None:
            return
        cmd = self._last_command
        q = self.data.qpos[self._qadr]
        v = self.data.qvel[self._vadr]
        tau_ff = cmd["torque"]
        cart = cmd.get("cartesian")
        if cart is not None:
            tau_ff = self._cartesian_tau_ff(tau_ff, cart)
        # Torque is computed at the FULL planned width and written only to the
        # joints that currently have an actuator: a module still in its nest is
        # commanded (the bridge does not know or care) and that command lands
        # nowhere, which is exactly right — it is a loose part.
        if _servo is None:
            _warn_python_pd_once()
            tau = (
                tau_ff
                + cmd["kp"] * (cmd["position"] - q)
                + cmd["kd"] * (cmd["velocity"] - v)
            )
            self.data.ctrl[self._act_id] = tau[self._driven]
            return
        # tau_ref is the plant's echo of the last ACCEPTED torque. On the FR3
        # that is state.tau_J_d; here it is the previous tick's applied ctrl
        # for these actuators — same meaning, and the fancy index hands back a
        # copy, so it cannot alias the assignment below.
        tau_ref = np.zeros(self.num_motors)
        tau_ref[self._driven] = self.data.ctrl[self._act_id]
        self.data.ctrl[self._act_id] = _servo.servo_torque(
            q=q,
            dq=v,
            q_des=cmd["position"],
            qd_des=cmd["velocity"],
            tau_ff=tau_ff,
            kp=cmd["kp"],
            kd=cmd["kd"],
            tau_ref=tau_ref,
            tau_limit=self._tau_limit,
            slew_per_tick=float(self.slew_per_tick),
        )[self._driven]

    def ee_wrench(self) -> np.ndarray:
        """External wrench on the EE body, WORLD frame, ``[fx,fy,fz,tx,ty,tz]``.

        The sim's analogue of the FR3's ``O_F_ext_hat_K`` — same frame (base /
        world), same sign (POSITIVE = the robot pushing on the world), so a
        consumer cannot tell the two plants apart.

        Source is ``mj_rnePostConstraint`` -> ``cfrc_ext``, NOT a sum over
        ``mj_contactForce``. Two reasons: cfrc_ext includes EQUALITY-constraint
        reactions, and in this scene the grasp fixture and the keyed dock mate
        ARE equalities while module<->dock contact pairs are masked off by the
        ground-only contact policy — a contact sum would read ~0 through the
        entire insertion. And cfrc_ext is a whole-body external total, which is
        structurally the quantity the FR3 estimates, rather than the per-pad
        normal that ``_pad_forces`` already answers.

        cfrc_ext is [torque; force] about the subtree CoM, so this reorders to
        force-first and shifts the moment to the EE body origin.
        """
        if self.data is None or self._ee_bid < 0 or not self._ee_tree.size:
            return np.zeros(6)
        # cfrc_ext is only meaningful after this call — mj_step leaves it stale
        # unless a sensor happened to need it.
        mujoco.mj_rnePostConstraint(self.model, self.data)
        cfrc = self.data.cfrc_ext[self._ee_tree].sum(axis=0)
        force = -cfrc[3:6]  # negate: cfrc_ext is world-ON-robot, libfranka is
        torque = -cfrc[0:3]  # robot-ON-world
        ref = self.data.subtree_com[self.model.body_rootid[self._ee_bid]]
        # Move the moment from `ref` to the EE origin: t_B = t_A + (A - B) x f
        torque = torque + np.cross(ref - self.data.body(self._ee_bid).xpos, force)
        return np.concatenate((force, torque))

    def step(self, command: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        self.load()
        if command is not None:
            self.apply_motor_command(command)
        # Re-close the PD loop every plant timestep within the control period —
        # ZOH across a slow command period limit-cycles the low-inertia wrist
        # (the DM firmware recomputes PD at multi-kHz under a 100 Hz stream).
        duration = self.control_period if self.control_period else self.model.opt.timestep
        steps = max(1, int(round(duration / self.model.opt.timestep)))
        for _ in range(steps):
            self._apply_pd()
            mujoco.mj_step(self.model, self.data)
        if self.scene is not None and getattr(self, "_gripper_act", None):
            # SENSOR-LOCAL topology physics — the plant senses and reacts to
            # its own world; the grasp PROTOCOL (close sequencing, success,
            # drop events) lives in the bridge's GraspGate:
            # - a nest fixture YIELDS once the gripper truly squeezes that
            #   module (a fixture's magnets give way to the hand's pull);
            # - the keyed mate COMMITS when the hand lets go with the
            #   connector actually at the seat (mm lead-in, no magnets), and
            #   committing re-grafts the module as a driven link.
            # "The hand is closing" — commanded away from its open stop. The
            # fixture-yield below is gated on it because a pad touch alone is
            # not a grasp: with several modules in the scene the arm BRUSHES a
            # neighbour in transit and the old first-touch rule handed that
            # module over (measured: reaching for s0 picked up s2, which the
            # open hand then immediately "released unseated"). Same test the
            # release branch uses, so the two can never disagree.
            hand_open = all(
                abs(float(self.data.ctrl[a]) - o) <= 0.004
                for a, o in zip(self._gripper_act, self._gripper_open_ctrl)
            )
            if self._held is None and not hand_open:
                # Yield at FIRST pad touch: the weld's only job is pre-grasp
                # presentation (the rounded hull drifts if the module just
                # stands). A real fixture seats the module magnetically — a
                # lateral push slides it off almost immediately, and the freed
                # module stands on the pedestal and SELF-CENTERS between the
                # closing pads. Any force threshold loses this race: with the
                # module rigidly anchored, the first (one-sided) contact twists
                # the soft wrist away — 8 deg at the old 10 N gate, still 7 deg
                # at a 3 N both-pad gate (measured live) — and the module is
                # handed over tilted, carried 24 mm off-nominal, and the mm
                # dock seat then rightly refuses the release.
                for slot in self.staged_slots:
                    eq = self._eq_id(slot.fixture_eq)
                    if eq < 0 or not self.data.eq_active[eq]:
                        continue
                    grip_n = float(sum(self._pad_forces(slot).values()))
                    if grip_n < 0.5:
                        continue
                    self._set_weld_active(slot.fixture_eq, False)
                    self._held = slot
                    ee_p = self._body_T(self.ee_body)[0]
                    mod_p = self._body_T(slot.body_name)[0]
                    conn_p = self.data.site(slot.passive_name).xpos
                    print(
                        f"[mujoco_backend] fixture yielded to the grip "
                        f"({grip_n:.1f} N) — {slot.slot} rides the hand; "
                        f"conn-in-EE {self._conn_in_ee_mm()} mm "
                        f"EE {np.round(ee_p * 1e3, 1).tolist()} "
                        f"mod {np.round(mod_p * 1e3, 1).tolist()} "
                        f"conn {np.round(conn_p * 1e3, 1).tolist()}",
                        flush=True,
                    )
                    break
            # "The hand let go" is a COMMANDED fact, not residual contact: at
            # the dock orientation gravity runs parallel to the pad faces, so
            # a released module SLIDES down the opening fingers and WEDGES
            # tilted between them with >2 N of contact — a force-decay
            # trigger then never fires and the module hangs stuck
            # (user-observed). Commanded-open fires at the seat, before the
            # module can slide. The 0.25 m band covers true mid-carry drops
            # (module gone while still commanded closed) for topology truth.
            elif self._held is not None and (
                hand_open
                or float(
                    np.linalg.norm(
                        self._body_T(self.ee_body)[0]
                        - self.data.site(self._held.grasp_name).xpos
                    )
                )
                > 0.25
            ):
                slot = self._held
                seated, why = self._dock_within_capture(slot)
                if seated:
                    self._commit_dock(slot)
                else:
                    # Hand open with the connector NOT seated: no magnets to
                    # forgive it — the module leaves the pads under plain
                    # physics. Topology reflects the fact; the bridge's gate
                    # reports the LOST/MISSED event.
                    self._held = None
                    print(
                        f"[mujoco_backend] released unseated: "
                        f"{slot.slot} free ({why})",
                        flush=True,
                    )
            if self._held is not None:
                # In-hand drift telemetry (~2 Hz). Standalone block: wedging
                # an `if` into the yield/release chain above re-bound the
                # release `elif` and silently disabled it (live-caught).
                self._grip_log_steps = getattr(self, "_grip_log_steps", 0) + 1
                if self._grip_log_steps % 100 == 0:
                    w = self.ee_wrench()
                    grip_n = float(sum(self._pad_forces(self._held).values()))
                    print(
                        f"[mujoco_backend] carry: conn-in-EE "
                        f"{self._conn_in_ee_mm()} mm grip {grip_n:.1f} N "
                        f"ee-wrench |f| {np.linalg.norm(w[:3]):.1f} N "
                        f"|t| {np.linalg.norm(w[3:]):.2f} N.m",
                        flush=True,
                    )
        if self.viewer is not None:
            # Sync at most ~15 Hz: sync() takes 14-21 ms under the GUI mutex
            # on this composed scene (measured), so 60 Hz still burned ~0.9 s
            # of every wall second and the sim fell ~25% behind real time —
            # the wall-clock executor then finished legs while the arm was up
            # to a radian behind, and the terminal PD yank whipped the wrist
            # (18 rad/s measured; flung the carried module). 15 Hz keeps the
            # physics real-time; the window is a debug view, not the mirror.
            now = time.monotonic()
            if now >= self._viewer_next_sync:
                self._viewer_next_sync = now + 1.0 / 15.0
                try:
                    self.viewer.sync()
                except Exception:
                    self.viewer = None  # window closed — keep simulating
        return self.motor_state()

    def motor_state(self) -> dict[str, np.ndarray]:
        self.load()
        position_cmd = (
            self._last_command["position"].copy()
            if self._last_command is not None
            else np.zeros(self.num_motors)
        )
        return {
            "position": self.data.qpos[self._qadr].copy(),
            "velocity": self.data.qvel[self._vadr].copy(),
            "position_cmd": position_cmd,
        }

    def apply_scene_state(self, new_state: SceneState) -> None:
        """Atomically rebuild the generic scene from a new runtime overlay."""
        if self.workcell_scene is None:
            raise RuntimeError("apply_scene_state requires a workcell scene")
        self.load()
        released: dict[str, tuple[Any, np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, attachment in self._applied_attachments.items():
            if name not in new_state.attachments:
                body = self.data.body(_prefixed(name, attachment.body))
                released[name] = (
                    attachment, body.xpos.copy(), body.xquat.copy(),
                    self.data.cvel[body.id].copy(),
                )
        self.scene_state = new_state
        self.model_revision += 1
        self._compile()
        for name, (attachment, pos, quat, cvel) in released.items():
            # Object-local free-joint names are intentionally not prescribed;
            # find the root free joint under the named detachable body.
            for joint_id in range(self.model.njnt):
                if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                joint = self.model.joint(joint_id)
                if joint.name.startswith(f"{name}__"):
                    qadr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
                    self.data.qpos[qadr : qadr + 3] = pos
                    self.data.qpos[qadr + 3 : qadr + 7] = quat
                    self.data.qvel[dadr : dadr + 3] = cvel[3:]
                    self.data.qvel[dadr + 3 : dadr + 6] = cvel[:3]
                    break
        mujoco.mj_forward(self.model, self.data)

    def frame_pose(self, name: str) -> list[float]:
        self.load()
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise KeyError(f"unknown scene frame: {name}")
        site = self.data.site(site_id)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, site.xmat)
        return [*map(float, site.xpos), *map(float, quat)]

    def body_pose(self, name: str) -> list[float]:
        self.load()
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"unknown scene body: {name}")
        body = self.data.body(body_id)
        return [*map(float, body.xpos), *map(float, body.xquat)]

    def set_constraint(self, name: str, active: bool) -> None:
        if self.scene_state is None:
            raise RuntimeError("set_constraint requires a workcell scene")
        self.load()
        equality = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name
        )
        if equality < 0:
            raise KeyError(f"unknown scene constraint: {name}")
        self.scene_state.set_constraint(name, active)
        self.data.eq_active[equality] = int(active)
        mujoco.mj_forward(self.model, self.data)

    def constraint_active(self, name: str) -> bool:
        """Return the current state of one named equality constraint."""
        self.load()
        equality = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name
        )
        if equality < 0:
            raise KeyError(f"unknown scene constraint: {name}")
        return bool(self.data.eq_active[equality])

    def equality_reaction(self, name: str) -> np.ndarray:
        """Return a named equality's translational then rotational reactions."""
        self.load()
        equality = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name
        )
        if equality < 0:
            raise KeyError(f"unknown scene constraint: {name}")
        if not self.data.eq_active[equality]:
            return np.zeros(6)
        rows = np.flatnonzero(
            (self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            & (self.data.efc_id == equality)
        )
        result = np.zeros(6)
        count = min(6, len(rows))
        result[:count] = self.data.efc_force[rows[:count]]
        return result

    def release_constraint_if_force_exceeds(
        self,
        name: str,
        axis_world,
        threshold_n: float,
    ) -> bool:
        """Disable a named equality once its axial reaction exceeds a threshold."""
        axis = np.asarray(axis_world, dtype=float).ravel()
        norm = float(np.linalg.norm(axis))
        threshold = float(threshold_n)
        if axis.shape != (3,) or not np.isfinite(axis).all() or norm <= 0.0:
            raise ValueError("axis_world must be a finite non-zero 3-vector")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold_n must be positive and finite")
        if not self.constraint_active(name):
            return False
        axial = abs(float(self.equality_reaction(name)[:3] @ (axis / norm)))
        if axial <= threshold:
            return False
        self.set_constraint(name, False)
        return True

    def aggregate_inertial(self, body_names: list[str]) -> dict[str, list[float] | float]:
        self.load()
        bodies = [self.data.body(name) for name in body_names]
        masses = np.asarray([self.model.body_mass[body.id] for body in bodies])
        mass = float(masses.sum())
        if mass <= 0.0:
            raise ValueError("aggregate bodies must have positive total mass")
        com = sum((m * body.xipos for m, body in zip(masses, bodies)), np.zeros(3)) / mass
        inertia = np.zeros((3, 3))
        for m, body in zip(masses, bodies):
            local = np.diag(self.model.body_inertia[body.id])
            rot = body.ximat.reshape(3, 3)
            delta = body.xipos - com
            inertia += rot @ local @ rot.T + m * (delta @ delta * np.eye(3) - np.outer(delta, delta))
        return {"mass": mass, "com": com.tolist(), "inertia": inertia.tolist()}

    # -- weld topology ----------------------------------------------------
    def _eq_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def _set_weld_active(self, name: str, active: bool) -> None:
        self.data.eq_active[self._eq_id(name)] = 1 if active else 0

    def _commit_dock(self, slot) -> None:
        """Commit the mate: record the edge, then rebuild the robot around it.

        The clocking is READ from the seated pose rather than assumed — the
        arm placed the module, so which of the four keys it actually landed on
        is a measurement. Snapping to the nearest key is BrickSim's move: the
        continuous pose is discarded the moment it has told us which discrete
        mate it is, so nothing downstream can accumulate drift.
        """
        clocking = self._seated_clocking(slot)
        port = self.dock_target_site
        self.chain.attach(
            slot=slot.slot,
            module_id=slot.module_id,
            clocking=clocking,
            active_port=slot.active_name,
        )
        self._held = None
        self.model_revision += 1
        self._compile()
        print(
            f"[mujoco_backend] docked {slot.slot} ({slot.module_id}) onto "
            f"{port} at clocking {clocking} ({clocking * 90}°) — chain is now "
            f"{'+'.join(self.chain.slots)}, next port {self.dock_target_site}, "
            f"rev {self.model_revision}",
            flush=True,
        )

    def _seated_clocking(self, slot) -> int:
        """Nearest keyed clocking of the module as currently presented.

        Roll about the mate axis, port x-axis to module x-axis, quantized to
        the four keys. On the bench this integer comes from the dock MCU's
        one-hot sensor word instead (``topology.clocking_from_onehot``) — same
        quantity, so sim and bench topology are directly comparable.
        """
        port = self.data.site(self.dock_target_site)
        mod = self.data.site(slot.passive_name)
        p_R = port.xmat.reshape(3, 3)
        m_R = mod.xmat.reshape(3, 3)
        angle = float(np.arctan2(m_R[:, 0] @ p_R[:, 1], m_R[:, 0] @ p_R[:, 0]))
        return int(round(angle / (np.pi / 2))) % 4

    def _body_T(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.data.body(name)
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()

    def _activate_body_weld(self, name: str, body1: str, body2: str) -> None:
        """Activate a body weld holding body2 at its CURRENT pose in body1."""
        eq = self._eq_id(name)
        p1, R1 = self._body_T(body1)
        p2, R2 = self._body_T(body2)
        rel_R = R1.T @ R2
        rel_p = R1.T @ (p2 - p1)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rel_R.ravel())
        data = self.model.eq_data[eq]
        data[:] = 0.0
        data[3:6] = rel_p
        data[6:10] = quat
        # torquescale 20: at 1.0 the weld's ROTATIONAL channel is so soft the
        # carried module leans 30-55° under its own gravity moment (measured;
        # the historical "carried-lean 28-34°" was THIS, not pedestal rock —
        # at 55° the new mate-true dock gate rightly rejected the release).
        # 20 holds the carry to ~1°.
        data[10] = 20.0
        self._set_weld_active(name, True)

    def inhand_pose(self) -> list | None:
        """Module pose in the EE-body frame, [x,y,z,qw,qx,qy,qz] (twin truth).

        The sim source for the in-hand monitor: on the bench the wrist
        camera's end-cap tag re-read supplies the same measurement. None in
        single-model mode (no module in the world).
        """
        if self.scene is None or self.data is None or self.module_body is None:
            return None
        p_ee, R_ee = self._body_T(self.ee_body)
        p_mod, R_mod = self._body_T(self.module_body)
        rel_R = R_ee.T @ R_mod
        rel_p = R_ee.T @ (p_mod - p_ee)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rel_R.ravel())
        return [float(v) for v in (*rel_p, *quat)]

    def connector_in_ee(self) -> list | None:
        """Carried module's PASSIVE port pose in the EE frame, [xyz, wxyz].

        The quantity that turns a dock port into an EE target: the arm must
        put THIS frame on the port. Sim reads it from the twin; the bench
        reads the same transform from the wrist camera's end-cap tag, so the
        dock leg is derived identically on both.
        """
        if self.data is None or self._held is None:
            return None
        p_ee, R_ee = self._body_T(self.ee_body)
        site = self.data.site(self._held.passive_name)
        rel_R = R_ee.T @ site.xmat.reshape(3, 3)
        rel_p = R_ee.T @ (site.xpos - p_ee)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rel_R.ravel())
        return [float(v) for v in (*rel_p, *quat)]

    def dock_port_body(self) -> str | None:
        """Body carrying the chain's open port — the dock request's parent.

        Walks out to the tip with the chain, which is why it cannot stay the
        scenario constant it used to be.
        """
        if self.model is None or self.chain is None:
            return None
        sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.dock_target_site
        )
        if sid < 0:
            return None
        return self.model.body(int(self.model.site_bodyid[sid])).name

    def module_pose_world(self, slot: str) -> list | None:
        """World pose of an inventory module's root body, [xyz, wxyz].

        The pick derives from THIS, never from the config nest pose: a module
        MJCF's root body carries its own offset (the row module's is
        pos="0 0 0.05"), so the configured world_pos is where the attachment
        FRAME goes, not where the body lands — 50 mm apart in the bench scene.
        It is also the same quantity perception publishes, so the sim and
        bench pick paths stay identical.
        """
        entry = self._slots.get(str(slot))
        if entry is None or self.data is None:
            return None
        pos, rot = self._body_T(entry.body_name)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rot.ravel())
        return [float(v) for v in (*pos, *quat)]

    def dock_port_world(self) -> list | None:
        """World pose of the chain's OPEN port, [xyz, wxyz].

        Where the next module mates. Walks out to the tip as the arm grows,
        which is what replaces the scenario's fixed dock waypoints.
        """
        if self.data is None or self.chain is None:
            return None
        site = self.data.site(self.dock_target_site)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, site.xmat.reshape(3, 3).ravel())
        return [float(v) for v in (*site.xpos, *quat)]

    def _conn_in_ee_mm(self) -> list:
        """Carried connector position in the EE frame (mm) — pad-slip probe.

        Nominal carry is ~[0, 45, 120]; the +y term is ALONG the module axis,
        the direction the pads cannot positively lock.
        """
        if self.module_site is None:
            return []
        p_ee, R_ee = self._body_T(self.ee_body)
        conn = self.data.site(self.module_site).xpos
        return np.round(R_ee.T @ (conn - p_ee) * 1e3, 1).tolist()

    def _dock_within_capture(self, slot) -> tuple[bool, str]:
        """Seat test against the chain's CURRENT open port, not a fixed dock."""
        dock = self.data.site(self.dock_target_site)
        mod = self.data.site(slot.passive_name)
        dist = float(np.linalg.norm(dock.xpos - mod.xpos))
        if dist > float(self.dock_capture_m):
            delta = (mod.xpos - dock.xpos) * 1e3
            return False, (
                f"dock gap {dist*1e3:.1f} mm > seat {self.dock_capture_m*1e3:.1f} mm "
                f"(connector - dock = [{delta[0]:.1f}, {delta[1]:.1f}, {delta[2]:.1f}] mm)"
            )
        z_dock = dock.xmat.reshape(3, 3)[:, 2]
        z_mod = mod.xmat.reshape(3, 3)[:, 2]
        # Signed alignment — a connector pointing AWAY from the dock must not
        # count as capturable, so no abs() here.
        angle = float(np.degrees(np.arccos(np.clip(z_dock @ z_mod, -1.0, 1.0))))
        if angle > float(self.dock_capture_deg):
            return False, f"dock axis off {angle:.0f}° > {self.dock_capture_deg:.0f}°"
        return True, ""

    def _pad_forces(self, slot) -> dict[str, float]:
        """Per-pad normal force (N) against ONE module."""
        out: dict[str, float] = {}
        wrench = np.zeros(6)
        if self.scene is None or slot is None:
            return out  # pad forces are a composed-scene concept
        prefix = slot.prefix
        for c in range(self.data.ncon):
            g1 = int(self.data.contact.geom1[c])
            g2 = int(self.data.contact.geom2[c])
            bodies = {
                self.model.body(self.model.geom_bodyid[g]).name or ""
                for g in (g1, g2)
            }
            is_finger = any(any(m in b for m in self.finger_body_match) for b in bodies)
            is_module = any(b.startswith(prefix) for b in bodies)
            if not (is_finger and is_module):
                continue
            key = next(b for b in bodies if any(m in b for m in self.finger_body_match))
            mujoco.mj_contactForce(self.model, self.data, c, wrench)
            out[key] = out.get(key, 0.0) + abs(float(wrench[0]))
        return out

    def topology_state(self) -> dict[str, Any]:
        """The assembly graph plus what the hand is holding.

        ``chain`` is the authority on the robot's shape; ``next_port`` is what
        the coordinator aims the next dock at, which is why the dock waypoints
        no longer need to be constants in the scenario config.
        """
        state = self.chain.state() if self.chain is not None else {}
        return {
            "schema": "topology_state",
            "revision": self.model_revision,
            "chain": state,
            "next_port": self.dock_target_site,
            "staged_slots": [s.slot for s in self.staged_slots],
            "held_slot": self._held.slot if self._held is not None else None,
        }


def _demo() -> None:
    """Self-check for the compiled servo law this plant closes.

    Run: ``python -m arm_control.simulation.mujoco_backend --demo``.
    The smallest thing that fails if the Cartesian math breaks — no scene, no
    physics, just the law.
    """
    if _servo is None:
        raise SystemExit(
            "arm_rt_servo is not importable — build it first:\n"
            "  pip install -e libs/arm_control/rt/bindings"
        )
    rng = np.random.default_rng(7)
    n = 6
    eye = np.eye(6)  # J = I: tau IS the EE wrench, so the checks read directly
    x = np.array([0.30, 0.10, 0.20])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    twist = np.zeros(6)
    zeros6 = np.zeros(6)
    # Task frame: X = the insertion axis, deliberately NOT a world axis, so a
    # law that quietly ignored R_task cannot pass.
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    R = np.column_stack((axis, np.array([-axis[1], axis[0], 0.0]), [0.0, 0.0, 1.0]))
    k_axial, k_lat = 2000.0, 200.0
    kc = np.array([k_axial, k_lat, k_lat, 20.0, 5.0, 5.0])
    dc = np.zeros(6)

    # (1) COMPATIBILITY: all-zero K_c/D_c must reduce EXACTLY to the plain
    # joint-space law. Random everything, so it is not passing by symmetry.
    for _ in range(50):
        args = {
            "q": rng.normal(size=n),
            "dq": rng.normal(size=n),
            "q_des": rng.normal(size=n),
            "qd_des": rng.normal(size=n),
            "kp": rng.uniform(0, 500, n),
            "kd": rng.uniform(0, 40, n),
            "tau_ref": rng.normal(size=n),
            "tau_limit": rng.uniform(5, 90, n),
            "slew_per_tick": 1.0,
        }
        tau_ff = rng.normal(size=n)
        folded = _servo.cartesian_torque(
            tau_ff=tau_ff, J=rng.normal(size=(6, n)), R_task=R, x=x, quat=quat,
            x_des=rng.normal(size=3), quat_des=quat, twist=rng.normal(size=6),
            twist_des=zeros6, kc=zeros6, dc=zeros6,
        )
        assert np.array_equal(folded, tau_ff), "zero K_c/D_c perturbed tau_ff"
        plain = _servo.servo_torque(tau_ff=tau_ff, **args)
        with_cart = _servo.servo_torque(tau_ff=folded, **args)
        assert np.array_equal(plain, with_cart), "zero K_c/D_c changed the output"

    # (2) DIRECTION: displace the EE 1 mm LATERALLY (task Y) and the restoring
    # wrench must come back mostly along task Y, at the LATERAL stiffness.
    d = 1e-3
    lat = _servo.cartesian_torque(
        tau_ff=np.zeros(n), J=eye, R_task=R, x=x + d * R[:, 1], quat=quat,
        x_des=x, quat_des=quat, twist=twist, twist_des=zeros6, kc=kc, dc=dc,
    )
    f_task = R.T @ lat[:3]
    assert abs(f_task[1] + k_lat * d) < 1e-9, f"lateral gain wrong: {f_task}"
    assert abs(f_task[0]) < 1e-12 and abs(f_task[2]) < 1e-12, (
        f"lateral push leaked onto other axes: {f_task}"
    )
    assert np.linalg.norm(lat[3:]) < 1e-12, "pure translation produced a moment"

    # ...and the SAME displacement along the stiff insertion axis must cost the
    # full stiffness ratio more force. That ratio is the whole design.
    ax = _servo.cartesian_torque(
        tau_ff=np.zeros(n), J=eye, R_task=R, x=x + d * R[:, 0], quat=quat,
        x_des=x, quat_des=quat, twist=twist, twist_des=zeros6, kc=kc, dc=dc,
    )
    ratio = np.linalg.norm(ax[:3]) / np.linalg.norm(lat[:3])
    assert abs(ratio - k_axial / k_lat) < 1e-9, f"stiffness ratio {ratio}"
    assert np.dot(ax[:3], R[:, 0]) < 0.0, "axial restoring force points outward"

    # (3) ORIENTATION error: a small rotation about task X must produce a
    # moment opposing it, and the SHORTEST-ARC sign must hold past 180 deg —
    # q and -q are the same rotation, so a naive difference springs the long
    # way round exactly there.
    ang = 0.02
    s, c = np.sin(ang / 2), np.cos(ang / 2)
    q_off = np.array([c, *(s * axis)])
    rot = _servo.cartesian_torque(
        tau_ff=np.zeros(n), J=eye, R_task=R, x=x, quat=q_off, x_des=x,
        quat_des=quat, twist=twist, twist_des=zeros6, kc=kc, dc=dc,
    )
    assert np.dot(rot[3:], axis) < 0.0, "rotational spring pushes the wrong way"
    assert abs(np.linalg.norm(rot[3:]) - kc[3] * ang) < 1e-5, f"rot gain {rot[3:]}"
    flipped = _servo.cartesian_torque(
        tau_ff=np.zeros(n), J=eye, R_task=R, x=x, quat=q_off, x_des=x,
        quat_des=-quat, twist=twist, twist_des=zeros6, kc=kc, dc=dc,
    )
    assert np.allclose(rot, flipped), "negating q_des changed the spring (arc sign)"

    # (4) DAMPING opposes motion, in the task frame like the stiffness does.
    v = 0.05
    damp = _servo.cartesian_torque(
        tau_ff=np.zeros(n), J=eye, R_task=R, x=x, quat=quat, x_des=x,
        quat_des=quat, twist=np.concatenate((v * R[:, 1], np.zeros(3))),
        twist_des=zeros6, kc=zeros6, dc=np.array([130.0, 40.0, 40.0, 0, 0, 0]),
    )
    assert abs((R.T @ damp[:3])[1] + 40.0 * v) < 1e-9, f"damping wrong: {damp[:3]}"

    print(
        f"mujoco_backend: servo law ok — zero K_c/D_c is bit-identical over 50 "
        f"random cases; lateral {np.linalg.norm(lat[:3]):.3f} N vs axial "
        f"{np.linalg.norm(ax[:3]):.3f} N for the same {d*1e3:.0f} mm "
        f"(ratio {ratio:.0f}x)"
    )


def _scene_demo() -> None:
    """Smallest proof that an arbitrary separable object composes correctly."""
    import tempfile

    from arm_control.scene import Attachment, load_scene

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "actor.xml").write_text(
            "<mujoco><worldbody><body name='root'><joint name='j' type='hinge'/>"
            "<geom type='sphere' size='.01'/><site name='tip'/></body></worldbody></mujoco>"
        )
        (root / "object.xml").write_text(
            "<mujoco><worldbody><body name='fixture'><geom type='sphere' size='.01'/>"
            "</body><body name='part'><freejoint name='free'/><geom type='sphere' "
            "size='.01'/><site name='inward'/></body></worldbody></mujoco>"
        )
        (root / "scene.yaml").write_text(
            "version: 1\nscene:\n  actors:\n"
            "    welder: {path: actor.xml, joints: [j], pos: [0, 0, 0], rpy: [0, 0, 0]}\n"
            "  objects:\n"
            "    item: {path: object.xml, pos: [1, 0, 0], rpy: [0, 0, 0]}\n"
        )
        scene = load_scene(root / "scene.yaml")
        state = scene.state()
        state.attach(
            scene,
            Attachment("item", "part", "welder__tip", "inward", (0, 0, 0, 1, 0, 0, 0)),
        )
        backend = MuJoCoBackend.from_workcell_scene(scene, state)
        backend.load()
        assert backend.body_pose("item__fixture")[0] == 1.0
        assert backend.frame_pose("item__inward")[:3] == [0.0, 0.0, 0.0]
        state.detach(scene, "item")
        backend.apply_scene_state(state)
        assert backend.body_pose("item__part")[0] == 0.0
    print("mujoco_backend: scene composition OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        _demo()
    elif "--scene-demo" in sys.argv:
        _scene_demo()
    else:
        print(__doc__)
