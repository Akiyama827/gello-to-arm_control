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
from arm_control.simulation.convex_decomp import replace_with_decomposition


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
class MuJoCoSceneSpec:
    arm: SceneModelSpec
    base: SceneModelSpec
    module: SceneModelSpec
    timestep: float = 0.001


def _load_model_spec(path: str | Path) -> mujoco.MjSpec:
    path = Path(path)
    if path.suffix.lower() == ".urdf":
        return mujoco.MjSpec.from_string(path.read_text())
    if path.suffix.lower() in (".xml", ".mjcf"):
        return mujoco.MjSpec.from_file(str(path))
    raise MuJoCoUnavailableError(
        f"{path}: MuJoCo scene models must be MJCF (.xml) or URDF — SDF-era "
        "paths belong to the removed Drake stack (use models/mjcf/…)"
    )


def compose_scene(
    spec_cfg: MuJoCoSceneSpec,
    ground_z: float | None = 0.0,
    static_boxes: list[dict] | None = None,
) -> mujoco.MjSpec:
    """mjSpec world: ground plane + the three models attached with prefixes.

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
    for model in (spec_cfg.arm, spec_cfg.base, spec_cfg.module):
        child = _load_model_spec(model.model_path)
        frame = spec.worldbody.add_frame(
            pos=list(model.world_pos), quat=list(_rpy_to_quat(*model.world_rpy))
        )
        spec.attach(child, prefix=f"{model.prefix}", frame=frame)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec

@dataclass
class MuJoCoBackend:
    """Sim plant with the old DrakeBackend's public surface.

    Two modes: a composed ``scene`` (assembler + base + free module, welds,
    ground) or a ``single_model_path`` (one MJCF/URDF, no welds/ground —
    the view/motion graphs' plant).
    """

    joint_names: list[str]  # actuated; PREFIXED in scene mode (<arm prefix>Joint1 …)
    scene: MuJoCoSceneSpec | None = None
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
    module_body: str | None = None
    dock_site: str | None = None
    module_site: str | None = None
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
    model_revision: int = 0

    model: Any = field(default=None, init=False, repr=False)
    data: Any = field(default=None, init=False, repr=False)
    viewer: Any = field(default=None, init=False, repr=False)
    _viewer_next_sync: float = field(default=0.0, init=False, repr=False)
    _qadr: np.ndarray | None = field(default=None, init=False, repr=False)
    _vadr: np.ndarray | None = field(default=None, init=False, repr=False)
    _act_id: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_command: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _grasped_modules: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _docked_modules: dict[str, dict[str, str]] = field(default_factory=dict, init=False, repr=False)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def num_motors(self) -> int:
        return len(self.joint_names)

    @property
    def model_path(self) -> str:
        if self.scene is not None:
            return str(self.scene.arm.model_path)
        return str(self.single_model_path)

    # -- lifecycle --------------------------------------------------------
    def _build_spec(self) -> mujoco.MjSpec:
        if self.scene is not None:
            return compose_scene(self.scene, self.ground_z, self.static_boxes)
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
            missing = [
                k
                for k in ("ee_body", "module_body", "dock_site", "module_site")
                if not getattr(self, k)
            ]
            if missing:
                raise ValueError(
                    f"scene mode requires {missing} (scene.welds.* in the "
                    "scenario config — model facts, no default robot)"
                )
        spec = self._build_spec()
        for name in self.joint_names:
            actuator = spec.add_actuator(
                name=f"{name}_motor", target=str(name), trntype=mujoco.mjtTrn.mjTRN_JOINT
            )
            actuator.gainprm[0] = 1.0  # ctrl IS torque; joint actuatorfrcrange clamps
        # Finger servos in BOTH modes: a single-model gripper (sim motion) is
        # otherwise a free prismatic pair — gravity slides a finger to its
        # stop, and gripper commands have no actuator to land on.
        gripper_present = []
        joint_names = {spec_joint.name for spec_joint in spec.joints}
        for name in self.gripper_joints:
            if name not in joint_names:
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
            spec.add_equality(
                name="dock_weld",
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                name1=self.dock_site,
                name2=self.module_site,
                active=False,
            )
            # Bench-fixture equivalent: the module is HELD at its presentation
            # pose until the grasp takes it. Contact-only emulation (pedestal,
            # pocket walls) was tried and always drifts/leans — the module's
            # collision hull has a rounded bottom; a real fixture's whole job
            # is that the module does not move.
            spec.add_equality(
                name="fixture_weld",
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1="world",
                name2=self.module_body,
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
                for mdl in (self.scene.arm, self.scene.base, self.scene.module)
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
                spec, bodies_containing=[self.scene.module.prefix],
                cache_dir=cache, mesh_search_dirs=dirs, threshold=0.06,
            )
        self.model = spec.compile()
        if self.scene is not None:
            # The free module's INTERNAL joint (passive-to-motor) is unactuated
            # in the MJCF, but 0.42 of the module's 0.49 kg hangs on it — free,
            # it turns the standing module into a pendulum that topples itself.
            # Real modules hold that joint by gearing when unpowered; emulate
            # with static friction. 5.0, not 1.0: carry-swing transients
            # back-drove the joint ~55° at 1.0 (measured), visually dangling
            # the motor half of the carried module.
            prefix = self.scene.module.prefix
            for jid in range(self.model.njnt):
                joint = self.model.joint(jid)
                if joint.name.startswith(prefix) and self.model.jnt_type[jid] in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ):
                    self.model.dof_frictionloss[joint.dofadr[0]] = max(
                        5.0, float(self.model.dof_frictionloss[joint.dofadr[0]])
                    )
        if not self.enable_self_collision:
            self._contacts_ground_only()
        self.data = mujoco.MjData(self.model)
        for name, value in (self.default_joint_positions or {}).items():
            joint = self.model.joint(str(name))
            self.model.qpos0[joint.qposadr[0]] = float(value)
        mujoco.mj_resetData(self.model, self.data)
        self._qadr = np.array(
            [self.model.joint(n).qposadr[0] for n in self.joint_names], dtype=int
        )
        self._vadr = np.array(
            [self.model.joint(n).dofadr[0] for n in self.joint_names], dtype=int
        )
        self._act_id = np.array(
            [self.model.actuator(f"{n}_motor").id for n in self.joint_names], dtype=int
        )
        self._gripper_act = [
            self.model.actuator(f"{n}_servo").id for n in gripper_present
        ]
        # The joints that ACTUALLY exist in this model, not the ones configured:
        # ``gripper_joints`` is a superset covering every arm (the DM assembler's
        # Gripper_1/2 and the FR3's fr3_finger_joint1/2), and only one pair is
        # present in any given model. Reading back through the configured names
        # would look up a joint this model has never heard of.
        self._gripper_names = list(gripper_present)
        for name in gripper_present:
            # The open stop is physical: without the limit, module contact
            # shoves the fingers to negative travel (outside the mechanism).
            self.model.jnt_limited[self.model.joint(name).id] = 1
        # Finger travel from the MODEL, not a constant: the clamp in
        # _set_gripper_targets must match whatever gripper this scene has.
        self._gripper_range = [
            (
                float(self.model.jnt_range[self.model.joint(n).id][0]),
                float(self.model.jnt_range[self.model.joint(n).id][1]),
            )
            for n in gripper_present
        ]
        # Servo targets start at the OPEN rest (ctrl defaults to 0 = closed).
        for act, name in zip(self._gripper_act, gripper_present):
            self.data.ctrl[act] = self.model.qpos0[
                self.model.joint(name).qposadr[0]
            ]
        mujoco.mj_forward(self.model, self.data)
        if self.scene is not None:
            self._activate_body_weld("fixture_weld", "world", self.module_body)
        if self.launch_viewer:
            try:
                from mujoco import viewer as mj_viewer

                self.viewer = mj_viewer.launch_passive(self.model, self.data)
                print("[mujoco_backend] viewer launched", flush=True)
            except Exception as exc:  # headless host — sim runs fine without it
                print(f"[mujoco_backend] viewer unavailable: {exc}", flush=True)

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
        module_prefix = self.scene.module.prefix if self.scene is not None else None
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
            elif module_prefix and body.startswith(module_prefix):
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

    def _apply_pd(self) -> None:
        if self._last_command is None:
            return
        cmd = self._last_command
        q = self.data.qpos[self._qadr]
        v = self.data.qvel[self._vadr]
        tau = (
            cmd["torque"]
            + cmd["kp"] * (cmd["position"] - q)
            + cmd["kd"] * (cmd["velocity"] - v)
        )
        self.data.ctrl[self._act_id] = tau

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
            # SENSOR-LOCAL weld physics — the plant senses and reacts to its
            # own world; the grasp PROTOCOL (close sequencing, success, drop
            # events) lives in the bridge's GraspGate:
            # - the fixture YIELDS once the gripper truly squeezes the module
            #   (a fixture's magnets give way to the hand's pull);
            # - the keyed dock mate ENGAGES when the hand lets go with the
            #   connector actually at the seat (mm lead-in, no magnets).
            pads = self._pad_forces()
            grip_n = float(sum(pads.values()))
            # Yield at FIRST pad touch: the weld's only job is pre-grasp
            # presentation (the rounded hull drifts if the module just
            # stands). A real fixture seats the module magnetically —
            # a lateral push slides it off almost immediately, and the
            # freed module stands on the pedestal and SELF-CENTERS between
            # the closing pads. Any force threshold loses this race: with
            # the module rigidly anchored, the first (one-sided) contact
            # twists the soft wrist away — 8 deg at the old 10 N gate,
            # still 7 deg at a 3 N both-pad gate (measured live) — and the
            # module is handed over tilted, carried 24 mm off-nominal, and
            # the mm dock seat then rightly refuses the release.
            if (
                self.data.eq_active[self._eq_id("fixture_weld")]
                and grip_n >= 0.5
            ):
                self._set_weld_active("fixture_weld", False)
                self._grasped_modules[self._module_id()] = self.ee_body
                ee_p = self._body_T(self.ee_body)[0]
                mod_p = self._body_T(self.module_body)[0]
                conn_p = self.data.site(self.module_site).xpos
                print(
                    f"[mujoco_backend] fixture yielded to the grip "
                    f"({grip_n:.1f} N) — module rides the hand; "
                    f"conn-in-EE {self._conn_in_ee_mm()} mm "
                    f"EE {np.round(ee_p * 1e3, 1).tolist()} "
                    f"mod {np.round(mod_p * 1e3, 1).tolist()} "
                    f"conn {np.round(conn_p * 1e3, 1).tolist()}",
                    flush=True,
                )
            # "The hand let go" is a COMMANDED fact, not residual contact: at
            # the dock orientation gravity runs parallel to the pad faces, so
            # a released module SLIDES down the opening fingers and WEDGES
            # tilted between them with >2 N of contact — a force-decay
            # trigger then never fires and the module hangs stuck
            # (user-observed). Commanded-open fires at the seat, before the
            # module can slide. The 0.25 m band covers true mid-carry drops
            # (module gone while still commanded closed) for topology truth.
            elif self._grasped_modules and (
                all(
                    float(self.data.ctrl[a]) <= 0.004 for a in self._gripper_act
                )
                or float(
                    np.linalg.norm(
                        self._body_T(self.ee_body)[0]
                        - self.data.site(
                            f"{self.scene.module.prefix}grasp_frame"
                        ).xpos
                    )
                )
                > 0.25
            ):
                docked, why = self._dock_within_capture()
                if docked:
                    self._activate_dock_weld()
                    for module_id in list(self._grasped_modules):
                        self._grasped_modules.pop(module_id, None)
                        self._docked_modules[module_id] = {
                            "parent_body": self.dock_site,
                            "child_body": self.module_body,
                            "mode": "weld",
                        }
                    self.model_revision += 1
                    print(
                        "[mujoco_backend] released at the seat — dock mate "
                        "engaged "
                        f"({self._module_id()} -> {self.dock_site})",
                        flush=True,
                    )
                else:
                    # Hand open with the connector NOT seated: no magnets to
                    # forgive it — the module leaves the pads under plain
                    # physics. Topology reflects the fact; the bridge's gate
                    # reports the LOST/MISSED event.
                    for module_id in list(self._grasped_modules):
                        self._grasped_modules.pop(module_id, None)
                        print(
                            f"[mujoco_backend] released unseated: "
                            f"{module_id} free ({why})",
                            flush=True,
                        )
            if self._grasped_modules:
                # In-hand drift telemetry (~2 Hz). Standalone block: wedging
                # an `if` into the yield/release chain above re-bound the
                # release `elif` and silently disabled it (live-caught).
                self._grip_log_steps = getattr(self, "_grip_log_steps", 0) + 1
                if self._grip_log_steps % 100 == 0:
                    print(
                        f"[mujoco_backend] carry: conn-in-EE "
                        f"{self._conn_in_ee_mm()} mm grip {grip_n:.1f} N",
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

    # -- weld topology ----------------------------------------------------
    def _eq_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def _set_weld_active(self, name: str, active: bool) -> None:
        self.data.eq_active[self._eq_id(name)] = 1 if active else 0

    def _activate_dock_weld(self) -> None:
        """Site weld: sites are pulled coincident — but only with a VALID
        relpose quat in eq_data (all-zero quat silently disables the pull)."""
        data = self.model.eq_data[self._eq_id("dock_weld")]
        data[:] = 0.0
        data[6] = 1.0  # identity relquat
        # torquescale 20, same as the body welds: the keyed 90° dock is
        # rotationally RIGID once seated — at 1.0 the docked
        # module visibly tilted under its own 0.43 N·m gravity moment after
        # release (user-observed).
        data[10] = 20.0
        self._set_weld_active("dock_weld", True)

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

    def _conn_in_ee_mm(self) -> list:
        """Carried connector position in the EE frame (mm) — pad-slip probe.

        Nominal carry is ~[0, 45, 120]; the +y term is ALONG the module axis,
        the direction the pads cannot positively lock.
        """
        p_ee, R_ee = self._body_T(self.ee_body)
        conn = self.data.site(self.module_site).xpos
        return np.round(R_ee.T @ (conn - p_ee) * 1e3, 1).tolist()

    def _dock_within_capture(self) -> tuple[bool, str]:
        dock = self.data.site(self.dock_site)
        mod = self.data.site(self.module_site)
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

    def _pad_forces(self) -> dict[str, float]:
        """Per-pad normal force (N) against the module."""
        out: dict[str, float] = {}
        wrench = np.zeros(6)
        if self.scene is None:
            return out  # pad forces are a composed-scene concept
        prefix = self.scene.module.prefix
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

    def _module_id(self) -> str:
        """Module id derived from the scene body name (mod_row_module_body ->
        row_module) — topology bookkeeping only; ids in EVENTS are bridge/
        orchestrator concerns."""
        name = str(self.module_body)
        prefix = self.scene.module.prefix if self.scene is not None else ""
        name = name.removeprefix(prefix)
        return name.removesuffix("_body")

    def topology_state(self) -> dict[str, Any]:
        return {
            "schema": "topology_state",
            "revision": self.model_revision,
            "docked_modules": sorted(self._docked_modules),
            "docks": dict(self._docked_modules),
            "grasped_modules": dict(self._grasped_modules),
        }
