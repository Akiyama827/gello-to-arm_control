"""Rerun operator preview: plan-review overlays + robot ghosts (no meshcat).

Everything the meshcat preview showed, as Rerun entities — which stream over
the network to the desk viewer, so the arm PC stays headless:

- ``preview/<name>``      module CAD ghost, target triads, EE path, scene cloud
- ``preview/plan_ghost``  the planned trajectory as a scrubbable ``plan_t``
                          timeline animation of the full robot
- ``measured/…``          translucent green robot at the live measured pose
- ``target/…``            solid robot at the teleop slider target

Robot rendering is Pinocchio visual-model FK + ``rr.Asset3D`` per geometry —
the same machinery the visualizer node uses. Rerun logs STL natively, so the
old STL→OBJ conversion cache is gone.

Sink: ``init_preview_stream`` joins the viewer that the visualizer node
spawned (default local gRPC), connects to a remote viewer (``connect: url``,
the bench topology), or spawns its own (``spawn: true``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

# ONE recording for every node that logs 3D state (plan preview, teleop, the
# sim twin mirror): same app + recording id means the streams MERGE in the
# viewer — the arm ghost, overlays, and the twin's true scene share one view
# instead of landing in separate recordings the operator has to switch between.
# The app id must NOT collide with the visualizer node's app ids: the viewer
# caches blueprints per app id, and an inherited plots/FK blueprint hides
# every entity these nodes log (its views only show robot/** — observed as an
# "empty" viewer).
DEFAULT_APP_ID = "arm_control_live"
DEFAULT_RECORDING_ID = "live"


def init_preview_stream(preview_cfg: dict | None, app_id: str = DEFAULT_APP_ID) -> None:
    """Initialise the process-global Rerun stream for preview logging.

    Sends an explicit whole-scene blueprint (one 3D view rooted at ``/``) so a
    stale cached layout can never filter the entities out of sight.

    CAVEAT (bench): the gRPC sink's batcher BLOCKS the logging thread once its
    channel fills while no viewer is attached (observed: 5 s+ stalls) — on the
    real rig that is a deadman hazard. Launch the viewer BEFORE arming, or use
    ``spawn: true``. A failed spawn falls back to a memory recording (preview
    dropped, control loop unaffected).
    """
    import rerun.blueprint as rrb

    cfg = dict(preview_cfg or {})
    rr.init(
        str(cfg.get("app_id", app_id)),
        recording_id=str(cfg.get("recording_id", DEFAULT_RECORDING_ID)),
        spawn=False,
    )
    if cfg.get("spawn"):
        try:
            rr.spawn(detach_process=False)
        except Exception as exc:  # headless host — never stall the control loop
            print(f"[preview] rerun spawn failed ({exc}) — preview disabled", flush=True)
            rr.memory_recording()
    elif cfg.get("connect"):
        rr.connect_grpc(str(cfg["connect"]))
    else:
        rr.connect_grpc()  # the viewer another node (visualizer) spawned
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="/")))


def log_frame_transform(entity_root: str, T: np.ndarray | None) -> None:
    """Pin an entity subtree to a parent frame (static, hierarchical).

    The preview/ghost entities are authored in the ARM-BASE frame while the
    sim mirror logs WORLD poses — without this transform on the roots the two
    families render mutually shifted by the arm's mount pose.
    """
    if T is None:
        return
    T = np.asarray(T, dtype=float)
    rr.log(
        entity_root,
        rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
        static=True,
    )


def _mesh_package_dirs(urdf_path: str | Path) -> list[str]:
    urdf_dir = Path(urdf_path).resolve().parent
    package_root = (
        urdf_dir.parent if (urdf_dir.parent / "package.xml").is_file() else urdf_dir
    )
    candidates = [urdf_dir, urdf_dir.parent, package_root, package_root.parent]
    return [str(path) for path in candidates if path.is_dir()]


def _rgba8(color) -> list[int]:
    rgba = np.clip(np.asarray(color, dtype=float), 0.0, 1.0)
    return [int(round(channel * 255.0)) for channel in rgba]


class RobotGhost:
    """One translucent (or solid) copy of the robot under a Rerun prefix.

    Pinocchio visual-model FK; assets logged once (static), per-geom
    transforms logged on every ``update``.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: list[str],
        prefix: str,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> None:
        import pinocchio as pin

        self._pin = pin
        self._prefix = str(prefix).rstrip("/")
        self.model, self.visual_model = pin.buildModelsFromUrdf(
            str(urdf_path),
            package_dirs=_mesh_package_dirs(urdf_path) or None,
            geometry_types=[pin.GeometryType.VISUAL],
        )
        self.data = self.model.createData()
        self.visual_data = self.visual_model.createData()
        self._q_idx = [
            self.model.joints[self.model.getJointId(str(n))].idx_q for n in joint_names
        ]
        self._rgba = rgba
        self._assets_logged = False

    def _ensure_assets(self) -> None:
        """Upload meshes lazily, on the FIRST posed update — logging them at
        construction leaves an unassembled pile of identity-transform parts at
        the origin until the ghost is first driven (e.g. the plan ghost sits
        unused until a confirm gate animates it)."""
        if self._assets_logged:
            return
        self._assets_logged = True
        for geom in self.visual_model.geometryObjects:
            mesh_path = Path(geom.meshPath)
            if not mesh_path.is_file():
                continue
            tint = _rgba8(self._rgba) if self._rgba is not None else _rgba8(geom.meshColor)
            rr.log(
                f"{self._prefix}/{geom.name}",
                rr.Asset3D(path=mesh_path, albedo_factor=tint),
                static=True,
            )

    def update(self, q_planned: np.ndarray) -> None:
        self._ensure_assets()
        q = self._pin.neutral(self.model)
        for idx, value in zip(self._q_idx, np.asarray(q_planned, dtype=float)):
            q[idx] = value
        self._pin.forwardKinematics(self.model, self.data, q)
        self._pin.updateGeometryPlacements(
            self.model, self.data, self.visual_model, self.visual_data, q
        )
        for geom, placement in zip(
            self.visual_model.geometryObjects, self.visual_data.oMg
        ):
            rr.log(
                f"{self._prefix}/{geom.name}",
                rr.Transform3D(
                    translation=placement.translation,
                    mat3x3=placement.rotation,
                    scale=geom.meshScale,
                ),
            )


class MeasuredGhost(RobotGhost):
    """The live measured robot in its ORIGINAL STL colors (user spec: the
    current status is the real thing, not a tinted ghost — tints are reserved
    for planned motion and targets)."""

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: list[str],
        world_T_arm: np.ndarray | None = None,
    ) -> None:
        super().__init__(urdf_path, joint_names, "measured", rgba=None)
        log_frame_transform("measured", world_T_arm)


class PreviewScene:
    """Operator-facing plan-review overlays (perceived module ghost, waypoint
    triads, planned EE path, scrubbable trajectory animation, scene cloud).

    All poses are in the ARM-BASE frame. ``world`` supplies FK for the EE
    path. Color legend (user spec): ORIGINAL STL = live measured robot,
    GREEN = played planned motion (playback ghost + EE path), ORANGE = the
    current planning TARGET posture, VIOLET = perceived module, white = sim
    truth, cyan = observed cloud.
    """

    def __init__(
        self,
        fk,
        *,
        urdf_path: str | Path,
        joint_names: list[str],
        ee_link: str,
        animate_fps: float = 30.0,
        world_T_arm: np.ndarray | None = None,
        gripper_joints: list[str] | None = None,
    ) -> None:
        # ``fk``: callable q -> 4x4 EE pose. Pass the planner's ``ik.fk`` — the
        # SAME forward kinematics the plan was solved against, so the drawn path
        # cannot disagree with the commanded one.
        #
        # This used to ask the MuJoCo collision world for the EE body's pose,
        # which breaks on any arm whose EE frame is behind a FIXED joint:
        # MuJoCo's URDF importer MERGES fixed-joint links into their parent, so
        # the frame simply is not a body. On the FR3, ``fr3_hand_tcp`` (and
        # fr3_hand, fr3_link8) vanish — valid bodies are only fr3_link1..7 plus
        # the fingers — and the lookup raised KeyError. Pinocchio keeps every
        # URDF frame, and it is the kinematics authority here anyway.
        self._fk = fk
        self._ee_link = str(ee_link)
        self._fps = float(animate_fps)
        # Ghosts carry the finger joints too: plans are 6-dim (the gripper is
        # a request, not a planned DOF), so ghost fingers MIRROR the live
        # measured aperture — URDF-default open fingers on a ghost while the
        # real gripper closes read as a rendering bug (user-reported).
        self._gripper_joints = list(gripper_joints or [])
        ghost_joints = list(joint_names) + self._gripper_joints
        self._fingers = 0.0
        self._plan_ghost = RobotGhost(
            urdf_path, ghost_joints, "preview/plan_ghost", rgba=(0.2, 0.85, 0.3, 0.9)
        )
        self._target_ghost = RobotGhost(
            urdf_path, ghost_joints, "preview/plan_target", rgba=(1.0, 0.5, 0.0, 0.55)
        )
        log_frame_transform("preview", world_T_arm)

    def set_finger_state(self, finger_m: float) -> None:
        """Live measured finger value mirrored into both plan ghosts."""
        self._fingers = float(finger_m)

    def _with_fingers(self, q: np.ndarray) -> np.ndarray:
        if not self._gripper_joints:
            return np.asarray(q, dtype=float)
        pad = np.full(len(self._gripper_joints), self._fingers)
        return np.append(np.asarray(q, dtype=float), pad)

    def show_module(
        self,
        name: str,
        mesh_path: str | Path,
        T: np.ndarray,
        rgba: tuple = (0.55, 0.3, 0.95, 0.9),
    ) -> None:
        """Perceived-object ghost: the module CAD at its estimated pose.

        VIOLET, deliberately not orange: orange is reserved for PLANNED
        motion (the trajectory ghost + EE path). Color legend: white = sim
        truth, green = measured, orange = planned, violet = perceived,
        cyan = observed cloud."""
        T = np.asarray(T, dtype=float)
        entity = f"preview/{name}"
        rr.log(entity, rr.Asset3D(path=Path(mesh_path), albedo_factor=_rgba8(rgba)), static=True)
        rr.log(entity, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))

    def show_target(self, name: str, T: np.ndarray, scale: float = 0.08) -> None:
        """RGB triad marking a target EE pose (x=red, y=green, z=blue)."""
        T = np.asarray(T, dtype=float)
        rr.log(
            f"preview/{name}",
            rr.Arrows3D(
                origins=[T[:3, 3]] * 3,
                vectors=[scale * T[:3, axis] for axis in range(3)],
                colors=[(255, 60, 60), (60, 255, 60), (70, 115, 255)],
            ),
        )

    def show_ee_path(self, name: str, positions: np.ndarray) -> None:
        """GREEN polyline of the EE positions along a planned joint trajectory."""
        pts = np.asarray(
            [self._fk(np.asarray(q, dtype=float))[:3, 3] for q in np.asarray(positions)]
        ).T
        if pts.shape[1] >= 2:
            rr.log(
                f"preview/{name}",
                rr.LineStrips3D([pts.T], colors=[(60, 220, 90)], radii=[0.005]),
            )

    def show_plan_pose(self, q_planned: np.ndarray) -> None:
        """One frame of the GREEN playback ghost on the LIVE timeline.

        The PLAY loop calls this at ~20 fps so the ghost flies the held or
        executing plan in the same view as everything else (MoveIt-style
        review) — no plan_t timeline scrubbing required."""
        self._plan_ghost.update(self._with_fingers(q_planned))

    def show_plan_target(self, q_goal: np.ndarray) -> None:
        """ORANGE static ghost at the CURRENT plan's goal posture."""
        self._target_ghost.update(self._with_fingers(q_goal))

    def animate(self, times: np.ndarray, positions: np.ndarray) -> None:
        """Scrubbable full-robot animation on the ``plan_t`` timeline."""
        last = None
        for t, q in zip(np.asarray(times, dtype=float), np.asarray(positions)):
            if last is not None and t - last < 1.0 / self._fps:
                continue
            last = t
            rr.set_time("plan_t", duration=float(t))
            self._plan_ghost.update(self._with_fingers(q))
        rr.set_time("plan_t", duration=float(times[-1]))
        self._plan_ghost.update(self._with_fingers(positions[-1]))
        rr.reset_time()  # detach subsequent logs from the plan timeline

    def show_cloud(
        self,
        name: str,
        points: np.ndarray,
        point_size: float = 0.004,
        rgba: tuple = (0.45, 0.85, 1.0, 1.0),
    ) -> None:
        """Observed scene cloud (arm-base frame) — the ground truth the module
        ghost and planned path are reviewed AGAINST."""
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return
        rr.log(
            f"preview/{name}",
            rr.Points3D(pts, colors=[_rgba8(rgba)], radii=[float(point_size)]),
        )
