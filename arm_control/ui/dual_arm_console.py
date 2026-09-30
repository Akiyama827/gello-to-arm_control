"""Library-owned dual-arm console for shared simulation or remote RT plants.

Planning runs in a child process. HTTP actions and control ticks are
serialized; heavy IK/OMPL work never runs in the controller critical section.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import signal
import threading
import tempfile
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from arm_control.config import arm_joints, ee_frame, gripper_joints
from arm_control import frames
from arm_control.console_server import ConsoleServer, file_asset
from arm_control.grasp_visual import VisualFK
from arm_control.multi_arm.closure import safe_closure_record
from arm_control.multi_arm.config import load_pair_config, scene_for_mode
from arm_control.multi_arm.coordinator import BimanualCoordinator
from arm_control.multi_arm.diagnostics import PairFaultTrace
from arm_control.multi_arm.simulation import SharedSimulation
from arm_control.planning.refinement import refinement_choice
from arm_control.multi_arm.cooperative import (
    CooperativeKinematics,
    RigidGrasps,
    plan_cooperative,
    pose_error,
    virtual_capture_reason,
)
from arm_control.node_utils import ShutdownFlag, install_signal_handlers, resolve_gains


def _vector(value, n, label):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{label}: expected {n} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{label}: expected numbers")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{label}: values must be finite")
    return result


def _planner_init():
    # The parent owns Ctrl-C, pair cancellation and orderly pool teardown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _scene_token(config, finger_positions, mode="independent", mode_revision=0, virtual_grasps=None):
    return hashlib.sha256(
        (
            config.scene_token
            + json.dumps([finger_positions, mode, mode_revision, virtual_grasps], sort_keys=True)
        ).encode()
    ).hexdigest()


def _pose_values(T):
    return [
        *(T[:3, 3] * 1000).tolist(),
        *Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True).tolist(),
    ]


def _pose_matrix(values):
    values = _vector(values, 6, "pose")
    T = np.eye(4)
    T[:3, 3] = values[:3] / 1000
    T[:3, :3] = Rotation.from_euler("xyz", values[3:], degrees=True).as_matrix()
    return T


def _pose_json(T):
    return dict(
        p=T[:3, 3].tolist(), q=Rotation.from_matrix(T[:3, :3]).as_quat().tolist()
    )


def _visuals(config):
    return {
        a.name: VisualFK(
            a.path,
            list(a.joints),
            ee_frame(config.robots[a.name]),
            gripper_joints(config.robots[a.name]),
            world_T_root=frames.world_T_arm(config.robots[a.name]),
        )
        for a in config.scene.actors
    }


def plan_request(
    config_path,
    expected_config,
    starts,
    targets,
    active,
    fingers,
    scene_token,
    object_start=None,
    object_target=None,
    mode="independent",
    prepare_grasps=False,
    refinement="none",
    virtual_grasps=None,
):
    """Picklable planning-process entry point; opens no device or HTTP endpoint."""
    from arm_control.dynamics import PinocchioDynamics
    from arm_control.planning.ik import PinocchioIK
    from arm_control.multi_arm.planning import PairPlanner

    config = load_pair_config(config_path)
    if config.scene_token != expected_config:
        raise ValueError("configuration or model changed; restart the console")
    scene = scene_for_mode(config, mode)
    kin = CooperativeKinematics(config) if config.shared_object else None
    if virtual_grasps is not None:
        if not (kin and config.shared_object.virtual and mode == "shared"):
            raise ValueError("captured grasps require virtual shared mode")
        captured = RigidGrasps(virtual_grasps)
        nominal = config.shared_object.grasps.object_T_tcp
        if set(captured.object_T_tcp) != set(nominal):
            raise ValueError("captured grasps must name both configured arms")
        reason = virtual_capture_reason({
            a: pose_error(T, nominal[a]) for a, T in captured.object_T_tcp.items()
        })
        if reason:
            raise ValueError(reason)
        kin.grasps = captured
    elif kin and config.shared_object.virtual and mode == "shared":
        raise ValueError("virtual shared planning requires captured measured grasps")
    if prepare_grasps and (kin is None or mode != "independent"):
        raise ValueError(
            "grasp preparation requires independent arms and a supported object"
        )
    if kin and object_start is None:
        raise ValueError("planning requires the current object pose")
    gains, velocity, acceleration, held = {}, {}, {}, {}
    goals = {a: np.array(q) for a, q in starts.items()}
    for actor in config.scene.actors:
        a, cfg = actor.name, config.robots[actor.name]
        n = len(actor.joints)
        resolved = resolve_gains(cfg, cfg.get("control_mode") or {}, list(actor.joints), n)
        gains[a] = (resolved["kp"], resolved["kd"])
        dynamics = PinocchioDynamics(str(actor.path), actor.joints)
        velocity[a] = np.minimum(
            np.minimum(config.velocity, dynamics.velocity_limits),
            cfg.raw["arm"].get("max_vel", np.inf),
        )
        acceleration[a] = np.minimum(
            np.full(n, config.acceleration), cfg.raw["arm"].get("max_acc", np.inf)
        )
        held.update({f"{a}__{j}": fingers[a] for j in gripper_joints(cfg)})
        if mode == "shared" or prepare_grasps or a not in active:
            continue
        target = targets[a]
        if target["kind"] == "joint":
            goals[a] = np.deg2rad(_vector(target["values"], n, f"{a} joints"))
        else:
            values = _vector(target["values"], 6, f"{a} pose")
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("xyz", values[3:], degrees=True).as_matrix()
            T[:3, 3] = values[:3] / 1000
            if target["frame"] == "world":
                T = np.linalg.inv(frames.world_T_arm(cfg)) @ T
            ik = PinocchioIK(actor.path, ee_frame(cfg), actor.joints)
            goals[a] = ik.solve(T, np.asarray(starts[a]))
            if goals[a] is None:
                raise ValueError(f"{a}: target pose has no IK solution")
    planner = PairPlanner(
        scene,
        scene.state(),
        model_tokens=config.model_tokens,
        scene_token=scene_token,
        gains=gains,
        max_velocity=velocity,
        max_acceleration=acceleration,
        held_positions=held,
        ground_z=config.ground_z,
        workspace_xy=config.workspace_xy,
        refinement=refinement,
    )
    if kin:
        planner.world.set_free_body_pose(config.shared_object.body, object_start)
    if mode == "shared":
        if object_start is None or object_target is None:
            raise ValueError("fixed shared grasps require an object target")
        motion = plan_cooperative(
            planner,
            kin,
            config.shared_object.body,
            starts,
            object_start,
            _pose_matrix(object_target),
        )
    else:
        if prepare_grasps:
            goals = kin.solve(object_start, starts)
            nearer = kin.nearer_posture(object_start, starts, goals)
            if not planner.world.in_collision(
                np.concatenate([nearer[a] for a in config.scene.actor_names])
            ):
                goals = nearer
        motion = planner.plan(starts, goals, active_arms=active)
    visual = _visuals(config)
    preview = []
    for t in np.linspace(
        0.0, motion.duration_sec, min(160, max(2, int(motion.duration_sec * 30) + 1))
    ):
        preview.append(
            dict(
                t=float(t),
                arms={
                    a: visual[a].poses(tr.sample_at(t).position, fingers[a])
                    for a, tr in motion.trajectories.items()
                },
            )
        )
        if mode == "shared":
            poses = kin.objects(
                {a: tr.sample_at(t).position for a, tr in motion.trajectories.items()}
            )
            preview[-1]["object"] = _pose_json(next(iter(poses.values())))
    return motion, preview


class DualArmConsole:
    def __init__(self, config_path, *, port=7520, start_server=True, runtime_factory=None):
        self.config = load_pair_config(config_path)
        if runtime_factory is None:
            if self.config.backend == "remote_rt":
                from arm_control.multi_arm.hardware import RemoteHardware

                runtime_factory = RemoteHardware
            else:
                runtime_factory = SharedSimulation
        self.runtime = runtime_factory(self.config)
        self.server = self._pool = self._mesh_cache = None
        try:
            self._initialize(port, start_server)
        except BaseException:
            # Covers visual/model setup failures as well as server startup.
            # Revoke hardware authority before waiting on other resources.
            self.runtime.close()
            if self.server:
                self.server.close()
            if self._pool:
                self._pool.shutdown(wait=False, cancel_futures=True)
            if self._mesh_cache:
                self._mesh_cache.cleanup()
            raise

    def _initialize(self, port, start_server):
        self.visual = _visuals(self.config)
        self.kin = getattr(self.runtime, "kinematics", None)
        if self.kin is None and self.config.shared_object:
            self.kin = CooperativeKinematics(self.config)
        from arm_control.simulation.rerun_scene import static_mesh_geoms

        self._mesh_cache = tempfile.TemporaryDirectory(prefix="dual-console-mesh-")
        self._static = static_mesh_geoms(
            self.runtime.scene_model,
            self.runtime.scene_data,
            root_T_world=np.eye(4),
            exclude_prefixes=tuple(f"{a}__" for a in self.visual),
            cache_dir=self._mesh_cache.name,
            body_names=True,
        )
        self._object_meshes = []
        if self.kin:
            prefix = self.config.shared_object.name + "__"
            inverse = np.linalg.inv(self.runtime.object_pose())
            self._object_meshes = [
                (name, label, path, inverse @ T)
                for name, label, path, T in self._static
                if name.startswith(prefix)
            ]
            self._static = [
                item for item in self._static if not item[0].startswith(prefix)
            ]
        self.session = uuid.uuid4().hex
        self._trace = PairFaultTrace(self.session)
        self.fault_trace = None
        self._fault_writer = None
        self.revision = 0
        self.refinement = "none"
        self.motion = None
        self.preview = []
        self.error = ""
        self.status = "idle"
        self._fingers = None
        self._object_start = None
        self._scene_token = self.config.scene_token
        self._future = None
        self._future_revision = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wall_tick = time.monotonic()
        self._client_seen = None
        self._stationary_since = None
        self._pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_planner_init,
        )
        self.coordinator = BimanualCoordinator(
            self.runtime.handles, clock=lambda: self.runtime.t, admission=self._admit,
            release_context=getattr(self.runtime, "release_context", None),
        )
        self.coordinator.on_pair_result(self._pair_result)
        if self.config.backend == "remote_rt":
            # In particular, VisualFK's native mesh loading can block Python
            # heartbeat threads. All expensive setup must precede admission.
            self.runtime.connect()
            if not self.runtime.fresh:
                raise ValueError("hardware feedback lost during console startup")
        if start_server:
            self.server = ConsoleServer(
                name="dual-arm console",
                bind="127.0.0.1",
                port=port,
                index="dual.html",
                get=self.get,
                post=self.post,
            )
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dual-arm-console"
        )
        self._thread.start()

    def _pair_result(self, state):
        self.status = state.lifecycle.value
        self.error = state.reason if state.ok() is False else ""

    def _admit(self, plan, states):
        if self.motion is None:
            raise ValueError("no current pair preview")
        self.motion.admit(plan, states, self._scene_token)
        fingers = self.runtime.finger_positions()
        if self._fingers is None or any(
            abs(fingers[a] - self._fingers[a]) > 0.001 for a in fingers
        ):
            raise ValueError("Hand position changed since planning; re-plan")
        if self.kin:
            distance, angle = pose_error(self._object_start, self.runtime.object_pose())
            if distance > 0.001 or angle > np.deg2rad(0.5):
                raise ValueError("shared object moved since planning; re-plan")
            if self.runtime.mode == "shared":
                self._check_closure()

    def _object_errors(self):
        state = getattr(self.runtime, "closure_joint_state", self.runtime.joint_state)()
        q = state["position"]
        inferred = self.kin.objects({a: q[sl] for a, sl in self.runtime.slices.items()})
        errors = [pose_error(T, self.runtime.object_pose()) for T in inferred.values()]
        return np.max(errors, axis=0)

    def _virtual_entry_errors(self):
        q = self.runtime.joint_state()["position"]
        measured = self.kin.tcp({a: q[sl] for a, sl in self.runtime.slices.items()})
        nominal = self.config.shared_object.grasps.targets(self.runtime.object_pose())
        return {a: pose_error(T, nominal[a]) for a, T in measured.items()}

    def _virtual_grasps(self):
        if self.config.shared_object and self.config.shared_object.virtual and self.runtime.mode == "shared":
            return {a: T.tolist() for a, T in self.kin.grasps.object_T_tcp.items()}
        return None

    def _check_closure(self, errors=None):
        distance, angle = self._object_errors() if errors is None else errors
        if (
            not np.isfinite([distance, angle]).all()
            or distance > 0.003
            or angle > np.deg2rad(2)
        ):
            raise ValueError(
                "TCP grasp agreement exceeded 3 mm / 2 degrees: "
                f"measured {distance * 1000:.3f} mm / {np.rad2deg(angle):.3f} degrees"
            )

    def _save_fault_trace(self, reason):
        if self.config.backend != "remote_rt":
            return
        try:
            context = dict(
                config=str(self.config.source),
                config_digest=self.config.content_digest,
                model_tokens=dict(self.config.model_tokens),
                mode=self.runtime.mode,
                pair=self.coordinator.pair_state().to_payload(),
                object_T_tcp=self._virtual_grasps(),
                world_T_base={a: frames.world_T_arm(cfg).copy() for a, cfg in self.config.robots.items()},
                trajectories=None if self.motion is None else {
                    a: dict(times=tr.times.copy(), positions=tr.positions.copy(),
                            velocities=None if tr.velocities is None else tr.velocities.copy())
                    for a, tr in self.motion.trajectories.items()
                },
            )
            # Entries are detached copies, never modified after record(). The
            # writer takes no console lock and cannot delay stop or shutdown.
            samples = list(self._trace.samples)
            recorded = self._trace.recorded

            def write():
                try:
                    self.fault_trace = str(self._trace.save(reason, context, samples=samples, recorded=recorded))
                    print(f"[dual_console] fault trace: {self.fault_trace}", flush=True)
                except Exception as exc:
                    print(f"[dual_console] fault trace could not be saved: {exc}", flush=True)

            self._fault_writer = threading.Thread(target=write, daemon=True, name="dual-fault-trace")
            self._fault_writer.start()
        except Exception as exc:
            # Diagnostics must never replace the fault or prevent stop/close.
            print(f"[dual_console] fault trace could not be saved: {exc}", flush=True)

    def _transition_reason(self, target):
        if self.kin is None:
            return "This setup has no shared object."
        if (
            self._future
            or self.coordinator.lifecycle.active
            or self.coordinator.pair_state().cancellation_pending
        ):
            return "Finish or cancel the active operation first."
        virtual = self.config.shared_object.virtual
        if self.runtime.hand_busy:
            return "Wait for both Hands to stop."
        if not (virtual and target == "independent") and not all(
            s.armed
            and s.ready
            and not s.executing
            and not s.frozen
            and not s.fault_reason
            for s in self.coordinator.arm_states().values()
        ):
            return "Arm both controllers and clear faults first."
        if self._stationary_since is None or self.runtime.t - self._stationary_since < 0.25:
            return "Wait for both arms, Hands and object to stop for 0.25 s."
        if target == "shared":
            if virtual:
                reason = virtual_capture_reason(self._virtual_entry_errors())
                if reason:
                    return reason
            else:
                distance, angle = self._object_errors()
                if (
                    not np.isfinite([distance, angle]).all()
                    or distance > 0.001
                    or angle > np.deg2rad(0.5)
                ):
                    return (
                        "Move both TCPs to the grasp poses (within 1 mm / 0.5 degrees). "
                        f"Measured: {distance * 1000:.2f} mm / {np.rad2deg(angle):.2f} degrees."
                    )
            measured = self.runtime.finger_positions()
            if not virtual and any(
                abs(measured[a] - width) > 0.0005
                or abs(self.runtime.fingers[a] - width) > 0.0005
                for a, width in self.config.shared_object.fingers.items()
            ):
                return "Close both Hands to the configured grasp widths."
        return ""

    def _switch_mode(self, target, body):
        if not isinstance(target, str) or target not in {"independent", "shared"}:
            raise ValueError("mode must be independent or shared")
        if target == self.runtime.mode:
            return
        reason = self._transition_reason(target)
        if reason:
            raise ValueError(reason)
        virtual = self.config.shared_object.virtual
        confirmation = (
            "empty_hands_confirmed" if target == "shared" else "virtual_release_confirmed"
        ) if virtual else (
            "grasps_confirmed" if target == "shared" else "support_confirmed"
        )
        if body.get(confirmation) is not True:
            raise ValueError(
                "Confirm empty Hands for virtual coupling."
                if virtual and target == "shared" else
                "Confirm return to independent arm targets."
                if virtual else
                "Confirm both grasps first."
                if target == "shared"
                else "Confirm external object support before releasing the grasps."
            )
        self._invalidate("control mode changed")
        self.runtime.set_shared_grasps(target == "shared")
        self._scene_token = _scene_token(
            self.config,
            self.runtime.finger_positions(),
            self.runtime.mode,
            self.runtime.mode_revision,
            self._virtual_grasps(),
        )
        self._stationary_since = None

    def _invalidate(self, reason):
        self.coordinator.cancel_pair(reason)
        self.motion, self.preview = None, []
        self.revision += 1
        self.status = "idle"
        self.error = ""

    def _run(self):
        try:
            while not self._stop.is_set():
                deadline = time.monotonic() + self.runtime.period
                with self._lock:
                    self.runtime.step()
                    aligned = getattr(self.runtime, "feedback_ready", True)
                    if not aligned and (
                        self.coordinator.lifecycle.active or self.motion is not None
                        or (self._future and self._future_revision == self.revision)
                    ):
                        self._invalidate("paired feedback interrupted while holding")
                        self.status = "cancelled"
                        self.error = (
                            "Feedback interrupted while holding; plan cancelled. "
                            "Wait for recovery, then re-plan."
                        )
                    self.coordinator.pump()
                    speeds = self.runtime.joint_state()["velocity"]
                    if (
                        np.isfinite(speeds).all()
                        and np.max(np.abs(speeds), initial=0) < 0.01
                    ):
                        if self._stationary_since is None:
                            self._stationary_since = self.runtime.t
                    else:
                        self._stationary_since = None
                    closure = (
                        self._object_errors()
                        if aligned and self.runtime.mode == "shared" and not self.runtime.paused else None
                    )
                    if self.config.backend == "remote_rt":
                        sample = self.runtime.diagnostic_sample()
                        pair_id = self.coordinator.active_pair_id
                        # CTRL-051 lineage: the requested pair identity, the
                        # admitted phase and the pair's DECLARED member plans.
                        # The plans are the identity the recorded chain joins
                        # on — a controller still holding a foreign plan must
                        # not read as this pair's selected reference.
                        stationary = self._stationary_since
                        sample.update(
                            closure_v1=safe_closure_record(
                                moving=stationary is None,
                                stationary_for_s=None if stationary is None else self.runtime.t - stationary,
                                planned=getattr(self.motion, "cooperative_metrics", None),
                                plan_ids=None if pair_id is None else
                                {m.arm_id: m.plan_id for m in self.coordinator.pair_state(pair_id).members},
                                arms=sample.get("arms"), common_time=closure,
                                alignment=sample.get("feedback_alignment")),
                            mode=self.runtime.mode, closure=closure,
                            pair_id=pair_id,
                            lifecycle=self.coordinator.lifecycle.value,
                            revision=self.revision,
                            pair_plans=(
                                {} if pair_id is None else
                                {m.arm_id: m.plan_id for m in
                                 self.coordinator.pair_state(pair_id).members}),
                        )
                        self._trace.record(sample)
                    if closure is not None:
                        self._check_closure(closure)
                    self._wall_tick = time.monotonic()
                    if not self.runtime.fresh and self.coordinator.lifecycle.active:
                        self._invalidate("hardware feedback stale or faulted")
                        self.status, self.error = "cancelled", "Hardware feedback lost; pair cancelled."
                    if (
                        self._client_seen is not None
                        and self._wall_tick - self._client_seen > 3.0
                        and (self.coordinator.lifecycle.active or self.runtime.hand_busy)
                    ):
                        self._invalidate("console disconnected")
                        self.runtime.cancel_hands()
                        self.status = "cancelled"
                        self.error = (
                            "Console disconnected; pair cancelled. Re-plan to continue."
                        )
                    if self._future and self._future.done():
                        future, self._future = self._future, None
                        if self._future_revision == self.revision:
                            try:
                                self.motion, self.preview = future.result()
                                self.status = "preview"
                            except Exception as exc:
                                self.status, self.error = "failed", str(exc)
                        else:
                            # A cancelled job may finish, but cannot restore its preview.
                            try:
                                future.result()
                            except Exception:
                                pass
                self._stop.wait(max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            with self._lock:
                self.coordinator.cancel_pair("console runtime failed")
                self.runtime.stop()
                self.status, self.error = "faulted", str(exc)
                if self.config.backend == "remote_rt" and "paired feedback" in str(exc):
                    # Alignment can fail before the normal sample is recorded.
                    # Preserve its raw packets too, after requesting stop.
                    sample = self.runtime.diagnostic_sample()
                    pair_id = self.coordinator.active_pair_id
                    sample.update(
                        closure_v1=safe_closure_record(
                            moving=self._stationary_since is None,
                            stationary_for_s=(None if self._stationary_since is None
                                              else self.runtime.t - self._stationary_since),
                            planned=getattr(self.motion, "cooperative_metrics", None),
                            plan_ids=None if pair_id is None else
                            {m.arm_id: m.plan_id for m in self.coordinator.pair_state(pair_id).members},
                            arms=sample.get("arms"), common_time=None,
                            alignment=sample.get("feedback_alignment")),
                        mode=self.runtime.mode, closure=None,
                        pair_id=pair_id,
                        lifecycle=self.coordinator.lifecycle.value,
                        revision=self.revision,
                        pair_plans=(
                            {} if pair_id is None else
                            {m.arm_id: m.plan_id for m in
                             self.coordinator.pair_state(pair_id).members}),
                    )
                    self._trace.record(sample)
                print(f"[dual_console] runtime fault: {exc}", flush=True)
                self._save_fault_trace(str(exc))

    def snapshot(self):
        with self._lock:
            self._client_seen = time.monotonic()
            arm_states = self.coordinator.arm_states()
            finger_positions = self.runtime.finger_positions()
            arms = {}
            for a, s in arm_states.items():
                cfg = self.config.robots[a]
                q = self.runtime.controllers[a].last_state.position.copy()
                pose = self.visual[a].poses(q, finger_positions[a])
                world_T = frames.world_T_arm(cfg)
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat(pose["ee"]["q"]).as_matrix()
                T[:3, 3] = pose["ee"]["p"]
                base_T = np.linalg.inv(world_T) @ T

                lo, hi = self.runtime.controllers[a].executor.joint_limits
                arms[a] = dict(
                    s.to_payload(),
                    joints=arm_joints(cfg),
                    q_deg=np.rad2deg(q).tolist(),
                    limits_deg=np.rad2deg(np.vstack([lo, hi])).T.tolist(),
                    visual=pose,
                    world_pose=_pose_values(T),
                    base_pose=_pose_values(base_T),
                    world_T_base=_pose_json(world_T),
                    finger_m=finger_positions[a],
                    target_finger_m=self.runtime.fingers[a],
                    finger_limits_mm=[v * 2000 for v in self.runtime.finger_limits[a]],
                    model_token=self.config.model_tokens[a],
                    tcp=ee_frame(cfg),
                )
            obj = None
            if self.kin:
                distance, angle = self._object_errors()
                obj = dict(
                    name=self.config.shared_object.name,
                    virtual=self.config.shared_object.virtual,
                    pose_source="inferred from first TCP; agreement checked against both" if self.config.shared_object.virtual else "simulated object",
                    pose=_pose_json(self.runtime.object_pose()),
                    values=_pose_values(self.runtime.object_pose()),
                    closure_mm=float(distance * 1000),
                    closure_deg=float(np.rad2deg(angle)),
                    entry_errors={
                        a: dict(position_mm=d * 1000, angle_deg=float(np.rad2deg(r)))
                        for a, (d, r) in self._virtual_entry_errors().items()
                    } if self.config.shared_object.virtual else None,
                    captured_offsets=getattr(self.runtime, "grasp_capture", None),
                    paused=self.runtime.paused,
                    validation=getattr(self.motion, "cooperative_metrics", None),
                    grasp_widths_mm={
                        a: width * 2000
                        for a, width in self.config.shared_object.fingers.items()
                    },
                    grasp_targets={
                        a: _pose_values(T)
                        for a, T in self.kin.grasps.targets(
                            self.runtime.object_pose()
                        ).items()
                    },
                )
            return dict(
                session=self.session,
                revision=self.revision,
                backend=self.runtime.backend_name,
                status=self.status,
                error=self.error,
                fault_trace=self.fault_trace,
                pair=self.coordinator.pair_state().to_payload(),
                connected=self.runtime.fresh and time.monotonic() - self._wall_tick < 0.5,
                telemetry=self.runtime.telemetry(),
                arms=arms,
                shared_object=obj,
                mode=self.runtime.mode,
                refinement=self.refinement,
                mode_revision=self.runtime.mode_revision,
                mode_settled=self._stationary_since is not None
                and self.runtime.t - self._stationary_since >= 0.25,
                transition_reason=self._transition_reason(
                    "shared" if self.runtime.mode == "independent" else "independent"
                ),
                planning=self._future is not None,
                preview_id=None if self.motion is None else self.motion.pair.pair_id,
                duration=None if self.motion is None else self.motion.duration_sec,
                timing=self.runtime.timing_status,
            )

    def get(self, route):
        if route == "state":
            return self.snapshot()
        if route == "geometry":
            return {
                "arms": {
                    a: self.visual[a].scene_json(mesh_prefix=f"mesh/{a}")
                    for a in self.visual
                },
                "static": [
                    dict(
                        mesh=f"static-mesh/{i}",
                        color=[0.55, 0.57, 0.60],
                        scale=[1, 1, 1],
                        p=T[:3, 3].tolist(),
                        q=Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
                    )
                    for i, (_, _, _, T) in enumerate(self._static)
                ],
                "object": [
                    dict(
                        mesh=f"object-mesh/{i}",
                        color=[0.55, 0.64, 0.75],
                        scale=[1, 1, 1],
                        **_pose_json(T),
                    )
                    for i, (_, _, _, T) in enumerate(self._object_meshes)
                ],
            }
        if route == "preview":
            with self._lock:
                return dict(
                    pair_id=None if self.motion is None else self.motion.pair.pair_id,
                    frames=self.preview,
                )
        if route.startswith("static-mesh/"):
            key = route.removeprefix("static-mesh/")
            if key.isdigit() and 0 <= int(key) < len(self._static):
                return file_asset(self._static[int(key)][2])
            return None
        if route.startswith("object-mesh/"):
            key = route.removeprefix("object-mesh/")
            if key.isdigit() and 0 <= int(key) < len(self._object_meshes):
                return file_asset(self._object_meshes[int(key)][2])
            return None
        if route.startswith("mesh/"):
            parts = route.split("/")
            if len(parts) != 3 or parts[1] not in self.visual or not parts[2].isdigit():
                return None
            return file_asset(self.visual[parts[1]].mesh_path(int(parts[2])))
        return None

    def post(self, route, body):
        if route != "action":
            return None
        if not isinstance(body, dict):
            raise ValueError("action body must be an object")
        with self._lock:
            if body.get("session") != self.session:
                raise ValueError("console session changed; reload")
            action = body.get("action")
            if action not in {
                "arm",
                "disarm",
                "gripper",
                "plan",
                "stage",
                "execute",
                "cancel",
                "invalidate",
                "set_mode",
                "plan_grasps",
                "set_refinement",
            }:
                raise ValueError("unknown action")
            if action not in {"cancel", "disarm"}:
                if body.get("revision") != self.revision:
                    raise ValueError("console revision changed; refresh and retry")
                if not self.runtime.fresh or time.monotonic() - self._wall_tick > 0.5:
                    raise ValueError("runtime feedback stale or faulted")
            self._client_seen = time.monotonic()
            if action == "set_refinement":
                choice = refinement_choice(body.get("refinement"))
                if (
                    self._future
                    or self.coordinator.lifecycle.active
                    or self.coordinator.pair_state().cancellation_pending
                ):
                    raise ValueError("finish or cancel the active operation first")
                if choice != self.refinement:
                    self._invalidate("refinement changed")
                    self.refinement = choice
            elif action == "set_mode":
                self._switch_mode(body.get("mode"), body)
            elif action in {"cancel", "invalidate"}:
                if action == "cancel":
                    self.runtime.cancel_hands()
                self._invalidate(
                    "operator cancelled" if action == "cancel" else "targets edited"
                )
            elif action in {"arm", "disarm", "gripper"}:
                a = body.get("arm_id")
                if a not in self.runtime.handles:
                    raise ValueError("unknown arm")
                width = None
                if action == "gripper":
                    if self.runtime.mode == "shared":
                        raise ValueError("fixed shared grasps lock both Hands")
                    controller = self.runtime.controllers[a]
                    if not controller.armed or not controller.ready_sent:
                        raise ValueError(
                            "arm the controller before commanding its Hand"
                        )
                    width = _vector([body.get("width_mm")], 1, "Hand width")[0] / 2000
                    if (
                        not self.runtime.finger_limits[a][0]
                        <= width
                        <= self.runtime.finger_limits[a][1]
                    ):
                        raise ValueError("Hand width outside model limits")
                if (
                    self.coordinator.lifecycle.active
                    or self.coordinator.pair_state().cancellation_pending
                ):
                    self.coordinator.cancel_pair("per-arm action")
                    if action != "disarm":
                        raise ValueError(
                            "pair cancellation requested; wait for confirmation and retry"
                        )
                self._invalidate(f"{a}: {action}")
                if action == "gripper":
                    self.runtime.set_fingers({a: width})
                else:
                    self.runtime.handles[a].set_armed(action == "arm")
            elif action in {"plan", "plan_grasps"}:
                if (
                    self._future is not None
                    or self.coordinator.lifecycle.active
                    or self.coordinator.pair_state().cancellation_pending
                ):
                    raise ValueError("finish or cancel the active operation first")
                targets, active = body.get("targets"), body.get("active_arms")
                object_target = body.get("object_target")
                if self.runtime.mode == "shared" and (
                    self._stationary_since is None
                    or self.runtime.t - self._stationary_since < 0.25
                ):
                    raise ValueError(
                        "Wait for the shared object and both arms to settle before planning."
                    )
                if action == "plan_grasps":
                    if self.kin is None or self.runtime.mode != "independent":
                        raise ValueError(
                            "Prepare grasp poses in independent mode."
                        )
                    widths = self.runtime.finger_positions()
                    if any(
                        widths[a] < w + 0.005
                        for a, w in self.config.shared_object.fingers.items()
                    ):
                        raise ValueError(
                            "Open both Hands at least 10 mm wider than the grasp width before approaching."
                        )
                    targets = {
                        a: dict(kind="pose", frame="world", values=_pose_values(T))
                        for a, T in self.config.shared_object.grasps.targets(
                            self.runtime.object_pose()
                        ).items()
                    }
                    active = list(self.runtime.handles)
                elif self.runtime.mode == "shared":
                    _vector(object_target, 6, "shared object target")
                    if targets is not None or active is not None:
                        raise ValueError(
                            "fixed shared grasps accept only an object target"
                        )
                    targets = {
                        a: dict(kind="pose", frame="world", values=[0] * 6)
                        for a in self.runtime.handles
                    }
                    active = list(self.runtime.handles)
                elif object_target is not None:
                    raise ValueError(
                        "enable shared-object mode before commanding an object target"
                    )
                if not isinstance(targets, dict) or set(targets) != set(
                    self.runtime.handles
                ):
                    raise ValueError("targets must name both arms")
                if (
                    not isinstance(active, list)
                    or not active
                    or len(set(active)) != len(active)
                    or set(active) - set(self.runtime.handles)
                ):
                    raise ValueError("select one or both arms")
                for a, target in targets.items():
                    if not isinstance(target, dict) or target.get("kind") not in {
                        "joint",
                        "pose",
                    }:
                        raise ValueError("target kind must be joint or pose")
                    if target.get("frame") not in {"world", "base"}:
                        raise ValueError("target frame must be world or base")
                    _vector(
                        target.get("values"),
                        len(arm_joints(self.config.robots[a]))
                        if target["kind"] == "joint"
                        else 6,
                        a,
                    )
                self._invalidate("new plan")
                self._fingers = self.runtime.finger_positions()
                self._scene_token = _scene_token(
                    self.config, self._fingers, self.runtime.mode, self.runtime.mode_revision,
                    self._virtual_grasps(),
                )
                self._object_start = self.runtime.object_pose()
                starts = {
                    a: c.last_state.position.copy()
                    for a, c in self.runtime.controllers.items()
                }
                self._future_revision = self.revision
                self._future = self._pool.submit(
                    plan_request,
                    str(self.config.source),
                    self.config.scene_token,
                    starts,
                    targets,
                    active,
                    self._fingers,
                    self._scene_token,
                    self._object_start,
                    object_target,
                    self.runtime.mode,
                    action == "plan_grasps",
                    self.refinement,
                    self._virtual_grasps(),
                )
                self.status = "planning"
            elif action == "stage":
                if (
                    self.motion is None
                    or body.get("pair_id") != self.motion.pair.pair_id
                ):
                    raise ValueError("preview changed; re-plan")
                if body.get("reviewed") is not True:
                    raise ValueError("review both paths before staging")
                self.coordinator.stage_pair(self.motion.pair)
                self.status = self.coordinator.lifecycle.value
            elif action == "execute":
                if (
                    self.motion is None
                    or body.get("pair_id") != self.motion.pair.pair_id
                ):
                    raise ValueError("preview changed; re-plan")
                self.coordinator.execute_pair(self.motion.pair.pair_id)
                self.status = self.coordinator.lifecycle.value
            return dict(ok=True, revision=self.revision)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        with self._lock:
            self.coordinator.cancel_pair("console shutdown")
            self.coordinator.pump()
        if self.server:
            self.server.close()
        self.runtime.close()
        if self._fault_writer:
            self._fault_writer.join(timeout=1)
            if self._fault_writer.is_alive():
                print("[dual_console] fault trace writer still pending; shutdown will not wait", flush=True)
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._mesh_cache.cleanup()


def main(config_path, *, port=7520):
    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    app = DualArmConsole(config_path, port=port)
    print(
        f"[dual_console] {app.config.backend}, DISARMED: http://127.0.0.1:{app.server.port}",
        flush=True,
    )
    try:
        while app._thread.is_alive() and not shutdown.stop_requested:
            time.sleep(0.2)
    finally:
        app.close()
