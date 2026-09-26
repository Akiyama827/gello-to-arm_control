"""Two existing controllers over session-bound RT links; no simulated feedback.

Each arm retains its RT authority and deadman. PC barrier release is not a
scheduled RT start. Rigid shared-object motion requires further qualification.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time

import numpy as np

from arm_control.config import arm_joints, gripper_joints
from arm_control.control.factory import build_arm_controller
from arm_control.end_effectors.franka_adapter import BridgeClient
from arm_control.messages import (
    pack_control_update, pack_json_message, pack_motor_state_dict,
    unpack_json_message, unpack_motor_command,
)
from arm_control.control.inprocess import QueueControllerNode
from arm_control.multi_arm.interface import LocalArmHandle
from arm_control.multi_arm.synchronization import SharedReferenceClock
from arm_control.multi_arm.feedback import FeedbackGap, align_feedback
from arm_control.multi_arm.cooperative import (
    CooperativeKinematics, RigidGrasps, pose_error, virtual_capture_reason,
)
from arm_control.plants.remote_rt.client import RtBackend
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld


class _HardwareNode(QueueControllerNode):
    def __init__(self, runtime, arm):
        super().__init__()
        self.runtime, self.arm = runtime, arm

    def send_output(self, topic, value):
        if topic == "arm":
            self.runtime._request_authority(self.arm, bool(unpack_json_message(value)["armed"]))
        else:
            super().send_output(topic, value)


class _HardwareHandle(LocalArmHandle):
    def __init__(self, controller, runtime):
        super().__init__(controller, clock=time.monotonic)
        self.runtime = runtime

    def set_armed(self, armed):
        if armed and self.runtime._pending[self.arm_id]:
            raise ValueError("wait for the current authority acknowledgement before arming")
        super().set_armed(armed)

    def state(self):
        state = super().state()
        health = self.runtime.links[self.arm_id].motor_health()
        return replace(state, armed=bool(health["armed"]),
                       ready=(state.ready and self.runtime.feedback_ready
                              and not self.runtime._pending[self.arm_id]),
                       fault_reason=health.get("latched_fault") or self.runtime._faults.get(self.arm_id))


class RemoteHardware:
    """Prepare locally, then connect after the viewer has loaded its assets.

    Nonblocking console ticks; bounded authority RPCs run per arm off-loop.
    """

    backend_name = "remote_rt"
    timing_status = "Hardware: common PC references; physical synchronization unqualified"
    mode = "independent"
    mode_revision = 0
    paused = False

    def __init__(self, config, *, backend_factory=None, hand_factory=None):
        if config.shared_object is not None and not config.shared_object.virtual:
            raise ValueError("physical hardware shared-object control is not qualified")
        self.config = config
        self.mode, self.mode_revision = "independent", 0
        self.kinematics = CooperativeKinematics(config) if config.shared_object else None
        self._object_pose = None if self.kinematics is None else config.shared_object.world_T_object.copy()
        self.grasp_capture = None
        self.controllers, self.handles, self.slices = {}, {}, {}
        self.links, self.hands, self._rpc = {}, {}, {}
        self._pending, self._seen, self._faults = {}, {}, {}
        self._authority_generation = {}
        self.fingers, self.finger_limits = {}, {}
        self._measured_fingers = {}
        self._hand_results = {}
        self._observations, self._observed_health = {}, {}
        self._closure_positions, self._feedback_alignment = None, None
        self._aligned_at = self._holding_since = None
        self._closed = False
        self._connect_started = False
        self.period = 0.01
        self.reference_clock = SharedReferenceClock()
        self.release_context = self.reference_clock.batch
        self._reference_time = None
        backend_factory = backend_factory or RtBackend.from_config
        self._hand_factory = hand_factory or BridgeClient
        offset = 0
        try:
            for actor in config.scene.actors:
                a, cfg = actor.name, config.robots[actor.name]
                joints = arm_joints(cfg)
                # Model binding is enforced for remote_rt by load_pair_config.
                c = build_arm_controller(
                    cfg, _HardwareNode(self, a), arm_id=a, require_policy=True,
                    clock=time.monotonic, reference_clock=self.reference_clock,
                    plant_reports_health=True,
                )
                if abs(c.execution_policy.period - self.period) > 1e-9:
                    raise ValueError("pair hardware controllers require a common 100 Hz policy")
                self.controllers[a] = c
                self.handles[a] = _HardwareHandle(c, self)
                self.slices[a] = slice(offset, offset + len(joints))
                offset += len(joints)
                self.links[a] = backend_factory(cfg)
                self._rpc[a] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{a}-authority")
                self._pending[a] = []
                self._authority_generation[a] = 0
                self._seen[a] = None
                self.finger_limits[a] = tuple(cfg.get("gripper_range_m") or (0., .04))
            # Native model/mesh loaders can hold Python's GIL longer than the
            # RT heartbeat deadline. Finish them before opening any transport.
            self.world = MuJoCoCollisionWorld.from_scene(
                config.scene, config.scene.state(), config.scene.actor_names,
                ground_z=config.ground_z, workspace_xy=config.workspace_xy,
            )
        except BaseException:
            self.close()
            raise

    def connect(self):
        """Admit both links and replace staged geometry state with measurements."""
        if self._closed or self._connect_started:
            raise ValueError("hardware runtime must be connected exactly once")
        self._connect_started = True
        try:
            # Admission of both links completes before presenting the console.
            opened = [self._rpc[a].submit(link.open) for a, link in self.links.items()]
            for future in opened:
                future.result()
            for a, cfg in self.config.robots.items():
                rt = cfg.get("rt") or {}
                hand = self._hand_factory(rt["host"], int(rt["hand_port"]))
                self.hands[a] = hand
                hand.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                self.step()
                if self.fresh:
                    break
                time.sleep(.01)
            if not self.fresh:
                raise ValueError("both arms and Hands must provide fresh measured feedback")
            self.fingers.update(self.finger_positions())
            self.world.set_held_positions({
                f"{a}__{j}": self.fingers[a]
                for a, cfg in self.config.robots.items() for j in gripper_joints(cfg)
            })
            self.world._fk(self.joint_state()["position"])
        except BaseException:
            self.close()
            raise

    @property
    def t(self):
        return time.monotonic()

    @property
    def scene_model(self):
        return self.world.model

    @property
    def scene_data(self):
        return self.world.data

    @property
    def fresh(self):
        return self.feedback_ready and len(self.hands) == 2 and all(
            self.links[a].motor_health()["state_fresh"]
            and not self.links[a].motor_health()["any_fault"]
            and not self._faults.get(a)
            and self.controllers[a]._last_state_at is not None
            and self.hands[a].snapshot().get("measured") is True
            for a in self.controllers
        )

    @property
    def feedback_ready(self):
        return self.mode != "shared" or not (self._feedback_alignment or {}).get("error")

    def joint_state(self):
        return {
            key: np.concatenate([getattr(c.last_state, key) for c in self.controllers.values()])
            for key in ("position", "velocity")
        }

    def _align_feedback(self, histories=None, *, allow_wait=False):
        if histories is None:
            histories = {a: link.recent_states() for a, link in self.links.items()}
        now = time.monotonic()
        try:
            positions, report = align_feedback(self.links, histories, now=now)
        except ValueError as exc:
            self._feedback_alignment = dict(error=str(exc), **getattr(exc, "details", {}))
            # Only a settled joint hold can wait briefly for a new bracket.
            # No trajectory clock is paused/resumed; pending plans lose admission.
            # Invalid data, unknown authority, active motion and prolonged loss
            # still reach the console's stop-both path immediately.
            if (isinstance(exc, FeedbackGap) and allow_wait
                    and self._aligned_at is not None and now - self._aligned_at <= .05):
                return
            raise
        self._closure_positions, self._feedback_alignment = positions, report
        # Re-evaluating unchanged packets must not renew the recovery budget.
        self._aligned_at = min(history[-1][1] for history in histories.values())

    def _update_holding(self, now):
        holding = all(
            c.armed and c.bridge_armed and c.ready_sent and c._health_ok
            and not c.stopped and not c.frozen
            and c._last_state_at is not None and not c._running
            and not c._jog.active and c.mode_law == "joint"
            and c.last_command is not None and not np.any(c.last_command.qd_des)
            and np.max(np.abs(c.last_state.velocity)) < .01
            for c in self.controllers.values()
        )
        if not holding:
            self._holding_since = None
        elif self._holding_since is None:
            self._holding_since = now
        return self._holding_since is not None and now - self._holding_since >= .25

    def closure_joint_state(self):
        """Shared closure uses common-time estimates; control uses raw states."""
        if self.mode != "shared":
            return self.joint_state()
        return {"position": np.concatenate(list(self._closure_positions.values()))}

    def finger_positions(self):
        for a, hand in self.hands.items():
            state = hand.snapshot()
            if state.get("measured") is True:
                self._measured_fingers[a] = float(state["width"]) / 2
        # A stale display retains its last measurement; fresh=False disables
        # motion. Reconnection must never display a fabricated open Hand.
        return dict(self._measured_fingers)

    def set_fingers(self, values):
        for a, width in values.items():
            if not np.isfinite(width) or not self.finger_limits[a][0] <= width <= self.finger_limits[a][1]:
                raise ValueError("Hand target outside configured limits")
            hand = self.hands[a]
            if not hand.send_line(
                f"MOVE {width * 2:.9g} 0.05",
                {"request_id": f"console-{time.monotonic_ns()}", "_position_move": True},
            ):
                raise ValueError(f"{a}: Hand unavailable, busy or missing active-stop support")
            self.fingers[a] = width
            self._hand_results.pop(a, None)

    def object_pose(self):
        if self.kinematics is None:
            return None
        if self.mode == "shared":
            # A virtual pose inferred from the first time-aligned TCP, never an
            # object sensor measurement. The console checks agreement of BOTH
            # TCP-inferred poses and stops if they diverge.
            poses = self.kinematics.objects(self._closure_positions)
            return next(iter(poses.values())).copy()
        return self._object_pose.copy()

    @property
    def hand_busy(self):
        return any(h.snapshot().get("busy") for h in self.hands.values())

    def cancel_hands(self):
        for hand in self.hands.values():
            hand.cancel_move()

    def stop(self):
        """Queue disarm immediately, even when no further console tick can run."""
        self.cancel_hands()
        for c in self.controllers.values():
            c.on_control(pack_control_update(cancel=True, arm=False, reason="runtime stop"))

    def _request_authority(self, a, armed):
        if armed and self._closed:
            raise ValueError("hardware runtime is closed")
        pending = self._pending[a]
        if not armed:
            if a in self.hands:
                self.hands[a].cancel_move()
            if any(not desired and not f.done() for f, desired in pending):
                return
            # The in-flight RPC is bounded by its ACK timeout. Drop all queued
            # enables; one disarm follows it and re-arm is blocked until ACK.
            for f, _ in pending:
                f.cancel()
        self._authority_generation[a] += 1
        generation = self._authority_generation[a]

        def apply():
            if armed and (self._closed or generation != self._authority_generation[a]):
                return None
            link = self.links[a]
            return link.enable_all() if armed else link.safe_stop()

        pending.append((self._rpc[a].submit(apply), armed))

    def set_shared_grasps(self, enabled):
        obj = self.config.shared_object
        if obj is None or not obj.virtual:
            raise ValueError("physical shared-object hardware motion is not qualified")
        mode = "shared" if enabled else "independent"
        if mode == self.mode:
            return
        if not self.fresh or self.hand_busy:
            raise ValueError("virtual mode needs fresh feedback and idle Hands")
        pose = self.object_pose()
        captured = None
        if enabled:
            self._align_feedback()
            measured = self.kinematics.tcp(self._closure_positions)
            nominal = obj.grasps.targets(pose)
            errors = {a: pose_error(T, nominal[a]) for a, T in measured.items()}
            reason = virtual_capture_reason(errors)
            if reason:
                raise ValueError(reason)
            captured = RigidGrasps({a: np.linalg.inv(pose) @ T for a, T in measured.items()})
        self.world.set_held_positions({f"{a}__{j}": width
                                       for a, width in self.finger_positions().items()
                                       for j in gripper_joints(self.config.robots[a])})
        self.world.set_free_body_pose(obj.body, pose)
        if self.world.in_collision(self.joint_state()["position"]):
            raise ValueError("current scene collides; cannot change virtual coupling")
        # No physical weld, Hand command or executor payload update.
        # Commit the measured frames only after every admission check passes.
        # They remain fixed for this shared-mode session; never re-fit in flight.
        self.kinematics.grasps = captured if enabled else obj.grasps
        self.grasp_capture = (
            {a: dict(position_mm=d * 1000, angle_deg=float(np.rad2deg(r)))
             for a, (d, r) in errors.items()} if enabled else None
        )
        self._object_pose = pose.copy()
        self.mode, self.mode_revision = mode, self.mode_revision + 1

    def telemetry(self):
        # Server timestamps remain in their native clocks, with no inferred
        # equivalence between two hosts or between PC receipt and capture.
        return {a: dict(link.motor_health(), authority_pending=bool(self._pending[a]),
                        hand=dict(self.hands[a].snapshot(), last_move=self._hand_results.get(a)))
                for a, link in self.links.items()}

    def diagnostic_sample(self):
        """Copy the exact packets consumed by the controllers on this PC tick."""
        now = time.monotonic()
        arms = {}
        for a, c in self.controllers.items():
            packet, received = self._observations.get(a, (None, None))
            motor = self.links[a].motor_state(state=packet) if packet is not None else None
            command = c.last_command
            arms[a] = dict(
                rt_host=self.links[a].config.host,
                received_at_pc=received,
                state_t_mono_ns=None if packet is None else packet.t_mono_ns,
                state_seq=None if packet is None else packet.state_seq,
                last_cmd_seq=None if packet is None else packet.last_cmd_seq,
                rt_session=None if packet is None else f"{packet.session_id:016x}",
                health=dict(self._observed_health.get(a, {})),
                controller_measurement_valid=c._last_state_at is not None,
                controller_observed_at_pc=c._last_state_at,
                q_controller=c.last_state.position.tolist(),
                dq_controller=c.last_state.velocity.tolist(),
                q_rt=None if motor is None else motor["position"].tolist(),
                dq_rt=None if motor is None else motor["velocity"].tolist(),
                q_cmd_rt=None if motor is None else motor["position_cmd"].tolist(),
                dq_cmd_rt=None if motor is None else motor["velocity_cmd"].tolist(),
                q_cmd_pc=None if command is None else command.q_des.tolist(),
                dq_cmd_pc=None if command is None else command.qd_des.tolist(),
                plan_id=c.current_plan_id,
                trajectory_elapsed_s=(c.reference.elapsed(self._reference_time)
                                      if c.reference is not None and self._reference_time is not None else None),
            )
        return dict(pc_monotonic=now, reference_time_pc=self._reference_time, arms=arms,
                    feedback_ready=self.feedback_ready,
                    feedback_alignment=self._feedback_alignment if self.mode == "shared" else None)

    def step(self):
        histories = {a: link.recent_states() for a, link in self.links.items()}
        for a, c in self.controllers.items():
            if a in self.hands:
                for result in self.hands[a].pop_move_results():
                    self._hand_results[a] = result
            link = self.links[a]
            pending = self._pending[a]
            while pending and pending[0][0].done():
                future, _ = pending.pop(0)
                if future.cancelled():
                    continue
                try:
                    if future.result() is False:
                        raise RuntimeError("RT stop acknowledgement missing")
                    self._faults.pop(a, None)
                except Exception as exc:
                    self._faults[a] = str(exc)
            health = link.motor_health()
            if self._faults.get(a):
                health = dict(health, any_fault=True, latched_fault=self._faults[a])
            self._observed_health[a] = health
            c.on_motor_health(pack_json_message("motor_health", health))
            state, received = histories[a][-1] if histories[a] else (None, 0.)
            if state is not None and health["state_fresh"] and state.state_seq != self._seen[a]:
                self._seen[a] = state.state_seq
                self._observations[a] = (state, received)
                c.on_motor_state(pack_motor_state_dict(link.motor_state(state=state)))
                # Preserve arrival freshness when the console loop was delayed.
                # A rejected non-finite sample deliberately clears this stamp.
                if c._last_state_at is not None:
                    c._last_state_at = received
        holding = self._update_holding(time.monotonic())
        if self.mode == "shared":
            if all(c.armed and c.bridge_armed and c.ready_sent
                   for c in self.controllers.values()):
                self._align_feedback(histories, allow_wait=holding)
            else:
                # ARM/DISARM changes each arm's authority epoch at a slightly
                # different instant, so a common time cannot be bracketed
                # across it -- a clean disarm used to end in "cannot bracket
                # common time" and a runtime fault. Nothing moves the pair
                # until both arms are armed and acknowledged; withhold
                # readiness (the console then cancels pending work and skips
                # closure) and align again once both are.
                self._feedback_alignment = dict(error="pair not armed")
        # Receive both arms before doing dynamics work; sample both trajectories
        # at the same PC time, then dispatch the two resulting commands together.
        # This removes computation-order skew, not network/physical-cycle skew.
        with self.reference_clock.batch() as sampled:
            self._reference_time = sampled
            for c in self.controllers.values():
                c.tick()
        for a, c in self.controllers.items():
            link = self.links[a]
            keep = []
            for topic, value in c.node.sent:
                if topic == "motor_command":
                    link.apply_command(unpack_motor_command(value, c.n_arm))
                elif topic == "controller_event":
                    keep.append((topic, value))
            c.node.sent = keep

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.cancel_hands()
        # Also covers partially constructed runtimes. Invalidate queued ARM
        # work before transport teardown joins the same executor queue.
        for a in self._pending:
            self._request_authority(a, False)
        closing = [self._rpc[a].submit(link.close) for a, link in self.links.items()]
        for future in closing:
            try:
                future.result(timeout=6)
            except Exception:
                pass
        for pool in self._rpc.values():
            pool.shutdown(wait=True, cancel_futures=True)
        for hand in self.hands.values():
            hand.close()
