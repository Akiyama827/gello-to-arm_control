"""The dual console's runtime over two arm supervisors: the only one (CTRL-053, operator 2026-10-09: every arm on its supervisor).

Each arm's control authority is its supervisor (the one process that owns the RT session, or the simulated pair's hosted
world, ``multi_arm.launch``): this runtime holds no controller and no RT link, only an ``ArmClient`` and a ``SdkArmHandle``
per arm, plus the Hands. ARM and DISARM run off the console loop and report through the handle's state.

Shared-object coupling is VIRTUAL, in simulation as on the bench (operator 2026-10-09): empty Hands, the object inferred from
both TCPs. Its closure uses common-time feedback: each supervisor's recent packets (state_history/1) bracketed at one RT stamp
(``supervised_feedback.align_arms``; one attested clock domain: one RT box, or this PC for the simulated pair). Pair legs start
at one PC-monotonic stamp (``release_context``, scheduled_release/1), so both references run on one phase as the old
in-process pair's did; each supervisor's controller anchors its trajectory there. A supervisor whose RT plays the reference
itself (native) refuses it: an RT-scheduled start is CTRL-040 group/1, not wired here. Physical synchronization stays
unqualified.
"""

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
import time

import numpy as np

from arm_control.config import arm_joints, gripper_joints
from arm_control.end_effectors.franka_adapter import BridgeClient
from arm_control.multi_arm.cooperative import CooperativeKinematics, RigidGrasps, pose_error, virtual_capture_reason
from arm_control.multi_arm.feedback import FeedbackGap
from arm_control.multi_arm.group import GROUP_BUDGET_S, GROUP_MIN_LEAD_MS
from arm_control.multi_arm.hands import HandControl
from arm_control.multi_arm.launch import pair_sockets
from arm_control.multi_arm.supervised import AsyncAuthorityHandle, SdkArmHandle
from arm_control.multi_arm.supervised_feedback import align_arms
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.execution.contract import Phase, Stamp
from arm_control.sdk import ArmClient, SupervisorError


class SupervisedHardware(HandControl):
    backend_name = 'supervised'
    timing_status = 'Supervisor: references through each arm supervisor; physical synchronization unqualified'
    mode = 'independent'
    mode_revision = 0
    period = 0.01

    def __init__(self, config, *, arm_factory=None, hand_factory=None, world_factory=None, env=None):
        import os

        self.config = config
        self.kinematics = CooperativeKinematics(config) if config.shared_object else None
        self._object_pose = None if self.kinematics is None else config.shared_object.world_T_object.copy()
        self.grasp_capture = None
        self.clients, self.handles, self.slices = {}, {}, {}
        self.hands, self.fingers, self.finger_limits = {}, {}, {}
        self._measured_fingers, self._hand_results = {}, {}
        self._sample = {}
        self._trace_seq, self._trace_dropped, self._trace_errors, self._trace_latest = {}, {}, {}, {}
        self._trace_ok = {}  # whether the last read of the supervisor's trace worked
        self._closed = False
        self._connected = False
        self._closure_positions, self._feedback_alignment = None, None
        self._aligned_at = self._holding_since = None
        self._hand_factory = hand_factory or BridgeClient
        sockets = pair_sockets(os.environ if env is None else env)  # the launcher's sockets; else each instance's default
        arm_factory = arm_factory or (lambda a, cfg: ArmClient((cfg.get('rt') or {}).get('instance_id', a),
                                                               socket_path=sockets.get(a)))
        offset = 0
        try:
            for actor in config.scene.actors:
                a, cfg = actor.name, config.robots[actor.name]
                n = len(arm_joints(cfg))
                client = arm_factory(a, cfg)
                self.clients[a] = client
                self.handles[a] = AsyncAuthorityHandle(
                    SdkArmHandle(client, a, model_token=config.model_tokens.get(a)),
                    is_closed=lambda: self._closed, on_disarm=self._cancel_hand)
                self.slices[a] = slice(offset, offset + n)
                self._sample[a] = (np.zeros(n), np.zeros(n), False)
                self._trace_seq[a] = self._trace_dropped[a] = self._trace_errors[a] = 0
                self._trace_ok[a] = True
                offset += n
                self.finger_limits[a] = tuple(cfg.get('gripper_range_m') or (0., .04))
            world_factory = world_factory or (lambda c: MuJoCoCollisionWorld.from_scene(
                c.scene, c.scene.state(), c.scene.actor_names, ground_z=c.ground_z, workspace_xy=c.workspace_xy))
            self.world = world_factory(config)
        except BaseException:
            self.close()
            raise

    def connect(self):
        """Claim both arms (nothing is armed or moved), start the Hands, wait for fresh measured feedback."""
        if self._closed or self._connected:
            raise ValueError('hardware runtime must be connected exactly once')
        self._connected = True
        try:
            unscheduled = [a for a, c in self.clients.items() if c.extensions.get('scheduled_release') != 1]
            if unscheduled:  # every pair leg starts at one stamp: a native-reference supervisor's RT plays its own start
                raise ValueError(f'{unscheduled}: the pair needs scheduled_release/1 (a native-reference supervisor '
                                 'cannot take one; an RT-side group start is not wired to the console)')
            for client in self.clients.values():
                client.claim()
            for a, cfg in self.config.robots.items():
                rt = cfg.get('rt') or {}
                hand = self._hand_factory(rt['host'], int(rt['hand_port']))
                self.hands[a] = hand
                hand.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                self.step()
                if self.fresh:
                    break
                time.sleep(.01)
            if not self.fresh:
                raise ValueError('both arms and Hands must provide fresh measured feedback')
            self.fingers.update(self.finger_positions())
            self.world.set_held_positions({
                f'{a}__{j}': self.fingers[a]
                for a, cfg in self.config.robots.items() for j in gripper_joints(cfg)})
            self.world._fk(self.joint_state()['position'])
        except BaseException:
            self.close()
            raise

    # -- observation ------------------------------------------------------------------------------------------
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
    def feedback_ready(self):
        return self.mode != 'shared' or not (self._feedback_alignment or {}).get('error')

    @property
    def fresh(self):
        return len(self.hands) == 2 and all(
            self._sample[a][2] and not self.handles[a].state().fault_reason
            and self.hands[a].snapshot().get('measured') is True
            for a in self.handles)

    def joint_state(self):
        return {key: np.concatenate([self._sample[a][i] for a in self.handles])
                for key, i in (('position', 0), ('velocity', 1))}

    def closure_joint_state(self):
        """Shared closure uses common-time estimates; control uses each arm's own state."""
        if self.mode != 'shared':
            return self.joint_state()
        return {'position': np.concatenate([self._closure_positions[a] for a in self.handles])}

    def object_pose(self):
        if self.kinematics is None:
            return None
        if self.mode == 'shared':
            # A virtual pose inferred from the first time-aligned TCP, never an object sensor measurement; the console checks
            # that BOTH TCP-inferred poses agree and stops if they diverge.
            return next(iter(self.kinematics.objects(self._closure_positions).values())).copy()
        return self._object_pose.copy()

    def _align_feedback(self, *, allow_wait=False):
        now = time.monotonic()
        try:
            positions, report = align_arms(self.clients)
        except ValueError as exc:
            self._feedback_alignment = dict(error=str(exc), **getattr(exc, 'details', {}))
            # Only a settled joint hold may wait briefly for a new bracket; motion, invalid data and prolonged loss reach the
            # console's stop-both path at once.
            if isinstance(exc, FeedbackGap) and allow_wait and self._aligned_at is not None and now - self._aligned_at <= .05:
                return
            raise
        # Re-evaluating unchanged packets must not renew the recovery budget: it runs from their receipt, not this call.
        self._closure_positions, self._feedback_alignment, self._aligned_at = positions, report, report['received_at']

    def _update_holding(self, now, states=None):
        states = [h.state() for h in self.handles.values()] if states is None else states
        holding = all(s.armed and s.ready and not s.executing and not s.stopped and not s.frozen and s.mode_law == 'joint'
                      and s.velocity is not None and np.max(np.abs(s.velocity)) < .01 for s in states)
        if not holding:
            self._holding_since = None
        elif self._holding_since is None:
            self._holding_since = now
        return self._holding_since is not None and now - self._holding_since >= .25

    def set_shared_grasps(self, enabled):
        """Virtual coupling on or off: the grasps are captured from the measured, time-aligned TCPs (never re-fit in flight);
        no weld, Hand command or payload change."""
        obj = self.config.shared_object
        if obj is None:
            raise ValueError('this setup has no virtual shared object')
        mode = 'shared' if enabled else 'independent'
        if mode == self.mode:
            return
        if not self.fresh or self.hand_busy:
            raise ValueError('virtual mode needs fresh feedback and idle Hands')
        pose = self.object_pose()
        captured = errors = None
        if enabled:
            self._align_feedback()
            measured = self.kinematics.tcp(self._closure_positions)
            nominal = obj.grasps.targets(pose)
            errors = {a: pose_error(T, nominal[a]) for a, T in measured.items()}
            reason = virtual_capture_reason(errors)
            if reason:
                raise ValueError(reason)
            captured = RigidGrasps({a: np.linalg.inv(pose) @ T for a, T in measured.items()})
        self.world.set_held_positions({f'{a}__{j}': width for a, width in self.finger_positions().items()
                                       for j in gripper_joints(self.config.robots[a])})
        self.world.set_free_body_pose(obj.body, pose)
        if self.world.in_collision(self.joint_state()['position']):
            raise ValueError('current scene collides; cannot change virtual coupling')
        self.kinematics.grasps = captured if enabled else obj.grasps
        self.grasp_capture = ({a: dict(position_mm=d * 1000, angle_deg=float(np.rad2deg(r)))
                               for a, (d, r) in errors.items()} if enabled else None)
        self._object_pose = pose.copy()
        self.mode, self.mode_revision = mode, self.mode_revision + 1

    @contextmanager
    def release_context(self):
        """The coordinator releases both members inside this. Their releases are collected, then sent together with one start
        stamp, a CTRL-040 budget ahead of the first supervisor's clock (both run on this PC's clock; each refuses a stamp
        outside 10-150 ms ahead or on another clock). Every release must be confirmed before the stamp minus the minimum lead:
        otherwise the members that were accepted are cancelled before they can move, and the release fails (one arm never
        plays a pair leg alone while its partner's release hangs or is refused). The stamp is taken when the releases are sent,
        so the collection does not eat into the budget."""
        inner = [h._inner for h in self.handles.values()]
        deferred = []
        for handle in inner:
            handle.deferred = deferred
        try:
            yield
        finally:
            for handle in inner:
                handle.deferred = None
        if len(deferred) == len(inner):  # an in-loop abort leaves fewer: a lone member never starts a pair leg
            now = next(iter(self.clients.values())).now()
            self._release_together(deferred, Stamp(now.ns + int(GROUP_BUDGET_S * 1e9), now.clock, now.domain))

    @staticmethod
    def _release_together(releases, stamp):
        pool = ThreadPoolExecutor(max_workers=len(releases), thread_name_prefix='pair-release')
        try:
            futures = [pool.submit(arm.release, eid, start_at=stamp) for arm, eid in releases]
            # the stamp is on this PC's monotonic clock (the supervisors' domain, checked by each of them)
            wait(futures, timeout=max(0.0, (stamp.ns - time.monotonic_ns()) / 1e9 - GROUP_MIN_LEAD_MS / 1e3))
        finally:
            pool.shutdown(wait=False)  # a hung release must not hold the console
        def outcome(future):
            """None: accepted for the stamp. A refusal at selection (not armed, start pose, stale observation) comes back
            as a terminal status, not an exception: it is a failure too."""
            if not future.done():
                return 'not confirmed before the scheduled start'
            if future.exception() is not None:
                return str(future.exception())
            status = future.result()
            if status.terminal or status.phase is Phase.ADMITTED:  # ADMITTED: the controller did not select it
                return f'{status.phase.value}: {status.reason or "not selected"}'
            return None

        reasons = {eid: outcome(f) for (_, eid), f in zip(releases, futures)}
        if not any(reasons.values()):
            return
        for arm, eid in releases:
            if reasons[eid] is None:
                arm.cancel(eid, 'partner release not accepted before the scheduled start')
        raise ValueError('pair release failed: ' + '; '.join(f'{eid}: {r}' for eid, r in reasons.items() if r))

    def step(self):
        states = []
        for a, handle in self.handles.items():
            state = handle.state()
            states.append(state)
            if state.position is not None and state.velocity is not None:
                self._sample[a] = (np.asarray(state.position, float), np.asarray(state.velocity, float), True)
            else:  # an invalid measurement keeps the last good pose for display but is no longer a valid one
                self._sample[a] = (*self._sample[a][:2], False)
            if a in self.hands:
                for result in self.hands[a].pop_move_results():
                    self._hand_results[a] = result
            self._read_trace(a)
        holding = self._update_holding(time.monotonic(), states)  # every step: re-entering shared mode needs a fresh hold
        if self.mode == 'shared':
            if all(s.armed and s.ready for s in states):
                self._align_feedback(allow_wait=holding)
            else:
                # ARM/DISARM changes each arm's authority epoch at a slightly different instant, so a common time cannot be
                # bracketed across it: withhold readiness until both arms are armed and acknowledged.
                self._feedback_alignment = dict(error='pair not armed')

    def _read_trace(self, a):
        """Read the supervisor's controller ticks since the last read (trace/1). The ring is bounded: a read that comes back
        late reports the ticks that fell off as ``dropped``, and they are counted, not hidden. A supervisor without trace/1
        has no trace; one that does not answer leaves the previous sample, marked ``trace_stale``. The client never
        reconnects, so a restarted supervisor is a lost channel, not a sequence that starts again."""
        client = self.clients[a]
        if client.extensions.get('trace') != 1:
            return
        try:
            reply = client.trace_samples(self._trace_seq[a])
        except (SupervisorError, TimeoutError, OSError):
            self._trace_ok[a] = False  # the held sample is now old: it is marked, never presented as current
            return
        self._trace_ok[a] = True
        self._trace_dropped[a] += reply['dropped']
        self._trace_errors[a] = max(self._trace_errors[a], reply['errors'])
        if reply['samples']:
            self._trace_latest[a] = reply['samples'][-1]
            self._trace_seq[a] = reply['samples'][-1]['seq']

    def telemetry(self):
        out = {}
        for a, handle in self.handles.items():
            state = handle.state()
            out[a] = dict(armed=bool(state.armed), ready=bool(state.ready), fault=state.fault_reason,
                          session=state.session_id, authority_pending=bool(handle._pending),
                          hand=dict(self.hands[a].snapshot(), last_move=self._hand_results.get(a)))
        return out

    def diagnostic_sample(self):
        """The supervisor's newest controller tick per arm (trace/1: the controller diagnostic, with its own
        sequence), the measured joints, and the count of ticks that fell off the supervisor's ring before they were read
        (``trace_dropped``). Without trace/1 only the measured joints are present, and nothing is filled in."""
        arms = {a: dict(dict(source='supervisor', q_measured=self._sample[a][0].tolist(),
                             dq_measured=self._sample[a][1].tolist(), measurement_valid=self._sample[a][2],
                             fault_reason=self.handles[a].state().fault_reason),
                        **(self._trace_latest.get(a) or {}), trace_seq=self._trace_seq[a],
                        trace_dropped=self._trace_dropped[a], trace_errors=self._trace_errors[a],
                        trace_stale=not self._trace_ok[a])
                for a in self.handles}
        return dict(pc_monotonic=time.monotonic(), reference_time_pc=None, arms=arms, feedback_ready=self.feedback_ready,
                    feedback_alignment=self._feedback_alignment if self.mode == 'shared' else None)

    # -- actions ----------------------------------------------------------------------------------------------
    def _cancel_hand(self, a):
        if a in self.hands:
            self.hands[a].cancel_move()

    def stop(self):
        """Queue a DISARM on every arm now, even when no further console tick can run."""
        self.cancel_hands()
        for handle in self.handles.values():
            handle.set_armed(False)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.cancel_hands()
        for handle in self.handles.values():
            try:
                handle.set_armed(False)
            except Exception:
                pass
        for handle in self.handles.values():
            handle.close()  # waits for the queued DISARM (bounded by the SDK timeout)
        for client in self.clients.values():
            client.close()
        for hand in self.hands.values():
            hand.close()
