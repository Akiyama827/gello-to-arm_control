"""The dual console's runtime over two arm supervisors (independent mode).

A drop-in for ``RemoteHardware`` where each arm's control authority is its supervisor (the one process that owns the RT
session): this runtime holds no controller and no RT link, only an ``ArmClient`` and a ``SdkArmHandle`` per arm, plus the
project's Hands. ARM and DISARM run off the console loop and report through the handle's state, as they do on
``RemoteHardware``. Shared-object coupling needs the PC-side closure and stays on ``RemoteHardware``: switching to it is
refused here. Selected by passing this class as the console's ``runtime_factory``; no default and no launch file changes.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time

import numpy as np

from arm_control.config import arm_joints, gripper_joints
from arm_control.end_effectors.franka_adapter import BridgeClient
from arm_control.multi_arm.cooperative import CooperativeKinematics
from arm_control.multi_arm.hands import HandControl
from arm_control.multi_arm.supervised import SdkArmHandle
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.sdk import ArmClient


class _AuthorityHandle:
    """A ``SdkArmHandle`` whose ARM/DISARM do not block the caller (the console holds its lock while calling): the request
    runs on this arm's single worker, ``state`` reports not-ready while it is pending, and a failure is the arm's fault
    reason until the next request succeeds. Everything else is the SDK handle's."""

    def __init__(self, inner: SdkArmHandle, runtime) -> None:
        self._inner, self._runtime = inner, runtime
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'{inner.arm_id}-authority')
        self._pending: list = []   # [(future, armed)], oldest first
        self._generation = 0
        self._fault: str | None = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def arm_id(self) -> str:
        return self._inner.arm_id

    def _reap(self) -> None:
        while self._pending and self._pending[0][0].done():
            future, _ = self._pending.pop(0)
            if future.cancelled():
                continue
            error = future.exception()
            self._fault = None if error is None else str(error)

    def state(self):
        self._reap()
        state = self._inner.state()
        return replace(state, ready=bool(state.ready) and not self._pending,
                       fault_reason=state.fault_reason or self._fault)

    def set_armed(self, armed: bool) -> None:
        self._reap()
        if armed and self._runtime._closed:
            raise ValueError('hardware runtime is closed')
        if armed and self._pending:
            raise ValueError('wait for the current authority acknowledgement before arming')
        if not armed:
            self._runtime._cancel_hand(self.arm_id)
            if any(not want and not f.done() for f, want in self._pending):
                return  # a DISARM is already in flight
            for f, _ in self._pending:
                f.cancel()  # queued enables are void; one DISARM follows the one in flight
        self._generation += 1
        generation = self._generation

        def apply():
            if armed and (self._runtime._closed or generation != self._generation):
                return
            self._inner.set_armed(armed)

        self._pending.append((self._pool.submit(apply), armed))

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


class SupervisedHardware(HandControl):
    backend_name = 'supervised'
    timing_status = 'Supervisor: references through each arm supervisor; physical synchronization unqualified'
    mode = 'independent'
    mode_revision = 0
    paused = False
    period = 0.01

    def __init__(self, config, *, arm_factory=None, hand_factory=None, world_factory=None):
        if config.shared_object is not None and not config.shared_object.virtual:
            raise ValueError('physical hardware shared-object control is not qualified')
        self.config = config
        self.kinematics = CooperativeKinematics(config) if config.shared_object else None
        self._object_pose = None if self.kinematics is None else config.shared_object.world_T_object.copy()
        self.grasp_capture = None
        self.clients, self.handles, self.slices = {}, {}, {}
        self.hands, self.fingers, self.finger_limits = {}, {}, {}
        self._measured_fingers, self._hand_results = {}, {}
        self._sample = {}
        self._closed = False
        self._connected = False
        self._hand_factory = hand_factory or BridgeClient
        arm_factory = arm_factory or (lambda a, cfg: ArmClient((cfg.get('rt') or {}).get('instance_id', a)))
        offset = 0
        try:
            for actor in config.scene.actors:
                a, cfg = actor.name, config.robots[actor.name]
                n = len(arm_joints(cfg))
                client = arm_factory(a, cfg)
                self.clients[a] = client
                self.handles[a] = _AuthorityHandle(
                    SdkArmHandle(client, a, model_token=config.model_tokens.get(a)), self)
                self.slices[a] = slice(offset, offset + n)
                self._sample[a] = (np.zeros(n), np.zeros(n), False)
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
        return True  # independent mode only: no common-time alignment to lose

    @property
    def fresh(self):
        return len(self.hands) == 2 and all(
            self._sample[a][2] and not self.handles[a].state().fault_reason
            and self.hands[a].snapshot().get('measured') is True
            for a in self.handles)

    def joint_state(self):
        return {key: np.concatenate([self._sample[a][i] for a in self.handles])
                for key, i in (('position', 0), ('velocity', 1))}

    def object_pose(self):
        return None if self.kinematics is None else self._object_pose.copy()

    def set_shared_grasps(self, enabled):
        raise ValueError('shared-object coupling needs the PC-side closure: it stays on the direct hardware runtime')

    def step(self):
        for a, handle in self.handles.items():
            state = handle.state()
            if state.position is not None and state.velocity is not None:
                self._sample[a] = (np.asarray(state.position, float), np.asarray(state.velocity, float), True)
            if a in self.hands:
                for result in self.hands[a].pop_move_results():
                    self._hand_results[a] = result

    def telemetry(self):
        out = {}
        for a, handle in self.handles.items():
            state = handle.state()
            out[a] = dict(armed=bool(state.armed), ready=bool(state.ready), fault=state.fault_reason,
                          session=state.session_id, authority_pending=bool(handle._pending),
                          hand=dict(self.hands[a].snapshot(), last_move=self._hand_results.get(a)))
        return out

    def diagnostic_sample(self):
        """What this runtime has: the supervisor's measured joints. It has no RT packets, PC commands or controller
        internals (those belong to the supervisor), and says so rather than filling them in."""
        arms = {a: dict(source='supervisor', q_measured=self._sample[a][0].tolist(), dq_measured=self._sample[a][1].tolist(),
                        measurement_valid=self._sample[a][2], health=self.handles[a].state().fault_reason)
                for a in self.handles}
        return dict(pc_monotonic=time.monotonic(), reference_time_pc=None, arms=arms, feedback_ready=True,
                    feedback_alignment=None)

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
