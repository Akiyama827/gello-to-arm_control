"""The Dora controller contract over ONE arm's supervisor: a drop-in for the ``arm_controller`` node in a console graph.

``arm_controller`` (``control.adapter``) is a Dora node around an in-process ``ArmController``. In the supervisor architecture the
controller lives in the supervisor, so this bridge takes the same Dora inputs (``plan``, ``control``, ``jog``, ``gripper``) and
produces the same outputs the console and planner read (``controller_event``, ``arm``, ``motor_state``, ``model_revision``) by talking to
the supervisor only through the SDK: one ``ArmClient``, one non-blocking ``AsyncAuthorityHandle``, and the console control adapter
(gains/Soft/payload become preset and payload calls). It owns no authority of its own and no control law.

Differences from the in-process node (stated, not hidden):
  * ``jog`` and ``gripper`` are refused: the supervisor has no raw-jog (CTRL-033/052 open) and no gripper; the console is told once a
    second, never silently ignored;
  * gains must be a named preset (``gain_presets`` in the control mode config); anything else is refused with a ``mode`` event;
  * ``hold`` cancels the live leg and leaves the arm held by the supervisor (there is no latched "frozen" controller state);
    ``stop`` cancels and DISARMs;
  * ``motor_state`` reports the supervisor's measured joints and its selected reference; gains and measured torque are zeros where the
    supervisor does not publish them;
  * a plan the supervisor cannot carry (Cartesian impedance) is refused through ``plan_staged`` with the reason.
"""
from __future__ import annotations

import os
import time

import numpy as np

from arm_control.contracts._arrow import pack_json_message
from arm_control.contracts.motion import pack_controller_event, unpack_control_update, unpack_plan
from arm_control.contracts.motor import pack_motor_state_dict
from arm_control.multi_arm.supervised import AsyncAuthorityHandle
from arm_control.multi_arm.types import ArmPlan
from arm_control.sdk import SupervisorError
from arm_control.supervisor.console_adapter import ConsoleControlAdapter

#: how often the unsupported-input refusals and the mode heartbeat may repeat
NOTICE_PERIOD_S = 1.0
#: the bridge publishes state and drains events at the controller's 100 Hz
PUMP_PERIOD_S = 0.01


def default_gains_and_presets(cfg):
    """``(kp, kd, presets)`` exactly as the supervisor resolves them from the same robot config (``resolve_gains`` with the config's
    ``control_mode``, then ``resolve_presets``): ONE recipe, so the bridge's preset table is the supervisor's, and ``check_config``
    proves it against the running supervisor."""
    from arm_control.config import arm_joints
    from arm_control.control.presets import resolve_presets
    from arm_control.node_utils import resolve_gains

    joints = arm_joints(cfg)
    n = len(joints)
    mode = cfg.get('control_mode') or {}
    gains = resolve_gains(cfg, mode, list(cfg.get('joint_names') or cfg.get('motor_names') or []), n)
    if gains is None:
        raise ValueError('the config names no motors: gains cannot be resolved the way the supervisor does')
    kp = np.asarray(gains['kp'], dtype=float).ravel()[:n]
    kd = np.asarray(gains['kd'], dtype=float).ravel()[:n]
    return kp, kd, resolve_presets(mode, joints, kp, kd)


class DoraArmBridge:
    def __init__(self, arm, handle: AsyncAuthorityHandle, presets, emit, *, arm_id: str = 'arm', clock=time.monotonic,
                 live=lambda: ()) -> None:
        """``emit(topic, arrow_value)`` sends one Dora output. ``presets``: ``control.presets.resolve_presets`` of the SAME config
        the supervisor runs (checked by :meth:`check_config`)."""
        self.arm, self.handle, self.emit, self.arm_id, self._clock = arm, handle, emit, arm_id, clock
        self.adapter = ConsoleControlAdapter(arm, presets, live=live)
        self.joints = tuple(arm.identity.joint_names)
        self._release_on_stage: set[str] = set()   # ungated plans: released as soon as the supervisor admits them
        self._staged: list[str] = []               # staged, not yet released: a new plan REPLACES these (as on the in-process controller)
        self._replaced: set[str] = set()           # plans cancelled by that replacement: their plan_cleared is not forwarded
        self._was_ready = False
        self._armed_sent: bool | None = None
        self._notice_at: dict[str, float] = {}
        self._mode_at = -1e9

    # -- startup ---------------------------------------------------------------------------------------------------
    def check_config(self, kp, kd) -> None:
        """The supervisor and this bridge must run the SAME configuration: refuse to start if the supervisor's current gains are
        not the defaults this bridge resolved from its own copy."""
        if self.arm.extensions.get('controller') != 1:
            raise RuntimeError('the supervisor does not advertise controller/1')
        state = self.arm.controller_state()
        if not (np.allclose(state.kp, kp) and np.allclose(state.kd, kd)):
            raise RuntimeError(f'configuration drift: the supervisor runs kp={list(state.kp)} kd={list(state.kd)}, this bridge '
                               f'resolved kp={list(kp)} kd={list(kd)}')

    def start(self) -> None:
        """Claim (never ARM) and announce the plant capabilities the console reads."""
        self.arm.claim()
        self.emit('model_revision', pack_json_message('plant_capabilities', {'supports_pose_hold': False}))
        self.emit('arm', pack_json_message('arm', {'armed': False}))
        self._armed_sent = False

    # -- inputs ----------------------------------------------------------------------------------------------------
    def on_plan(self, value) -> None:
        try:
            body = unpack_plan(value)
        except (ValueError, KeyError, TypeError) as exc:
            self._event('plan_staged', '', False, f'unreadable plan: {exc}')
            return
        plan_id = str(body['plan_id'])
        # The in-process controller replaces a staged-but-unreleased plan when a new one arrives (it only refuses while one RUNS);
        # the supervisor refuses a second admitted plan until the first is superseded or cancelled: do the replacement here.
        for old in list(self._staged):
            self._staged.remove(old)
            self._replaced.add(old)
            self._release_on_stage.discard(old)
            self.handle.cancel('replaced by a new plan', plan_id=old)
        if not body.get('gated'):
            self._release_on_stage.add(plan_id)
            body = dict(body, gated=True)
        self.handle.stage(ArmPlan(self.arm_id, plan_id, body))

    def on_control(self, value) -> None:
        try:
            update = unpack_control_update(value)
        except (ValueError, KeyError, TypeError) as exc:
            self._event('mode', '', False, f'unreadable control update: {exc}')
            return
        result = self.adapter.apply(update, send=self._send_group)
        for line in result.refusals:
            self._mode_event(ok=False, reason=line)
        if result.applied:
            self._mode_event(ok=True)

    def on_jog(self, value) -> None:
        self._notice('jog', 'jog is not available through the supervisor (no raw-jog)')

    def on_gripper(self, value) -> None:
        self._notice('gripper', 'the gripper is not served by an arm supervisor')

    # -- one control group, in on_control order ------------------------------------------------------------------------
    def _send_group(self, group: dict) -> None:
        if 'arm' in group:
            self.handle.set_armed(bool(group['arm']))
        elif 'execute' in group:
            if str(group['execute']) in self._staged:
                self._staged.remove(str(group['execute']))
            self.handle.execute(str(group['execute']))
        elif 'cancel' in group:
            self._staged.clear()  # a cancel clears whatever was staged (a targeted one names it)
            self.handle.cancel(str(group.get('reason') or 'cancelled'), plan_id=group.get('cancel_plan_id'))
        elif 'hold' in group:
            self._staged.clear()
            self.handle.cancel(str(group.get('reason') or 'hold'))
        elif 'stop' in group:
            self._staged.clear()
            self.handle.cancel(str(group.get('reason') or group['stop'] or 'stop'))
            self.handle.set_armed(False)

    # -- periodic: events, authority edge, measured state -----------------------------------------------------------
    def pump(self) -> None:
        for event in self.handle.poll_events():
            self._forward(event)
        state = self.handle.state()
        armed = bool(state.armed)
        if armed != self._armed_sent:
            self._armed_sent = armed
            self.emit('arm', pack_json_message('arm', {'armed': armed}))
        ready = bool(state.ready) and state.position is not None
        if ready and not self._was_ready:  # the first fresh measured pose after ARM: where a sequence must start from
            self.emit('controller_event', pack_controller_event(
                kind='ready', q=state.position, qd=state.velocity, runtime_state=self._runtime_state(state)))
        self._was_ready = ready
        if self._clock() - self._mode_at >= NOTICE_PERIOD_S:
            self._mode_event(ok=True, heartbeat=True)
        self._publish_motor_state()

    def _forward(self, event: dict) -> None:
        plan_id = str(event.get('plan_id') or '')
        if event['kind'] == 'plan_cleared' and plan_id in self._replaced:
            self._replaced.discard(plan_id)  # the new plan replaced it: the in-process controller reports no clear for that
            return
        if event['kind'] == 'plan_staged' and event.get('ok', True):
            self._staged.append(plan_id)
        if event['kind'] == 'plan_staged' and plan_id in self._release_on_stage:
            self._release_on_stage.discard(plan_id)
            if event.get('ok', True):
                self._staged = [p for p in self._staged if p != plan_id]
                self.handle.execute(plan_id)  # an ungated plan starts when admitted, as on the in-process controller
        self._event(event['kind'], plan_id, bool(event.get('ok', True)), str(event.get('reason') or ''))

    def _publish_motor_state(self) -> None:
        try:
            robot = self.arm.state()
        except (SupervisorError, TimeoutError, OSError):
            return
        if robot.q is None or robot.dq is None:
            return  # nothing measured: nothing is published, never zeros
        zeros = np.zeros(len(robot.q))
        q_cmd = robot.q if robot.selected_q is None else robot.selected_q
        qd_cmd = zeros if robot.selected_qd is None else robot.selected_qd
        tau_cmd = zeros if robot.output_tau is None else robot.output_tau
        self.emit('motor_state', pack_motor_state_dict({
            'position': robot.q, 'velocity': robot.dq, 'position_cmd': q_cmd, 'velocity_cmd': qd_cmd,
            'torque_cmd': tau_cmd, 'kp': zeros, 'kd': zeros, 'torque': zeros if robot.tau is None else robot.tau}))

    # -- events ------------------------------------------------------------------------------------------------------
    def _runtime_state(self, state=None) -> dict:
        state = state if state is not None else self.handle.state()
        return dict(session_id=str(state.session_id or self.arm.session), n_arm=len(self.joints), armed=bool(state.armed),
                    ready=bool(state.ready), executing=bool(state.executing), stopped=False, frozen=bool(state.frozen),
                    model_token=state.model_token, plan_id=state.running_plan_id)

    def _event(self, kind: str, plan_id: str, ok: bool, reason: str) -> None:
        self.emit('controller_event', pack_controller_event(
            kind=kind, plan_id=plan_id, ok=ok, reason=reason, runtime_state=self._runtime_state()))

    def _mode_event(self, *, ok: bool, reason: str | None = None, heartbeat: bool = False) -> None:
        try:
            mode = self.arm.controller_state()
            fields = dict(kind='mode', ok=ok, reason=mode.law if reason is None else reason,
                          runtime_state=self._runtime_state())
            event = pack_controller_event(**fields, mode_state=dict(law=mode.law, kp=mode.kp, kd=mode.kd))
        except (SupervisorError, TimeoutError, OSError, ValueError, TypeError):
            if heartbeat:
                return  # a missing heartbeat is how the console learns the mode is not fresh
            event = pack_controller_event(kind='mode', ok=ok, reason=str(reason or 'mode'),
                                          runtime_state=self._runtime_state())
        self.emit('controller_event', event)
        self._mode_at = self._clock()

    def _notice(self, key: str, text: str) -> None:
        now = self._clock()
        if now - self._notice_at.get(key, -1e9) >= NOTICE_PERIOD_S:
            self._notice_at[key] = now
            self._mode_event(ok=False, reason=text)


__all__ = ['DoraArmBridge']


# -- the Dora seam -------------------------------------------------------------------------------------------------------
def main() -> None:
    """Dora node: ``arm_controller``'s contract served by the arm's supervisor (``ARM_CONTROL_CONFIG`` picks the arm, ``ARM_ID`` names
    it, ``ARM_CONTROL_SOCKET`` the supervisor socket; default: the per-user socket of ``rt.instance_id``)."""
    from dora import Node

    from arm_control.config import load_robot_config
    from arm_control.multi_arm.supervised import SdkArmHandle
    from arm_control.node_utils import ShutdownFlag, install_signal_handlers, next_event_gil_friendly
    from arm_control.sdk import ArmClient

    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    cfg = load_robot_config()
    instance_id = str((cfg.get('rt') or {}).get('instance_id', ''))
    if not instance_id:
        raise SystemExit('config has no rt.instance_id: this bridge serves one supervised arm')
    arm = ArmClient(instance_id, socket_path=os.environ.get('ARM_CONTROL_SOCKET') or None)
    node = Node()
    handle = AsyncAuthorityHandle(SdkArmHandle(arm, os.environ.get('ARM_ID', 'arm'), model_token=None))
    kp, kd, presets = default_gains_and_presets(cfg)
    bridge = DoraArmBridge(arm, handle, presets, node.send_output, arm_id=os.environ.get('ARM_ID', 'arm'))
    bridge.check_config(kp, kd)
    bridge.start()
    handlers = {'plan': bridge.on_plan, 'control': bridge.on_control, 'jog': bridge.on_jog, 'gripper': bridge.on_gripper,
                'plan_replay': bridge.on_plan, 'control_replay': bridge.on_control}
    print(f"[supervised_arm] {instance_id}: DISARMED, the supervisor owns control (nothing moves without an operator)", flush=True)
    last_pump = 0.0
    try:
        while not shutdown.stop_requested:
            event = next_event_gil_friendly(node)
            if event is not None and event['type'] == 'STOP':
                break
            if event is not None and event['type'] == 'INPUT' and event['id'] in handlers:
                try:
                    handlers[event['id']](event['value'])
                except Exception as exc:  # noqa: BLE001 -- one bad message must not end the node
                    print(f"[supervised_arm] {event['id']} failed: {exc!r}", flush=True)
            if time.monotonic() - last_pump >= PUMP_PERIOD_S:
                last_pump = time.monotonic()
                bridge.pump()
    finally:
        try:  # leave the arm DISARMED: the supervisor abandons a vanished owner and stays armed
            handle.set_armed(False)
        except Exception:  # noqa: BLE001
            pass
        handle.close()
        arm.close()


def cli() -> None:
    main()


if __name__ == '__main__':
    cli()
