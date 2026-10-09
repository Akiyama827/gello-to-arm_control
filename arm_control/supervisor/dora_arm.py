"""The Dora controller contract over ONE arm's supervisor: a drop-in for the ``arm_controller`` node in a console graph.

``arm_controller`` (``control.adapter``) is a Dora node around an in-process ``ArmController``. In the supervisor architecture the
controller lives in the supervisor, so this bridge takes the same Dora inputs (``plan``, ``control``, ``jog``, ``gripper``) and
produces the same outputs the console and planner read (``controller_event``, ``arm``, ``motor_state``, ``model_revision``) by talking to
the supervisor only through the SDK: one ``ArmClient``, one non-blocking ``AsyncAuthorityHandle``, and the console control adapter
(gains/Soft/payload become preset and payload calls). It owns no authority of its own and no control law.

Differences from the in-process node (stated, not hidden):
  * ``jog`` is a supervisor setpoint (setpoint/1; refused with the reason, at most once a second, e.g. while a plan runs or on a
    native-reference supervisor); ``gripper`` is refused (an arm supervisor serves no gripper: the graph runs its own Hand
    node); neither is silently ignored;
  * gains must be a named preset (``gain_presets`` in the control mode config); anything else is refused with a ``mode`` event;
  * ``hold`` cancels the live leg and leaves the arm held by the supervisor (there is no latched "frozen" controller state);
    ``stop`` cancels and DISARMs;
  * ``motor_state`` reports the supervisor's measured joints and its selected reference; gains and measured torque are zeros where the
    supervisor does not publish them;
  * a plan the supervisor refuses (e.g. a Cartesian-impedance leg on a plant without the law, or on the native reference path)
    is refused through ``plan_staged`` with the supervisor's reason.
"""
from __future__ import annotations

import os
import time

import numpy as np

from arm_control.contracts._arrow import pack_json_message
from arm_control.contracts.motion import pack_controller_event, unpack_control_update, unpack_jog, unpack_plan
from arm_control.contracts.motor import pack_motor_state_dict
from arm_control.multi_arm.supervised import AsyncAuthorityHandle
from arm_control.multi_arm.types import ArmPlan
from arm_control.sdk import Refused, SupervisorError
from arm_control.supervisor.console_adapter import ConsoleControlAdapter

#: how often the unsupported-input refusals and the mode heartbeat may repeat
NOTICE_PERIOD_S = 1.0
#: the bridge publishes state and drains events at the controller's 100 Hz
PUMP_PERIOD_S = 0.01
#: superseded plan ids remembered for a late execute
SUPERSEDED_KEPT = 64
#: motor_health is published on change and at least this often (the console treats silence as stale)
HEALTH_PERIOD_S = 0.5


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
        self._health_at = -1e9
        self._health_sent: tuple | None = None
        self._fault_text = ''                      # a recoverable supervisor fault, shown through motor_health like a server latch
        self.lost: str | None = None               # why the supervisor is unreachable (terminal for this node)
        self._superseded: dict[str, None] = {}     # staged plans a jog cancelled (oldest first): an execute naming one fails its leg

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
        self._superseded.pop(plan_id, None)  # a plan staged again under this id is that plan now
        now = self.handle.state()
        running = now.running_plan_id if now.executing else None
        if running:  # as in-process: a plan is refused while one RUNS (only a staged, unreleased one is replaced)
            self._event('plan_staged', plan_id, False, f'{running} is still executing')
            return
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
        self._staged.append(plan_id)  # at stage time, not when the event is forwarded: an execute right behind it must find it
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
        """A console jog is a supervisor setpoint (setpoint/1): the same deadman, the same hold at the measured pose on expiry.

        A staged, unreleased plan was planned from the pose this jog leaves: it is cancelled (``plan_cleared``; the supervisor
        refuses setpoints while any execution is live). An ``execute`` that later names it answers a failed ``leg_result``
        ('superseded by a jog'), so a console that kept it as pending is not left waiting. The in-process controller kept such
        a plan and ran it at execute if the jog stayed within start tolerance, else failed it."""
        try:
            q = unpack_jog(value)['q']
        except (ValueError, KeyError, TypeError) as exc:
            self._notice('jog', f'unreadable jog: {exc}')
            return
        for old in self._staged:
            self._release_on_stage.discard(old)
            self._superseded[old] = None
            while len(self._superseded) > SUPERSEDED_KEPT:  # oldest first: a console keeps one pending plan
                del self._superseded[next(iter(self._superseded))]
            self.handle.cancel('superseded by a jog', plan_id=old)
        self._staged.clear()
        try:
            self.arm.setpoint(q)
        except Refused as exc:  # nothing changed; the console re-sends
            self._notice('jog', f'jog refused: {exc}')
        except (SupervisorError, TimeoutError, OSError) as exc:  # lost, garbled or unanswered: the outcome is unknown
            self._notice('jog', f'jog not confirmed (outcome unknown): {exc}')

    def on_gripper(self, value) -> None:
        self._notice('gripper', 'the gripper is not served by an arm supervisor')

    # -- one control group, in on_control order ------------------------------------------------------------------------
    def _send_group(self, group: dict) -> None:
        if 'arm' in group:
            self.handle.set_armed(bool(group['arm']))
        elif 'execute' in group:
            plan_id = str(group['execute'])
            if plan_id in self._superseded:
                del self._superseded[plan_id]
                self._event('leg_result', plan_id, False, 'superseded by a jog: plan again from the new pose')
                return
            if plan_id in self._staged:
                self._staged.remove(plan_id)
            self.handle.execute(plan_id)
        elif 'cancel' in group:
            self._staged.clear()  # a cancel clears whatever was staged (a targeted one names it)
            self._cancel_leg(str(group.get('reason') or 'cancelled'), group.get('cancel_plan_id'))
        elif 'hold' in group:
            self._staged.clear()
            self._cancel_leg(str(group.get('reason') or 'hold'), None)
        elif 'stop' in group:
            self._staged.clear()
            self._cancel_leg(str(group.get('reason') or group['stop'] or 'stop'), None)
            self.handle.set_armed(False)

    def _cancel_leg(self, reason: str, plan_id) -> None:
        """Cancel, and say what the in-process controller says: a cancelled RUNNING leg is a failed leg_result, not only a clear."""
        now = self.handle.state()
        running = now.running_plan_id if now.executing else None
        self.handle.cancel(reason, plan_id=plan_id)
        if running and plan_id in (None, running):
            self._event('leg_result', str(running), False, f'cancelled: {reason}')

    # -- periodic: events, authority edge, measured state -----------------------------------------------------------
    def pump(self) -> None:
        for event in self.handle.poll_events():
            self._forward(event)
        state = self.handle.state()
        if state.armed is None:
            # The supervisor cannot be read. Unknown is never reported as DISARMED: the arm may still be armed and holding (an owner
            # that vanishes leaves it so). Terminal for this node: the SDK channel never reconnects.
            self._lose(state.fault_reason or 'supervisor unreachable')
            return
        if state.fault_reason is not None:  # a failed ARM/DISARM or a supervisor fault is the arm's latch until it clears
            self._fault_text = str(state.fault_reason)
        else:
            self._fault_text = ''
        armed = bool(state.armed)
        if armed != self._armed_sent:
            self._armed_sent = armed
            self.emit('arm', pack_json_message('arm', {'armed': armed}))
        self._publish_health(state, armed)
        ready = bool(state.ready) and state.position is not None
        if ready and not self._was_ready:  # the first fresh measured pose after ARM: where a sequence must start from
            self.emit('controller_event', pack_controller_event(
                kind='ready', q=state.position, qd=state.velocity, runtime_state=self._runtime_state(state)))
        self._was_ready = ready
        if self._clock() - self._mode_at >= NOTICE_PERIOD_S:
            self._mode_event(ok=True, heartbeat=True)
        self._publish_motor_state()

    def _lose(self, reason: str) -> None:
        if self.lost is None:
            self.lost = reason
            self.emit('controller_event', pack_controller_event(kind='fault', ok=False, reason=f'supervisor lost: {reason}'))
            self.emit('motor_health', pack_json_message('motor_health', dict(
                armed=False, state_fresh=False, latched_fault=f'supervisor lost: {reason}', any_fault=True,
                supports_pose_hold=False, unknown_authority=True)))

    def _publish_health(self, state, armed: bool) -> None:
        """The console reads armed and its fault latch from ``motor_health`` (authoritative, and the only thing that CLEARS a latch)."""
        fault = self._fault_text
        key = (armed, fault)
        if key != self._health_sent or self._clock() - self._health_at >= HEALTH_PERIOD_S:
            self._health_sent, self._health_at = key, self._clock()
            self.emit('motor_health', pack_json_message('motor_health', dict(
                armed=armed, state_fresh=state.position is not None, latched_fault=fault, any_fault=bool(fault),
                supports_pose_hold=False)))

    def _forward(self, event: dict) -> None:
        plan_id = str(event.get('plan_id') or '')
        if event['kind'] == 'fault':
            # A supervisor fault is recoverable (DISARM then ARM) and the console clears a latch only through motor_health; the
            # controller_event `fault` of the in-process node is the terminal stop. Surface it as a latch, and in the log.
            self._fault_text = str(event.get('reason') or 'supervisor fault')
            self._event('mode', '', False, f'supervisor fault: {self._fault_text}')
            return
        if event['kind'] == 'plan_cleared' and plan_id in self._replaced:
            self._replaced.discard(plan_id)  # the new plan replaced it: the in-process controller reports no clear for that
            return
        if event['kind'] == 'plan_staged' and not event.get('ok', True) and plan_id in self._staged:
            self._staged.remove(plan_id)  # refused: nothing is staged
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


def release_after_loss(instance_id: str, socket_path=None) -> str:
    """Best effort once the supervisor channel was lost: a supervisor that loses its owner CANCELS and HOLDS, staying armed. A new
    session claims the robot and DISARMS it. Returns what happened (never raises): the caller is already failing."""
    from arm_control.sdk import ArmClient

    try:
        fresh = ArmClient(instance_id, socket_path=socket_path, timeout_s=3.0)
    except Exception as exc:  # noqa: BLE001
        return f'no supervisor to release ({exc}); if the arm was armed it is held by its servo: DISARM it'
    try:
        fresh.claim()
        fresh.disarm(timeout_s=3.0)
        return 'the arm was DISARMED through a new session'
    except Exception as exc:  # noqa: BLE001
        return f'could not DISARM through a new session ({exc!r}); DISARM it explicitly'
    finally:
        fresh.close()


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
                if bridge.lost is not None:
                    break
    finally:
        if bridge.lost is None:
            try:  # leave the arm DISARMED: the supervisor abandons a vanished owner and stays armed
                handle.set_armed(False)
            except Exception:  # noqa: BLE001
                pass
        handle.close()
        arm.close()
        if bridge.lost is not None:
            print(f"[supervised_arm] supervisor lost: {bridge.lost}; {release_after_loss(instance_id, os.environ.get('ARM_CONTROL_SOCKET') or None)}", flush=True)
    if bridge.lost is not None:
        raise SystemExit(1)


def cli() -> None:
    main()


if __name__ == '__main__':
    cli()
