"""An :class:`~arm_control.multi_arm.interface.ArmHandle` over one arm's supervisor (``ArmClient``).

The dual-arm migration (CTRL-053 step 2): the same pair coordinator that drives in-process controllers can drive two
supervisors, one per arm, through the SDK. ``SdkArmHandle`` is the seam:

* ``stage(ArmPlan)`` converts the plan body into a gated ``TrajectoryRequest`` (``Start.ON_RELEASE``) and submits it;
  the admission result becomes the ``plan_staged`` event. Plan fields the SDK request cannot carry (``cartesian``, the
  Cartesian-impedance law) REFUSE the plan with a reason, never silently dropped; ``cartesian_poses`` (visualisation
  FK) is dropped and recorded in ``dropped``.
* ``execute(plan_id)`` releases it; ``cancel`` cancels the execution and answers with ``plan_cleared`` only when the
  supervisor confirmed it; a terminal execution becomes the ``leg_result`` event; a robot fault becomes ``fault``.
* ``state()`` is built from measured SDK state only (unknown stays ``None``, never zero). ``model_token`` is the
  planning bundle token the caller passes in: the supervisor enforces the same token at admission.

What stays out: the Hand/gripper and collision world (project composition, per the card), Soft (the SDK path needs a
Soft-capable plant: ``set_mode('soft')`` is refused), arbitrary gains (only named presets, through the console adapter),
and timing across arms (a pair's start skew belongs to the CTRL-040 group execution, not to this handle). The calls
that talk to the supervisor block up to the SDK timeouts (``set_armed`` up to ``arm_timeout_s``).
"""
from __future__ import annotations

import time

import numpy as np

from arm_control.execution.contract import Completion, JointGains, Lifecycle, Phase, Start, TrajectoryRequest, Validity
from arm_control.motion import JointTrajectory
from arm_control.multi_arm.types import ArmControlState
from arm_control.sdk import SupervisorError

_PLAN_KEYS = {'schema', 'plan_id', 'phase', 'gated', 'times', 'positions', 'velocities', 'kp', 'kd', 'cartesian_poses',
              'cartesian', 'completion', 'model_token'}


def plan_body(payload) -> dict:
    """The decoded plan body of a wire payload (pyarrow array) or an already decoded dict."""
    if isinstance(payload, dict):
        body = dict(payload)
        for key in ('times', 'positions', 'velocities', 'kp', 'kd'):
            body[key] = np.asarray(body[key], dtype=float)
        return body
    from arm_control.contracts.motion import unpack_plan  # needs pyarrow (the supervisor extra)
    return unpack_plan(payload)


def plan_request(body: dict, instance_id: str, joint_names, *, start=Start.ON_RELEASE):
    """``(TrajectoryRequest, dropped_fields)``; ValueError names what cannot be carried."""
    unknown = set(body) - _PLAN_KEYS
    if unknown:
        raise ValueError(f'plan carries fields this handle does not understand: {sorted(unknown)}')
    if body.get('cartesian') is not None:
        raise ValueError('Cartesian-impedance plans are not supported through the supervisor SDK')
    if body.get('gated') is not True:
        raise ValueError('a pair member plan must be gated')
    spec = body.get('completion')
    completion = None if spec is None else Completion(
        vel_tol=float(spec['vel_tol']), goal_q=spec.get('goal_q'), residual_max=spec.get('residual_max'))
    kp, kd = np.asarray(body['kp'], dtype=float), np.asarray(body['kd'], dtype=float)
    request = TrajectoryRequest(
        str(body['plan_id']), instance_id, tuple(joint_names),
        JointTrajectory(body['times'], body['positions'], body['velocities']),
        gains=JointGains(kp, kd) if kp.size else None, model_token=body.get('model_token'), start=start,
        completion=completion)
    return request, (['cartesian_poses'] if body.get('cartesian_poses') is not None else [])


class SdkArmHandle:
    def __init__(self, arm, arm_id: str, *, model_token: str | None = None, presets=None, clock=time.monotonic,
                 arm_timeout_s: float = 5.0):
        self._arm, self._id, self._token, self._presets = arm, arm_id, model_token, presets
        self._clock, self._arm_timeout = clock, float(arm_timeout_s)
        self._plans: dict[str, dict] = {}     # plan_id -> {'execution': id, 'released': bool, 'reported': bool}
        self._events: list[dict] = []
        self._fault_reported = False
        self.dropped: dict[str, list] = {}

    @property
    def arm_id(self) -> str:
        return self._id

    # -- events -------------------------------------------------------------------------------------------
    def _runtime_state(self) -> dict:
        return dict(session_id=self._arm.session, model_token=self._token)

    def _event(self, kind: str, plan_id: str = '', ok: bool = True, reason: str = '') -> None:
        self._events.append(dict(kind=kind, plan_id=plan_id, ok=ok, reason=reason, runtime_state=self._runtime_state()))

    def poll_events(self) -> list[dict]:
        """New events: results of terminal executions and robot faults, then everything queued by stage/cancel."""
        for plan_id, record in self._plans.items():
            if record['released'] and not record['reported']:
                try:
                    status = self._arm.status(record['execution'])
                except (SupervisorError, TimeoutError, OSError):
                    continue
                if status.terminal:
                    record['reported'] = True
                    self._event('leg_result', plan_id, status.phase is Phase.COMPLETED,
                                '' if status.phase is Phase.COMPLETED else f'{status.phase.value}: {status.reason}')
        try:
            authority = self._arm.authority()
        except (SupervisorError, TimeoutError, OSError):
            authority = None
        if authority is not None and authority.fault is not None and not self._fault_reported:
            self._fault_reported = True
            self._event('fault', ok=False, reason=str(authority.fault))
        events, self._events = self._events, []
        return events

    # -- state --------------------------------------------------------------------------------------------
    def _loaded(self):
        """The plan id the supervisor currently holds for this handle and whether it is running."""
        for plan_id, record in self._plans.items():
            try:
                status = self._arm.status(record['execution'])
            except (SupervisorError, TimeoutError, OSError):
                return plan_id, record['released']
            if not status.terminal:
                return plan_id, record['released'] and status.phase is not Phase.ADMITTED
        return None, False

    def state(self) -> ArmControlState:
        now = self._clock()  # read FIRST: observed_at must not be later than a coordinator reading taken before this call
        try:
            robot, authority, stamp = self._arm.state(), self._arm.authority(), self._arm.now()
        except (SupervisorError, TimeoutError, OSError) as exc:
            return ArmControlState(self._id, session_id=self._arm.session, model_token=self._token,
                                   fault_reason=f'supervisor unreachable: {exc}')
        measured = robot.validity is Validity.VALID and robot.q is not None and robot.dq is not None
        age = None
        if robot.received is not None and (robot.received.clock, robot.received.domain) == (stamp.clock, stamp.domain):
            age = max(0.0, (stamp.ns - robot.received.ns) / 1e9)
        plan_id, running = self._loaded()
        law = None
        if self._arm.extensions.get('controller') == 1:
            law = self._arm.controller_state().law
        else:
            law = 'joint'  # no controller/1: the supervisor only runs its joint law
        return ArmControlState(
            self._id, position=robot.q if measured else None, velocity=robot.dq if measured else None,
            armed=authority.lifecycle is Lifecycle.ARMED, ready=measured and authority.lifecycle is Lifecycle.ARMED,
            executing=running, running_plan_id=plan_id, mode_law=law, stopped=False,
            frozen=authority.lifecycle is Lifecycle.FAULTED,
            fault_reason=None if authority.fault is None else str(authority.fault),
            session_id=self._arm.session, model_token=self._token,
            observed_at=now - age if measured and age is not None else None,
            state_age_sec=age if measured and age is not None else None)

    # -- plans --------------------------------------------------------------------------------------------
    def stage(self, plan) -> None:
        try:
            body = plan_body(plan.payload)
            if body.get('plan_id') != plan.plan_id:
                raise ValueError('plan_id does not match the member plan id')
            request, dropped = plan_request(body, self._arm.identity.instance_id, self._arm.identity.joint_names)
            status = self._arm.execute(request)
        except (ValueError, KeyError, TypeError) as exc:
            self._event('plan_staged', plan.plan_id, False, f'refused: {exc}')
            return
        except SupervisorError as exc:
            self._event('plan_staged', plan.plan_id, False, f'refused: {exc}')
            return
        if status.phase is Phase.REJECTED or status.terminal:
            self._event('plan_staged', plan.plan_id, False, f'{status.phase.value}: {status.reason}')
            return
        self._plans[plan.plan_id] = dict(execution=status.execution_id, released=False, reported=False)
        if dropped:
            self.dropped[plan.plan_id] = dropped
        self._event('plan_staged', plan.plan_id, True)

    def execute(self, plan_id: str) -> None:
        record = self._plans.get(plan_id)
        if record is None:
            raise ValueError(f'{plan_id}: no such staged plan')
        self._arm.release(record['execution'])  # a refusal raises Refused to the coordinator (aborts the pair)
        record['released'] = True

    def cancel(self, reason: str = '', *, plan_id: str | None = None) -> None:
        targets = [plan_id] if plan_id is not None else list(self._plans)
        for pid in targets:
            record = self._plans.get(pid)
            if record is None:
                # nothing by that id is loaded: cleared (as the in-process controller answers) unless another plan is
                other = self._loaded()[0]
                self._event('plan_cleared', pid or '', other is None, '' if other is None else 'another plan is loaded')
                continue
            try:
                status = self._arm.cancel(record['execution'], reason or 'cancelled')
            except (SupervisorError, TimeoutError, OSError) as exc:
                self._event('plan_cleared', pid, False, f'cancel not confirmed: {exc}')
                continue
            record['reported'] = True  # a cancelled plan reports plan_cleared, not leg_result
            self._event('plan_cleared', pid, status.terminal, '' if status.terminal else 'not terminal')

    # -- lifecycle and mode --------------------------------------------------------------------------------
    def set_armed(self, armed: bool) -> None:
        if armed:
            self._arm.arm(self._arm_timeout)
        else:
            self._arm.disarm(self._arm_timeout)

    def set_mode(self, law: str, spec=None) -> None:
        if law == 'soft':
            raise ValueError('Soft is not available through the supervisor SDK on this plant')
        if law != 'joint':
            raise ValueError(f'unknown control law {law!r} (joint or soft)')
        if self._arm.extensions.get('controller') == 1 and self._arm.controller_state().law != 'joint':
            self._arm.select_control_preset('track')

    def set_gains(self, kp=None, kd=None) -> None:
        if self._presets is None:
            raise ValueError('gains need the named presets (pass presets=)')
        from arm_control.supervisor.console_adapter import ConsoleControlAdapter
        result = ConsoleControlAdapter(self._arm, self._presets).apply(
            {'gains': {'kp': None if kp is None else list(kp), 'kd': None if kd is None else list(kd)}})
        if result.refusals:
            raise ValueError('; '.join(result.refusals))

    def command_gripper(self, finger_m: float) -> None:
        raise ValueError('the Hand/gripper stays with the project console, not an arm supervisor')
