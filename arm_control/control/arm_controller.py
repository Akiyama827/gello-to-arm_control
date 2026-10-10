"""The BOUNDED half of the arm stack: servo, hold, report. Never blocked.

Splits out of what used to be one node that planned *and* servoed. That node
planned inside its Dora event handler, so while OMPL solved it serviced no
inputs and sent no ``motor_command``; the plant's idle timeout then zeroed the
arm's gains and the arm went limp. Measured in one workcell-add run: **12 limp
windows, every one at a phase boundary**, one of them in the same second as the
dock verify. The same bug had already been found and fixed once, on the base
(``base_tilt_planner``), for the same reason.

A thread would not have fixed it -- it keeps the search in the controller's
process, GIL and scheduling jitter. The node boundary is what separates them.

So this class does a FIXED amount of work per event and can always meet a
deadline. It steps a trajectory it was handed, holds its anchor when it has
none, and reports what happened. It never calls IK, never runs OMPL, never
loads a scene, and never decides what to do next.

The one rule that makes the split safe is ``plan_id``:

1. It executes a NAMED plan, never "the current plan". A plan arriving while
   another is executing is refused, not swapped.
2. A gated plan is loaded for review and runs only when an ``execute`` naming
   that same id arrives -- which is what stops a stale plan, one reviewed and
   then superseded by a re-plan, from ever running.
3. Every result carries its plan_id back, so a late reply from a superseded
   plan is dropped rather than credited to the current one.

Its checks are ``tools/check/check_arm_controller.py`` (run it directly) and
the doubles they share are ``arm_control.control.doubles``. They used to be the
last third of this file.
"""

from __future__ import annotations

import math
import time
import uuid

import numpy as np
from arm_control.contracts.impedance import pose_hold_values, unpack_pose_hold_values

from arm_control.control.jog_hold import JogHold, JogRamp
from arm_control.control.leg_settling import SETTLED, TIMED_OUT, LegSettling
from arm_control.control.integral import IntegralState
from arm_control.control.reference import TrajectoryReference
from arm_control.control.trajectory_executor import gain_error
from arm_control.motion import JointServoCommand, JointState, JointTrajectory
from arm_control.contracts import pack_json_message, unpack_json_message
from arm_control.contracts.motion import pack_controller_event, unpack_jog, unpack_control_update, unpack_plan
from arm_control.contracts.motor import pack_cartesian_block, pack_motor_command, unpack_motor_state


class ArmController:
    """Owns the executor and the plant stream. Bounded work per event."""

    def __init__(
        self,
        node,
        executor,
        *,
        arm_id: str = "arm",
        state_period_s: float | None = None,
        plant_reports_health: bool = True,
        jog_timeout_s: float = 0.2,
        clock=None,
        reference_clock=None,
        execution_policy=None,
        settle_timeout_s: float = 5.0,
        settle_dwell_s: float = 0.2,
        model_token: str | None = None,
        pose_provenance: str = "unbound",
        reference_source=None,
        integral_state: IntegralState | None = None,
    ) -> None:
        self.node = node
        # Model binding (arm_control.model_bundle). None = unbound graph (sims,
        # legacy entries): plans are not checked against a model.
        self.model_token = model_token
        self.pose_provenance = str(pose_provenance)
        self.executor = executor
        self._settling = LegSettling(timeout_s=settle_timeout_s, dwell_s=settle_dwell_s)
        self._completion_sample_at = None
        # Soft's stale-stream fallback must remain a real joint hold, even
        # when the preceding operator mode was Float (zero joint stiffness).
        self._track_kp = executor.kp.copy()
        self._track_kd = executor.kd.copy()
        self.arm_id = str(arm_id)
        self.n_arm = executor.n_joints
        self.integral = integral_state if integral_state is not None else IntegralState(self.n_arm)
        if self.integral.bias.shape != (self.n_arm,):
            raise ValueError("integral state must match controller joints")
        self._cycle_id = 0
        # TWO clock policies, and the choice is `motor_state_period_s`:
        #
        #   set    -> PLANT clock. Trajectory time advances one nominal period
        #             per received motor_state, i.e. by how far the PLANT
        #             actually progressed, never by wall time. A slow sim then
        #             plays the plan in lockstep slow-motion instead of letting
        #             the setpoint race ahead and discharge the gap as one PD
        #             yank (18 rad/s whip, measured).
        #   unset  -> WALL clock, time.monotonic().
        #
        # The fallback is the fix for a real bug, not a convenience. This used
        # to advance _plant_t ONLY when a period was configured, while
        # exec_time always returned _plant_t -- so with the key absent the
        # clock sat at 0.0 forever, the executor never left its first sample,
        # and no leg could ever complete. The caller's sim config set it
        # (0.05) and its real config did NOT, so every green sim run hid a
        # controller that would have hung on the first bench move. An arm that cannot finish a motion is not an acceptable
        # response to a missing optional key.
        #
        # On the bench states tick at wall rate, so the two policies agree
        # there; the plant clock stays the right choice for a sim that cannot
        # keep up with real time.
        self._state_period = None if state_period_s is None else float(state_period_s)
        if self._state_period is not None and not (
            math.isfinite(self._state_period) and self._state_period > 0.0
        ):
            raise ValueError(
                f"motor_state_period_s must be finite and positive, "
                f"got {state_period_s!r}"
            )
        self.session_id = uuid.uuid4().hex
        self._clock = clock or time.monotonic
        # Local paired controllers may sample one common reference time while
        # safety checks above/below continue to use the advancing wall clock.
        if reference_clock is not None and self._state_period is not None:
            raise ValueError("reference_clock cannot be combined with motor_state_period_s")
        self._reference_clock = reference_clock or self._clock
        self._reference_source = reference_source
        self.execution_policy = execution_policy
        if execution_policy is not None:
            if execution_policy.torque_limits.shape != (self.n_arm,):
                raise ValueError("execution_policy torque limits must match the arm joint count")
            if self._state_period is not None:
                raise ValueError("deadline execution_policy requires wall time, not motor_state_period_s")
        self._health_at: float | None = None
        self._health_ok = not bool(plant_reports_health)
        self._rt_holding = False
        self._observation_paused = False
        self._plant_t = 0.0
        # A jog is a hold whose anchor MOVES, and which dies of old age. The
        # timeout is the safety property: the console re-sends while a button
        # is held, so a closed tab, a wedged console, a dropped network or a
        # crashed browser all look identical to the controller -- setpoints
        # stop arriving and the arm stops. Nothing has to notice and send a
        # stop; not sending IS the stop.
        self._jog = JogHold(timeout_s=jog_timeout_s, n_joints=self.n_arm)
        # A streamed target (console jog, setpoint/1) moves no faster than planned motion may: saturated per tick at the
        # policy's velocity/acceleration limits (operator 2026-10-09). No policy limits: unsaturated.
        policy = execution_policy
        self._jog_ramp = (JogRamp(policy.velocity_limits, policy.acceleration_limits,
                                  lower=policy.position_lower, upper=policy.position_upper)
                          if policy is not None and policy.velocity_limits is not None
                          and policy.acceleration_limits is not None else None)

        self.last_state = JointState(np.zeros(self.n_arm), np.zeros(self.n_arm))
        self._last_state_at: float | None = None
        self._mode_report_at: float | None = None
        self.last_command: JointServoCommand | None = None
        self._hold_anchor: JointServoCommand | None = None
        # Never further out (operator 2026-10-09): a joint RESTING past its limit when the arm is ARMed -- gravity, disarmed,
        # on a stop -- may be held where it rests and brought back in. `_arm_pose` is where the arm rested: it follows the
        # measured pose until the episode's first command (the plant is limp until then, and that hold is built from a
        # sample taken then). Joints outside the envelope there follow their outermost measured pose since ARM (`_been`:
        # on the stop, a hold re-anchoring at the measured pose must not fault); joints inside it never widen, so an
        # excursion while armed (overshoot, sag) grants nothing.
        self._arm_pose: np.ndarray | None = None
        self._arm_pose_fixed = False
        self._been: np.ndarray | None = None
        self._cartesian_now: dict | None = None   # spec for the loaded plan
        self.last_law: str | None = None           # the law of the last word sent
        self._pose_hold: dict | None = None
        self.supports_pose_hold = False

        self.armed = False
        # Does this plant have a safety bridge that ACKs the arm? A bridge
        # publishes motor_state while DISARMED, so a queued pre-arm sample can
        # arrive after our arm message and we would anchor on a stale or zero
        # pose; waiting for the armed health edge is what makes a first state
        # provably post-enable. A bare sim plant has no gate and no health
        # topic -- every state it sends is live -- and waiting for an edge that
        # cannot come means it never streams at all. That is a fact about the
        # GRAPH, so the graph says it (the node reads ARM_CONTROL_PLANT_HEALTH),
        # and it is opt-OUT: a real bridge must never be assumed absent.
        self._needs_health = bool(plant_reports_health)
        self.bridge_armed = not self._needs_health
        self.ready_sent = False
        self.frozen = False         # milestone hold / failed settle; cancel clears
        self.stopped = False
        self.stop_reason = ""

        # The one loaded plan, its staged trajectory, and once released the
        # reference being executed (trajectory, start time, sampling).
        self._plan: dict | None = None
        self._pending_traj: JointTrajectory | None = None
        self._reference: TrajectoryReference | None = None
        self._running = False
        # Transport acceptance, not command construction, starts the local
        # streamed deadman. Only the supervisor can report a successful send.
        self._stream_gap_limit_s: float | None = None
        self._stream_sent_at: float | None = None
        self._stream_sent_target: np.ndarray | None = None

    def set_stream_hold_deadline(self, hold_ms: float, *, margin_ms: float = 20.0) -> None:
        """Bind the negotiated RT hold threshold for legacy streamed motion."""
        if not (np.isfinite(hold_ms) and hold_ms > margin_ms > 0):
            raise ValueError("RT hold_ms must exceed the streamed guard margin")
        self._stream_gap_limit_s = (hold_ms - margin_ms) / 1000.0

    def streamed_command_sent(self, plan_id: str, q_des, sent_at: float) -> None:
        """Record a transport-accepted command for the still-running leg."""
        if self._running and self.current_plan_id == plan_id and self._reference_source is None:
            self._stream_sent_target = np.asarray(q_des, dtype=float).copy()
            self._stream_sent_at = sent_at

    # -- inputs --------------------------------------------------------------
    @property
    def current_plan_id(self) -> str:
        return "" if self._plan is None else self._plan["plan_id"]

    @property
    def mode_law(self) -> str:
        return "soft" if self._pose_hold is not None else "joint"

    @property
    def runtime_state(self) -> dict:
        """Controller identity and observed lifecycle, carried by heartbeats."""
        return dict(session_id=self.session_id, model_token=self.model_token,
                    n_arm=self.n_arm,
                    armed=bool(self.armed), ready=bool(self.ready_sent),
                    executing=bool(self._running), stopped=bool(self.stopped),
                    frozen=bool(self.frozen), plan_id=self.current_plan_id)

    @property
    def reference(self) -> TrajectoryReference | None:
        """The released reference being executed, or None."""
        return self._reference

    def _stage_ack(self, plan_id: str, *, ok: bool, reason: str = "") -> None:
        self.node.send_output("controller_event", pack_controller_event(
            kind="plan_staged", plan_id=plan_id, ok=ok, reason=reason,
            runtime_state=self.runtime_state))

    def _refuse_plan(self, plan, reason: str, *, report: bool) -> None:
        """Say WHY, always, on the one path an operator can actually read.

        A plan that vanishes without a word is the worst failure this class
        has: the console still shows its own "plan OK" (it planned locally),
        and the refusal only surfaces seconds later as Execute reporting
        "loaded plan is None" -- which names the symptom and not one word of
        the cause. Measured on the bench 2026-09-15: a settling timeout froze
        the controller, every later plan was dropped by the `frozen` guard,
        and no log anywhere said so.

        ``report`` keeps the leg_result contract exactly as it was. Emitting
        one where none was emitted before would tell a waiting planner that
        ITS leg failed, so the extra events stay out of this fix; stdout costs
        nothing and is where the evidence was missing.
        """
        print(f"[arm_controller] REFUSED plan {plan['plan_id']} "
              f"({plan['phase']}): {reason}", flush=True)
        self._stage_ack(plan["plan_id"], ok=False, reason=reason)
        if report:
            self.node.send_output("controller_event", pack_controller_event(
                kind="leg_result", plan_id=plan["plan_id"], ok=False, reason=reason))

    def on_plan(self, value) -> None:
        """Load one leg. Refused if another is mid-flight (never swapped)."""
        try:
            plan = unpack_plan(value)
        except (KeyError, TypeError, ValueError) as exc:
            # A plan that names itself but does not decode (a Cartesian spec outside the envelope, mismatched rows) is
            # refused like any other, never allowed to take the controller down. One that cannot name itself still raises.
            body = unpack_json_message(value, expected_schema="plan")
            self._refuse_plan(dict(plan_id=str(body["plan_id"]), phase=str(body.get("phase"))),
                              f"malformed plan: {exc}", report=True)
            return
        if self._pose_hold is not None:
            self._refuse_plan(
                plan, "Soft pose hold active; select Track before planning",
                report=True)
            return
        if self.stopped or self.frozen:
            # Terminal states, and the reason they are terminal is why this has
            # to speak: `stop` latches and `hold`/a failed settle freeze, so
            # EVERY later plan lands here until the operator cancels (which now
            # clears `frozen`) or the graph restarts.
            self._refuse_plan(
                plan,
                "controller is stopped" if self.stopped else
                "controller is frozen (hold or failed settle) - press Stop to clear",
                report=False)
            return
        if self._running:
            self._refuse_plan(
                plan, f"{self._plan['plan_id']} is still executing", report=False)
            return
        if self.model_token is not None and plan.get("model_token") != self.model_token:
            # Planned against another bundle, revision or scene (or unbound):
            # its joint targets and collision clearance were computed for a
            # model this controller is not running.
            self._refuse_plan(
                plan, f"plan model {plan.get('model_token')!r} != controller model "
                      f"{self.model_token!r}; re-plan", report=True)
            return
        if self.execution_policy is not None:
            reason = self._observation_error() or self.execution_policy.plan_error(
                plan, self.last_state.position, self.n_arm, been=self._where_been())
            if reason:
                self._refuse_plan(plan, reason, report=True)
                return
        # Checked here as well as in set_gains: this path must REPORT a bad
        # law back to the planner, not raise inside a Dora handler. Outside
        # the execution_policy block on purpose — the graphs that configure no
        # policy are exactly the ones with no other admission check.
        reason = gain_error(plan["kp"], plan["kd"])
        if reason:
            self._refuse_plan(plan, reason, report=True)
            return
        traj = JointTrajectory(
            times=plan["times"],
            positions=plan["positions"],
            velocities=plan["velocities"],
        )
        self.executor.set_gains(kp=plan["kp"], kd=plan["kd"])
        self._plan = plan
        self._cartesian_now = plan.get("cartesian")
        self._pending_traj = traj
        self._running = False
        self._stage_ack(plan["plan_id"], ok=True)
        if not plan["gated"]:
            self._release(plan["plan_id"])

    def on_control(self, value) -> None:
        fields = unpack_control_update(value)
        if "arm" in fields:
            self._set_arm(bool(fields["arm"]))
        if "payload" in fields:
            payload = fields["payload"] or {}
            self.executor.set_payload(
                float(payload.get("mass_kg", 0.0)),
                com_offset=payload.get("com_ee"),
            )
        if "execute" in fields:
            self._release(str(fields["execute"]))
        if "gains" in fields:
            self._set_gains(fields["gains"] or {})
        if "pose_hold" in fields:
            self._set_pose_hold(fields["pose_hold"])
        if fields.get("cancel"):
            requested = fields.get("cancel_plan_id")
            if requested is None or not self.current_plan_id or requested == self.current_plan_id:
                cleared = self.current_plan_id or str(requested or "")
                self._cancel(str(fields.get("reason") or "cancelled"))
                if requested is not None:
                    self.node.send_output("controller_event", pack_controller_event(
                        kind="plan_cleared", plan_id=cleared,
                        runtime_state=self.runtime_state))
            else:
                # A delayed pair cancel must never clear a different plan.
                self.node.send_output("controller_event", pack_controller_event(
                    kind="plan_cleared", plan_id=str(requested), ok=False,
                    reason="another plan is loaded", runtime_state=self.runtime_state))
        if fields.get("hold"):
            # Milestone reached: hold this pose forever rather than disarming.
            self.frozen = True
            self._running = False
            self._reference = None
            print(
                f"[arm_controller] holding: {fields.get('reason', 'milestone')}",
                flush=True,
            )
        if "stop" in fields:
            self._stop(str(fields.get("reason") or fields["stop"]))

    def on_jog(self, value) -> str | None:
        """Take a live setpoint, if this is a moment a jog may move at all; else return why not (None = taken).

        Refused while a plan runs: a reviewed plan owns the arm until it ends
        or is cancelled, and a jog arriving underneath it would fight the
        trajectory for the same joints. The supervisor returns the reason
        (setpoint/1).
        """
        refusal = self.jog_refusal()
        if refusal is None:
            if self._jog_ramp is not None and not self._jog.active:
                # a NEW jog (none live: never, expired, cleared, a leg released or cancelled) starts at the measured pose, at
                # rest -- never from a ramp a cleared jog left behind, even within the tick it was cleared in
                self._jog_ramp.reset()
            self._jog.set(unpack_jog(value)["q"], self._clock())
        return refusal

    def jog_refusal(self) -> str | None:
        """Why a jog would be dropped now (None = it would be taken). Pure."""
        return ("stopped" if self.stopped else "frozen after a failed settle (cancel/hold, or DISARM, clears it)" if self.frozen
                else "not armed" if not self.armed else "no fresh measured pose since ARM" if not self.ready_sent
                else "Soft (pose hold) is selected" if self._pose_hold is not None
                else "a plan is running" if self._running else None)

    @property
    def jog_timeout_s(self) -> float:
        """How long a jog/setpoint stays live without renewal."""
        return self._jog.timeout_s

    def clear_jog(self) -> None:
        """Drop a live jog and hold the MEASURED pose (as its expiry does): its sender is gone."""
        if self._jog.active:
            self._jog.clear()
            self._hold_anchor = self._static_hold()

    def _jog_target(self) -> tuple[np.ndarray | None, bool]:
        """Select the live jog target and report expiry; do not update bias."""
        q_des = self._jog.target(self._clock())
        expired = self._jog.take_expired()
        if expired:
            # Anchor at the MEASURED pose, explicitly. Clearing the anchor is
            # not enough: _anchored_hold rebuilds it from the last COMMANDED
            # target whenever that command was quiescent, and a jog command is
            # quiescent by construction (a hold with zero qd_des). So the arm
            # would have carried on to the last setpoint it was asked for --
            # exactly the extra travel the timeout exists to prevent. That
            # heuristic is right for a leg that SETTLED at its target and wrong
            # for a jog that was cut off mid-flight.
            self._hold_anchor = self._static_hold()
            print(
                f"[arm_controller] jog expired after {self._jog.timeout_s:.2f}s "
                f"— holding",
                flush=True,
            )
        return q_des, expired

    def on_motor_health(self, value) -> None:
        payload = unpack_json_message(value)
        if self.execution_policy is not None:
            self._health_at = self._clock()
            self._health_ok = (payload.get("state_fresh") is True
                               and not payload.get("any_fault", False)
                               and not payload.get("latched_fault"))
            holding = payload.get("holding") is True
            if (self._reference_source is None and holding and not self._rt_holding
                    and self._health_ok):
                # RT has replaced the moving command with a measured hold.
                # Retire a running leg before wall-clock sampling resumes;
                # inherit that exact RT hold target for the PC keepalive. A
                # later measurement may differ after residual motion, but
                # relatching it would introduce another q_des step.
                hold_q = np.asarray(payload.get("holding_q"), dtype=float)
                if self._running:
                    self._cancel("RT command-staleness hold — re-plan from measured pose")
                if hold_q.shape == (self.n_arm,) and np.isfinite(hold_q).all():
                    self._hold_anchor = self.executor.hold_command(
                        self.last_state, q_des=hold_q, integral_bias=self.integral.bias)
            self._rt_holding = holding
            if not self._health_ok:
                self._pause_observation("plant health is faulted or stale")
        self.supports_pose_hold = bool(payload.get("supports_pose_hold", False))
        armed = bool(payload.get("armed", False))
        # Latch the plant's armed edge only while THIS controller wants
        # authority: health can still read armed for a sample or two after a
        # DISARM request, and latching that stale edge made the next ARM
        # fault on its own not-yet-armed sample ("plant reported disarmed").
        if armed and not self.bridge_armed and self.armed:
            self.bridge_armed = True
        elif not armed and self.bridge_armed and self.armed and not self.stopped:
            self._fault("plant reported disarmed")

    def on_motor_state(self, value) -> None:
        previous_state_at = self._last_state_at
        self._last_state_at = self._clock()
        if self.stopped:
            return
        if self._state_period is not None:
            self._plant_t += self._state_period  # one message == one plant period
        state = unpack_motor_state(value, self.n_arm)
        arm = JointState(position=state["position"].copy(), velocity=state["velocity"].copy())
        if self.execution_policy is not None:
            if not (np.isfinite(arm.position).all() and np.isfinite(arm.velocity).all()):
                self._last_state_at = None
                self._pause_observation("non-finite motor state")
                return
            if (previous_state_at is not None
                    and self._clock() - previous_state_at > self.execution_policy.state_timeout_sec):
                self._pause_observation(
                    "motor state gap — re-plan",
                    preserve_rt_hold=self._reference_source is None,
                )
        self.last_state = arm
        if self.armed and np.isfinite(arm.position).all():
            q = np.asarray(arm.position, dtype=float)
            if not self._arm_pose_fixed:
                self._arm_pose = q.copy()  # where it rests, until the first command
            self._been = (np.vstack([q, q]) if self._been is None
                          else np.vstack([np.minimum(self._been[0], q), np.maximum(self._been[1], q)]))
        if self.execution_policy is None:
            self._stream()
        if (np.isfinite(arm.position).all() and np.isfinite(arm.velocity).all()
                and (self._mode_report_at is None
                     or self._last_state_at - self._mode_report_at >= 0.25)):
            self._report_mode(heartbeat=True)

    def _report_mode(self, *, ok=True, reason=None, heartbeat=False) -> None:
        law = "soft" if self._pose_hold is not None else "joint"
        fields = dict(kind="mode", ok=ok, reason=law if reason is None else reason,
                      runtime_state=self.runtime_state)
        try:
            event = pack_controller_event(**fields, mode_state=dict(
                law=law, kp=self.executor.kp, kd=self.executor.kd))
        except (TypeError, ValueError):
            # Invalid executor gains cannot certify a mode. Preserve existing
            # transition/refusal events, but let heartbeat freshness expire.
            if heartbeat:
                return
            event = pack_controller_event(**fields)
        self.node.send_output("controller_event", event)
        self._mode_report_at = self._clock()

    def _observation_error(self):
        policy = self.execution_policy
        now = self._clock()
        if self._last_state_at is None or now - self._last_state_at > policy.state_timeout_sec:
            return "motor state stale — re-plan"
        if self._needs_health and (
            not self._health_ok or self._health_at is None
            or now - self._health_at > policy.health_timeout_sec
        ):
            return "motor health stale or faulted — re-plan"
        return None

    def _pause_observation(self, reason, *, preserve_rt_hold=False):
        anchor = (self._hold_anchor if preserve_rt_hold and self._rt_holding
                  and self._health_ok else None)
        if (not self._observation_paused or self._plan is not None
                or self._running or self._jog.active or self._pose_hold is not None):
            self._cancel(reason)
        self._observation_paused = True
        self.ready_sent = False
        self.last_command = None
        self._hold_anchor = anchor

    def tick(self):
        """One deadline-driven command: the supervisor's servo loop calls it once per period."""
        if self.execution_policy is None or self.stopped:
            return
        reason = self._observation_error()
        if reason:
            self._pause_observation(reason)
            return
        self._observation_paused = False
        self._stream()

    def _stream(self) -> None:
        arm = self.last_state

        if not self.armed:
            # Disarmed: track the pose, stream nothing. The bridge zero-holds.
            return
        if not self.ready_sent:
            if not self.bridge_armed:
                # Waiting for the armed health edge -- see _needs_health.
                return
            self.ready_sent = True
            self.node.send_output(
                "controller_event",
                pack_controller_event(kind="ready", q=arm.position, qd=arm.velocity,
                                      runtime_state=self.runtime_state),
            )
            # Fall through: the planner needs a hold streaming from right now,
            # because it is about to spend seconds planning the first leg.
        if not (np.isfinite(arm.position).all() and np.isfinite(arm.velocity).all()):
            return
        if (self._running and self._reference_source is None
                and self._stream_gap_limit_s is not None
                and self._stream_sent_at is not None
                and self._clock() - self._stream_sent_at > self._stream_gap_limit_s):
            # A delayed state/health frame must not let the resumed PC tick
            # sample a trajectory that advanced while RT held the old target.
            target = self._stream_sent_target.copy()
            self._cancel("PC streamed command gap exceeded RT hold margin — re-plan from measured pose")
            self._hold_anchor = self.executor.hold_command(
                arm, q_des=target, integral_bias=self.integral.bias)
            self._stream_sent_at = None
            self._stream_sent_target = None
            self._send(self._hold_anchor, False)
            return
        self._cycle_id += 1
        cycle = (self._cycle_id, self._clock())
        if (self.execution_policy is not None and self._running
                and self._reference is not None):
            # Ordinary joint-goal completion must retain the planned endpoint.
            # Re-latching measured q here baked the allowed settle residual into
            # a new target and added another static tracking error at each stop.
            # Task-specific/contact completion keeps its measured hold contract.
            goal = (self._reference.end.position.copy()
                    if self._plan.get("completion") is None else None)
            if self._check_completion(arm):
                if not self.frozen:
                    self.last_command = self._hold_anchor = (
                        self.executor.hold_command(arm, q_des=goal, integral_bias=self.integral.bias)
                        if goal is not None else None
                    )
                self._cartesian_now = None
        if self._running and self._reference is not None:
            point = self._reference.sample(self.exec_time)
            # A Cartesian leg's stiffness is K_c's (operator 2026-10-10): its joint integral is FROZEN -- updated with zero
            # error, so it neither accumulates against a push or a contact nor jumps; the bias it entered with (the hold's
            # gravity residual) stays, so the leg starts without a torque step.
            q_int = arm.position if self._cartesian_now is not None else point.position
            bias = self.integral.update(*cycle, q_int, arm.position, self.executor.kp)
            command = self.executor.step(point, arm, integral_bias=bias)
            self._hold_anchor = None  # motion streams: next hold re-anchors
            if self._send(command, True) is False:
                return
            if self.execution_policy is None:
                self._check_completion(arm)
            return
        jog, expired = self._jog_target()
        if self._jog_ramp is not None:
            if jog is None:
                self._jog_ramp.reset()  # no jog (none yet, expired, cleared, a leg released): the next starts at rest
            elif np.isfinite(jog).all():
                jog = self._jog_ramp.step(jog, self.last_state.position, self.execution_policy.period)
        if jog is not None:
            # Invalid streamed targets used to reach _send's fault path.
            # Refuse before the atomic integral boundary without changing
            # that lifecycle response or allowing a partial state advance.
            if self.execution_policy is not None and not np.isfinite(jog).all():
                self._fault("non-finite or invalid servo command")
                return
            bias = self.integral.update(*cycle, jog, arm.position, self.executor.kp)
            command = self.executor.hold_command(arm, q_des=jog, integral_bias=bias)
            self._hold_anchor = None   # a jog moves: the next hold re-anchors
            self._send(command, False)
            return
        # No trajectory released: hold the anchor. This is the keepalive the
        # plant's deadman needs while the planner searches, while an operator
        # reviews a gated plan, and while a gripper/tilt/graft request is out.
        command = self._anchored_hold(cycle, refresh=expired)
        if command is not None:
            self._send(command, False)

    # -- leg lifecycle -------------------------------------------------------
    @property
    def exec_time(self) -> float:
        """Trajectory time under whichever clock policy is configured.

        Both are monotonic and both are anchored the same way (``_release``
        stamps ``t_start`` from this property), so the executor never sees the
        absolute value -- only differences.
        """
        return self._plant_t if self._state_period is not None else self._reference_clock()

    def select_reference_window(self, plan_id, trajectory, *, terminal):
        """Selected native stream revision; the reference owns endpoint/termination."""
        from dataclasses import replace
        if self.current_plan_id != plan_id or self._reference is None or self._reference.sample_source is None:
            raise ValueError('selected native reference required')
        self._reference = replace(self._reference, trajectory=trajectory, terminal=terminal)

    def bind_reference_source(self, source) -> None:
        """Disarmed construction binding for a negotiated plant reference."""
        if self.armed or self._plan is not None:
            raise ValueError("bind reference source before ARM or plan admission")
        self._reference_source = source

    def staged_trajectory(self, plan_id: str) -> JointTrajectory:
        """Snapshot the trajectory admitted by this controller, for plant upload."""
        if self.current_plan_id != plan_id or self._pending_traj is None or self._running:
            raise ValueError("the named trajectory is not staged and unselected")
        traj = self._pending_traj
        return JointTrajectory(traj.times, traj.positions, traj.velocities)

    def release_error(self, plan_id: str) -> str:
        """The existing selection checks, shared with asynchronous RT admission."""
        if self._plan is None or self._plan["plan_id"] != plan_id:
            return "the named plan is no longer staged"
        if not self.armed or not self.ready_sent:
            return "not armed"
        if self.execution_policy is not None:
            # on_plan certified the staged trajectory. Selection
            # only needs the current observation and start-pose check; repeating
            # continuous Hermite bounds here delayed the first native envelope.
            return self._observation_error() or self.execution_policy.start_error(
                self._plan["positions"][0], self.last_state.position)
        return ""

    def _release(self, plan_id: str, start: float | None = None) -> None:
        """Start a named plan. Anchors t_start NOW, never at plan time.

        ``start`` (on ``exec_time``'s clock) schedules the anchor instead: a pair's two supervisors anchor both members at
        one stamp so their reference phases agree (the reference holds its first point until then).

        Review time must not be charged against the trajectory: anchoring at
        the planner's send made an arbitrarily long gate pause play out as
        elapsed plan time (the bug that hung lift_module for 40 days of it).
        """
        if self._plan is None or self._plan["plan_id"] != plan_id:
            have = None if self._plan is None else self._plan["plan_id"]
            print(
                f"[arm_controller] execute {plan_id!r} REFUSED — loaded plan "
                f"is {have!r} (a superseded plan never runs)",
                flush=True,
            )
            return
        if not self.armed or not self.ready_sent:
            # Not a race we can wait out: the plant is limp (or has not ACKed
            # the arm), so the pose this plan was built from is already stale.
            # FAIL the leg rather than sit on it -- the planner is blocked on a
            # leg_result and would otherwise wait for one that never comes.
            print(
                f"[arm_controller] execute {plan_id!r} REFUSED — "
                f"armed={self.armed} ready={self.ready_sent}",
                flush=True,
            )
            self._finish_leg(ok=False, reason="not armed")
            return
        if self._running:
            return
        if self.execution_policy is not None:
            reason = self.release_error(plan_id)
            if reason:
                self._finish_leg(ok=False, reason=reason)
                self._pending_traj = None
                return
        self._jog.clear()  # a released plan owns the arm: a jog set before it must not resume after the leg
        start = self.exec_time if start is None else start
        source = self._reference_source
        self._reference = TrajectoryReference(
            self._pending_traj, None if source else start, sample_source=source)
        self._settling.begin(None if source else start + self._reference.duration_sec)
        self._completion_sample_at = None
        self._stream_sent_at = None
        self._stream_sent_target = None
        self._running = True
        # Stamped (the reference clock; a pair's supervisors share the PC's monotonic clock): `start` is when the trajectory
        # begins, `taken` when this controller took the release -- a pair's start skew on the PC side is the difference of
        # its two lines. Not the physical skew at the robots.
        print(
            f"[arm_controller] executing {self._plan['phase']} "
            f"({plan_id}, {self._pending_traj.duration_sec:.2f}s) start={start:.6f} taken={self.exec_time:.6f}",
            flush=True,
        )

    def validate_pose_hold(self, spec: dict) -> dict:
        """Pure admission shared with the supervisor; never retires execution."""
        if self.execution_policy is not None and self._observation_error():
            raise ValueError("Soft needs fresh, healthy measured state")
        spec = unpack_pose_hold_values(pose_hold_values(spec))
        if self.pose_provenance == "calibrated":
            # The RT pose hold closes on the plant's own (vendor) task
            # model; no calibrated RT task space exists. Mixing a
            # calibrated PC model with vendor feedback is refused.
            raise ValueError("Soft uses the RT vendor task model; unavailable "
                             "while a calibrated model bundle is active")
        if (not self.supports_pose_hold or not self.armed or not self.ready_sent
                or self.stopped or self.frozen):
            raise ValueError("Soft needs an armed, ready, capable plant")
        if (self._last_state_at is None or self._clock() - self._last_state_at > 1.0
                or not np.isfinite(self.last_state.position).all()
                or not np.isfinite(self.last_state.velocity).all()):
            raise ValueError("Soft needs fresh, finite measured state")
        if self._running or self._jog.active or np.max(np.abs(self.last_state.velocity)) > 0.05:
            raise ValueError("Stop and settle before selecting Soft")
        return spec

    def _set_pose_hold(self, spec: dict) -> None:
        try:
            spec = self.validate_pose_hold(spec)
        except (TypeError, ValueError) as exc:
            self._report_mode(ok=False, reason=str(exc))
            return
        self._cancel("entering Soft")
        self.executor.set_gains(kp=self._track_kp.copy(), kd=self._track_kd.copy())
        self.last_command = None
        self._cartesian_now = None
        self._pose_hold = spec
        self._reset_integral()  # the pose-hold law ignores tau_ff
        self._report_mode()

    def _set_gains(self, gains: dict) -> None:
        """Change the control law under an operator's hand.

        A running leg is CANCELLED first, always. The plan was reviewed at one
        stiffness and released at that stiffness; swapping the law underneath it
        mid-flight means the arm is no longer doing the thing that was approved
        -- and the most useful preset, float, would drop kp to zero on a moving
        trajectory and let the arm fall through the rest of its path.
        """
        kp = gains.get("kp")
        kd = gains.get("kd")
        if kp is None and kd is None:
            return
        # BEFORE the cancel: a preset that will be refused must not also cost
        # the operator the running leg.
        reason = gain_error(self.executor.kp if kp is None else kp,
                            self.executor.kd if kd is None else kd)
        if reason:
            print(f"[arm_controller] REFUSED gains: {reason}", flush=True)
            self._report_mode()
            return
        if self._pose_hold is not None:
            self._cancel("leaving Soft")
            self.last_command = None
        if self._running:
            self._cancel("gains changed mid-leg")
        self.executor.set_gains(
            kp=None if kp is None else np.asarray(kp, dtype=float),
            kd=None if kd is None else np.asarray(kd, dtype=float),
        )
        # The anchor was built at the OLD stiffness; a softer law holding a
        # stiffer law's target is how an arm sags at a gate.
        self._hold_anchor = None
        self._report_mode()
        print(
            f"[arm_controller] gains set: kp={None if kp is None else np.round(kp, 2)} "
            f"kd={None if kd is None else np.round(kd, 2)}",
            flush=True,
        )

    def _cancel(self, reason: str) -> None:
        """Abort the current leg and hold, still ARMED and still runnable.

        None of the three existing stops means this. ``stop`` is terminal (it
        disarms and latches). ``hold`` freezes -- it is the milestone signal.
        Disarming is a lifecycle reset that makes the plant go limp. An
        operator pressing Stop on a console means none of those: put the arm
        down where it is, forget the plan, and let me try again without
        re-arming.

        Which is why cancel CLEARS ``frozen`` (2026-09-15). It used to freeze
        forever with nothing to undo it -- ``frozen = False`` appeared only in
        __init__ -- so one failed settle wedged a manual session until the
        graph was restarted, and every plan after it was dropped in silence.
        "Let me try again" cannot mean "restart the process".

        ``stopped`` is deliberately NOT cleared here: it latches on purpose and
        has already disarmed the plant, so recovering from it is an explicit
        re-arm, not a cancel. Clearing ``frozen`` cannot start motion by
        itself either -- it only lets a LATER plan be admitted, which still
        needs Plan, Execute, ARM and a fresh observation to move anything.

        Dropping ``_hold_anchor`` is what makes it land in the right place. The
        anchor normally follows the last COMMANDED target so a settled hold
        does not dip, but a leg aborted mid-motion has a moving command, so
        ``_anchored_hold`` falls through to a static hold at the MEASURED
        pose -- which is where the arm actually is when the button is pressed.
        A staged, unreleased joint plan has not moved the arm: cancelling only
        that plan preserves the existing hold anchor, subject to normal relatch
        and observation guards.
        """
        keep_anchor = (
            self._plan is not None and self._pending_traj is not None and not self._running
            and not self._jog.active and self._pose_hold is None
            and self._cartesian_now is None
            and not self.frozen and not self.stopped
            and self.last_command is not None and not np.any(self.last_command.qd_des)
            and np.isfinite(self.last_state.velocity).all()
            and np.max(np.abs(self.last_state.velocity)) < .01
        )
        if self.execution_policy is not None:
            self._cartesian_now = None
        if self._pose_hold is not None:
            self._pose_hold = None
            self.last_command = None
            self._reset_integral()  # accumulated under a law that ignored it
            self._report_mode()
        if self._running:
            self._finish_leg(ok=False, reason=reason)
        was_frozen, self.frozen = self.frozen, False
        self._plan = None
        self._pending_traj = None
        self._running = False
        self._reference = None
        self._jog.clear()
        if not keep_anchor:
            self._hold_anchor = None
        print(f"[arm_controller] cancelled: {reason}"
              f"{' (cleared frozen hold)' if was_frozen else ''}", flush=True)

    def _leg_done(self, state: JointState) -> bool:
        """Has this leg finished? The RULE arrives with the plan.

        No rule means the default: the reference played out and the
        arm is at the target. A ``completion`` spec means the SETTLE rule
        instead -- the plan played out, the arm has genuinely STOPPED, and it
        did not stop absurdly far from the plan. That rule exists for legs
        whose success is not the arm's joint residual (a leg that drives a part
        into a fixture succeeds at the PART's pose), but this node is not the
        one that gets to know which legs those are, or what tolerances their
        physics deserves. Both numbers ride in with the leg.
        """
        # LegSettling consults this only once the reference has played out
        # (its leg end is the reference's start + duration).
        ref = self._reference
        if ref is None:
            return True
        spec = (self._plan or {}).get("completion")
        if spec is None:
            return self.executor.at_goal(ref.end.position, state)
        if float(np.max(np.abs(np.asarray(state.velocity)))) > spec["vel_tol"]:
            return False  # still moving: not settled yet
        goal = spec.get("goal_q")
        if goal is None:
            return True
        residual = float(np.max(np.abs(np.asarray(state.position) - goal)))
        return residual <= spec["residual_max"]

    def _check_completion(self, state: JointState) -> bool:
        """Act on the settle verdict: report progress, accept, or freeze.

        The TIMING lives in LegSettling. What stays here is everything that
        touches the arm -- the freeze on a timed-out settle -- plus the
        measurement the operator needs to see, which needs the plan and the
        executor's tolerances that this class owns and that one does not.
        """
        if self._reference is not None and not self._reference.terminal:
            return False
        now = self.exec_time
        fresh = self._completion_sample_at != self._last_state_at
        if fresh:
            self._completion_sample_at = self._last_state_at
        verdict = self._settling.poll(
            now=now, fresh=fresh, done=lambda: self._leg_done(state),
            reference_ended=(self._reference is not None
                             and self._reference.sample_source is not None
                             and self._reference.elapsed(now) >= self._reference.end_phase),
        )
        if verdict is None:
            return False
        if verdict.state == SETTLED:
            self._finish_leg(ok=True, reason='')
            return True
        spec = self._plan.get('completion')
        pos_tol, vel_tol = self.executor.completion_tolerances
        goal = self._reference.end.position
        if spec is not None:
            vel_tol = spec['vel_tol']
            goal, pos_tol = spec.get('goal_q'), spec.get('residual_max')
        error = None if goal is None else np.abs(state.position - goal)
        joint = None if error is None else int(np.argmax(error)) + 1
        residual = None if error is None else float(np.max(error))
        speed = float(np.max(np.abs(state.velocity)))
        progress = dict(state=verdict.state, elapsed_s=verdict.elapsed_s,
                        remaining_s=verdict.remaining_s,
                        joint=joint, error_rad=residual, position_tolerance_rad=pos_tol,
                        speed_rad_s=speed, velocity_tolerance_rad_s=vel_tol)
        if verdict.state == TIMED_OUT:
            detail = ('' if residual is None else
                      f'joint {joint} error {residual:.5f} rad (limit {pos_tol:.5f}); ')
            reason = (f'Completion timeout after {self._settling.timeout_s:g}s settling: '
                      f'{detail}speed {speed:.5f} rad/s (limit {vel_tol:.5f}). '
                      'Holding measured pose; completion criteria not met.')
            # A failed settle must not turn into another motion or release an
            # attached object. Freeze at the measured pose with zero velocity.
            self.last_command = self._hold_anchor = self._static_hold()
            self._cartesian_now = None
            self._pending_traj = None
            self.frozen = True
            self._finish_leg(ok=False, reason=reason, completion=progress)
            print(f'[arm_controller] {reason}', flush=True)
            return True
        first = not self._settling.started
        if self._settling.due_to_report(now):
            if first:
                print(f'[arm_controller] settling {self._plan["phase"]}: '
                      f'up to {self._settling.timeout_s:g}s, '
                      f'dwell {self._settling.dwell_s:g}s', flush=True)
            self.node.send_output('controller_event', pack_controller_event(
                kind='leg_progress', plan_id=self._plan['plan_id'], completion=progress))
        return False

    def _finish_leg(self, *, ok: bool, reason: str, completion=None) -> None:
        plan_id = "" if self._plan is None else self._plan["plan_id"]
        self._stream_sent_at = None
        self._stream_sent_target = None
        self._reference = None
        self._running = False
        self._plan = None
        self.node.send_output(
            "controller_event",
            pack_controller_event(
                kind="leg_result", plan_id=plan_id, ok=ok, reason=reason,
                completion=completion,
            ),
        )

    # -- outputs -------------------------------------------------------------
    def _where_been(self) -> np.ndarray | None:
        """The poses the position envelope widens to (rows), operator 2026-10-09 "never further out": per joint, where it
        rested at ARM if that was inside the envelope (so nothing widens), else -- on the side it rests past -- the outermost
        pose it has been at since ARM (it rests on its stop: holds re-anchoring at the measured pose, and samples a command was built from as it
        creeps back in, lie there). None while disarmed or before the first sample: the envelope as configured."""
        policy = self.execution_policy
        if not self.armed or self._arm_pose is None or policy is None or policy.position_lower is None:
            return None
        rest = self._arm_pose
        been = self._been if self._been is not None else np.vstack([rest, rest])
        # per SIDE: a joint resting past its lower limit never gains anything past its upper one
        return np.vstack([np.where(rest < policy.position_lower, been[0], rest),
                          np.where(rest > policy.position_upper, been[1], rest)])

    def envelope_band(self):
        """``(lower, upper)`` every command and setpoint must stay inside, or None without a position envelope."""
        return None if self.execution_policy is None else self.execution_policy.envelope(self._where_been())

    def _static_hold(self) -> JointServoCommand:
        """Measured-position gravity hold, reading this cycle's existing bias.

        Replaying a moving command would retain its velocity/feedforward;
        constructing this fallback must not advance integral state again.
        """
        return self.executor.hold_command(self.last_state, integral_bias=self.integral.bias)

    def _anchored_hold(self, cycle, *, refresh=False) -> JointServoCommand | None:
        """The gate/keepalive hold, ANCHORED once per hold episode.

        Rebuilding the hold from the live measured pose every tick ratchets:
        each tick's gravity sag becomes the next tick's setpoint and the arm
        walks downhill (user-observed slow drift at a held gate -- the PD was
        fine, the target was moving). The anchor resets whenever a trajectory
        command streams.

        Anchor at the last COMMANDED target, not the measured pose: at leg end
        the measurement sags below the target by the tracking/stiction band,
        and anchoring there made the arm visibly dip at every gate. Falls back
        to the measured pose when the last command was not quiescent.
        """
        if self.execution_policy is not None:
            q = self.last_state.position
            tolerance = self.execution_policy.relatch_tolerance(self.executor.kp)
            if self._hold_anchor is None or np.any(np.abs(q - self._hold_anchor.q_des) > tolerance):
                self._hold_anchor = self._static_hold()
            # Keep only the target fixed: feedforward uses the measured pose.
            target = self._hold_anchor.q_des
            bias = self.integral.update(*cycle, target, q, self.executor.kp)
            return self.executor.hold_command(self.last_state, q_des=target, integral_bias=bias)
        if self._hold_anchor is None or refresh:
            last = self.last_command
            # Legacy cached holds historically evaluated both gravity and
            # integral at this synthetic commanded state. Preserve that cadence
            # and observation choice; a reused cache does not advance state.
            state = self.last_state
            if not refresh and last is not None and np.max(np.abs(last.qd_des)) < 0.05:
                state = JointState(np.asarray(last.q_des, dtype=float).copy(), np.zeros(self.n_arm))
            bias = self.integral.update(*cycle, state.position, state.position, self.executor.kp)
            self._hold_anchor = self.executor.hold_command(state, integral_bias=bias)
        return self._hold_anchor

    def _send(self, command: JointServoCommand, cartesian: bool) -> bool:
        """``cartesian``: this word belongs to the running leg and carries its Cartesian spec (if it has one). The law lives
        with its leg (execution and settling) and nowhere else: every hold, jog and keepalive is the joint law only, as the RT
        box's own hold is, so a gain change, a disarm or a finished leg can never leave a spring on under other joint gains.
        The spec has no pose: the plant aims the spring at FK(q_des) with its own model."""
        aborted = False
        if self.execution_policy is not None:
            if self.armed and self._arm_pose is not None:
                self._arm_pose_fixed = True  # the episode's first command: from here the band is fixed
            if not self.execution_policy.command_valid(command, self.n_arm, self._where_been()):
                self._fault("non-finite or invalid servo command")
                return False
            if (self._pose_hold is None
                    and self.execution_policy.tracking_error_exceeded(command, self.last_state.position)):
                self._cancel("tracking error exceeded — holding measured pose")
                self.last_command = None
                self._hold_anchor = self._static_hold()
                command = self._hold_anchor
                if command is None or not self.execution_policy.command_valid(command, self.n_arm, self._where_been()):
                    self._fault("invalid measured hold command")
                    return False
                cartesian = False
                aborted = True
        self.last_command = command
        spec = None
        if self._cartesian_now is not None and cartesian:
            spec = self._cartesian_now
        # Which law this word asks the plant for (trace/1's law_pc: "did this leg run compliant or stiff?").
        self.last_law = "cartesian_impedance" if spec is not None else "pose_hold" if self._pose_hold else "joint"
        self.node.send_output(
            "motor_command",
            pack_motor_command(
                np.asarray(command.q_des, dtype=float),
                np.asarray(command.qd_des, dtype=float),
                np.asarray(command.tau_ff, dtype=float),
                np.asarray(command.kp, dtype=float),
                np.asarray(command.kd, dtype=float),
                None if spec is None else pack_cartesian_block(spec),
                pose_hold=self._pose_hold,
            ),
        )
        return not aborted

    def _set_arm(self, armed: bool) -> None:
        armed = bool(armed)
        if self.armed and not armed:
            self._disarm_reset()
        if armed and not self.armed:
            # the pose it rests at as it ARMs (the hold anchors there); none yet: the first sample after
            fresh = (self._last_state_at is not None and self.execution_policy is not None
                     and self._clock() - self._last_state_at <= self.execution_policy.state_timeout_sec)
            self._arm_pose = (np.asarray(self.last_state.position, dtype=float).copy()
                              if fresh and np.isfinite(self.last_state.position).all() else None)
            self._arm_pose_fixed = False
            self._been = None if self._arm_pose is None else np.vstack([self._arm_pose, self._arm_pose])
        self.armed = armed
        self.node.send_output("arm", pack_json_message("arm", {"armed": self.armed}))

    def _disarm_reset(self) -> None:
        """Disarming is a lifecycle event, not just a flag.

        The plant goes limp the moment the bridge sees ``armed=False``, so
        everything the controller still believes is stale: the running leg will
        never finish (and the planner blocks on its leg_result), a loaded plan
        was built from a pose the arm has since sagged out of, the hold anchor
        names a target nothing is holding, and ``ready_sent`` would suppress
        the fresh ready the planner needs to re-anchor. Re-arming must look
        exactly like arming for the first time -- including waiting for the
        plant to ACK again before a single command streams.
        """
        if self._running:
            self._finish_leg(ok=False, reason="disarmed mid-leg")
        self._plan = None
        self._cartesian_now = None
        self._pending_traj = None
        self._running = False
        self._reference = None
        self.ready_sent = False
        self.bridge_armed = not self._needs_health
        self._hold_anchor = None
        self._jog.clear()
        self.last_command = None
        self._pose_hold = None
        self._reset_integral()  # the plant went limp; its bias is gone

    def _reset_integral(self) -> None:
        self.integral.reset()

    def _fault(self, reason: str) -> None:
        if self._running:
            self._finish_leg(ok=False, reason=reason)
        self._stop(reason)

    def _stop(self, reason: str) -> None:
        if self.stopped:
            return
        self.stopped, self.stop_reason = True, str(reason)
        if self.armed:
            self._set_arm(False)
        self.node.send_output(
            "controller_event",
            pack_controller_event(kind="fault", ok=False, reason=reason),
        )
        print(f"[arm_controller] stopped: {reason}", flush=True)
