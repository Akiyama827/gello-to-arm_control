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
"""

from __future__ import annotations

import math
import time

import numpy as np
from arm_control.contracts.impedance import pose_hold_values, unpack_pose_hold_values

from arm_control.control.trajectory_executor import gain_error
from arm_control.motion import JointServoCommand, JointState, JointTrajectory
from arm_control.joint_motor_map import (
    pack_arm_gripper_command,
    unpack_motor_state_to_joint,
)
from arm_control.messages import (
    pack_controller_event,
    unpack_jog,
    unpack_motor_command,
    pack_json_message,
    unpack_control_update,
    unpack_json_message,
    unpack_plan,
)


class ArmController:
    """Owns the executor and the plant stream. Bounded work per event."""

    def __init__(
        self,
        node,
        executor,
        *,
        arm_id: str = "arm",
        gripper: dict,
        state_period_s: float | None = None,
        plant_reports_health: bool = True,
        jog_timeout_s: float = 0.2,
        clock=None,
        execution_policy=None,
        settle_timeout_s: float = 5.0,
        settle_dwell_s: float = 0.2,
    ) -> None:
        self.node = node
        self.executor = executor
        self._settle_timeout = float(settle_timeout_s)
        self._settle_dwell = float(settle_dwell_s)
        if not (math.isfinite(self._settle_timeout) and math.isfinite(self._settle_dwell)
                and 0 <= self._settle_dwell < self._settle_timeout):
            raise ValueError('settling requires 0 <= dwell < finite positive timeout')
        self._settled_since = self._settle_report_at = None
        self._completion_sample_at = None
        self._leg_end = 0.
        # Soft's stale-stream fallback must remain a real joint hold, even
        # when the preceding operator mode was Float (zero joint stiffness).
        self._track_kp = executor.kp.copy()
        self._track_kd = executor.kd.copy()
        self.arm_id = str(arm_id)
        self.gripper = dict(gripper)
        self.n_arm = executor.n_joints
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
        self._clock = clock or time.monotonic
        self.execution_policy = execution_policy
        if execution_policy is not None:
            if execution_policy.torque_limits.shape != (self.n_arm,):
                raise ValueError("execution_policy torque limits must match the arm joint count")
            if self._state_period is not None:
                raise ValueError("deadline execution_policy requires wall time, not motor_state_period_s")
        self._health_at: float | None = None
        self._health_ok = not bool(plant_reports_health)
        self._observation_paused = False
        self._plant_t = 0.0
        self._finger_m = 0.0
        # Operator-commanded finger target, or None to hold the configured open
        # width. The assembly path leaves this None on purpose: the bridge's
        # grasp gate owns the gripper slot from close_gripper onward, and a
        # second writer would fight it. A manual console has no grasp gate, so
        # its slider has to reach the jaws somehow -- this is that path, and it
        # is the capability the trajectory executor used to provide.
        self._finger_cmd: float | None = None
        # A jog is a hold whose anchor MOVES, and which dies of old age. The
        # timeout is the safety property: the console re-sends while a button
        # is held, so a closed tab, a wedged console, a dropped network or a
        # crashed browser all look identical to the controller -- setpoints
        # stop arriving and the arm stops. Nothing has to notice and send a
        # stop; not sending IS the stop.
        self._jog_q: np.ndarray | None = None
        self._jog_at = 0.0
        self._jog_timeout = float(jog_timeout_s)
        if not (self._jog_timeout > 0.0):
            raise ValueError("jog_timeout_s must be positive")

        self.last_state = JointState(np.zeros(self.n_arm), np.zeros(self.n_arm))
        self._last_state_at: float | None = None
        self._mode_report_at: float | None = None
        self.last_command: JointServoCommand | None = None
        self._hold_anchor: JointServoCommand | None = None
        self._cartesian_now: dict | None = None   # spec for the loaded plan
        self._cartesian_poses: np.ndarray | None = None
        self._last_cartesian_pose: np.ndarray | None = None
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
        self.frozen = False         # holding forever (milestone reached)
        self.stopped = False

        # The one loaded plan and whether it has been released to run.
        self._plan: dict | None = None
        self._pending_traj: JointTrajectory | None = None
        self._running = False

    # -- clock ---------------------------------------------------------------
    @property
    def plant_time(self) -> float:
        return self._plant_t

    # -- inputs --------------------------------------------------------------
    def on_plan(self, value) -> None:
        """Load one leg. Refused if another is mid-flight (never swapped)."""
        plan = unpack_plan(value)
        if self._pose_hold is not None:
            self.node.send_output("controller_event", pack_controller_event(
                kind="leg_result", plan_id=plan["plan_id"], ok=False,
                reason="Soft pose hold active; select Track before planning"))
            return
        if self.stopped or self.frozen:
            return
        if self._running:
            print(
                f"[arm_controller] REFUSED plan {plan['plan_id']} "
                f"({plan['phase']}): {self._plan['plan_id']} is still executing",
                flush=True,
            )
            return
        if self.execution_policy is not None:
            reason = self._observation_error() or self.execution_policy.plan_error(
                plan, self.last_state.position, self.n_arm)
            if reason:
                self.node.send_output("controller_event", pack_controller_event(
                    kind="leg_result", plan_id=plan["plan_id"], ok=False, reason=reason))
                return
        # Checked here as well as in set_gains: this path must REPORT a bad
        # law back to the planner, not raise inside a Dora handler. Outside
        # the execution_policy block on purpose — the graphs that configure no
        # policy are exactly the ones with no other admission check.
        reason = gain_error(plan["kp"], plan["kd"])
        if reason:
            print(f"[arm_controller] REFUSED plan {plan['plan_id']}: {reason}",
                  flush=True)
            self.node.send_output("controller_event", pack_controller_event(
                kind="leg_result", plan_id=plan["plan_id"], ok=False, reason=reason))
            return
        traj = JointTrajectory(
            times=plan["times"],
            positions=plan["positions"],
            velocities=plan["velocities"],
        )
        self.executor.set_gains(kp=plan["kp"], kd=plan["kd"])
        self._plan = plan
        self._cartesian_now = plan.get("cartesian")
        self._cartesian_poses = plan.get("cartesian_poses")
        self._pending_traj = traj
        self._running = False
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
            self._cancel(str(fields.get("reason") or "cancelled"))
        if fields.get("hold"):
            # Milestone reached: hold this pose forever rather than disarming.
            self.frozen = True
            self._running = False
            self.executor.clear_trajectory()
            print(
                f"[arm_controller] holding: {fields.get('reason', 'milestone')}",
                flush=True,
            )
        if "stop" in fields:
            self._stop(str(fields.get("reason") or fields["stop"]))

    def on_jog(self, value) -> None:
        """Take a live setpoint, if this is a moment a jog may move at all.

        Refused while a plan runs: a reviewed plan owns the arm until it ends
        or is cancelled, and a jog arriving underneath it would fight the
        trajectory for the same joints.
        """
        if self.stopped or self.frozen or not self.armed or not self.ready_sent or self._pose_hold is not None:
            return
        if self._running:
            return
        q = np.asarray(unpack_jog(value)["q"], dtype=float).ravel()
        if q.shape != (self.n_arm,):
            raise ValueError(f"jog q must have {self.n_arm} values, got {q.size}")
        self._jog_q = q
        self._jog_at = self._clock()

    def _jog_command(self) -> JointServoCommand | None:
        """The servo command for a FRESH jog, or None once it has gone stale."""
        if self._jog_q is None:
            return None
        if self._clock() - self._jog_at > self._jog_timeout:
            self._jog_q = None
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
                f"[arm_controller] jog expired after {self._jog_timeout:.2f}s "
                f"— holding",
                flush=True,
            )
            return None
        hold = getattr(self.executor, "hold_command", None)
        if hold is None:
            return None
        return hold(self.last_state, q_des=self._jog_q)

    def on_gripper(self, value) -> None:
        """A commanded finger width, in metres, from an operator console."""
        command = unpack_motor_command(value, 2)
        self._finger_cmd = float(np.asarray(command["position"], dtype=float)[0])

    def on_motor_health(self, value) -> None:
        payload = unpack_json_message(value)
        if self.execution_policy is not None:
            self._health_at = self._clock()
            self._health_ok = (payload.get("state_fresh") is True
                               and not payload.get("any_fault", False)
                               and not payload.get("latched_fault"))
            if not self._health_ok:
                self._pause_observation("plant health is faulted or stale")
        self.supports_pose_hold = bool(payload.get("supports_pose_hold", False))
        armed = bool(payload.get("armed", False))
        if armed and not self.bridge_armed:
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
        arm, finger_m = unpack_motor_state_to_joint(
            value, self.gripper["n_motors"], self.gripper["mimic"], n_arm=self.n_arm
        )
        if self.execution_policy is not None:
            if not (np.isfinite(arm.position).all() and np.isfinite(arm.velocity).all()):
                self._last_state_at = None
                self._pause_observation("non-finite motor state")
                return
            if (previous_state_at is not None
                    and self._clock() - previous_state_at > self.execution_policy.state_timeout_sec):
                self._pause_observation("motor state gap — re-plan")
        self.last_state = arm
        if finger_m is not None:
            self._finger_m = float(finger_m)
        if self.execution_policy is None:
            self._stream()
        if (np.isfinite(arm.position).all() and np.isfinite(arm.velocity).all()
                and (self._mode_report_at is None
                     or self._last_state_at - self._mode_report_at >= 0.25)):
            self._report_mode(heartbeat=True)

    def _report_mode(self, *, ok=True, reason=None, heartbeat=False) -> None:
        law = "soft" if self._pose_hold is not None else "joint"
        fields = dict(kind="mode", ok=ok, reason=law if reason is None else reason)
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

    def _pause_observation(self, reason):
        if (not self._observation_paused or self._plan is not None
                or self._running or self._jog_q is not None or self._pose_hold is not None):
            self._cancel(reason)
        self._observation_paused = True
        self.ready_sent = False
        self.last_command = None
        self._hold_anchor = None

    def tick(self):
        """One deadline-driven command; unused by event-driven deployments."""
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
                pack_controller_event(kind="ready", q=arm.position, qd=arm.velocity),
            )
            # Fall through: the planner needs a hold streaming from right now,
            # because it is about to spend seconds planning the first leg.
        if (self.execution_policy is not None and self._running
                and self.executor.has_trajectory and self._check_completion(arm)):
            # Legacy ordering: completed means measured idle hold NOW, not
            # one more trajectory sample before dropping into hold.
            # _check_completion reports the result and anchors a timeout hold.
            if not self.frozen:
                self.last_command = self._hold_anchor = None
            self._cartesian_now = self._cartesian_poses = self._last_cartesian_pose = None
        if self._running and self.executor.has_trajectory:
            command = self.executor.step(self.exec_time, arm)
            self._hold_anchor = None  # motion streams: next hold re-anchors
            if self._send(command, self._cartesian_for_time(self.exec_time)) is False:
                return
            if self.execution_policy is None:
                self._check_completion(arm)
            return
        jog = self._jog_command()
        if jog is not None:
            self._hold_anchor = None   # a jog moves: the next hold re-anchors
            self._send(jog, self._last_cartesian_pose)
            return
        # No trajectory released: hold the anchor. This is the keepalive the
        # plant's deadman needs while the planner searches, while an operator
        # reviews a gated plan, and while a gripper/tilt/graft request is out.
        command = self._anchored_hold()
        if command is not None:
            self._send(command, self._last_cartesian_pose)

    # -- leg lifecycle -------------------------------------------------------
    @property
    def exec_time(self) -> float:
        """Trajectory time under whichever clock policy is configured.

        Both are monotonic and both are anchored the same way (``_release``
        stamps ``t_start`` from this property), so the executor never sees the
        absolute value -- only differences.
        """
        return self._plant_t if self._state_period is not None else self._clock()

    def _release(self, plan_id: str) -> None:
        """Start a named plan. Anchors t_start NOW, never at plan time.

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
            reason = self._observation_error() or self.execution_policy.plan_error(
                self._plan, self.last_state.position, self.n_arm)
            if reason:
                self._finish_leg(ok=False, reason=reason)
                self._pending_traj = None
                return
        start = self.exec_time
        self.executor.load_trajectory(self._pending_traj, t_start=start)
        self._leg_end = start + self._pending_traj.duration_sec
        self._settled_since = self._settle_report_at = self._completion_sample_at = None
        self._running = True
        print(
            f"[arm_controller] executing {self._plan['phase']} "
            f"({plan_id}, {self._pending_traj.duration_sec:.2f}s)",
            flush=True,
        )

    def on_plant_capabilities(self, value) -> None:
        self.supports_pose_hold = bool(unpack_json_message(value).get("supports_pose_hold", False))

    def _set_pose_hold(self, spec: dict) -> None:
        try:
            if self.execution_policy is not None and self._observation_error():
                raise ValueError("Soft needs fresh, healthy measured state")
            spec = unpack_pose_hold_values(pose_hold_values(spec))
            if (not self.supports_pose_hold or not self.armed or not self.ready_sent
                    or self.stopped or self.frozen):
                raise ValueError("Soft needs an armed, ready, capable plant")
            if (self._last_state_at is None or self._clock() - self._last_state_at > 1.0
                    or not np.isfinite(self.last_state.position).all()
                    or not np.isfinite(self.last_state.velocity).all()):
                raise ValueError("Soft needs fresh, finite measured state")
            if self._running or self._jog_q is not None or np.max(np.abs(self.last_state.velocity)) > 0.05:
                raise ValueError("Stop and settle before selecting Soft")
        except (TypeError, ValueError) as exc:
            self._report_mode(ok=False, reason=str(exc))
            return
        self._cancel("entering Soft")
        self.executor.set_gains(kp=self._track_kp.copy(), kd=self._track_kd.copy())
        self.last_command = None
        self._cartesian_now = None
        self._last_cartesian_pose = None
        self._pose_hold = spec
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
        disarms and latches). ``hold`` freezes forever -- it is the milestone
        signal, and a frozen controller refuses every later plan. Disarming is
        a lifecycle reset that makes the plant go limp. An operator pressing
        Stop on a console means none of those: put the arm down where it is,
        forget the plan, and let me try again without re-arming.

        Dropping ``_hold_anchor`` is what makes it land in the right place. The
        anchor normally follows the last COMMANDED target so a settled hold
        does not dip, but a leg aborted mid-motion has a moving command, so
        ``_anchored_hold`` falls through to a static hold at the MEASURED
        pose -- which is where the arm actually is when the button is pressed.
        """
        if self.execution_policy is not None:
            self._cartesian_now = self._cartesian_poses = self._last_cartesian_pose = None
        if self._pose_hold is not None:
            self._pose_hold = None
            self.last_command = None
            self._report_mode()
        if self._running:
            self._finish_leg(ok=False, reason=reason)
        self._plan = None
        self._pending_traj = None
        self._running = False
        self.executor.clear_trajectory()
        self._jog_q = None
        self._hold_anchor = None
        print(f"[arm_controller] cancelled: {reason}", flush=True)

    def _leg_done(self, state: JointState) -> bool:
        """Has this leg finished? The RULE arrives with the plan.

        No rule means the executor's own done(): the plan played out and the
        arm is at the target. A ``completion`` spec means the SETTLE rule
        instead -- the plan played out, the arm has genuinely STOPPED, and it
        did not stop absurdly far from the plan. That rule exists for legs
        whose success is not the arm's joint residual (a leg that drives a part
        into a fixture succeeds at the PART's pose), but this node is not the
        one that gets to know which legs those are, or what tolerances their
        physics deserves. Both numbers ride in with the leg.
        """
        spec = (self._plan or {}).get("completion")
        if spec is None:
            return self.executor.done(self.exec_time, state)
        if not self.executor.done(
            self.exec_time, state, pos_tol=float("inf"), vel_tol=float("inf")
        ):
            return False  # the plan has not played out yet
        if float(np.max(np.abs(np.asarray(state.velocity)))) > spec["vel_tol"]:
            return False  # still moving: not settled yet
        goal = spec.get("goal_q")
        if goal is None:
            return True
        residual = float(np.max(np.abs(np.asarray(state.position) - goal)))
        return residual <= spec["residual_max"]

    def _check_completion(self, state: JointState) -> bool:
        """Bounded settling after reference end; never accept a stopped miss."""
        now = self.exec_time
        if now < self._leg_end:
            return False
        fresh = self._completion_sample_at != self._last_state_at
        if fresh:
            self._completion_sample_at = self._last_state_at
            if self._leg_done(state):
                if self._settled_since is None:
                    self._settled_since = now
            else:
                self._settled_since = None
        elapsed = max(0., now - self._leg_end)
        if (fresh and self._settled_since is not None
                and now - self._settled_since >= self._settle_dwell
                and elapsed <= self._settle_timeout):
            self._finish_leg(ok=True, reason='')
            return True
        spec = self._plan.get('completion')
        pos_tol, vel_tol = self.executor.completion_tolerances
        goal = self._plan['positions'][-1]
        if spec is not None:
            vel_tol = spec['vel_tol']
            goal, pos_tol = spec.get('goal_q'), spec.get('residual_max')
        error = None if goal is None else np.abs(state.position - goal)
        joint = None if error is None else int(np.argmax(error)) + 1
        residual = None if error is None else float(np.max(error))
        speed = float(np.max(np.abs(state.velocity)))
        progress = dict(state='settling', elapsed_s=elapsed,
                        remaining_s=max(0., self._settle_timeout - elapsed),
                        joint=joint, error_rad=residual, position_tolerance_rad=pos_tol,
                        speed_rad_s=speed, velocity_tolerance_rad_s=vel_tol)
        if elapsed >= self._settle_timeout:
            progress['state'] = 'timed_out'
            detail = ('' if residual is None else
                      f'joint {joint} error {residual:.5f} rad (limit {pos_tol:.5f}); ')
            reason = (f'Completion timeout after {self._settle_timeout:g}s settling: '
                      f'{detail}speed {speed:.5f} rad/s (limit {vel_tol:.5f}). '
                      'Holding measured pose; completion criteria not met.')
            # A failed settle must not turn into another motion or release an
            # attached object. Freeze at the measured pose with zero velocity.
            self.last_command = self._hold_anchor = self._static_hold()
            self._cartesian_now = self._cartesian_poses = self._last_cartesian_pose = None
            self._pending_traj = None
            self.frozen = True
            self._finish_leg(ok=False, reason=reason, completion=progress)
            print(f'[arm_controller] {reason}', flush=True)
            return True
        if self._settle_report_at is None or now - self._settle_report_at >= .25:
            if self._settle_report_at is None:
                print(f'[arm_controller] settling {self._plan["phase"]}: '
                      f'up to {self._settle_timeout:g}s, dwell {self._settle_dwell:g}s', flush=True)
            self._settle_report_at = now
            self.node.send_output('controller_event', pack_controller_event(
                kind='leg_progress', plan_id=self._plan['plan_id'], completion=progress))
        return False

    def _finish_leg(self, *, ok: bool, reason: str, completion=None) -> None:
        plan_id = "" if self._plan is None else self._plan["plan_id"]
        self.executor.clear_trajectory()
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
    def _static_hold(self) -> JointServoCommand | None:
        """Zero-velocity gravity hold at the MEASURED pose (safe freeze).

        Replaying ``last_command`` verbatim would carry a mid-trajectory qd_des
        and motion tau_ff into the freeze; fall back to it only when the
        executor cannot build a hold.
        """
        hold = getattr(self.executor, "hold_command", None)
        if hold is None:
            return self.last_command
        return hold(self.last_state)

    def _anchored_hold(self) -> JointServoCommand | None:
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
            return self.executor.hold_command(self.last_state, q_des=self._hold_anchor.q_des)
        if self._hold_anchor is None:
            last = self.last_command
            hold = getattr(self.executor, "hold_command", None)
            if (
                last is not None
                and hold is not None
                and float(np.max(np.abs(last.qd_des))) < 0.05
            ):
                self._hold_anchor = hold(
                    JointState(
                        position=np.asarray(last.q_des, dtype=float).copy(),
                        velocity=np.zeros(self.n_arm),
                    )
                )
            else:
                self._hold_anchor = self._static_hold()
        return self._hold_anchor

    def _cartesian_for_time(self, t_now: float):
        """The precomputed world EE pose for the sample being commanded.

        The controller owns no kinematics, so the Cartesian impedance target
        (which is FK(q_des)) rides along with the plan rather than being
        re-derived here from a model this node is not allowed to load.
        """
        poses = self._cartesian_poses
        if poses is None or self._plan is None:
            return None
        times = self._plan["times"]
        tau = np.clip(self.executor.elapsed(t_now), times[0], times[-1])
        idx = int(np.searchsorted(times, tau, side="right")) - 1
        return poses[max(0, min(idx, len(poses) - 1))]

    def _send(self, command: JointServoCommand, cartesian_pose) -> bool:
        aborted = False
        if self.execution_policy is not None:
            if not self.execution_policy.command_valid(command, self.n_arm):
                self._fault("non-finite or invalid servo command")
                return False
            if (self._pose_hold is None
                    and self.execution_policy.tracking_error_exceeded(command, self.last_state.position)):
                self._cancel("tracking error exceeded — holding measured pose")
                self.last_command = None
                self._hold_anchor = self._static_hold()
                command = self._hold_anchor
                if command is None or not self.execution_policy.command_valid(command, self.n_arm):
                    self._fault("invalid measured hold command")
                    return False
                cartesian_pose = None
                aborted = True
        self.last_command = command
        cartesian = None
        if self._cartesian_now is not None and cartesian_pose is not None:
            self._last_cartesian_pose = np.asarray(cartesian_pose, dtype=float)
            cartesian = {
                "pose": self._last_cartesian_pose,
                "task_R": self._cartesian_now["task_R"],
                "kc": self._cartesian_now["kc"],
                "dc": self._cartesian_now["dc"],
            }
        self.node.send_output(
            "motor_command",
            pack_arm_gripper_command(
                command.q_des,
                command.qd_des,
                command.tau_ff,
                command.kp,
                command.kd,
                self.gripper["open_finger_m"]
                if self._finger_cmd is None
                else self._finger_cmd,
                self.gripper["gains"],
                self.gripper["mimic"],
                cartesian=cartesian,
                pose_hold=self._pose_hold,
            ),
        )
        return not aborted

    def _set_arm(self, armed: bool) -> None:
        armed = bool(armed)
        if self.armed and not armed:
            self._disarm_reset()
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
        self._pending_traj = None
        self._running = False
        self.executor.clear_trajectory()
        self.ready_sent = False
        self.bridge_armed = not self._needs_health
        self._hold_anchor = None
        self._jog_q = None
        self.last_command = None
        self._pose_hold = None

    def _fault(self, reason: str) -> None:
        if self._running:
            self._finish_leg(ok=False, reason=reason)
        self._stop(reason)

    def _stop(self, reason: str) -> None:
        if self.stopped:
            return
        self.stopped = True
        if self.armed:
            self._set_arm(False)
        self.node.send_output(
            "controller_event",
            pack_controller_event(kind="fault", ok=False, reason=reason),
        )
        print(f"[arm_controller] stopped: {reason}", flush=True)

    # -- run loop ------------------------------------------------------------
    def run(self, *, shutdown=None) -> None:
        handlers = {
            "motor_state": self.on_motor_state,
            "motor_health": self.on_motor_health,
            "plant_capabilities": self.on_plant_capabilities,
            "plan": self.on_plan,
            "control": self.on_control,
            "gripper": self.on_gripper,
            "jog": self.on_jog,
            # A second motion source speaks the same contract under its own
            # topic name (dora maps one producer per input, so an alias is how
            # two of them reach one handler -- the trajectory executor did the
            # same for its replay input).
            "plan_replay": self.on_plan,
            "control_replay": self.on_control,
        }
        seen_unknown: set[str] = set()
        period = None if self.execution_policy is None else self.execution_policy.period
        next_deadline = None if period is None else self._clock() + period
        while shutdown is None or not shutdown.stop_requested:
            timeout = .05 if period is None else max(0.0, min(.05, next_deadline - self._clock()))
            event = self.node.next(timeout=timeout)
            if shutdown is not None and shutdown.stop_requested:
                break
            if event is not None and event["type"] == "STOP":
                break
            if event is not None and event["type"] == "INPUT":
                handler = handlers.get(event["id"])
                # A graph can wire an input this controller has no handler for
                # -- renaming `trajectory` to `plan` did exactly that -- and a
                # silent drop makes it look like the producer is broken. Say it
                # once per topic; every tick would be a log flood.
                if handler is None and event["id"] not in seen_unknown:
                    seen_unknown.add(event["id"])
                    print(
                        f"[arm_controller] IGNORING input {event['id']!r}: no "
                        f"handler (known: {sorted(handlers)})",
                        flush=True,
                    )
                if handler is not None:
                    handler(event["value"])
            if self.stopped:
                break
            if period is not None:
                now = self._clock()
                if now >= next_deadline - .1 * period:
                    # Legacy deadline/slack schedule: one command after a
                    # stall, never a burst of catch-up commands.
                    next_deadline = (next_deadline + period
                                     if now - next_deadline < period else now + period)
                    self.tick()


# -- self-check ---------------------------------------------------------------
class _FakeNode:
    """Collects outputs; iterating it ends the run loop immediately."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []

    def send_output(self, topic: str, arrow) -> None:
        self.sent.append((topic, arrow))

    def topics(self) -> list[str]:
        return [t for t, _ in self.sent]


class _FakeExecutor:
    """Just enough executor to exercise the controller's own decisions."""

    completion_tolerances = (0.01, 0.05)

    def __init__(self, n: int = 7) -> None:
        self.n_joints = n
        self._traj = None
        self._t_start = 0.0
        self.kp = np.full(n, 1200.0)
        self.kd = np.full(n, 30.0)
        self.payload = (0.0, None)
        self.steps = 0
        # The real executor reads these off the URDF it loaded; a double with
        # no model states an honest, permissive envelope rather than omitting
        # the property, so control/adapter.py can keep requiring it.
        self.joint_limits = (np.full(n, -np.pi), np.full(n, np.pi))

    def set_gains(self, kp=None, kd=None) -> None:
        if kp is not None:
            self.kp = np.asarray(kp, dtype=float)
        if kd is not None:
            self.kd = np.asarray(kd, dtype=float)

    def set_payload(self, mass_kg, frame_name="", com_offset=None) -> None:
        self.payload = (float(mass_kg), com_offset)

    @property
    def has_trajectory(self) -> bool:
        return self._traj is not None

    def load_trajectory(self, traj, t_start: float) -> None:
        self._traj, self._t_start = traj, float(t_start)

    def elapsed(self, t_now: float) -> float:
        return float(t_now) - self._t_start

    def clear_trajectory(self) -> None:
        self._traj = None

    def step(self, t_now, state) -> JointServoCommand:
        self.steps += 1
        n = self.n_joints
        return JointServoCommand(
            q_des=np.zeros(n), qd_des=np.zeros(n), tau_ff=np.zeros(n),
            kp=self.kp, kd=self.kd,
        )

    def hold_command(self, state, q_des=None) -> JointServoCommand:
        n = self.n_joints
        q = np.asarray(state.position, dtype=float) if q_des is None else np.asarray(q_des)
        return JointServoCommand(
            q_des=q, qd_des=np.zeros(n), tau_ff=np.zeros(n), kp=self.kp, kd=self.kd
        )

    def done(self, t_now, state, pos_tol=None, vel_tol=None) -> bool:
        return self._traj is not None and t_now - self._t_start >= self._traj.duration_sec


_GRIPPER = {"n_motors": 7, "mimic": None, "open_finger_m": 0.04, "gains": (100.0, 1.0)}


def _controller(**kw) -> "ArmController":
    node = _FakeNode()
    # PLANT clock by default: these checks fast-forward by assigning _plant_t,
    # which only means anything under that policy. The WALL policy -- what a
    # config without motor_state_period_s gets -- has its own check below.
    kw.setdefault("state_period_s", 0.01)
    kw.setdefault('settle_dwell_s', 0.)  # Legacy lifecycle checks; dwell has its own replay.
    c = ArmController(node, _FakeExecutor(), gripper=dict(_GRIPPER), **kw)
    c._set_arm(True)
    c.bridge_armed = True
    c.on_motor_state(_state())   # the post-arm state that earns `ready`
    node.sent.clear()
    return c


def _state(n: int = 7, vel: float = 0.0, pos: float = 0.0):
    from arm_control.messages import pack_motor_state_dict

    return pack_motor_state_dict(
        {k: np.zeros(n) for k in
         ("position", "velocity", "position_cmd", "velocity_cmd",
          "torque_cmd", "kp", "kd", "torque")}
        | {"position": np.full(n, pos), "velocity": np.full(n, vel)}
    )


def _plan_msg(plan_id: str, *, gated: bool, completion=None):
    from arm_control.messages import pack_plan

    times = np.array([0.0, 1.0])
    q = np.zeros((2, 7))
    return pack_plan(
        plan_id=plan_id, phase="move", gated=gated, times=times,
        positions=q, velocities=q, kp=np.full(7, 1200.0), kd=np.full(7, 30.0),
        completion=completion,
    )


def _check_clock_policies() -> None:
    """Either clock must actually advance, and a bad period must be refused.

    The regression this pins: _plant_t advanced only when a period was
    configured, while exec_time always returned it -- so a config WITHOUT
    motor_state_period_s (which is every real/*.yaml here) froze trajectory
    time at 0.0 and no leg could ever finish. It passed every sim run because
    the sim config happens to set the key.
    """
    # WALL policy: no period configured. Time must move without any state.
    ticks = iter([10.0, 10.5, 11.25])
    c = ArmController(
        _FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER), clock=lambda: next(ticks)
    )
    assert c.exec_time == 10.0
    assert c.exec_time == 10.5
    assert c.exec_time == 11.25, "wall clock must advance with no motor_state"

    # PLANT policy: time moves per state message and NOT with wall time.
    frozen = ArmController(
        _FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
        state_period_s=0.02, clock=lambda: 999.0,
    )
    assert frozen.exec_time == 0.0
    frozen.on_motor_state(_state())
    frozen.on_motor_state(_state())
    assert abs(frozen.exec_time - 0.04) < 1e-12, frozen.exec_time

    # A period that is present but nonsense is a config error, not a fallback:
    # silently switching clocks on a typo is how the original bug felt normal.
    for bad in (0.0, -0.01, float("inf"), float("nan")):
        try:
            ArmController(_FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
                          state_period_s=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"motor_state_period_s={bad} must be refused")


def _self_check() -> None:
    import inspect
    from arm_control.messages import pack_control_update, unpack_controller_event

    assert 'qd' in inspect.signature(pack_controller_event).parameters, \
        'ready event cannot report measured velocity'
    old = unpack_controller_event(pack_controller_event(kind='ready', q=[.1, .2]))
    assert 'qd' not in old and np.array_equal(old['q'], [.1, .2])
    measured = unpack_controller_event(pack_controller_event(
        kind='ready', q=[.1, .2], qd=[-.3, .4]))
    assert np.array_equal(measured['qd'], [-.3, .4])
    for q, qd in (([0., 0.], [np.nan, 0.]), ([0., 0.], [0., np.inf]),
                  ([0., 0.], [0.]), ([0., 0.], [[0., 0.]]),
                  ([0., 0.], 'invalid'), ([0.], 0.), ([], []),
                  (None, [0.]), ([[0., 0.]], [0., 0.])):
        for decode in (False, True):
            try:
                if decode:
                    unpack_controller_event(pack_json_message('controller_event', dict(
                        kind='ready', q=q, qd=qd)))
                else:
                    pack_controller_event(kind='ready', q=q, qd=qd)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f'invalid ready velocity accepted: q={q!r}, qd={qd!r}')

    # 1. A gated plan does NOT run until an execute names it -- and while it
    #    waits, the controller still streams. That stream is the whole point of
    #    the split: it is what the plant's deadman sees while a human reviews.
    c = _controller()
    c.on_plan(_plan_msg("p1", gated=True))
    c.on_motor_state(_state())
    assert c._running is False, "a gated plan ran without an execute"
    assert c.node.topics().count("motor_command") == 1, c.node.topics()
    assert c.executor.steps == 0, "the executor stepped a gated plan"

    # 2. An execute naming a DIFFERENT plan is refused: a plan reviewed and then
    #    superseded by a re-plan must never run.
    c.on_control(pack_control_update(execute="p-stale"))
    assert c._running is False
    c.on_control(pack_control_update(execute="p1"))
    assert c._running is True

    # 3. A plan arriving mid-flight is refused, not swapped.
    c.on_plan(_plan_msg("p2", gated=False))
    assert c._plan["plan_id"] == "p1", c._plan["plan_id"]

    # 4. Running: the executor steps, and the leg reports back under ITS id.
    c.on_motor_state(_state())
    assert c.executor.steps == 1
    c._plant_t = 5.0  # past the 1 s trajectory
    c.on_motor_state(_state())
    events = [unpack_controller_event(a) for t, a in c.node.sent if t == "controller_event"]
    legs = [e for e in events if e["kind"] == "leg_result"]
    assert len(legs) == 1 and legs[0]["plan_id"] == "p1" and legs[0]["ok"], legs
    assert c._running is False and c._plan is None

    # 5. The settle rule, with the tolerances the LEG carried in: the plan
    #    playing out is not enough -- a still-moving arm is not settled, and a
    #    residual beyond the bound is not either.
    settle = {"vel_tol": 0.002, "goal_q": np.zeros(7), "residual_max": 0.01}
    c = _controller()
    c.on_plan(_plan_msg("s1", gated=False, completion=settle))
    c._plant_t = 5.0
    c.on_motor_state(_state(vel=0.05))          # moving: 25x the settle gate
    assert c._running is True, "settle leg completed while still moving"
    c.on_motor_state(_state(vel=0.0))
    assert c._running is False, "stopped settle leg never completed"

    # A settle leg parked far from its plan must NOT be called done.
    c = _controller()
    c.on_plan(_plan_msg("s2", gated=False,
                        completion=dict(settle, goal_q=np.full(7, 0.5))))
    c._plant_t = 5.0
    c.on_motor_state(_state(vel=0.0))
    assert c._running is True, "settle leg completed 0.5 rad from its plan"

    # 6. Payload crosses as a control message, not as config.
    c = _controller()
    c.on_control(pack_control_update(payload={"mass_kg": 0.4051, "com_ee": [0.013, 0.0, 0.0]}))
    assert abs(c.executor.payload[0] - 0.4051) < 1e-9, c.executor.payload

    # 7. Disarmed, no commands stream (the bridge zero-holds); mode feedback
    #    still reports gains. Commands wait for the health ACK after arming.
    node = _FakeNode()
    c = ArmController(node, _FakeExecutor(), gripper=dict(_GRIPPER),
                      state_period_s=0.01)
    c.on_motor_state(_state())
    assert node.topics() == ["controller_event"], node.topics()
    assert unpack_controller_event(node.sent[-1][1])["kind"] == "mode"
    c._set_arm(True)
    node.sent.clear()
    c.on_motor_state(_state(pos=-.4, vel=.7))
    assert node.topics() == [], "streamed before the plant ACKed the arm"
    c.bridge_armed = True
    c.on_motor_state(_state(pos=.2, vel=-.3))
    readies = [unpack_controller_event(a) for t, a in node.sent if t == "controller_event"]
    assert [e["kind"] for e in readies] == ["ready"], readies
    assert np.array_equal(readies[0]['q'], np.full(7, .2))
    assert np.array_equal(readies[0]['qd'], np.full(7, -.3))
    assert np.array_equal(readies[0]['q'], c.last_state.position)
    assert np.array_equal(readies[0]['qd'], c.last_state.velocity)
    assert "motor_command" in node.topics(), "no hold streaming after ready"

    # 8. Disarming is a lifecycle reset, not a flag flip: the running leg fails
    #    back (the planner is blocked on it), the loaded plan is dropped, and
    #    re-arming re-earns both the plant ACK and `ready` before anything runs.
    c = _controller()
    c.on_plan(_plan_msg("d1", gated=False))
    assert c._running is True
    c.node.sent.clear()
    c.on_control(pack_control_update(arm=False))
    legs = [unpack_controller_event(a) for t, a in c.node.sent
            if t == "controller_event"]
    assert [(e["plan_id"], e["ok"]) for e in legs] == [("d1", False)], legs
    assert c._plan is None and c._running is False
    assert not c.ready_sent and not c.bridge_armed, "re-arm would skip the ACK"

    c.on_control(pack_control_update(arm=True))
    c.node.sent.clear()
    c.on_motor_state(_state())
    assert c.node.topics() == [], "streamed on a re-arm without a fresh ACK"
    c.bridge_armed = True
    c.on_motor_state(_state())
    kinds = [unpack_controller_event(a)["kind"] for t, a in c.node.sent
             if t == "controller_event"]
    assert kinds == ["ready"], kinds

    # 9. An execute that lands while disarmed is refused AND fails the leg, so
    #    the planner learns instead of waiting on a leg that will never run.
    #    (Flag set directly: a real disarm would have dropped the plan already,
    #    which is check 8 -- this isolates the release guard itself.)
    c = _controller()
    c.on_plan(_plan_msg("x1", gated=True))
    c.armed = False
    c.node.sent.clear()
    c.on_control(pack_control_update(execute="x1"))
    assert c._running is False, "started a leg while disarmed"
    legs = [unpack_controller_event(a) for t, a in c.node.sent
            if t == "controller_event"]
    assert [(e["kind"], e["ok"]) for e in legs] == [("leg_result", False)], legs

    _check_clock_policies()

    # N. Cancel is the OPERATOR stop, and it is none of the other three.
    #    A running leg aborts, the plan is dropped, and the arm stays armed and
    #    still accepts the next plan -- unlike `hold` (frozen forever) and
    #    `stop` (terminal), both of which refuse everything after.
    c = _controller()
    c.on_plan(_plan_msg("run1", gated=False))
    c.on_motor_state(_state())
    c.on_motor_state(_state(vel=0.5))
    assert c._running, "leg should be running before the cancel"
    c.on_control(pack_control_update(cancel=True, reason="operator stop"))
    assert not c._running and c._plan is None, (c._running, c._plan)
    assert c.armed and not c.stopped and not c.frozen, (c.armed, c.stopped, c.frozen)
    # It reports the abort rather than letting a caller block on a leg that
    # will never finish.
    kinds = [unpack_controller_event(a)["kind"] for _t, a in c.node.sent
             if _t == "controller_event"]
    assert "leg_result" in kinds, kinds
    # And the arm is still usable: the next plan loads and runs.
    c.on_plan(_plan_msg("run2", gated=False))
    c.on_motor_state(_state())
    assert c._running, "a cancelled controller must still accept a new plan"

    # Contrast, so the three cannot quietly converge: `hold` refuses the next
    # plan outright.
    c = _controller()
    c.on_motor_state(_state())
    c.on_control(pack_control_update(hold=True, reason="milestone"))
    c.on_plan(_plan_msg("after_hold", gated=False))
    assert c._plan is None, "a frozen controller must not load a plan"

    # N+1. A plant with no health topic still streams; one WITH a bridge still
    #      waits for the armed edge. Getting this backwards is silent either
    #      way -- an arm that never moves, or one anchored on a pre-arm pose.
    # (built raw, not via _controller(), which pre-sets bridge_armed for the
    #  checks above -- that shortcut is the very thing under test here)
    c = ArmController(
        _FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER),
        state_period_s=0.01, plant_reports_health=False,
    )
    c._set_arm(True)
    c.on_motor_state(_state())
    assert c.ready_sent, "a bridgeless plant must not wait for a health edge"
    assert any(t == "motor_command" for t, _a in c.node.sent), c.node.topics

    c = ArmController(
        _FakeNode(), _FakeExecutor(), gripper=dict(_GRIPPER), state_period_s=0.01,
    )  # the default: there IS a bridge
    c._set_arm(True)
    c.on_motor_state(_state())
    assert not c.ready_sent, "a bridged plant must wait for the armed edge"
    c.on_motor_health(pack_json_message("motor_health", {"armed": True}))
    c.on_motor_state(_state())
    assert c.ready_sent, "the armed edge should have released it"

    # N+2. The console's gripper slider reaches the jaws. Needs an arm whose
    #      gripper IS a motor slot (mimic_cfg=None means the jaws are their own
    #      device and no slot is packed at all -- the FR3 case). Default is the
    #      configured open width: the assembly path never sends this input,
    #      because the bridge's grasp gate owns the slot there and two writers
    #      would fight over it.
    from arm_control.joint_motor_map import gripper_finger_to_motor
    from arm_control.messages import pack_motor_command
    from arm_control.messages import unpack_motor_command as _umc

    mimic = {"source": "Gripper", "motor_open": 0.0, "motor_closed": -4.985,
             "lower": 0.0, "upper": 0.0439}
    node = _FakeNode()
    c = ArmController(
        node, _FakeExecutor(n=6), state_period_s=0.01,
        gripper={"n_motors": 7, "mimic": mimic, "open_finger_m": 0.04,
                 "gains": (40.0, 2.0)},
    )
    c._set_arm(True)
    c.bridge_armed = True
    c.on_motor_state(_state(n=7))
    held = _umc([a for t, a in node.sent if t == "motor_command"][-1], 7)
    assert held["position"][6] == gripper_finger_to_motor(0.04, mimic), held["position"][6]

    zeros2 = np.zeros(2)
    c.on_gripper(pack_motor_command([0.01, 0.01], zeros2, zeros2, zeros2, zeros2))
    c.on_motor_state(_state(n=7))
    moved = _umc([a for t, a in node.sent if t == "motor_command"][-1], 7)
    assert moved["position"][6] == gripper_finger_to_motor(0.01, mimic), moved["position"][6]
    assert moved["position"][6] != held["position"][6]

    # N+3. Gains change the control law, so a running leg is cancelled first --
    #      float (kp=0) applied to a moving trajectory would drop the arm
    #      through the rest of its path.
    c = _controller()
    c.on_plan(_plan_msg("gainrun", gated=False))
    c.on_motor_state(_state())
    c.on_motor_state(_state(vel=0.5))
    assert c._running
    c.on_control(pack_control_update(gains={"kp": [0.0] * 7, "kd": [1.0] * 7}))
    assert not c._running, "a gains change must not ride on top of a running leg"
    assert float(np.max(np.abs(c.executor.kp))) == 0.0, c.executor.kp
    assert c.armed and not c.stopped, "and it must leave the arm armed"

    # N+4. Jog: it moves the arm, and it DIES IF IT STOPS ARRIVING. That
    #      expiry is the whole safety story -- a closed tab, a wedged console
    #      and a cut network are indistinguishable here, and all three must
    #      stop the arm without anything having to notice and send a stop.
    from arm_control.messages import pack_jog

    now = [1000.0]
    c = _controller(clock=lambda: now[0])
    c.on_jog(pack_jog(q=[0.4] * 7))
    c.on_motor_state(_state())
    cmd = [a for t, a in c.node.sent if t == "motor_command"][-1]
    served = _umc(cmd, 7)["position"][:7]
    assert float(np.max(np.abs(served - 0.4))) < 1e-9, served

    # still fresh just inside the timeout
    now[0] += 0.19
    c.node.sent.clear()
    c.on_motor_state(_state())
    served = _umc([a for t, a in c.node.sent if t == "motor_command"][-1], 7)["position"][:7]
    assert float(np.max(np.abs(served - 0.4))) < 1e-9, "a fresh jog stopped early"

    # ...and dead just past it: the arm holds where it IS (measured = zeros),
    # not where the last setpoint was still asking it to go.
    now[0] += 0.5
    c.node.sent.clear()
    c.on_motor_state(_state())
    served = _umc([a for t, a in c.node.sent if t == "motor_command"][-1], 7)["position"][:7]
    assert float(np.max(np.abs(served))) < 1e-9, f"a stale jog kept commanding: {served}"
    assert c.armed, "expiry holds the arm; it does not disarm it"

    # A jog must never fight a reviewed plan for the same joints.
    c = _controller(clock=lambda: now[0])
    c.on_plan(_plan_msg("owns", gated=False))
    c.on_motor_state(_state())
    assert c._running
    c.on_jog(pack_jog(q=[0.9] * 7))
    assert c._jog_q is None, "a jog was accepted while a plan was running"

    print("arm_controller self-check ok")


if __name__ == "__main__":
    _self_check()
