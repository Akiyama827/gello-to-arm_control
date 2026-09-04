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

from arm_control.execution.trajectory_executor import (
    JointServoCommand,
    JointState,
)
from arm_control.planning.trajectory import JointTrajectory
from arm_control.joint_motor_map import (
    pack_arm_gripper_command,
    unpack_motor_state_to_joint,
)
from arm_control.messages import (
    pack_controller_event,
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
        clock=None,
    ) -> None:
        self.node = node
        self.executor = executor
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
        # and no leg could ever complete. configs/sim/franka_workcell_add.yaml
        # sets it (0.05) and configs/real/franka.yaml does NOT, so every green
        # sim run hid a controller that would have hung on the first bench
        # move. An arm that cannot finish a motion is not an acceptable
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
        self._plant_t = 0.0
        self._finger_m = 0.0

        self.last_state = JointState(np.zeros(self.n_arm), np.zeros(self.n_arm))
        self.last_command: JointServoCommand | None = None
        self._hold_anchor: JointServoCommand | None = None
        self._cartesian_now: dict | None = None   # spec for the loaded plan
        self._cartesian_poses: np.ndarray | None = None
        self._last_cartesian_pose: np.ndarray | None = None

        self.armed = False
        self.bridge_armed = False   # bridge-confirmed (motor_health armed edge)
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
        if self.stopped or self.frozen:
            return
        if self._running:
            print(
                f"[arm_controller] REFUSED plan {plan['plan_id']} "
                f"({plan['phase']}): {self._plan['plan_id']} is still executing",
                flush=True,
            )
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

    def on_motor_health(self, value) -> None:
        payload = unpack_json_message(value)
        armed = bool(payload.get("armed", False))
        if armed and not self.bridge_armed:
            self.bridge_armed = True
        elif not armed and self.bridge_armed and self.armed and not self.stopped:
            self._fault("plant reported disarmed")

    def on_motor_state(self, value) -> None:
        if self.stopped:
            return
        if self._state_period is not None:
            self._plant_t += self._state_period  # one message == one plant period
        arm, finger_m = unpack_motor_state_to_joint(
            value, self.gripper["n_motors"], self.gripper["mimic"], n_arm=self.n_arm
        )
        self.last_state = arm
        if finger_m is not None:
            self._finger_m = float(finger_m)

        if not self.armed:
            # Disarmed: track the pose, stream nothing. The bridge zero-holds.
            return
        if not self.ready_sent:
            if not self.bridge_armed:
                # The bridge publishes motor_state while disarmed, so a queued
                # pre-arm sample can arrive after our arm message. Only a state
                # received AFTER the armed health edge is provably post-enable;
                # starting from anything earlier could plan from a stale/zero
                # pose and jump the arm.
                return
            self.ready_sent = True
            self.node.send_output(
                "controller_event",
                pack_controller_event(kind="ready", q=arm.position),
            )
            # Fall through: the planner needs a hold streaming from right now,
            # because it is about to spend seconds planning the first leg.
        if self._running and self.executor.has_trajectory:
            command = self.executor.step(self.exec_time, arm)
            self._hold_anchor = None  # motion streams: next hold re-anchors
            self._send(command, self._cartesian_for_time(self.exec_time))
            if self._leg_done(arm):
                self._finish_leg(ok=True, reason="")
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
        self.executor.load_trajectory(self._pending_traj, t_start=self.exec_time)
        self._running = True
        print(
            f"[arm_controller] executing {self._plan['phase']} "
            f"({plan_id}, {self._pending_traj.duration_sec:.2f}s)",
            flush=True,
        )

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

    def _finish_leg(self, *, ok: bool, reason: str) -> None:
        plan_id = "" if self._plan is None else self._plan["plan_id"]
        self.executor.clear_trajectory()
        self._running = False
        self._plan = None
        self.node.send_output(
            "controller_event",
            pack_controller_event(
                kind="leg_result", plan_id=plan_id, ok=ok, reason=reason
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
        tau = np.clip(t_now - self.executor._t_start, times[0], times[-1])
        idx = int(np.searchsorted(times, tau, side="right")) - 1
        return poses[max(0, min(idx, len(poses) - 1))]

    def _send(self, command: JointServoCommand, cartesian_pose) -> None:
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
                self.gripper["open_finger_m"],
                self.gripper["gains"],
                self.gripper["mimic"],
                cartesian=cartesian,
            ),
        )

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
        self.bridge_armed = False
        self._hold_anchor = None
        self.last_command = None

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
    def run(self) -> None:
        handlers = {
            "motor_state": self.on_motor_state,
            "motor_health": self.on_motor_health,
            "plan": self.on_plan,
            "control": self.on_control,
        }
        for event in self.node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            handler = handlers.get(event["id"])
            if handler is not None:
                handler(event["value"])
            if self.stopped:
                break


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

    def __init__(self, n: int = 7) -> None:
        self.n_joints = n
        self._traj = None
        self._t_start = 0.0
        self.kp = np.full(n, 1200.0)
        self.kd = np.full(n, 30.0)
        self.payload = (0.0, None)
        self.steps = 0

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
    c = ArmController(node, _FakeExecutor(), gripper=dict(_GRIPPER), **kw)
    c._set_arm(True)
    c.bridge_armed = True
    c.on_motor_state(_state())   # the post-arm state that earns `ready`
    node.sent.clear()
    return c


def _state(n: int = 7, vel: float = 0.0):
    from arm_control.messages import pack_motor_state_dict

    return pack_motor_state_dict(
        {k: np.zeros(n) for k in
         ("position", "velocity", "position_cmd", "velocity_cmd",
          "torque_cmd", "kp", "kd", "torque")}
        | {"velocity": np.full(n, vel)}
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
    from arm_control.messages import pack_control_update, unpack_controller_event

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

    # 7. Disarmed, it streams NOTHING (the bridge zero-holds); armed but before
    #    the health ACK it streams nothing either, then announces ready once.
    node = _FakeNode()
    c = ArmController(node, _FakeExecutor(), gripper=dict(_GRIPPER),
                      state_period_s=0.01)
    c.on_motor_state(_state())
    assert node.topics() == [], node.topics()
    c._set_arm(True)
    node.sent.clear()
    c.on_motor_state(_state())
    assert node.topics() == [], "streamed before the plant ACKed the arm"
    c.bridge_armed = True
    c.on_motor_state(_state())
    readies = [unpack_controller_event(a) for t, a in node.sent if t == "controller_event"]
    assert [e["kind"] for e in readies] == ["ready"], readies
    assert readies[0]["q"] is not None
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

    print("arm_controller self-check ok")


if __name__ == "__main__":
    _self_check()
