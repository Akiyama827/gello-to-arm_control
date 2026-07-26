"""Command-mode safety layer for the DM hardware bridge."""
from __future__ import annotations

import numpy as np

from arm_control.node_utils import _zero_command


class SafetyController:
    """Command-mode safety layer for the DM hardware bridge.

    The real arm holds its last MIT command if the host dies, so safety is
    software-owned and fails safe.  Disarmed is the default: motors stay
    disabled (DM_ENABLE is sent on *arm*, never silently on open) and any
    incoming setpoint is replaced by a zero-torque hold.  Arming enables the
    motors and forwards real setpoints.  The deadman watchdog, the fault reflex,
    and the per-joint torque clamp all fall back to one safe-stop primitive
    (zero torque + DM_DISABLE, latched disarm).

    Pure orchestration over the backend + a wall clock injected as ``now`` so it
    is unit-testable against a fake transport with no Dora runtime.
    """

    def __init__(
        self,
        backend,
        *,
        deadman_timeout_s: float,
        temp_limit_c: float,
        torque_limits: np.ndarray,
        arm_ramp_sec: float = 0.0,
    ) -> None:
        self.backend = backend
        self.n = backend.num_motors
        self.deadman_timeout_s = float(deadman_timeout_s)
        self.temp_limit_c = float(temp_limit_c)
        self.arm_ramp_sec = float(arm_ramp_sec)
        self._armed_at: float | None = None
        self.torque_limits = np.asarray(torque_limits, dtype=np.float64)
        if self.torque_limits.size != self.n:
            raise ValueError(
                f"torque_limits length {self.torque_limits.size} != {self.n} motors"
            )
        self.armed = False
        self.latched_fault: str | None = None
        self._last_command_t: float | None = None
        self._first_command_received = False
        self._clamp_joints: list[int] = []

    # -- operator arm / disarm ------------------------------------------------
    def arm(self, now: float) -> None:
        if self.latched_fault is not None:
            print(
                f"[safety] arm refused — latched fault: {self.latched_fault}",
                flush=True,
            )
            return
        self.backend.enable_all()  # the fixed no-DM_ENABLE-in-command-mode gap
        self.armed = True
        self._armed_at = now  # soft-start ramp anchor
        self._last_command_t = now  # start the deadman clock at arm time
        self._first_command_received = False  # awaiting the orchestrator's 1st command
        print("[safety] ARMED — motors enabled", flush=True)

    def disarm(self, reason: str | None = None) -> None:
        # ponytail: single-step to zero gains (safe_stop sends a kp=kd=0 frame)
        # then DM_DISABLE, rather than the design's gradual kp/kd ramp — DISABLE
        # removes torque anyway, so the ramp only softens a mechanical jerk on a
        # <0.3 kg arm. Upgrade to a multi-tick ramp here if the bench shows one.
        self.backend.safe_stop()  # zero torque + DM_DISABLE, channel stays up
        self.armed = False
        self._first_command_received = False
        if reason is not None:
            self.latched_fault = reason
            print(f"[safety] DISARM latched: {reason}", flush=True)
        else:
            print("[safety] DISARMED (operator)", flush=True)

    # -- command path ---------------------------------------------------------
    def on_command(self, command: dict, now: float) -> None:
        self._last_command_t = now
        if self.latched_fault is not None:
            return  # faulted: forward nothing (motors already disabled)
        if not self.armed:
            self.backend.apply_command(_zero_command(self.n))  # zero-torque hold
            return
        self._first_command_received = True  # real setpoint flowing: keepalive can stop
        self.backend.apply_command(self._soft_start(self._clamp_torque(command), now))

    @property
    def awaiting_first_command(self) -> bool:
        """True while armed but no real setpoint has been forwarded yet.

        In this arm->first-command window the bridge sends a zero-torque MIT
        keepalive (``listen_step``) so the DM motors solicit a fresh reply — giving
        the orchestrator the TRUE measured pose to start its trajectory from — while
        commanding no motion.  It stops once the first real command arrives.
        """
        return self.armed and not self._first_command_received

    def _clamp_torque(self, command: dict) -> dict:
        tau = np.asarray(command["torque"], dtype=np.float64).copy()
        lim = self.torque_limits
        over = np.abs(tau) > lim
        self._clamp_joints = [int(i) for i in np.nonzero(over)[0]]
        if self._clamp_joints:
            tau = np.clip(tau, -lim, lim)
            print(
                f"[safety] tau_ff clamp engaged on joints {self._clamp_joints}",
                flush=True,
            )
        out = dict(command)
        out["torque"] = tau
        return out

    def _soft_start(self, command: dict, now: float) -> dict:
        """Scale kp/kd/tau_ff 0 -> 1 over ``arm_ramp_sec`` after arming.

        Going from disabled to full gains + 100% gravity ff in one tick turns
        any gravity-model residual into an instant torque step (a visible
        twitch). The ramp eases the arm onto the servo instead.
        """
        if self.arm_ramp_sec <= 0 or self._armed_at is None:
            return command
        scale = (now - self._armed_at) / self.arm_ramp_sec
        if scale >= 1.0:
            return command
        scale = max(0.0, scale)
        out = dict(command)
        for key in ("kp", "kd", "torque"):
            out[key] = np.asarray(command[key], dtype=np.float64) * scale
        return out

    # -- deadman watchdog (fires from the node tick, not on command arrival) --
    def tick(self, now: float) -> None:
        if not self.armed or self._last_command_t is None:
            return
        if now - self._last_command_t > self.deadman_timeout_s:
            self.disarm(
                f"deadman: no motor_command for >{self.deadman_timeout_s:g}s"
            )

    # -- fault reflex + observable health -------------------------------------
    def check_health(self, now: float) -> dict:
        health = self.backend.motor_health()
        errors = [int(e) for e in health["error"]]
        t_mos = [float(t) for t in health["t_mos"]]
        t_rotor = [float(t) for t in health["t_rotor"]]
        # DM ERR nibble: 0=disabled, 1=enabled, 8..E hardware faults (2..7
        # undocumented -> treat as faults). 0 while armed = the motor dropped
        # out of enable (no holding torque); guarded on first-command so the
        # arm-instant race with a stale pre-enable frame can't latch.
        fault_joints = [
            i
            for i, e in enumerate(errors)
            if e not in (0, 1)
            or (e == 0 and self.armed and self._first_command_received)
        ]
        hot_joints = [
            i
            for i in range(self.n)
            if t_mos[i] > self.temp_limit_c or t_rotor[i] > self.temp_limit_c
        ]
        if (fault_joints or hot_joints) and self.latched_fault is None:
            self.disarm(
                f"fault reflex — err_joints={fault_joints} hot_joints={hot_joints}"
            )
        return {
            "armed": self.armed,
            "latched_fault": self.latched_fault or "",
            "error": errors,
            "t_mos": t_mos,
            "t_rotor": t_rotor,
            "temp_limit_c": self.temp_limit_c,
            "fault_joints": fault_joints,
            "hot_joints": hot_joints,
            "clamp_joints": list(self._clamp_joints),
            "any_fault": bool(fault_joints or hot_joints),
        }
