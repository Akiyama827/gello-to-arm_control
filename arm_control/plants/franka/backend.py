"""Franka Research 3 backend — READ-ONLY arm state over libfranka.

THERE IS NO MOTION PATH IN THIS BACKEND, deliberately (user decision
2026-07-26): the FR3 is driven in TORQUE mode only, and the torque servo law
runs as a C++ 1 kHz loop on the realtime machine — a Python process on a stock
kernel cannot hold libfranka's <300 µs tick budget, and the position-control
mode this file once carried was a debug path. What remains here is everything
that is NOT the servo loop:

* state streaming (``read_once`` → the bridge's ``motor_state`` shape),
* health/fault reporting (``robot_mode`` + ``current_errors`` + the
  ``control_command_success_rate`` packet-loss canary),
THE HAND IS NOT HERE. It is owned by ``rt/src/hand_bridge.cpp`` on the RT
box and reached through ``end_effectors/franka_adapter.py`` — PC-side
pylibfranka cannot read it since the network migration (server-push UDP does
not cross the PC-side NAT: moves worked, every read timed out, bench
2026-07-28). This file carried a second, direct ``fr.Gripper`` implementation
until 2026-09-10; it was unreachable from every graph and could not have
worked. Likewise payload: ``Robot.set_load`` is a NON-REALTIME command,
illegal once a control session is open, so only the RT box's startup
``--ee-mass`` can declare a fixed load, and a GRASPED module's weight rides
the controller's ``tau_ff`` (``JointTrajectoryExecutor.set_payload``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

N_JOINTS = 7  # FR3 is a 7-DoF arm; fixed by the hardware, not a config knob


class FrankaBackendUnavailableError(RuntimeError):
    """Raised when pylibfranka or the robot itself cannot be reached."""


def _import_pylibfranka():
    try:
        import pylibfranka
    except ImportError as exc:  # pragma: no cover - bench dependency
        raise FrankaBackendUnavailableError(
            "pylibfranka is not installed in this environment. The Control stack "
            "runs in conda BASE; install it there (it must match the system "
            "libfranka — /usr/lib/libfranka.so)."
        ) from exc
    return pylibfranka


@dataclass(frozen=True)
class FrankaConfig:
    """Everything the FR3 backend reads from the runtime YAML (``franka:`` block)."""

    ip: str = "172.16.0.2"
    # kEnforce demands an RT-scheduled thread and refuses to run without one.
    # kIgnore lets bring-up proceed on a stock kernel at the cost of occasional
    # communication_constraints_violation reflexes — fine for hovering and
    # calibration, NOT for a real dock.
    enforce_realtime: bool = True
    # Collision thresholds: torque (7) and cartesian force (6). Low values make
    # the arm stop on light contact — safer for bring-up, and the reflex is
    # recoverable via automatic_error_recovery.
    lower_torque_thresholds: tuple[float, ...] = (20.0,) * N_JOINTS
    upper_torque_thresholds: tuple[float, ...] = (40.0,) * N_JOINTS
    lower_force_thresholds: tuple[float, ...] = (20.0,) * 6
    upper_force_thresholds: tuple[float, ...] = (40.0,) * 6
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg) -> "FrankaConfig":
        raw = dict(cfg.get("franka") or {})

        def _vec(key, default, n):
            value = raw.get(key, default)
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size == 1:
                arr = np.full(n, float(arr[0]))
            if arr.shape != (n,):
                raise ValueError(f"franka.{key} must have {n} values, got {arr.size}")
            return tuple(float(v) for v in arr)

        return cls(
            ip=str(raw.get("ip", "172.16.0.2")),
            enforce_realtime=bool(raw.get("enforce_realtime", True)),
            lower_torque_thresholds=_vec(
                "lower_torque_thresholds", cls.lower_torque_thresholds, N_JOINTS
            ),
            upper_torque_thresholds=_vec(
                "upper_torque_thresholds", cls.upper_torque_thresholds, N_JOINTS
            ),
            lower_force_thresholds=_vec(
                "lower_force_thresholds", cls.lower_force_thresholds, 6
            ),
            upper_force_thresholds=_vec(
                "upper_force_thresholds", cls.upper_force_thresholds, 6
            ),
            raw=raw,
        )


def active_error_names(errors: Any) -> list[str]:
    """Names of the set flags on a ``franka::Errors``.

    The binding exposes one bool attribute per error; libfranka has no iterator,
    so reflect over the public names. Used for the fault reason string — a bare
    "reflex" tells an operator nothing, ``joint_reflex`` tells them where to look.
    """
    out = []
    for name in dir(errors):
        if name.startswith("_"):
            continue
        try:
            if bool(getattr(errors, name)):
                out.append(name)
        except Exception:
            continue
    return out


class FrankaHardwareBackend:
    """FR3 arm + Franka Hand behind the DM backend's method surface."""

    def __init__(self, config: FrankaConfig, joint_names: list[str] | None = None) -> None:
        self.config = config
        self.joint_names = list(joint_names or [f"fr3_joint{i + 1}" for i in range(N_JOINTS)])
        if len(self.joint_names) != N_JOINTS:
            raise ValueError(f"FR3 has {N_JOINTS} joints, got {len(self.joint_names)}")
        self._fr = None
        self.robot = None
        self._last_state = None
        self._state_errors = 0          # consecutive read_once failures
        self._latched_fault: str | None = None
        self._armed = False
        self.last_close_errors: list = []

    @classmethod
    def from_config(cls, cfg) -> "FrankaHardwareBackend":
        from arm_control.config import arm_joints

        return cls(FrankaConfig.from_config(cfg), joint_names=arm_joints(cfg))

    @property
    def num_motors(self) -> int:
        return N_JOINTS

    # -- lifecycle ------------------------------------------------------------
    def open(self) -> None:
        fr = _import_pylibfranka()
        self._fr = fr
        cfg = self.config
        try:
            self.robot = fr.Robot(
                cfg.ip,
                fr.RealtimeConfig.kEnforce
                if cfg.enforce_realtime
                else fr.RealtimeConfig.kIgnore,
            )
        except Exception as exc:
            raise FrankaBackendUnavailableError(
                f"cannot reach the FR3 at {cfg.ip}: {exc}. Check the cable, that "
                "the robot is unlocked in Desk, and that FCI mode is active."
            ) from exc
        self.robot.set_collision_behavior(
            list(cfg.lower_torque_thresholds),
            list(cfg.upper_torque_thresholds),
            list(cfg.lower_force_thresholds),
            list(cfg.upper_force_thresholds),
        )
        self._last_state = self.robot.read_once()
        print(
            f"[franka] connected to {cfg.ip} — read-only arm state "
            f"(motion runs on the RT machine; "
            f"realtime={'enforced' if cfg.enforce_realtime else 'IGNORED'})",
            flush=True,
        )

    def enable_all(self) -> None:
        """Arm: clear any latched reflex. Nothing can move from this process —
        arming only clears a latched reflex and marks the monitor live."""
        if self.robot is None:
            raise FrankaBackendUnavailableError("enable_all() before open()")
        self.robot.automatic_error_recovery()
        self._latched_fault = None
        self._armed = True

    def safe_stop(self) -> None:
        """Zero-authority stop: halt any active motion and disarm."""
        if self.robot is not None:
            try:
                self.robot.stop()
            except Exception as exc:
                print(f"[franka] stop: {exc}", flush=True)
        self._armed = False

    def close(self) -> None:
        self.last_close_errors = []
        try:
            self.safe_stop()
        except Exception as exc:
            self.last_close_errors.append(exc)
        self.robot = None

    # -- feedback -------------------------------------------------------------
    def motor_state(self) -> dict[str, np.ndarray]:
        """Measured state in the bridge's ``motor_state`` shape.

        ``read_once`` is a network round trip and can raise while a motion is
        active; the last good state is reused so one hiccup never presents the
        orchestrator with a zeroed pose (which it would plan from).
        """
        state = self._read_state()
        if state is None:
            zeros = np.zeros(N_JOINTS)
            return {
                "position": zeros, "velocity": zeros, "position_cmd": zeros,
                "velocity_cmd": zeros, "torque_cmd": zeros,
                "kp": zeros, "kd": zeros, "torque": zeros,
            }
        q = np.asarray(state.q, dtype=float)
        return {
            "position": q,
            "velocity": np.asarray(state.dq, dtype=float),
            "position_cmd": np.asarray(state.q_d, dtype=float),
            "velocity_cmd": np.asarray(state.dq_d, dtype=float),
            # tau_J_d is the controller's own desired torque — the FR3 analogue
            # of the DM torque_cmd, and useful for the same tracking plots.
            "torque_cmd": np.asarray(state.tau_J_d, dtype=float),
            # No gains to echo: this process closes no loop over the arm.
            "kp": np.zeros(N_JOINTS),
            "kd": np.zeros(N_JOINTS),
            "torque": np.asarray(state.tau_J, dtype=float),
        }

    def _read_state(self):
        if self.robot is None:
            return self._last_state
        try:
            self._last_state = self.robot.read_once()
            self._state_errors = 0
        except Exception as exc:
            self._state_errors += 1
            if self._state_errors in (1, 50, 500):
                print(
                    f"[franka] read_once failed ({self._state_errors}x): {exc}",
                    flush=True,
                )
        return self._last_state

    def motor_health(self) -> dict:
        """Health in the ``motor_health`` shape the orchestrator consumes.

        The FR3 owns its own safety, so this REPORTS rather than enforces:
        ``robot_mode`` Reflex/UserStopped and any set error flag are faults, and
        ``control_command_success_rate`` is the packet-loss canary that precedes
        a ``communication_constraints_violation``.
        """
        state = self._last_state
        if state is None:
            return {
                "armed": self._armed, "latched_fault": self._latched_fault or "",
                "any_fault": False, "robot_mode": "unknown", "errors": [],
                "success_rate": 0.0, "state_read_errors": self._state_errors,
            }
        errors = active_error_names(state.current_errors)
        mode = getattr(state.robot_mode, "name", str(state.robot_mode))
        success_rate = float(getattr(state, "control_command_success_rate", 1.0))
        faulted = bool(errors) or mode in ("Reflex", "UserStopped")
        if faulted and self._latched_fault is None:
            self._latched_fault = f"{mode}: {', '.join(errors) or 'stopped'}"
            self._armed = False
            print(f"[franka] FAULT — {self._latched_fault}", flush=True)
        return {
            "armed": self._armed,
            "latched_fault": self._latched_fault or "",
            "any_fault": faulted,
            "robot_mode": mode,
            "errors": errors,
            "success_rate": success_rate,
            "state_read_errors": self._state_errors,
        }

    def set_payload(self, mass_kg: float, com_m=(0.0, 0.0, 0.0)) -> None:
        """Tell libfranka about the held module (the FR3's payload feedforward).

        This replaces the DM path's RNEA payload torque: the FR3 compensates the
        load inside its own controller, so the mass must be declared here rather
        than added to tau_ff.
        """
        if self.robot is None:
            return
        com = np.asarray(com_m, dtype=float).ravel()[:3]
        # NOT a point mass: the control box rejects a nonzero mass with a zero
        # inertia tensor ("Set Load command rejected: invalid argument!",
        # probed 2026-08-06), and the failure lands in the except below where
        # it becomes a log line and an UNDECLARED payload. Solid sphere of
        # radius R about the CoM, I = 2/5 m R^2 — diagonal, always valid, and
        # irrelevant to the gravity compensation this call exists for.
        inertia = 0.4 * float(mass_kg) * 0.05**2
        try:
            self.robot.set_load(
                float(mass_kg), com.tolist(),
                [inertia, 0, 0, 0, inertia, 0, 0, 0, inertia],
            )
        except Exception as exc:
            print(f"[franka] set_load failed: {exc}", flush=True)



def _demo() -> None:
    """Self-check for the pure parts — config parsing and error reflection.

    Everything that talks to the robot needs an FR3; everything below does not.
    """
    cfg = FrankaConfig.from_config(
        {
            "franka": {
                "ip": "10.0.0.2",
                "enforce_realtime": False,
                "upper_torque_thresholds": 35,   # scalar broadcasts to 7
            }
        }
    )
    assert cfg.ip == "10.0.0.2"
    assert cfg.enforce_realtime is False
    assert cfg.upper_torque_thresholds == (35.0,) * 7
    # Defaults survive an empty block.
    empty = FrankaConfig.from_config({})
    assert empty.ip == "172.16.0.2" and len(empty.upper_torque_thresholds) == N_JOINTS

    # A wrong-length vector must be rejected, not silently broadcast.
    try:
        FrankaConfig.from_config({"franka": {"upper_torque_thresholds": [1, 2, 3]}})
    except ValueError as exc:
        assert "7 values" in str(exc), exc
    else:
        raise AssertionError("wrong-length upper_torque_thresholds was accepted")

    # Error reflection picks out exactly the set flags.
    class FakeErrors:
        joint_reflex = True
        cartesian_reflex = False
        self_collision_avoidance_violation = True

    assert sorted(active_error_names(FakeErrors())) == [
        "joint_reflex",
        "self_collision_avoidance_violation",
    ]
    assert active_error_names(type("E", (), {})()) == []
    print("franka_backend: ok")


if __name__ == "__main__":
    _demo()
