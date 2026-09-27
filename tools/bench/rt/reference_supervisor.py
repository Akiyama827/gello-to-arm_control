"""Native supervisor/provider integration against the fake-only RT runtime."""

from pathlib import Path
import subprocess
import re
import sys
import tempfile
import time
import threading
from dataclasses import replace
import numpy as np

from arm_control.config import load_robot_config
from arm_control.control.factory import build_arm_controller
from arm_control.control.inprocess import QueueControllerNode
from arm_control.execution.contract import (
    Start,
    Completion,
    Lifecycle,
    Phase,
    TrajectoryRequest,
    Stamp,
    Clock,
)
from arm_control.motion import JointTrajectory, TrajectoryPoint
from arm_control.supervisor.runtime import RobotRuntime
from native_reference import unused_port
from arm_control.plants.remote_rt import protocol as p
from arm_control.plants.remote_rt.client import RtLinkError
import socket

# Reuse the existing analytical fixture; no robot asset or deployed model.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_sdk_supervisor import write_config


def until(fn, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        out = fn()
        if out:
            return out
        time.sleep(0.01)
    raise AssertionError("timed out: " + str(fn()))


def main(binary):
    udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode="w+") as log:
        proc = subprocess.Popen([binary, str(udp), str(tcp)], stdout=log, stderr=log)
        path = write_config(Path(tmp), udp, tcp)
        # Dedicated fake server uses j0,j1,j2 and id=fake. Names are changed in
        # both the URDF and config, leaving all analytic dynamics unchanged.
        for file in (path, Path(tmp) / "three.urdf"):
            txt = file.read_text().replace("sdk_arm", "fake")
            file.write_text(re.sub(r"\bj([123])\b", lambda m: f"j{int(m[1])-1}", txt))
        cfg = load_robot_config(path)
        runtime = RobotRuntime(cfg)
        # This is an analytical test binding, not project hardware staging.
        runtime.controller.model_token = "analytical-three-joint-fixture"
        oracle_executor = build_arm_controller(
            cfg, QueueControllerNode(), arm_id="fake", require_policy=True
        ).executor
        step = runtime.controller.executor.step
        parity_samples = []

        def checked_step(point, measured, *, integral_bias):
            reference = runtime.controller.reference
            assert reference.t_start is None
            expected = reference.trajectory.sample_at(point.time)
            if point.time >= reference.duration_sec:
                expected = TrajectoryPoint(
                    point.time, expected.position, np.zeros(3), np.zeros(3)
                )
            actual = step(point, measured, integral_bias=integral_bias)
            oracle = oracle_executor.step(expected, measured, integral_bias=integral_bias)
            for field in ("q_des", "qd_des", "tau_ff", "kp", "kd"):
                assert (
                    getattr(actual, field).tobytes() == getattr(oracle, field).tobytes()
                ), field
            parity_samples.append(point.time)
            return actual

        runtime.controller.executor.step = checked_step
        try:
            time.sleep(0.2)
            runtime.start()
            assert runtime._native is not None
            runtime.claim("test")
            gen = runtime.arm("test")
            assert (
                runtime.await_authority(gen, armed=True, wait_s=3).lifecycle
                is Lifecycle.ARMED
            )
            q0 = runtime.state().q
            t = np.arange(321) * 0.01
            delta = np.array([0.08, -0.06, 0.04])
            u = t / t[-1]
            q = q0 + (3 * u * u - 2 * u * u * u)[:, None] * delta
            v = (6 * u - 6 * u * u)[:, None] * delta / t[-1]
            request = TrajectoryRequest(
                "long",
                "fake",
                ("j0", "j1", "j2"),
                JointTrajectory(t, q, v),
                model_token=runtime.controller.model_token,
            )
            status = runtime.submit("test", request)
            assert status.phase is Phase.ADMITTED, status
            until(lambda: runtime.status(status.execution_id).terminal, 8)
            done = runtime.status(status.execution_id)
            assert done.phase is Phase.COMPLETED, done
            phases = [r.phase for r in runtime.receipts(status.execution_id)]
            assert phases == [
                Phase.REQUESTED,
                Phase.ADMITTED,
                Phase.SELECTED,
                Phase.SENT,
                Phase.APPLIED,
                Phase.COMPLETED,
            ], phases
            state, _ = runtime.link.latest_state()
            assert state.reference["revision"] >= 4, state.reference
            np.testing.assert_array_equal(state.reference["q_ref"], q[-1])
            time.sleep(0.2)
            assert runtime.authority().lifecycle is Lifecycle.ARMED
            # Cancellation is an explicit retirement followed by the same
            # controller-owned measured, compensated hold as streamed mode.
            q0 = runtime.state().q
            tr = JointTrajectory(
                [0.0, 1.0, 2.0], np.stack([q0, q0 + 0.04, q0 + 0.08]), np.zeros((3, 3))
            )
            req = TrajectoryRequest(
                "cancel",
                "fake",
                ("j0", "j1", "j2"),
                tr,
                model_token=runtime.controller.model_token,
            )
            ex = runtime.submit("test", req)
            until(lambda: Phase.APPLIED in runtime.status(ex.execution_id).reached)
            runtime.cancel("test", ex.execution_id)
            assert runtime.status(ex.execution_id).phase is Phase.CANCELLED
            time.sleep(0.3)
            assert runtime.authority().lifecycle is Lifecycle.ARMED, runtime.authority()
            state, _ = runtime.link.latest_state()
            assert state.reference["status"] == 2, state.reference
            assert np.max(np.abs(state.dq)) < 0.05, state.dq
            # A delayed initial COMMIT cannot outlive the PC selection deadline.
            original_commit = runtime.link.commit_reference

            def delayed_commit(*args, **kw):
                time.sleep(0.18)
                return original_commit(*args, **kw)

            runtime.link.commit_reference = delayed_commit
            q0 = runtime.state().q
            tr = JointTrajectory(
                [0.0, 0.5, 1.0], np.stack([q0, q0 + 0.01, q0 + 0.02]), np.zeros((3, 3))
            )
            deadline = Stamp(
                time.monotonic_ns() + 150_000_000, Clock.PC, runtime.domain
            )
            req = replace(
                req, request_id="late-commit", trajectory=tr, admit_by=deadline
            )
            before_id = runtime.link.latest_state()[0].reference["reference_id"]
            ex = runtime.submit("test", req)
            until(
                lambda: (
                    runtime._native.selection_requested
                    and runtime._native.key == ex.execution_id
                )
            )
            replacement = runtime.submit(
                "test",
                replace(
                    req,
                    request_id="racing-supersede",
                    supersedes=ex.execution_id,
                    admit_by=None,
                ),
            )
            assert (
                replacement.phase is Phase.REJECTED
                and "in flight" in replacement.reason
            ), replacement
            until(lambda: runtime.status(ex.execution_id).terminal)
            assert runtime.status(ex.execution_id).phase is Phase.EXPIRED, (
                runtime.status(ex.execution_id)
            )
            assert Phase.SELECTED not in runtime.status(ex.execution_id).reached
            assert runtime.link.latest_state()[0].reference["reference_id"] == before_id
            runtime.link.commit_reference = original_commit

            # A PC sample older than the envelope budget follows normal RT
            # freshness fallback; it must not raise into supervisor DISARM.
            send_envelope = runtime.link.apply_reference_envelope

            def stale_envelope(command, **kw):
                kw["sample_received_at"] = time.monotonic() - 1.0
                return send_envelope(command, **kw)

            runtime.link.apply_reference_envelope = stale_envelope
            time.sleep(0.09)
            state, _ = runtime.link.latest_state()
            assert state.armed and not state.faulted and state.flags & p.FLAG_HOLDING
            assert not runtime.controller.stopped
            runtime.link.apply_reference_envelope = send_envelope
            until(lambda: not runtime.link.latest_state()[0].flags & p.FLAG_HOLDING)

            # Cancelling an unselected upload ABORTs only that candidate. The
            # previous terminal endpoint and its compensated hold stay selected.
            entered = threading.Event()
            exchange = runtime.link._reference_exchange

            def slow_chunk(operation, **kw):
                if operation == p.CTL_REFERENCE_CHUNK:
                    entered.set()
                    time.sleep(0.04)
                return exchange(operation, **kw)

            runtime.link._reference_exchange = slow_chunk
            q0 = runtime.state().q
            tr = JointTrajectory(
                t, np.repeat(q0[None, :], len(t), axis=0), np.zeros((len(t), 3))
            )
            ex = runtime.submit(
                "test",
                replace(req, request_id="abort-upload", trajectory=tr, admit_by=None),
            )
            until(entered.is_set)
            runtime.cancel("test", ex.execution_id)
            until(lambda: not runtime._native.busy)
            runtime.link._reference_exchange = exchange
            time.sleep(0.1)
            state, _ = runtime.link.latest_state()
            assert runtime.status(ex.execution_id).phase is Phase.CANCELLED
            assert (
                state.reference["reference_id"] == before_id
                and state.reference["status"] == p.REFERENCE_END
            )
            assert state.armed and not state.faulted

            # Missing continuation data is an execution failure. It stays an
            # armed, zero-PC-feedforward RT hold; only a NEW admission recovers.
            horizon = runtime._native
            stage_window = horizon._stage_window
            horizon._stage_window = lambda first: (
                stage_window(first) if first == 0 else None
            )
            q0 = runtime.state().q
            tr = JointTrajectory(
                t,
                q0 + np.repeat((0.004 * t)[:, None], 3, axis=1),
                np.full((len(t), 3), 0.004),
            )
            ex = runtime.submit(
                "test",
                replace(req, request_id="underflow", trajectory=tr, admit_by=None),
            )
            until(lambda: runtime.status(ex.execution_id).terminal)
            status = runtime.status(ex.execution_id)
            assert (
                status.phase is Phase.FAULTED and status.reason == "REFERENCE_UNDERFLOW"
            ), status
            old_id = horizon.id
            time.sleep(0.05)
            state, _ = runtime.link.latest_state()
            assert (
                state.armed
                and not state.faulted
                and state.reference["status"] == p.REFERENCE_UNDERFLOW
            ), state
            assert (
                horizon.id == old_id
                and runtime.authority().lifecycle is Lifecycle.ARMED
            )
            horizon._stage_window = stage_window
            q0 = runtime.state().q
            tr = JointTrajectory(
                [0.0, 0.3, 0.6],
                np.stack([q0, q0 + 0.002, q0 + 0.004]),
                np.zeros((3, 3)),
            )
            ex = runtime.submit(
                "test",
                replace(req, request_id="recovery", trajectory=tr, admit_by=None),
            )
            until(lambda: runtime.status(ex.execution_id).terminal)
            assert runtime.status(ex.execution_id).phase is Phase.COMPLETED, (
                runtime.status(ex.execution_id)
            )
            assert horizon.id > old_id
            assert len(parity_samples) > 400
            print(
                f"PASS {len(parity_samples)} PC dynamics commands byte-identical for RT/Python samples"
            )

            assert (
                runtime.link.apply_reference_envelope(
                    {},
                    reference_id=horizon.id,
                    revision=horizon.revision,
                    phase=horizon.point.time,
                    sample_received_at=time.monotonic() - 1.0,
                    valid_for_s=0.05,
                    phase_window_s=0.05,
                    session_id=horizon.session_id,
                    arm_epoch=horizon.arm_epoch,
                )
                is None
            )
            # Task-specific settling retains its existing measured compensated
            # hold, through a new explicitly admitted constant reference.
            q0 = runtime.state().q
            tr = JointTrajectory(
                [0.0, 0.3, 0.6], np.stack([q0, q0 + 0.01, q0 + 0.02]), np.zeros((3, 3))
            )
            ex = runtime.submit(
                "test",
                replace(
                    req,
                    request_id="custom-settle",
                    trajectory=tr,
                    admit_by=None,
                    completion=Completion(vel_tol=0.05),
                ),
            )
            until(lambda: runtime.status(ex.execution_id).terminal)
            assert runtime.status(ex.execution_id).phase is Phase.COMPLETED
            selected = next(
                r
                for r in runtime.receipts(ex.execution_id)
                if r.phase is Phase.SELECTED
            )
            until(
                lambda: (
                    horizon.key is None
                    and horizon.selected
                    and horizon.feedback["status"] == p.REFERENCE_END
                )
            )
            assert horizon.id > selected.evidence["reference_id"]
            np.testing.assert_array_equal(
                horizon.point.position, runtime.controller.last_command.q_des
            )

            # If PC release checks fail AFTER RT selection, receipts must
            # retain that selection and fault it, never claim unselected refusal.
            release_error = runtime.controller.release_error

            def changed_observation(eid):
                if horizon.key == eid and horizon.selected:
                    return "test observation changed after RT selection"
                return release_error(eid)

            runtime.controller.release_error = changed_observation
            q0 = runtime.state().q
            tr = JointTrajectory(
                [0.0, 0.3, 0.6],
                np.stack([q0, q0 + 0.001, q0 + 0.002]),
                np.zeros((3, 3)),
            )
            ex = runtime.submit(
                "test",
                replace(
                    req, request_id="selected-refusal", trajectory=tr, admit_by=None
                ),
            )
            until(lambda: runtime.status(ex.execution_id).terminal)
            assert [r.phase for r in runtime.receipts(ex.execution_id)] == [
                Phase.REQUESTED,
                Phase.ADMITTED,
                Phase.SELECTED,
                Phase.FAULTED,
            ]
            runtime.controller.release_error = release_error

            # Upload the controller's admitted snapshot, never a later-mutated
            # caller request that bypasses its plan validation.
            q0 = runtime.state().q
            tr = JointTrajectory(
                [0.0, 0.3, 0.6],
                np.stack([q0, q0 + 0.001, q0 + 0.002]),
                np.zeros((3, 3)),
            )
            endpoint = tr.positions[-1].copy()
            ex = runtime.submit(
                "test",
                replace(
                    req,
                    request_id="immutable-admission",
                    trajectory=tr,
                    admit_by=None,
                    start=Start.ON_RELEASE,
                ),
            )
            tr.positions[:] = 20.0  # invalid, but NOT the already admitted plan
            runtime.release("test", ex.execution_id)
            until(lambda: runtime.status(ex.execution_id).terminal)
            assert runtime.status(ex.execution_id).phase is Phase.COMPLETED
            np.testing.assert_array_equal(
                runtime.link.latest_state()[0].reference["q_ref"], endpoint
            )

            # Lifecycle invalidates both old uploads and dynamic envelopes.
            old_session, old_epoch = horizon.session_id, horizon.arm_epoch
            gen = runtime.disarm("test")
            runtime.await_authority(gen, armed=False, wait_s=3)
            gen = runtime.arm("test")
            runtime.await_authority(gen, armed=True, wait_s=3)
            for stale in (
                lambda: runtime.link.upload_reference(
                    tr, reference_id=999, session_id=old_session, arm_epoch=old_epoch
                ),
                lambda: runtime.link.commit_reference(
                    999, reference_id=999, session_id=old_session, arm_epoch=old_epoch
                ),
                lambda: runtime.link.apply_reference_envelope(
                    {},
                    reference_id=999,
                    revision=1,
                    phase=0.0,
                    sample_received_at=time.monotonic(),
                    valid_for_s=0.05,
                    phase_window_s=0.05,
                    session_id=old_session,
                    arm_epoch=old_epoch,
                ),
            ):
                try:
                    stale()
                except RtLinkError:
                    pass
                else:
                    raise AssertionError("old ARM epoch survived")
            q0 = runtime.state().q
            tr = JointTrajectory(
                [0.0, 1.0, 2.0], np.stack([q0, q0 + 0.02, q0 + 0.04]), np.zeros((3, 3))
            )
            ex = runtime.submit(
                "test",
                replace(req, request_id="owner-loss", trajectory=tr, admit_by=None),
            )
            until(lambda: Phase.APPLIED in runtime.status(ex.execution_id).reached)
            runtime.close_session("test")
            until(lambda: runtime.status(ex.execution_id).terminal)
            time.sleep(0.2)
            assert runtime.status(ex.execution_id).phase is Phase.CANCELLED
            assert runtime.authority().lifecycle is Lifecycle.ARMED, runtime.authority()
            print(
                "PASS deadline in-flight refusal, supersession race, underflow/new-admission recovery, old-epoch rejection, owner loss"
            )
            print(
                "PASS supervisor: 321 knots, RT source, SELECTED/SENT/APPLIED/COMPLETED, endpoint retention, cancel/compensated hold"
            )
        except BaseException:
            log.seek(0)
            print(log.read())
            raise
        finally:
            runtime.close()
            proc.terminate()
            proc.wait(3)


if __name__ == "__main__":
    main(str(Path(sys.argv[1]).resolve()))
