"""PC window transport + RT execution: long immutable curve, no hardware."""

from pathlib import Path
import sys
import time

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt.reference import ReferenceHorizon
from arm_control.plants.remote_rt import protocol as p
from _envelope_age import EnvelopeAgeMonitor
from native_reference import fake


def main(binary, speed=0.05):
    with fake(binary) as rt:
        ages = EnvelopeAgeMonitor(rt)
        with ages:
            _run(rt, speed)
        ages.report(f'reference_horizon {speed:g} rad/s')
    stalled_running(binary)
    stalled_end(binary)


def _command():
    return dict(position=np.zeros(3), velocity=np.zeros(3), torque=np.zeros(3),
                kp=np.ones(3) * 2, kd=np.ones(3) * 0.1)


def _drive_to_selection(rt, horizon, publish, *, stop_at_end=False):
    """Poll/approve until selected (and, optionally, until END)."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state, received = rt.latest_state()
        horizon.poll(state, received)
        assert not horizon.error, horizon.error
        if horizon.ready and not horizon.selected:
            horizon.approve()
        ref = None if state is None else state.reference
        if horizon.selected and ref is not None and ref['reference_id'] == horizon.id:
            if publish:
                horizon.publish(_command(), reference_id=horizon.id,
                                revision=ref['revision'], phase=horizon.point.time,
                                sample_received_at=received)
            if not stop_at_end:
                return
            if ref['status'] == p.REFERENCE_END:
                return
        time.sleep(0.005)
    raise AssertionError('reference never reached the requested phase')


def stalled_running(binary):
    """CTRL-034 B1: ONE decision, then no further control ticks.

    The fixed-rate sender may repeat a decision but never re-stamps it, so a
    RUNNING reference must be retired at the ceiling — and then, because no new
    decision ever arrives, `CMD_LOST` must follow at `fault_ms`.
    """
    with fake(binary) as rt:
        horizon = ReferenceHorizon(rt)
        try:
            horizon.stage('stalled-running', JointTrajectory(
                [0., 2.], np.array([[0.] * 3, [0.8] * 3]), np.full((2, 3), 0.4)),
                terminal=False)
            _drive_to_selection(rt, horizon, publish=False)
            state, received = rt.latest_state()
            horizon.publish(_command(), reference_id=horizon.id,
                            revision=state.reference['revision'],
                            phase=horizon.point.time, sample_received_at=received)
            t0 = time.monotonic()
            while time.monotonic() < t0 + 2.0:
                state = rt.latest_state()[0]
                ref = None if state is None else state.reference
                if (ref is not None and ref['reference_id'] == horizon.id
                        and ref['status'] == p.REFERENCE_CANCELLED):
                    break
                time.sleep(0.005)
            else:
                raise AssertionError('RUNNING reference was never retired')
            retire_ms = (time.monotonic() - t0) * 1000.0
            assert retire_ms <= 100 + 40, f'retired {retire_ms:.0f} ms after one decision'
            while time.monotonic() < t0 + 2.0:
                state = rt.latest_state()[0]
                if state.faulted:
                    break
                time.sleep(0.005)
            else:
                raise AssertionError('no CMD_LOST after the decision stream stopped')
            assert state.fault_code == p.FAULT_CMD_LOST, state.fault_code
            print(f'  stalled-running: one decision -> retired at {retire_ms:.0f} ms, '
                  f'CMD_LOST at fault_ms')
        finally:
            horizon.close()


def stalled_end(binary):
    """At END the stream must stop and `CMD_LOST` must follow with no new decision.

    The previous (rejected) sender re-stamped its sample and held indefinitely.
    """
    with fake(binary) as rt:
        horizon = ReferenceHorizon(rt)
        try:
            horizon.stage('stalled-end', JointTrajectory(
                [0., 0.3], np.array([[0.] * 3, [0.05] * 3]), np.zeros((2, 3))),
                terminal=True)
            _drive_to_selection(rt, horizon, publish=True, stop_at_end=True)
            t0 = time.monotonic()
            last_status = None
            while time.monotonic() < t0 + 3.0:
                state = rt.latest_state()[0]
                if state.faulted:
                    break
                last_status = state.reference['status']
                time.sleep(0.005)
            else:
                raise AssertionError('no CMD_LOST after the frozen END decision')
            assert state.fault_code == p.FAULT_CMD_LOST, state.fault_code
            # The stream stopped at the ceiling while the reference was already
            # stationary at END: the RT does not retire END, but the deadman
            # still lapses and CMD_LOST follows. (The latch itself then retires.)
            assert last_status == p.REFERENCE_END, last_status
            print(f'  stalled-end: stream stopped at the ceiling, CMD_LOST '
                  f'{(time.monotonic() - t0) * 1000:.0f} ms later')
        finally:
            horizon.close()


def _run(rt, speed):
    horizon = ReferenceHorizon(rt)
    t = np.arange(321) * 0.01
    q = np.repeat((speed * t)[:, None], 3, axis=1)
    trajectory = JointTrajectory(t, q, np.full_like(q, speed))
    horizon.stage("long-plan", trajectory)
    revisions = set()
    previous = None
    previous_revision = None
    revision_switches = 0
    running_hold_frames = 0
    deadline = time.monotonic() + 6
    samples = 0
    try:
        while time.monotonic() < deadline:
            state, received = rt.latest_state()
            horizon.poll(state, received)
            assert not horizon.error, horizon.error
            if horizon.ready and not horizon.selected:
                horizon.approve()
            command = dict(
                position=np.zeros(3),
                velocity=np.zeros(3),
                torque=np.zeros(3),
                kp=np.ones(3) * 2,
                kd=np.ones(3) * 0.1,
            )
            if horizon.selected:
                ref, point = horizon.feedback, horizon.point
                revisions.add(ref["revision"])
                if (previous_revision is not None
                        and ref["revision"] != previous_revision):
                    # CTRL-034 F3/B2: a revision switch is not a
                    # first-envelope gap. The prefix is byte-identical, so the
                    # previous revision's envelopes stay admissible until the
                    # new one is in flight: no hold frame.
                    revision_switches += 1
                    assert ref["status"] == p.REFERENCE_RUNNING, ref
                    assert not horizon.error, horizon.error
                    assert not state.holding, "revision switch held"
                previous_revision = ref["revision"]
                if (ref["status"] == p.REFERENCE_RUNNING and state.holding
                        and ref["envelope_age_us"] != 0xFFFFFFFF):
                    # Startup holds before the first envelope has authority,
                    # where RT reports no age at all, are not stream lapses.
                    running_hold_frames += 1
                oracle = trajectory.sample_at(point.time)
                np.testing.assert_allclose(
                    point.position, oracle.position, rtol=0, atol=1e-15
                )
                if ref["status"] == p.REFERENCE_RUNNING:
                    np.testing.assert_allclose(
                        point.velocity, oracle.velocity, rtol=0, atol=1e-15
                    )
                    np.testing.assert_allclose(
                        point.acceleration, oracle.acceleration, rtol=0, atol=1e-13
                    )
                else:
                    assert np.array_equal(point.velocity, np.zeros(3))
                if previous is not None:
                    assert point.time >= previous
                previous = point.time
                # CTRL-034 B1: the horizon's fixed-rate sender repeats this
                # decision but never re-stamps it; the loop publishes the
                # decision's own phase and sample time.
                horizon.publish(command, reference_id=horizon.id,
                                revision=ref["revision"], phase=point.time,
                                sample_received_at=received)
                samples += 1
                if ref["status"] == p.REFERENCE_END:
                    assert 4 <= len(revisions) <= 5 and revisions == set(
                        range(1, len(revisions) + 1)
                    ), revisions
                    assert not state.faulted and state.armed
                    assert running_hold_frames == 0, running_hold_frames
                    break
            else:
                rt.apply_command(command)
            time.sleep(0.005)
        else:
            raise AssertionError("long reference did not reach terminal end")
        print(
            f"PASS 321 knots through {len(revisions)} bounded revisions "
            f"({revision_switches} switches, {running_hold_frames} RUNNING hold frames); "
            f"{samples} feedback samples match original curve, monotonic RT phase"
        )
    finally:
        horizon.close()


if __name__ == "__main__":
    main(str(Path(sys.argv[1]).resolve()),
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.05)
