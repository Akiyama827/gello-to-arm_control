"""PC window transport + RT execution: long immutable curve, no hardware."""

from pathlib import Path
import sys
import time

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt.reference import ReferenceHorizon
from arm_control.plants.remote_rt import protocol as p
from envelope_age import EnvelopeAgeMonitor
from native_reference import fake


def main(binary):
    with fake(binary) as rt:
        ages = EnvelopeAgeMonitor(rt)
        with ages:
            _run(rt)
        ages.report('reference_horizon')


def _run(rt):
    horizon = ReferenceHorizon(rt)
    t = np.arange(321) * 0.01
    q = np.repeat((0.05 * t)[:, None], 3, axis=1)
    trajectory = JointTrajectory(t, q, np.full_like(q, 0.05))
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
                # CTRL-034 B1: the horizon's fixed-rate sender owns the v7
                # stream; this loop only publishes the decision.
                horizon.publish(command, reference_id=horizon.id,
                                revision=ref["revision"])
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
    main(str(Path(sys.argv[1]).resolve()))
