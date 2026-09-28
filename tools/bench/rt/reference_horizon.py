"""PC window transport + RT execution: long immutable curve, no hardware."""

from pathlib import Path
import sys
import time

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt.reference import ReferenceHorizon
from arm_control.plants.remote_rt import protocol as p
from native_reference import fake


def main(binary):
    with fake(binary) as rt:
        horizon = ReferenceHorizon(rt)
        t = np.arange(321) * 0.01
        q = np.repeat((0.05 * t)[:, None], 3, axis=1)
        trajectory = JointTrajectory(t, q, np.full_like(q, 0.05))
        horizon.stage("long-plan", trajectory)
        revisions = set()
        previous = None
        previous_revision = None
        revision_switches = 0
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
                        # CTRL-034 F3: a normal revision switch is not a
                        # first-envelope gap. The new revision's first envelope
                        # is admitted well inside the 50 ms ceiling, so the RT
                        # never retires it and the run continues RUNNING.
                        revision_switches += 1
                        assert ref["status"] == p.REFERENCE_RUNNING, ref
                        assert not horizon.error, horizon.error
                    previous_revision = ref["revision"]
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
                    rt.apply_reference_envelope(
                        command,
                        reference_id=horizon.id,
                        revision=ref["revision"],
                        phase=point.time,
                        sample_received_at=received,
                        valid_for_s=0.05,
                        phase_window_s=0.05,
                    )
                    samples += 1
                    if ref["status"] == p.REFERENCE_END:
                        assert 4 <= len(revisions) <= 5 and revisions == set(
                            range(1, len(revisions) + 1)
                        ), revisions
                        assert not state.faulted and state.armed
                        break
                else:
                    rt.apply_command(command)
                time.sleep(0.005)
            else:
                raise AssertionError("long reference did not reach terminal end")
            print(
                f"PASS 321 knots through {len(revisions)} bounded revisions "
                f"({revision_switches} switches stayed RUNNING); "
                f"{samples} feedback samples match original curve, monotonic RT phase"
            )
        finally:
            horizon.close()


if __name__ == "__main__":
    main(str(Path(sys.argv[1]).resolve()))
