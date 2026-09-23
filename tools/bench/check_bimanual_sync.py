"""Send skew and timeout checks. These do not measure physical motion timing."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from check_bimanual_contract import Harness, measured
from check_execution_policy import setup, commands
from arm_control.multi_arm.synchronization import SharedReferenceClock, StartRecord
from arm_control.multi_arm.types import PairLifecycle
from arm_control.multi_arm.feedback import align_feedback
from arm_control.plants.remote_rt.protocol import State, FLAG_ARMED, FLAG_FAULTED


def main():
    check_reference_phase()
    check_reference_safety()
    check_feedback_alignment()
    h = Harness()
    h.coord.stage_pair(h.pair())
    for event in h.handles["fr3_1"].poll_events():
        h.coord.on_member_event("fr3_1", event)
    h.t += 0.03
    h.coord.pump()
    assert abs(h.coord.pair_state().stage_skew_sec - 0.03) < 1e-10
    original = h.handles["other_arm"].execute

    def delayed(plan_id):
        h.t += 0.02
        original(plan_id)

    h.handles["other_arm"].execute = delayed
    h.coord.execute_pair("pair")
    assert h.coord.lifecycle is PairLifecycle.EXECUTING
    assert abs(h.coord.pair_state().release_skew_sec - 0.02) < 1e-10
    record = StartRecord("pair")
    assert record.requested_start_ns is None and record.release_skew_sec is None
    h.coord.cancel_pair()
    h.coord.pump()
    print("bimanual timing: PASS (distinct send stamps; no physical-start claim)")


def check_reference_phase():
    """Reproduce sequential release/tick skew, then remove it on a moving path."""
    errors = []
    for shared in (False, True):
        reference = SharedReferenceClock(lambda: h.t)
        h = Harness(controller_kwargs=dict(
            state_period_s=None, reference_clock=reference if shared else None,
        ))
        if shared:
            h.coord._release_context = reference.batch
        h.coord.stage_pair(h.pair(distance=.1))
        h.coord.pump()
        right = h.handles["other_arm"]
        execute = right.execute

        def delayed(plan_id):
            h.t += .02
            execute(plan_id)

        right.execute = delayed
        h.coord.execute_pair("pair")
        assert h.coord.lifecycle is PairLifecycle.EXECUTING
        assert abs(h.coord.pair_state().release_skew_sec - .02) < 1e-10
        starts = [c.executor._t_start for c in h.controllers.values()]
        assert (abs(starts[0] - starts[1]) < 1e-12) == shared
        # Make the fake executor evaluate the actual cubic so assertions cover
        # commands produced by ArmController, not just clock accessor values.
        for c in h.controllers.values():
            executor = c.executor
            original = executor.step

            def sample(t, state, executor=executor, original=original):
                command = original(t, state)
                point = executor._traj.sample_at(executor.elapsed(t))
                return replace(command, q_des=point.position, qd_des=point.velocity)

            executor.step = sample
        h.t += .4
        with reference.batch() if shared else nullcontext():
            for c in h.controllers.values():
                c.on_motor_state(measured(c.n_arm))
                h.t += .01  # deliberate computation delay between arm ticks
        sent = [c.last_command.q_des[0] for c in h.controllers.values()]
        error = abs(sent[0] - sent[1])
        errors.append(error)
        if shared:
            np.testing.assert_allclose(sent[0], sent[1], atol=1e-12)
        else:
            assert error > .001
    print(f"Delayed release/tick reference mismatch: {errors[0]:.6f} -> {errors[1]:.6f} rad: PASS")


def check_reference_safety():
    reference = SharedReferenceClock(lambda: wall[0])
    c, wall = setup(reference_clock=reference, health=False)
    with reference.batch() as t:
        wall[0] += .3
        assert c.exec_time == t
        c.tick()
        assert c._observation_paused and not commands(c)
        with reference.batch():
            assert reference() == t
    assert reference() == wall[0]
    try:
        with reference.batch():
            wall[0] += .1
            raise RuntimeError("batch aborted")
    except RuntimeError:
        pass
    assert reference() == wall[0], "failed batch left reference time frozen"
    try:
        setup(reference_clock=reference, state_period_s=.01)
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("ambiguous plant/reference clock accepted")
    print("Shared reference clock retains wall-time freshness and resets on failure: PASS")


def check_feedback_alignment():
    def sample(t, offset=0., **kw):
        return State(n=1, state_seq=int(t * 1000), t_mono_ns=round(t * 1e9),
                     session_id=7, flags=FLAG_ARMED,
                     q=[.4 * (t - 10) + offset], dq=[.4], **kw), 20.

    def history(offset=0.):
        return {"a": [sample(10.010), sample(10.020)],
                "b": [sample(10.005, offset), sample(10.015, offset)]}

    def motor_state(*, state):
        return {"position": state.q, "velocity": state.dq}

    links = {a: SimpleNamespace(config=SimpleNamespace(host="same-host"),
                                motor_health=lambda: dict(session_id="7", arm_epoch=0),
                                motor_state=motor_state) for a in "ab"}
    raw = history()
    positions, report = align_feedback(links, raw, now=20.01)
    assert abs(raw["a"][-1][0].q[0] - raw["b"][-1][0].q[0]) > .0019
    np.testing.assert_allclose(positions["a"], positions["b"], atol=1e-12)
    assert report["t_mono_ns"] == 10_015_000_000
    positions, report = align_feedback(links, history(.01), now=20.01)
    np.testing.assert_allclose(positions["b"] - positions["a"], [.01], atol=1e-12)
    for a, bracket in report["brackets"].items():
        f = bracket["fraction"]
        expected = (1-f)*np.array(bracket["q_before"]) + f*np.array(bracket["q_after"])
        np.testing.assert_allclose(positions[a], expected)

    def refused(raw, text, now=20.01):
        try:
            align_feedback(links, raw, now=now)
        except ValueError as exc:
            assert text in str(exc), exc
        else:
            raise AssertionError(f"alignment accepted {text}")

    refused(history(), "stale", now=20.06)
    refused(history(), "stale", now=19.99)
    raw = history()
    raw["a"] = raw["a"][-1:]
    refused(raw, "cannot bracket")  # no extrapolation
    raw = history()
    raw["a"][0][0].arm_epoch += 1
    refused(raw, "cannot bracket")
    raw = history()
    raw["a"] = [sample(9.98), sample(10.02)]
    refused(raw, "interpolation gap")
    raw = history()
    raw["a"][-1][0].t_mono_ns += 30_000_000
    refused(raw, "publication skew")
    raw = history()
    raw["a"][-1][0].q[0] = np.nan
    refused(raw, "invalid joints")
    raw = history()
    raw["a"][-1][0].dq[0] = np.nan
    refused(raw, "invalid joints")
    for attr in ("session_id", "arm_epoch"):
        raw = history()
        for s, _ in raw["a"]:
            setattr(s, attr, 99)
        refused(raw, "authority changed")
    raw = history()
    for h in raw.values():
        for s, _ in h:
            s.flags = 0
            s.arm_epoch = 99  # RT retains its historical epoch after DISARM.
    align_feedback(links, raw, now=20.01)
    raw = history()
    raw["a"][-1][0].flags |= FLAG_FAULTED
    refused(raw, "plant faulted")
    raw = history()
    raw["a"].insert(1, sample(10.021))
    refused(raw, "not increasing")
    raw = history()
    raw["b"].insert(1, sample(10.016))
    refused(raw, "not increasing")  # also reject a reordered exact-time sample
    links["b"].config.host = "different-host"
    refused(history(), "one RT host clock")
    print("Common-time feedback removes timestamp skew, retains real offsets, refuses invalid/stale/unbracketed states: PASS")


if __name__ == "__main__":
    main()
