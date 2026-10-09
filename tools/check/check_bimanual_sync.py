"""Send skew and timeout checks. These do not measure physical motion timing."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from check_bimanual_contract import Harness
from arm_control.multi_arm.synchronization import StartRecord
from arm_control.multi_arm.types import PairLifecycle
from arm_control.multi_arm.feedback import FeedbackGap, align_feedback
from arm_control.multi_arm import supervised_hardware
from arm_control.multi_arm.supervised_hardware import SupervisedHardware
from arm_control.plants.remote_rt.protocol import State, FLAG_ARMED, FLAG_FAULTED


def main():
    check_feedback_alignment()
    check_idle_feedback_wait()
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
    # Replay the exact timestamp/sequence spacing from the operator's fault.
    raw = history()
    stamps = {"a": [(2537345944240057, 17613370), (2537345968247231, 17613394)],
              "b": [(2537345948579137, 14949686), (2537345969104630, 14949689)]}
    for a, h in raw.items():
        for (s, _), (stamp, seq) in zip(h, stamps[a]):
            s.t_mono_ns, s.state_seq = stamp, seq
    try:
        align_feedback(links, raw, now=20.01)
    except FeedbackGap as exc:
        assert exc.details["arm_id"] == "b"
        assert abs(exc.details["interval_s"] - .020525493) < 1e-12
        assert exc.details["after_seq"] - exc.details["before_seq"] == 3
    else:
        raise AssertionError("operator's 20.525493 ms bracket was accepted")
    # CTRL-041 clock domain: attested domains decide; the host string alone does not.
    domain = ("mono", "a" * 12, "b" * 12)
    for a in "ab":
        links[a].clock_domain = domain
    _, report = align_feedback(links, history(), now=20.01)
    assert report["clock_domain"] == ":".join(domain)
    links["b"].config.host = "another-name"  # same attested clock under another name: accepted
    align_feedback(links, history(), now=20.01)
    links["b"].clock_domain = ("mono", "a" * 12, "c" * 12)  # same host, rebooted: refused
    refused(history(), "one RT clock domain")
    links["b"].clock_domain = None  # attested on one arm only: refused
    refused(history(), "one arm only")
    links["b"].config.host = "same-host"
    for a in "ab":
        links[a].clock_domain = None  # both unattested: the host-string rule, labeled
    _, report = align_feedback(links, history(), now=20.01)
    assert report["clock_domain"] == "unattested"
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


def check_idle_feedback_wait():
    """An idle hold may wait; invalid data, moving and prolonged gaps must fail (SupervisedHardware, the pair's runtime)."""
    runtime = SupervisedHardware.__new__(SupervisedHardware)
    runtime.mode = "shared"
    runtime._holding_since = runtime._aligned_at = None
    runtime._closure_positions = runtime._feedback_alignment = None
    runtime.clients = {a: None for a in "ab"}
    held = dict(armed=True, ready=True, executing=False, stopped=False, frozen=False, mode_law="joint",
                velocity=np.zeros(7))
    states = {a: dict(held) for a in "ab"}
    runtime.handles = {a: SimpleNamespace(state=lambda a=a: SimpleNamespace(**states[a])) for a in "ab"}
    links = {a: SimpleNamespace(
        config=SimpleNamespace(host="same-host"),
        motor_health=lambda: dict(session_id="7", arm_epoch=0),
        motor_state=lambda *, state: dict(position=state.q, velocity=state.dq),
    ) for a in "ab"}

    def sample(stamp):
        return State(n=1, state_seq=stamp, t_mono_ns=stamp * 1_000_000,
                     session_id=7, flags=FLAG_ARMED, q=[0.], dq=[0.]), 20.

    good = {"a": [sample(10), sample(20)], "b": [sample(5), sample(15)]}
    gap = {"a": [sample(1), sample(25)], "b": [sample(5), sample(15)]}
    assert not runtime._update_holding(19.74)
    assert runtime._update_holding(20.)

    def apply(raw, now, allowed, expected_error=None):
        with patch.object(supervised_hardware, "time", SimpleNamespace(monotonic=lambda: now)), patch.object(
                supervised_hardware, "align_arms", lambda clients: align_feedback(links, raw, now=now)):
            try:
                runtime._align_feedback(allow_wait=allowed)
            except ValueError as exc:
                assert expected_error and expected_error in str(exc), exc
            else:
                assert expected_error is None, "unsafe feedback wait was accepted"

    apply(good, 20., False)
    held_positions = runtime._closure_positions
    apply(gap, 20.01, True)
    assert not runtime.feedback_ready and runtime._closure_positions is held_positions
    assert runtime._aligned_at == 20., "a gap must not reset its own deadline"
    apply(gap, 20.04, True)
    apply(gap, 20.051, True, "stale")
    # Fresh packet receipt still cannot extend the 50 ms alignment deadline.
    fresh_gap = {a: [(s, 20.04) for s, _ in h] for a, h in gap.items()}
    apply(fresh_gap, 20.051, True, "interpolation gap")
    apply(good, 20.02, True)
    assert runtime.feedback_ready
    apply(good, 20.04, True)
    assert runtime._aligned_at == 20., "re-aligning unchanged packets renewed the recovery budget"
    apply(fresh_gap, 20.051, True, "interpolation gap")
    for flag, bad in (("executing", True), ("frozen", True), ("stopped", True), ("armed", False), ("ready", False),
                      ("mode_law", "cartesian"), ("velocity", np.full(7, .011))):
        saved = states["a"][flag]
        states["a"][flag] = bad
        allowed = runtime._update_holding(20.03)
        assert not allowed, flag
        apply(gap, 20.03, allowed, "interpolation gap")
        states["a"][flag] = saved
        assert not runtime._update_holding(20.03)
        assert runtime._update_holding(20.30)
    good["a"][-1][0].dq[0] = np.nan
    apply(good, 20.03, True, "invalid joints")
    good["a"][-1][0].dq[0] = 0.
    good["a"][-1][0].session_id = 99
    apply(good, 20.03, True, "authority changed")
    print("Idle feedback wait: bounded 50 ms, no extrapolation; motion/authority/invalid/stale guards retained: PASS")


if __name__ == "__main__":
    main()
