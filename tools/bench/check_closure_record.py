"""CTRL-041 closure record: five quantities kept apart, absent ones say why, JSON-safe, no gating."""
import json
import math

from arm_control.multi_arm.closure import CLOSURE_SCHEMA, QUANTITIES, closure_record


def alignment():
    return dict(t_mono_ns=10_015_000_000, latest_skew_s=.0005, clock_domain="mono:aaaaaaaaaaaa:bbbbbbbbbbbb",
                brackets={a: dict(before_seq=1, after_seq=2, before_t_mono_ns=1, after_t_mono_ns=2,
                                  fraction=.5, q_before=[0.], q_after=[0.], q=[0.]) for a in "ab"})


def main():
    arms = {a: dict(q_rt=[.1, .2], q_cmd_rt=[.1, .25], q_cmd_pc=[.1, .3], state_seq=7, state_t_mono_ns=99)
            for a in "ab"}
    full = closure_record(moving=True, planned=dict(closure_m=.0004, closure_rad=.001, path_m=.0002, path_rad=.0005),
                          arms=arms, common_time=(.002, .01), alignment=alignment(),
                          independent=dict(value=dict(mm=.4), source="mocap fixture", frame="world", time="t0"))
    assert full["schema"] == CLOSURE_SCHEMA and full["window"] == "motion" and full["stationary_for_s"] is None
    assert set(QUANTITIES) <= set(full)
    assert full["planned_reference"]["value"]["closure_m"] == .0004
    assert abs(full["tracking"]["value"]["a"]["vs_rt_command_rad"] - .05) < 1e-12
    assert abs(full["tracking"]["value"]["a"]["vs_pc_command_rad"] - .1) < 1e-12
    ct = full["common_time_inferred"]
    assert ct["value"] == dict(distance_m=.002, angle_rad=.01)
    assert "mono:aaaaaaaaaaaa:bbbbbbbbbbbb" in ct["basis"]["time"] and ct["basis"]["latest_skew_s"] == .0005
    assert full["physical"]["value"] == dict(mm=.4) and full["physical"]["basis"]["source"] == "mocap fixture"
    # RT-selected reference is absent in streamed mode and says so; nothing is zero-filled.
    assert full["rt_selected_reference"]["value"] is None and "streamed" in full["rt_selected_reference"]["absent"]
    # Everything unavailable: each slot carries its own reason, none a value.
    empty = closure_record(moving=False, stationary_for_s=3.)
    assert empty["window"] == "stationary" and empty["stationary_for_s"] == 3.
    for name in QUANTITIES:
        assert empty[name]["value"] is None and empty[name]["absent"], name
    # An estimate without its alignment report, or a non-finite value, is absent, never a number.
    assert closure_record(moving=True, common_time=(.1, .1))["common_time_inferred"]["value"] is None
    bad = closure_record(moving=True, common_time=(math.nan, .1), alignment=alignment(),
                         planned=dict(closure_m=math.inf, closure_rad=0, path_m=0, path_rad=0))
    assert bad["common_time_inferred"]["value"] is None and bad["planned_reference"]["value"] is None
    # Every record is plain JSON with no NaN.
    for record in (full, empty, bad):
        json.dumps(record, allow_nan=False)
    assert len(full["assumptions"]) == 4 and any("gates nothing" in x for x in full["assumptions"])
    print("closure record: five quantities separate, absences explained, JSON-safe: PASS")


if __name__ == "__main__":
    main()
