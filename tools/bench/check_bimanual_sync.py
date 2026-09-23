"""Send skew and timeout checks. These do not measure physical motion timing."""

from check_bimanual_contract import Harness
from arm_control.multi_arm.synchronization import StartRecord
from arm_control.multi_arm.types import PairLifecycle


def main():
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


if __name__ == "__main__":
    main()
