"""Assert console Hand gates without a robot; run with PYTHONPATH=."""
from unittest.mock import patch
from pathlib import Path

import yaml

from arm_control.ui.arm_console import ControlPanel


class EmptyFK:
    def poses(self, q, grip):
        return {"geoms": [], "ee": {"p": [0, 0, 0], "q": [0, 0, 0, 1]}}


def main():
    clock = [100.0]
    cfg = {"force_n": 40., "grasp_width_m": .045, "speed_mps": .05,
           "epsilon_inner_m": .02, "epsilon_outer_m": .02, "open_width_m": .075}
    assert "set_hand_state" in vars(ControlPanel), "console has no force-grasp gate yet"
    with patch("arm_control.ui.arm_console.time.monotonic", side_effect=lambda: clock[0]):
        panel = ControlPanel(["joint"], [-1], [1], 0, EmptyFK(),
                             grip_range=(0, .04), gripper_cfg=cfg)
        try:
            def sample(seq, **overrides):
                state = dict(width=.078, is_grasped=False, available=True,
                    busy=False, measured=True, force_grasp=True, sample_seq=seq)
                state.update(overrides)
                panel.set_hand_state(state)

            def refused(mode="close", **overrides):
                payload = dict(mode=mode, width_m=.04, force_n=40)
                payload.update(overrides)
                try:
                    panel._post("hand", payload)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"unsafe request accepted: {payload}")

            sample(1)
            assert panel._state()["hand"]["status"] == "Idle"
            refused()  # Unknown ARM is not confirmation.
            panel.set_armed(False)
            refused()
            panel.set_armed(True, "fault")
            refused()
            panel.set_armed(True)
            for fields in ({"force_n": float("nan")}, {"force_n": 0},
                           {"force_n": 1000}, {"width_m": float("inf")},
                           {"width_m": -.001}, {"width_m": .081}, {"mode": "home"}):
                refused(**fields)
            assert panel.pop_hand() is None, "parameter errors enqueued motion"
            panel._post("hand", {"mode": "close", "width_m": .042, "force_n": 40})
            refused()  # One click, no backlog.
            original_slider = panel.gripper_value()
            panel.set_gripper(.01, dirty=True)
            assert panel.pop_gripper() is None
            assert panel.gripper_value() == original_slider
            req = panel.pop_hand()
            assert req["mode"] == "close" and req["width_m"] == .042
            assert req["force_n"] == 40 and req["request_id"]
            assert panel.pop_hand() is None
            panel.set_hand_result(dict(request_id="old", ok=True, reason="grasped"))
            refused()  # A stale result cannot release the one-in-flight gate.
            panel.set_hand_result(dict(request_id=req["request_id"], ok=True, reason="grasped"))
            assert panel._state()["hand"]["status"] == "Held"
            sample(2)
            panel._post("hand", {"mode": "release"})
            release = panel.pop_hand()
            assert release["mode"] == "release"
            assert release["request_id"] != req["request_id"]
            panel.set_hand_result(dict(request_id=release["request_id"], ok=True, reason="released"))
            assert "accepted" in panel._state()["hand"]["status"].lower()
            refused()  # Open ack is not physical completion.
            panel.set_gripper(.001, dirty=True)
            assert panel.pop_gripper() is None, "slider accepted behind an Open acknowledgement"
            sample(3, width=.075)
            assert panel._state()["hand"]["status"] == "Open"
            clock[0] += 1
            sample(3)  # Republishing the SAME sample cannot keep the gate fresh.
            refused()
            assert panel._state()["hand"]["measured_width_mm"] is None
            sample(4)
            panel._post("hand", {"mode": "close", "width_m": .04, "force_n": 40})
            panel.set_hand_state(dict(available=False, measured=False, force_grasp=True, sample_seq=4))
            sample(5)
            assert panel.pop_hand() is None, "known disconnect replayed an unsent request after recovery"
            panel._post("hand", {"mode": "close", "width_m": .04, "force_n": 40})
            panel._click("DISARM")
            assert panel.pop_hand() is None, "DISARM did not drop unconsumed click"
            assert panel.pop_gripper() is None
            panel.set_armed(False)
            panel.set_armed(True)
            panel.set_hand_state({"force_grasp": False, "width": .078})
            refused()
            assert not panel._state()["hand"]["enabled"]
        finally:
            panel.close()
    print("console grasp: authority, validation, single flight, freshness, correlation, disarm OK")
    root = Path(__file__).resolve().parents[2]
    html = (root / "arm_control/ui/static/console.html").read_text()
    assert html.index('id="grip"') < html.index('id="hand-grasp"') < html.index('id="hand-open"')
    nodes = {n["id"]: n for n in yaml.safe_load((root / "dataflows/real_franka_motion.yml").read_text())["nodes"]}
    assert nodes["franka_gripper"]["inputs"]["grasp_request"] == "arm_console/grasp_request"
    assert nodes["arm_console"]["inputs"]["grasp_result"] == "franka_gripper/grasp_result"
    for node in nodes.values():
        for source in node.get("inputs", {}).values():
            producer, topic = source.split("/")
            assert topic in nodes[producer]["outputs"], source
    print("console grasp: below-slider markup and real Hand graph contracts OK")


if __name__ == "__main__":
    main()
