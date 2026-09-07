"""Assert the arm console interaction policy; no hardware and no motion.

Run from the repo: PYTHONPATH=. python tools/bench/check_console_authority.py
"""
import time

from arm_control.ui.arm_console import ControlPanel, JOG_STALE_S


class _Visual:
    def poses(self, values, grip):
        return {"geoms": [], "ee": None}


def main():
    for initially_armed in (False, True):
        panel = ControlPanel(
            ["joint"], [-1.0], [1.0], 0, _Visual(), grip_range=(0, 0.04),
            jog_speed_m_s=0.012, jog_joint_speed_rad_s=0.2,
        )
        try:
            state = panel._state()
            assert "deadman" not in state
            assert state["jog"]["speed_m_s"] == 0.012
            assert state["jog"]["joint_speed_rad_s"] == 0.2
            panel._click("Execute")
            assert not panel.clicked("Execute"), "unknown authority must refuse"
            panel.set_armed(initially_armed)
            panel._click("Execute")
            assert panel.clicked("Execute") is initially_armed
            panel.set_armed(True)
            panel._click("Execute")
            assert panel.clicked("Execute")
            panel.set_gripper(0.02, dirty=True)
            assert panel.pop_gripper() == 0.02
            assert not panel.clicked("Stop (hold)")
            assert not panel.clicked("DISARM")
            assert panel._post("deadman", {"held": True}) is None
            panel.set_jog("x", 1, True)
            assert panel.jog_held() == ("x", 1)
            panel.set_jog("x", 1, False)
            assert panel.jog_held() is None
            panel.set_jog("x", -1, True)
            panel._jog_at = time.monotonic() - JOG_STALE_S - 0.01
            assert panel.jog_held() is None
            assert not panel.clicked("DISARM"), "jog release must not disarm"
            panel.set_control_mode("soft")
            panel._click("Plan + preview")
            panel._click("Execute")
            assert not panel.clicked("Plan + preview") and not panel.clicked("Execute")
            panel.set_jog("x", 1, True)
            assert panel.jog_held() is None
            panel.set_control_mode("joint")
            for stop in ("Stop (hold)", "DISARM"):
                panel.set_jog("x", 1, True)
                panel._click(stop)
                assert panel.jog_held() is None, "stop must clear the held jog"
                assert panel.clicked(stop)
            panel.set_armed(False)
            panel.set_gripper(0.01, dirty=True)
            assert panel.pop_gripper() is None
            panel.set_armed(True, "fault")
            panel._click("Execute")
            assert not panel.clicked("Execute")
            panel.set_gripper(0.03, dirty=True)
            assert panel.pop_gripper() is None
        finally:
            panel._server.close()
    print("console authority: armed execute, gripper gate, jog expiry OK")


if __name__ == "__main__":
    main()
