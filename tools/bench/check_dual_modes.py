"""Exercise the runtime mode transition through the console action interface."""

import argparse
import time
from pathlib import Path

import numpy as np

from arm_control.multi_arm.cooperative import pose_error
from arm_control.ui.dual_arm_console import DualArmConsole, _pose_matrix


def check(path, refinement="none"):
    app = DualArmConsole(path, start_server=False)

    def action(name, *, refused=False, **extra):
        state = app.snapshot()
        try:
            result = app.post(
                "action",
                dict(
                    session=state["session"],
                    revision=state["revision"],
                    action=name,
                    **extra,
                ),
            )
        except ValueError as exc:
            if not refused:
                raise
            return str(exc)
        assert not refused, (name, result)
        return result

    def wait(predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = app.snapshot()
            assert state["connected"] and state["status"] != "faulted", state["error"]
            if predicate(state):
                return state
            time.sleep(0.04)
        raise AssertionError(state)

    def execute():
        state = wait(lambda s: s["status"] in {"preview", "failed"})
        assert state["status"] == "preview", state["error"]
        pair_id = state["preview_id"]
        action("stage", pair_id=pair_id, reviewed=True)
        wait(lambda s: s["pair"]["lifecycle"] == "ready")
        action("execute", pair_id=pair_id)
        state = wait(
            lambda s: s["pair"]["lifecycle"] in {"complete", "faulted", "cancelled"}
        )
        assert state["pair"]["lifecycle"] == "complete", state["error"]
        return pair_id

    def switch(target, **confirmation):
        with app._lock:
            qpos = app.runtime.backend.data.qpos.copy()
            plant, controllers = app.runtime.backend.model, dict(app.runtime.controllers)
            action("set_mode", mode=target, **confirmation)
            assert app.runtime.backend.model is plant and app.runtime.controllers == controllers
            np.testing.assert_array_equal(app.runtime.backend.data.qpos, qpos)
        state = app.snapshot()
        assert state["mode"] == target and state["preview_id"] is None

    try:
        initial = app.snapshot()
        assert initial["mode"] == "independent" and initial["shared_object"]
        for actor in app.config.scene.actors:
            np.testing.assert_allclose(
                np.deg2rad(initial["arms"][actor.name]["q_deg"]), actor.q, atol=1e-5
            )
        assert app.runtime.backend.constraint_active(app.config.shared_object.support)
        assert not any(
            app.runtime.backend.constraint_active(n)
            for n in app.config.shared_object.grasp_welds
        )
        action("set_mode", mode="shared", grasps_confirmed=True, refused=True)
        for arm in app.runtime.handles:
            action("arm", arm_id=arm)
        wait(lambda s: all(a["ready"] for a in s["arms"].values()))
        action("set_refinement", refinement=refinement)
        action("set_mode", mode="shared", grasps_confirmed=True, refused=True)
        action("plan_grasps")
        assert "active operation" in action(
            "set_mode", mode="shared", grasps_confirmed=True, refused=True
        )
        approach_id = execute()
        state = app.snapshot()
        assert state["mode"] == "independent"
        print("Current joints -> reviewed grasp approach: PASS", flush=True)
        action("set_mode", mode="shared", grasps_confirmed=True, refused=True)
        for arm, width in state["shared_object"]["grasp_widths_mm"].items():
            action("gripper", arm_id=arm, width_mm=width)
        wait(lambda s: not s["transition_reason"])
        action("set_mode", mode="shared", refused=True)
        switch("shared", grasps_confirmed=True)
        action("stage", pair_id=approach_id, reviewed=True, refused=True)
        action("gripper", arm_id=next(iter(app.runtime.handles)), width_mm=80, refused=True)
        print(
            "Confirmed grasp transition preserves joints/object/controllers and clears old preview: PASS",
            flush=True,
        )
        wait(lambda s: s["mode_settled"])
        state = app.snapshot()
        goal = state["shared_object"]["values"].copy()
        goal[2] += 40
        goal[5] += 10
        action("plan", object_target=goal)
        shared_id = execute()
        wait(lambda s: not s["transition_reason"])
        state = app.snapshot()
        error = pose_error(
            _pose_matrix(state["shared_object"]["values"]), _pose_matrix(goal)
        )
        assert error[0] < 0.0005 and error[1] < np.deg2rad(0.2), error
        assert state["shared_object"]["closure_mm"] < 0.5
        print(
            f"Shared 40 mm / 10 degree move: PASS; error {error[0] * 1000:.4f} mm / {np.rad2deg(error[1]):.5f} degrees",
            flush=True,
        )
        action("set_mode", mode="independent", refused=True)
        switch("independent", support_confirmed=True)
        assert app.runtime.backend.constraint_active(app.config.shared_object.support)
        assert not any(
            app.runtime.backend.constraint_active(n)
            for n in app.config.shared_object.grasp_welds
        )
        assert all(c.executor._payload == (0.0, None) for c in app.runtime.controllers.values())
        action("stage", pair_id=shared_id, reviewed=True, refused=True)
        action("plan", object_target=goal, refused=True)
        # The support anchors the CURRENT object pose after a lift/rotation.
        supported_pose = app.runtime.object_pose().copy()
        for arm in app.runtime.handles:
            action("gripper", arm_id=arm, width_mm=80)
        wait(lambda s: all(a["finger_m"] > 0.039 for a in s["arms"].values()))
        held_error = pose_error(app.runtime.object_pose(), supported_pose)
        assert held_error[0] < 0.0001 and held_error[1] < 0.001, held_error
        print(
            "Confirmed support -> independent mode, stale preview rejection and Hand release: PASS",
            flush=True,
        )
    finally:
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("examples/configs/dual_arm_object.yaml"),
    )
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    check(args.config, refinement="trajopt" if args.refine else "none")
