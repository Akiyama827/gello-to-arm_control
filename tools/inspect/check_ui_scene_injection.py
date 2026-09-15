"""Headless UI injection check; run with the library on PYTHONPATH."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from arm_control import frames
import pinocchio as pin

from arm_control.config import RobotConfig
from arm_control.grasp_visual import VisualFK
from arm_control.planning import preview_rerun as preview
from arm_control.ui import arm_console as console
from arm_control.viz import visualizer


def main() -> None:
    transform = np.eye(4)
    transform[:3, :3] = pin.rpy.rpyToMatrix(0.2, -0.3, 0.7)
    transform[:3, 3] = [0.125, -0.25, 0.75]
    geoms = [("fixture", "mesh", Path(__file__), transform)]
    vfk = VisualFK.__new__(VisualFK)
    vfk._pin, vfk._static, vfk._geom_ids = pin, [], [0, 1]
    with patch.object(preview, "static_scene_geoms", return_value=geoms) as resolve:
        vfk.add_static_scene({}, geoms=[])
        assert vfk.static_json() == []
        resolve.assert_not_called()
        vfk.add_static_scene({}, geoms=geoms)
        resolve.assert_not_called()
        item = vfk.static_json()[0]
        assert item["mesh"].startswith("mesh/2?")
        assert vfk.mesh_path(2) == Path(__file__)
        np.testing.assert_allclose(item["p"], transform[:3, 3], atol=1e-5)
        np.testing.assert_allclose(
            item["q"], pin.Quaternion(transform[:3, :3]).coeffs(), atol=1e-6,
        )
        vfk.add_static_scene({})
        resolve.assert_called_once_with({})

    sink = Mock()
    sink.Transform3D.side_effect = lambda **kw: SimpleNamespace(**kw)
    with patch.object(preview, "rr", sink), patch.object(
        preview, "static_scene_geoms", return_value=geoms,
    ) as resolve:
        preview.log_static_scene({}, geoms=[])
        resolve.assert_not_called()
        sink.log.assert_not_called()
        preview.log_static_scene({}, geoms=geoms)
        resolve.assert_not_called()
        logged = sink.log.call_args.args[1]
        np.testing.assert_array_equal(logged.translation, transform[:3, 3])
        np.testing.assert_array_equal(logged.mat3x3, transform[:3, :3])
        preview.log_static_scene({})
        resolve.assert_called_once_with({})

    cfg = RobotConfig(
        num_motors=1, joint_names=["joint"], motor_names=["joint"],
        raw={"arm": {"joints": ["joint"], "ee_frame": "tcp", "gripper_joints": []}},
    )
    world = SimpleNamespace(lower=np.array([-1.0]), upper=np.array([1.0]))
    with patch.object(console, "_run") as run, patch.object(
        console, "install_signal_handlers",
    ):
        console.main(cfg=cfg, collision_world=world, static_geoms=geoms)
        assert run.call_args.kwargs == {
            "cfg": cfg, "collision_world": world, "static_geoms": geoms,
        }
        console.main()

    # Stop immediately after startup scene logging: no Dora or HTTP connection.
    class StartupComplete(Exception):
        pass

    for supplied in (True, False):
        with ExitStack() as stack:
            load = stack.enter_context(patch.object(console, "load_robot_config", return_value=cfg))
            stack.enter_context(patch.object(console, "_load_mode_config", return_value={}))
            stack.enter_context(patch.object(console, "resolve_gains", return_value={"kp": [1], "kd": [1]}))
            build = stack.enter_context(patch.object(console, "build_collision_stack", return_value=(world, None)))
            legacy = stack.enter_context(patch.object(preview, "scene_obstacle_geoms", return_value=[]))
            stack.enter_context(patch.object(preview, "init_preview_stream"))
            stack.enter_context(patch.object(preview.rr, "disconnect"))
            visual = stack.enter_context(patch.object(console, "VisualFK"))
            log = stack.enter_context(patch.object(preview, "log_static_scene"))
            stack.enter_context(patch.object(preview, "RobotGhost"))
            pin_log = stack.enter_context(patch.object(preview, "log_frame_transform"))
            measured = stack.enter_context(
                patch.object(preview, "MeasuredGhost", side_effect=StartupComplete))
            try:
                with ExitStack() as cleanup:
                    console._run(
                        console.ShutdownFlag(), cleanup, cfg=cfg if supplied else None,
                        collision_world=world if supplied else None,
                        static_geoms=geoms if supplied else None,
                    )
            except StartupComplete:
                pass
            else:
                raise AssertionError("console did not reach static scene logging")
            assert build.call_args.kwargs["collision_world"] is (world if supplied else None)
            assert load.call_count == legacy.call_count == int(not supplied)
            visual.return_value.add_static_scene.assert_called_once_with(
                cfg, geoms=geoms if supplied else None,
            )
            log.assert_called_once_with(cfg, geoms=geoms if supplied else None)
            # Both ghost roots carry the arm mount. They are authored in the
            # ARM-BASE frame while the static scene and the planning boundary
            # are logged in WORLD; unpinned they render mutually rotated and
            # shifted by the mount, which on a 90-degree mount reads as a
            # boundary that excludes the robot standing inside it.
            expected = frames.world_T_arm(cfg if supplied else load.return_value)
            np.testing.assert_allclose(measured.call_args.kwargs["world_T_arm"], expected)
            target_pins = [c for c in pin_log.call_args_list if c.args[0] == "target"]
            assert len(target_pins) == 1, target_pins
            np.testing.assert_allclose(target_pins[0].args[1], expected)

    viz_cfg = {"arm": {"gripper_joints": []}}
    for supplied in (True, False):
        with ExitStack() as stack:
            load = stack.enter_context(patch.object(visualizer, "_load_cfg", return_value=viz_cfg))
            for name in ("_init_rerun", "_setup_series_style", "_setup_blueprint"):
                stack.enter_context(patch.object(visualizer, name))
            stack.enter_context(patch.object(visualizer, "_build_render_model", return_value=None))
            stack.enter_context(patch.object(visualizer, "Node", return_value=[]))
            stack.enter_context(patch.object(visualizer.signal, "signal"))
            stack.enter_context(patch.object(visualizer.rr, "disconnect"))
            log = stack.enter_context(patch.object(visualizer, "log_static_scene"))
            visualizer.main(cfg=viz_cfg if supplied else None, static_geoms=[] if supplied else None)
            assert load.call_count == int(not supplied)
            log.assert_called_once_with(viz_cfg, geoms=[] if supplied else None)
    print("UI scene injection: PASS (defaults, supplied empty scene, mesh poses, wiring)")


if __name__ == "__main__":
    main()
