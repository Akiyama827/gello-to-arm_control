"""Device-free rigid-grasp planning, collision, HTTP and execution checks."""

from dataclasses import replace
from pathlib import Path
import json
import os
import signal
import time
import urllib.error
import urllib.request

import numpy as np

from arm_control import frames
from arm_control.multi_arm.cooperative import pose_error
from arm_control.multi_arm.simulation import SharedSimulation
from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
from arm_control.scene import ObstacleSpec
from arm_control.ui.dual_arm_console import DualArmConsole, _pose_matrix, plan_request


def main():
    app = DualArmConsole(Path("examples/configs/dual_arm_object.yaml"), port=0)
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{app.server.port}"

    def fetch(route="state", body=None, expected=200):
        request = urllib.request.Request(
            base + "/" + route, data=None if body is None else json.dumps(body).encode()
        )
        try:
            response = client.open(request, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            content = response.read()
            assert response.status == expected, (response.status, content)
            return (
                json.loads(content)
                if "json" in response.headers.get("Content-Type", "")
                else content
            )

    def action(name, expected=200, **extra):
        s = fetch()
        return fetch(
            "action",
            dict(session=s["session"], revision=s["revision"], action=name, **extra),
            expected,
        )

    def wait(predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = fetch()
            assert s["connected"], s
            if predicate(s):
                return s
            time.sleep(0.04)
        raise AssertionError(s)

    try:
        obj = app.config.shared_object
        # Non-identity object pose and asymmetric grasp orientations round-trip.
        T = frames.T_from_spec(dict(pos=[0.3, -0.2, 0.7], rpy=[0.2, -0.3, 0.4]))
        for inferred in obj.grasps.infer(obj.grasps.targets(T)).values():
            np.testing.assert_allclose(inferred, T, atol=1e-12)
        first = fetch()["shared_object"]
        assert not first["paused"] and fetch()["mode"] == "independent"
        time.sleep(0.12)
        assert (
            pose_error(
                _pose_matrix(fetch()["shared_object"]["values"]),
                _pose_matrix(first["values"]),
            )[0]
            < 0.0001
        )
        geometry = fetch("geometry")
        assert geometry["object"] and not geometry["static"]
        fetch("object-mesh/0")
        fetch("object-mesh/999999", expected=404)
        fetch("object-mesh/../../etc/passwd", expected=404)
        action("plan", expected=400, targets={}, active_arms=["left"])
        action("plan", expected=400, object_target=[0] * 5)
        action("plan", expected=400, object_target=[float("nan")] * 6)
        for a in app.runtime.handles:
            action("arm", arm_id=a)
        wait(lambda s: all(a["ready"] for a in s["arms"].values()))
        action("plan_grasps")
        s = wait(lambda s: s["status"] in {"preview", "failed"})
        assert s["status"] == "preview", s["error"]
        pid = s["preview_id"]
        action("stage", pair_id=pid, reviewed=True)
        wait(lambda s: s["pair"]["lifecycle"] == "ready")
        action("execute", pair_id=pid)
        wait(lambda s: s["pair"]["lifecycle"] == "complete")
        for a, width in obj.fingers.items():
            action("gripper", arm_id=a, width_mm=width * 2000)
        wait(lambda s: not s["transition_reason"])
        action("set_mode", mode="shared", grasps_confirmed=True)
        wait(lambda s: s["mode_settled"])
        action("gripper", expected=400, arm_id="left", width_mm=50)
        initial = fetch()["shared_object"]["values"]
        goal = initial.copy()
        goal[2] += 40
        goal[5] += 10
        action("plan", object_target=goal)
        s = wait(lambda s: s["status"] in {"preview", "failed"})
        assert s["status"] == "preview", s["error"]
        metrics = s["shared_object"]["validation"]
        assert metrics["closure_m"] < 0.001
        assert metrics["closure_rad"] < np.deg2rad(0.5)
        assert all("object" in frame for frame in fetch("preview")["frames"])
        pid = s["preview_id"]
        action("stage", expected=400, pair_id=pid, reviewed=False)
        action("stage", pair_id=pid, reviewed=True)
        wait(lambda s: s["pair"]["lifecycle"] == "ready")
        action("execute", pair_id=pid)
        done = wait(
            lambda s: s["pair"]["lifecycle"] in {"complete", "faulted", "cancelled"}
        )
        assert done["pair"]["lifecycle"] == "complete", done["error"]
        measured = _pose_matrix(done["shared_object"]["values"])
        error = pose_error(measured, _pose_matrix(goal))
        assert error[0] < 0.0005 and error[1] < np.deg2rad(0.2), error
        assert done["shared_object"]["closure_mm"] < 0.5
        print(
            f"Shared object 40 mm lift + 10 deg yaw: PASS; error={error[0] * 1000:.4f} mm/{np.rad2deg(error[1]):.5f} deg; closure={done['shared_object']['closure_mm']:.4f} mm"
        )

        # The plate alone intersects a small central obstacle; neither arm does.
        scene = app.config.scene
        obstacle = ObstacleSpec(
            "object_only", "box", (0.02, 0.02, 0.004), (0.45, 0, 0.4), (0, 0, 0)
        )
        scene = replace(scene, obstacles=(obstacle,))
        held = {name: 0.006 for name in app.runtime.finger_names}
        world = MuJoCoCollisionWorld.from_scene(
            scene, scene.state(), scene.actor_names, held_positions=held
        )
        q = np.concatenate([a.q for a in scene.actors])
        assert not world.in_collision(q), world.collision_contacts(q)
        world.set_free_body_pose(obj.body, obj.world_T_object)
        contacts = world.collision_contacts(q)
        assert world.in_collision(q) and any(
            obj.body in c["bodies"] for c in contacts
        ), contacts
        print(
            "Object-only collision, exact grasp permissions, transform round-trip: PASS"
        )

        try:
            bad_sim = SharedSimulation(replace(app.config, scene=scene))
        except ValueError as exc:
            assert "initial scene collides" in str(exc), str(exc)
        else:
            bad_sim.close()
            raise AssertionError("colliding already-held initial scene admitted")

        # Endpoint IK is not enough: a broken initial rigid relation is refused.
        with app._lock:
            starts = {
                a: c.last_state.position.copy() for a, c in app.runtime.controllers.items()
            }
            object_start = app.runtime.object_pose()
            fingers = app.runtime.finger_positions()
        starts["left"][0] += 0.04
        try:
            plan_request(
                app.config.source,
                app.config.scene_token,
                starts,
                {},
                list(starts),
                fingers,
                "test",
                object_start,
                goal,
                mode="shared",
            )
        except ValueError as exc:
            assert "closure" in str(exc), str(exc)
        else:
            raise AssertionError("broken shared grasp admitted")

        # Staging is also bound to the measured object, not only arm joints.
        goal[2] += 10
        action("plan", object_target=goal)
        s = wait(lambda s: s["status"] in {"preview", "failed"})
        assert s["status"] == "preview", s["error"]
        with app._lock:
            app._object_start[0, 3] += 0.01
        action("stage", expected=400, pair_id=s["preview_id"], reviewed=True)
        action("cancel")
        action("disarm", arm_id="left")
        assert fetch()["shared_object"]["paused"]
        action("disarm", arm_id="right")
        worker_pid = app._pool.submit(os.getpid).result(timeout=5)
        os.kill(worker_pid, signal.SIGINT)
        assert app._pool.submit(pow, 2, 3).result(timeout=5) == 8
        print(
            "HTTP fixed-grasp lock, invalid input, stale object admission, disarm pause, worker Ctrl-C ownership: PASS"
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
