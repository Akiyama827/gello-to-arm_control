"""Device-free HTTP -> planner process -> two stock controllers regression."""

from pathlib import Path
import json
import selectors
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from arm_control.ui.dual_arm_console import DualArmConsole, plan_request


def main():
    app = DualArmConsole(Path("examples/configs/dual_arm.yaml"), port=0)
    local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{app.server.port}"

    def fetch(path="state", body=None, expected=200, headers=None):
        request = urllib.request.Request(
            f"{base}/{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers=headers or {},
        )
        try:
            reply = local.open(request, timeout=5)
        except urllib.error.HTTPError as exc:
            reply = exc
        with reply:
            content = reply.read()
            assert reply.status == expected, (reply.status, content)
            if "json" in reply.headers.get("Content-Type", ""):
                return json.loads(content)
            return content

    def action(name, *, expected=200, **extra):
        state = fetch()
        return fetch(
            "action",
            dict(
                session=state["session"],
                revision=state["revision"],
                action=name,
                **extra,
            ),
            expected=expected,
        )

    def wait(predicate, *, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = fetch()
            assert state["connected"], state
            if predicate(state):
                return state
            time.sleep(0.04)
        raise AssertionError(state)

    def plan(delta=3, active=None):
        state = fetch()
        targets = {}
        for a, arm in state["arms"].items():
            q = arm["q_deg"].copy()
            q[0] += delta
            targets[a] = dict(kind="joint", frame="base", values=q)
        action("plan", targets=targets, active_arms=active or list(targets))
        state = wait(lambda s: s["status"] in {"preview", "failed"})
        assert state["status"] == "preview", state["error"]
        return state, targets

    try:
        # A cancel holds the measured fingers even when soft MuJoCo limits let
        # them read past the model range (seen: 1.7 mm over the 40 mm limit).
        with app._lock:
            limits = app.runtime.finger_limits
            app.runtime.finger_positions = lambda: {a: hi + 0.002 for a, (_, hi) in limits.items()}
            try:
                app.runtime.cancel_hands()
            finally:
                del app.runtime.finger_positions
            assert app.runtime.fingers == {a: hi for a, (_, hi) in limits.items()}
        # The console reads limits from the model and poses through the handles, never a controller: the displayed limits are
        # the controller's own, a rejected sample keeps the last good pose, and before any measurement a plan is refused.
        state = fetch()
        for a_, arm_ in state["arms"].items():
            lo, hi = app.runtime.controllers[a_].executor.joint_limits
            np.testing.assert_allclose(arm_["limits_deg"], np.rad2deg(np.vstack([lo, hi])).T.tolist(), atol=1e-6)
        with app._lock:
            from dataclasses import replace
            real = app.runtime.handles["left"].state
            good = app._measured("left")
            app.runtime.handles["left"].state = lambda: replace(real(), position=None, velocity=None)
            try:
                np.testing.assert_allclose(app._measured("left"), good)
                app._last_q.pop("left")
                try:
                    app._measured("left")
                except ValueError as exc:
                    assert "no measured pose" in str(exc), exc
                else:
                    raise AssertionError("a pose was invented before any measurement")
            finally:
                del app.runtime.handles["left"].state
            app._measured("left")
        assert b"Dual-arm control" in fetch("")
        assert b"currentPreview" in fetch("static/dual.js")
        geometry = fetch("geometry")
        assert set(geometry["arms"]) == {"left", "right"}
        fetch("mesh/left/0")
        fetch("mesh/left/../../etc/passwd", expected=404)
        fetch("mesh/left/999999", expected=404)
        fetch("action", [], expected=400)
        fetch("action", {}, expected=403, headers={"Origin": "http://evil.example"})
        fetch(headers={"Host": "evil.example"}, expected=403)
        fetch("action", {"action": "arm", "session": "old"}, expected=400)
        state = fetch()
        fetch(
            "action",
            dict(session=state["session"], revision=-1, action="arm", arm_id="left"),
            expected=400,
        )
        action("gripper", arm_id="left", width_mm=40, expected=400)
        for arm in ("left", "right"):
            action("arm", arm_id=arm)
        wait(lambda s: all(a["armed"] and a["ready"] for a in s["arms"].values()))
        action("gripper", arm_id="left", width_mm=900, expected=400)
        action("gripper", arm_id="left", width_mm=float("nan"), expected=400)
        state, targets = plan()
        pair_id = state["preview_id"]
        assert fetch("preview")["pair_id"] == pair_id
        action("stage", pair_id=pair_id, reviewed=False, expected=400)
        action("execute", pair_id=pair_id, expected=400)
        action("stage", pair_id=pair_id, reviewed=True)
        wait(lambda s: s["pair"]["lifecycle"] == "ready")
        action("execute", pair_id=pair_id)
        done = wait(lambda s: s["pair"]["lifecycle"] == "complete")
        error = max(
            np.max(np.abs(np.array(a["q_deg"]) - targets[k]["values"]))
            for k, a in done["arms"].items()
        )
        assert error < 0.1, error
        action("stage", pair_id=pair_id, reviewed=True, expected=400)
        print(f"HTTP paired execution: PASS, max joint residual={error:.6g}deg")

        # Moving one arm keeps the other on the same timeline at its measured q.
        state, targets = plan(active=["left"])
        with app._lock:
            held = app.motion.trajectories["right"]
            assert np.all(held.positions == held.positions[0])
            assert np.all(held.velocities == 0)
        old_id = state["preview_id"]
        action("gripper", arm_id="left", width_mm=40)
        assert fetch()["preview_id"] is None
        action("stage", pair_id=old_id, reviewed=True, expected=400)
        time.sleep(0.3)

        # Cancelling a pending worker cannot resurrect its preview.
        action("plan", targets=targets, active_arms=["left"])
        action("cancel")
        state = wait(lambda s: not s["planning"])
        assert state["preview_id"] is None

        state, _ = plan()
        action("stage", pair_id=state["preview_id"], reviewed=True)
        wait(lambda s: s["pair"]["lifecycle"] == "ready")
        # No GET /state or actions while the operator lease expires.
        time.sleep(3.4)
        state = fetch()
        assert state["pair"]["lifecycle"] == "cancelled", state
        assert not state["pair"]["cancellation_pending"], state
        assert all(not a["running_plan_id"] for a in state["arms"].values())

        # The worker refuses a changed config rather than attaching old provenance.
        try:
            plan_request(app.config.source, "old-config", {}, {}, [], {}, "scene")
        except ValueError as exc:
            assert "configuration or model changed" in str(exc)
        else:
            raise AssertionError("stale worker config admitted")
        print(
            "dual console: PASS (HTTP boundaries, preview review, stale IDs, held member, cancellation, disconnect lease, config binding)"
        )
        for arm in ("left", "right"):
            action("disarm", arm_id=arm)
        wait(lambda s: all(not a["armed"] for a in s["arms"].values()))
    finally:
        app.close()


def check_launcher_shutdown():
    """SIGTERM must close the server/plant/worker, as Ctrl-C does."""
    process = subprocess.Popen(
        [sys.executable, "-B", "examples/run_console.py", "--dual", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=15), "launcher did not become ready"
            line = process.stdout.readline()
        assert "[dual_console] shared_sim, DISARMED: http://127.0.0.1:" in line, line
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=15)
        assert process.returncode == 0, output
        print("dual launcher: PASS (SIGTERM exits through resource cleanup)")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
    check_launcher_shutdown()
