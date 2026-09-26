"""Dual hardware console against two C++ fake RT servers, never robot addresses.

Pass a staged remote_rt pair config and a locally built arm_rt_server binary.
The fake plant starts at zero; a test-only coordinate offset maps that zero to
the configured ready joints. Measured motion, authority and faults use the wire.
Hand feedback is a local double; physical Hand transport is not qualified here.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from arm_control.config import gripper_joints
from arm_control.multi_arm.config import load_pair_config
from arm_control.multi_arm import diagnostics
from arm_control.multi_arm import hardware
from arm_control.multi_arm.cooperative import pose_error, virtual_capture_reason
from arm_control.multi_arm.hardware import RemoteHardware
from arm_control.plants.remote_rt.client import RtBackend, RtConfig
from arm_control.ui.dual_arm_console import DualArmConsole, _pose_matrix
from arm_control.ui import dual_arm_console
from arm_control.simulation import rerun_scene


class FakeHand:
    def __init__(self, *_):
        self.width = .08
        self.fresh = True
        self.busy = False
        self.cancellations = 0
        self.results = []
        self.moves = 0

    def start(self):
        pass

    def snapshot(self):
        return dict(width=self.width, measured=self.fresh, available=True,
                    active_stop=True, busy=self.busy)

    def send_line(self, line, request=None):
        self.moves += 1
        self.width = float(line.split()[1])
        self.busy = True
        return True

    def cancel_move(self):
        self.busy = False
        self.cancellations += 1

    def close(self):
        pass

    def pop_move_results(self):
        results, self.results = self.results, []
        return results


class OffsetFakeRt(RtBackend):
    def __init__(self, config, joints, offset):
        assert config.host == "127.0.0.1"
        super().__init__(config, joints)
        self.offset = np.asarray(offset)
        self.invalid = False
        self.measurement_delta = np.zeros_like(self.offset)

    def motor_state(self, *, state=None):
        state = super().motor_state(state=state)
        for key in ("position", "position_cmd"):
            state[key] = state[key] + self.offset
        state["position"] += self.measurement_delta
        if self.invalid:
            state["position"][0] = np.nan
        return state

    def apply_command(self, command):
        super().apply_command(dict(command, position=command["position"] - self.offset))


def free_port(kind):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check_virtual(app, action, wait, refinement):
    """Exercise virtual grasps with empty Hands over real RT wire protocols."""
    assert app.config.shared_object.virtual and not app.config.scene.welds
    try:
        RemoteHardware(replace(app.config, shared_object=replace(app.config.shared_object, virtual=False)))
    except ValueError as exc:
        assert "not qualified" in str(exc)
    else:
        raise AssertionError("physical shared-object hardware was admitted")

    def refused(name, **kw):
        try:
            action(name, **kw)
        except ValueError:
            return
        raise AssertionError(f"{name} was admitted")

    def execute():
        state = wait(lambda s: s["status"] in {"preview", "failed", "faulted"}, timeout=60)
        assert state["status"] == "preview", state["error"]
        pair = state["preview_id"]
        action("stage", pair_id=pair, reviewed=True)
        wait(lambda s: s["pair"]["lifecycle"] == "ready")
        right = list(app.runtime.handles.values())[1]
        execute_member = right.execute

        def delayed_release(plan_id):
            time.sleep(.02)
            execute_member(plan_id)

        with patch.object(right, "execute", delayed_release):
            action("execute", pair_id=pair)
        with app._lock:
            starts = [c.reference.t_start for c in app.runtime.controllers.values()]
            assert starts[0] == starts[1], starts
            assert app.coordinator.pair_state().release_skew_sec >= .02
        state = wait(lambda s: s["pair"]["lifecycle"] in {"complete", "faulted", "cancelled"}, timeout=60)
        assert state["pair"]["lifecycle"] == "complete", state["error"]
        wait(lambda s: not s["transition_reason"])
        return pair

    action("set_refinement", refinement=refinement)
    refused("set_mode", mode="shared", empty_hands_confirmed=True)
    action("plan_grasps")
    approach = execute()
    # A stopped pose can miss nominal grasp alignment while satisfying joint
    # completion. Reproduce a small residual with a reviewed fake-only move.
    state = app.snapshot()
    targets = {a: dict(kind="joint", frame="base", values=s["q_deg"].copy())
               for a, s in state["arms"].items()}
    first = next(iter(targets))
    targets[first]["values"][0] += .4
    action("plan", targets=targets, active_arms=list(targets))
    execute()
    errors = app.snapshot()["shared_object"]["entry_errors"]
    assert errors[first]["position_mm"] > 1, errors
    assert not virtual_capture_reason({a: (e["position_mm"] / 1000, np.deg2rad(e["angle_deg"])) for a, e in errors.items()})
    for invalid in ((.0101, 0), (0, np.deg2rad(2.01)), (np.nan, 0)):
        assert virtual_capture_reason({first: invalid})
    refused("set_mode", mode="shared", grasps_confirmed=True)
    nominal = app.config.shared_object.grasps
    with patch.object(app.runtime.world, "in_collision", return_value=True):
        refused("set_mode", mode="shared", empty_hands_confirmed=True)
    assert app.runtime.kinematics.grasps is nominal and app.runtime.grasp_capture is None
    hands = {a: (h.width, h.moves) for a, h in app.runtime.hands.items()}
    with app._lock:
        before = app.runtime.joint_state()["position"].copy()
        object_before = app.runtime.object_pose().copy()
        action("set_mode", mode="shared", empty_hands_confirmed=True)
        np.testing.assert_array_equal(app.runtime.joint_state()["position"], before)
        np.testing.assert_allclose(app.runtime.object_pose(), object_before, atol=1e-12)
        assert app.runtime.kinematics.grasps is not nominal
        captured = app._virtual_grasps()
        assert captured and app.snapshot()["shared_object"]["captured_offsets"][first]["position_mm"] > 1
        assert app.snapshot()["shared_object"]["closure_mm"] < 1e-8
    assert app.snapshot()["mode"] == "shared" and app.snapshot()["preview_id"] is None
    assert hands == {a: (h.width, h.moves) for a, h in app.runtime.hands.items()}
    assert all(h.width == .08 for h in app.runtime.hands.values())
    assert all(c.executor._payload == (0.0, None) for c in app.runtime.controllers.values())
    refused("stage", pair_id=approach, reviewed=True)
    refused("gripper", arm_id=next(iter(hands)), width_mm=10)
    print("Opposite-edge preparation -> explicit empty-Hand coupling, no weld/payload/Hand command: PASS", flush=True)
    wait(lambda s: s["mode_settled"])
    goal = app.snapshot()["shared_object"]["values"].copy()
    goal[2] += 10
    goal[5] += 2
    action("plan", object_target=goal)
    state = wait(lambda s: s["status"] in {"preview", "failed"}, timeout=60)
    assert state["status"] == "preview", state["error"]
    idle_pair = state["preview_id"]
    action("stage", pair_id=idle_pair, reviewed=True)
    wait(lambda s: s["pair"]["lifecycle"] == "ready")
    align, invalidate = hardware.align_feedback, app._invalidate
    injected, cancelled = [], []
    holds = {a: c.last_command.q_des.copy() for a, c in app.runtime.controllers.items()}

    def gap_once(links, histories, **kwargs):
        if not injected:
            injected.append(True)
            a, b = links
            left, right = histories[a][-1], histories[b][-1]
            target = min(left[0].t_mono_ns, right[0].t_mono_ns)
            histories = {
                a: [(replace(left[0], t_mono_ns=target), left[1])],
                b: [(replace(right[0], t_mono_ns=target - 20_000_000,
                             state_seq=right[0].state_seq - 3), right[1]),
                    (replace(right[0], t_mono_ns=target + 525_493), right[1])],
            }
        return align(links, histories, **kwargs)

    def observe_cancel(reason):
        if "feedback interrupted" in reason:
            assert not app.runtime.feedback_ready and not app.runtime.fresh
            assert all(not h.state().ready for h in app.runtime.handles.values())
            cancelled.append(True)
        invalidate(reason)

    with patch.object(hardware, "align_feedback", gap_once), patch.object(app, "_invalidate", observe_cancel):
        state = wait(lambda s: s["status"] in {"cancelled", "faulted"})
        assert state["status"] == "cancelled", state["error"]
    assert injected and cancelled and app.fault_trace is None
    assert state["pair"]["lifecycle"] == "cancelled" and state["preview_id"] is None
    assert all(a["released_at"] is None for a in state["pair"]["members"])
    wait(lambda s: app.runtime.feedback_ready and app.runtime.fresh)
    with app._lock:
        for a, c in app.runtime.controllers.items():
            assert c.armed and not c._running and c._plan is None
            np.testing.assert_array_equal(c.last_command.q_des, holds[a])
    refused("execute", pair_id=idle_pair)
    print("Idle 20.525493 ms feedback gap cancels staged pair, preserves hold, requires new plan: PASS", flush=True)
    action("plan", object_target=goal)
    first_executor = next(iter(app.runtime.controllers.values())).executor
    step = first_executor.step

    def delayed_dynamics(*args, **kwargs):
        time.sleep(.003)
        return step(*args, **kwargs)

    with patch.object(first_executor, "step", delayed_dynamics):
        shared = execute()
    state = app.snapshot()
    error = pose_error(_pose_matrix(state["shared_object"]["values"]), _pose_matrix(goal))
    assert error[0] < .001 and error[1] < np.deg2rad(.5), error
    assert state["shared_object"]["closure_mm"] < 3
    assert app._virtual_grasps() == captured, "captured grasps changed during motion"
    print(f"Virtual 10 mm / 2 degree move ({refinement}): PASS; error {error[0]*1000:.4f} mm / {np.rad2deg(error[1]):.5f} degrees", flush=True)
    moving = [s for s in app._trace.samples if s.get("closure") is not None
              and all(a["plan_id"] == f"{shared}:{name}" and a["trajectory_elapsed_s"] is not None
                      for name, a in s["arms"].items())]
    assert moving, "no moving shared-object samples captured"
    for s in moving:
        phases = [a["trajectory_elapsed_s"] for a in s["arms"].values()]
        np.testing.assert_allclose(phases[0], phases[1], atol=1e-12)
    peak = max(s["closure"][0] for s in moving) * 1000
    print(f"20 ms release / 3 ms dynamics delays: equal reference phases, peak closure {peak:.4f} mm: PASS", flush=True)
    refused("set_mode", mode="independent", support_confirmed=True)
    action("set_mode", mode="independent", virtual_release_confirmed=True)
    assert app.runtime.kinematics.grasps is nominal and app.runtime.grasp_capture is None
    frozen = app.runtime.object_pose().copy()
    time.sleep(.1)
    np.testing.assert_array_equal(app.runtime.object_pose(), frozen)
    refused("stage", pair_id=shared, reviewed=True)
    wait(lambda s: not s["transition_reason"])
    action("set_mode", mode="shared", empty_hands_confirmed=True)
    wait(lambda s: s["mode_settled"])
    goal = app.snapshot()["shared_object"]["values"].copy()
    goal[2] += 10
    action("plan", object_target=goal)
    state = wait(lambda s: s["status"] in {"preview", "failed"}, timeout=60)
    assert state["status"] == "preview", state["error"]
    pair = state["preview_id"]
    action("stage", pair_id=pair, reviewed=True)
    wait(lambda s: s["pair"]["lifecycle"] == "ready")
    action("execute", pair_id=pair)
    time.sleep(.04)
    # Change only measured feedback, not the fake plant's command coordinates.
    # This breaks TCP agreement and must stop both RT servers immediately.
    peer = list(app.runtime.links.values())[1]
    peer.measurement_delta[0] = .05
    state = wait(lambda s: s["status"] == "faulted")
    assert "TCP grasp agreement" in state["error"], state["error"]
    deadline = time.monotonic() + 3
    while any(link.motor_health()["armed"] for link in app.runtime.links.values()) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not any(link.motor_health()["armed"] for link in app.runtime.links.values())
    assert "measured " in state["error"]
    app._fault_writer.join(timeout=3)
    trace = json.loads(Path(app.fault_trace).read_text())
    assert trace["reason"] == state["error"]
    assert 1 < len(trace["samples"]) <= 500
    assert trace["context"]["pair"]["pair_id"] == pair
    assert trace["context"]["trajectories"]
    last = trace["samples"][-1]
    alignment = last["feedback_alignment"]
    inferred = app.kin.objects({a: np.asarray(s["q"]) for a, s in alignment["brackets"].items()})
    measured = pose_error(*inferred.values())
    np.testing.assert_allclose(last["closure"], measured, atol=1e-12)
    assert measured[0] > .003 or measured[1] > np.deg2rad(2)
    assert all(s["state_seq"] and s["state_t_mono_ns"] and s["q_cmd_rt"] is not None
               and s["q_cmd_pc"] is not None for s in last["arms"].values())
    with patch.object(app._trace, "save", side_effect=OSError("test disk failure")):
        app._save_fault_trace(state["error"])
        app._fault_writer.join(timeout=3)
    assert app.error == state["error"]
    entered, release = threading.Event(), threading.Event()

    def blocked_save(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise OSError("test blocked writer released")

    with patch.object(app._trace, "save", side_effect=blocked_save), ThreadPoolExecutor(max_workers=1) as pool:
        app._save_fault_trace(state["error"])
        assert entered.wait(1)
        closing = pool.submit(app.close)
        try:
            closing.result(timeout=3)
        finally:
            release.set()
        app._fault_writer.join(timeout=3)
    assert app.runtime._closed
    print("Blocked diagnostic writer does not block transport shutdown: PASS", flush=True)
    print("Fault trace retains triggering measurements/references/timestamps and preserves stop on write failure: PASS", flush=True)
    print("Virtual release freezes pose; TCP divergence stops both RT servers: PASS", flush=True)


def main(pair_path, binary, refinement="none"):
    trace = diagnostics.PairFaultTrace("self-check")
    for i in range(510):
        trace.record(dict(index=i, value=float("nan")))
    assert len(trace.samples) == 500 and trace.samples[0]["index"] == 10
    with tempfile.TemporaryDirectory() as trace_root, patch.object(diagnostics, "CONTROL_ROOT", Path(trace_root)):
        report = json.loads(trace.save("test", {}).read_text())
        assert report["samples"][-1]["value"] is None
    # A queued enable must not run after close begins, even if the authority
    # worker is occupied. This regression needs no socket or controller tick.
    entered, release = threading.Event(), threading.Event()
    calls = []
    rt = RemoteHardware.__new__(RemoteHardware)
    rt._closed, rt.hands = False, {}
    rt._pending, rt._authority_generation = {"test": []}, {"test": 0}
    rt._rpc = {"test": ThreadPoolExecutor(max_workers=1)}
    rt.links = {"test": SimpleNamespace(enable_all=lambda: calls.append("ARM"),
                                       safe_stop=lambda: calls.append("STOP"),
                                       close=lambda: calls.append("CLOSE"))}

    def block_worker():
        entered.set()
        assert release.wait(3)

    rt._rpc["test"].submit(block_worker)
    assert entered.wait(1)
    rt._request_authority("test", True)
    closing = threading.Thread(target=rt.close)
    closing.start()
    deadline = time.monotonic() + 1
    while not rt._closed and time.monotonic() < deadline:
        time.sleep(.01)
    assert rt._closed
    release.set()
    closing.join(timeout=3)
    assert not closing.is_alive() and calls == ["STOP", "CLOSE"], calls
    print("Shutdown cancels queued ARM before transport close: PASS", flush=True)
    config = load_pair_config(pair_path)
    assert config.backend == "remote_rt"
    ports, servers = {}, []
    app = None
    with tempfile.TemporaryDirectory(prefix="dual-hardware-check-") as root, patch.object(
        diagnostics, "CONTROL_ROOT", Path(root)
    ):
        logs = []
        try:
            for actor in config.scene.actors:
                udp, tcp = free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM)
                ports[actor.name] = (udp, tcp)
                log = Path(root) / f"{actor.name}.log"
                logs.append(log)
                with log.open("w") as output:
                    servers.append(subprocess.Popen(
                        [binary, "--backend", "fake", "--n", str(len(actor.joints)),
                         "--instance-id", actor.name, "--joint-names", ",".join(actor.joints),
                         "--bind", "127.0.0.1", "--udp-port", str(udp), "--tcp-port", str(tcp)],
                        stdout=output, stderr=subprocess.STDOUT,
                    ))
            time.sleep(.3)
            offsets = {a.name: a.q for a in config.scene.actors}

            def backend(cfg):
                rt = RtConfig.from_config(cfg)
                udp, tcp = ports[rt.instance_id]
                return OffsetFakeRt(replace(rt, host="127.0.0.1", udp_port=udp, tcp_port=tcp),
                                    list(cfg.joint_names), offsets[rt.instance_id])

            runtimes = []
            opened = []

            def runtime(cfg):
                rt = RemoteHardware(cfg, backend_factory=backend, hand_factory=FakeHand)
                runtimes.append(rt)
                for a, link in rt.links.items():
                    original = link.open

                    def open_link(a=a, original=original):
                        opened.append(a)
                        return original()

                    link.open = open_link
                return rt

            with patch("arm_control.ui.dual_arm_console._visuals", side_effect=ValueError("visual failure")):
                try:
                    DualArmConsole(pair_path, start_server=False, runtime_factory=runtime)
                except ValueError as exc:
                    assert str(exc) == "visual failure"
                else:
                    raise AssertionError("visual failure was not raised")
            assert runtimes[-1]._closed
            assert all(link._tcp is None for link in runtimes[-1].links.values())
            assert not opened
            print("Visual setup failure opens no RT or Hand transport: PASS", flush=True)
            with patch("arm_control.ui.dual_arm_console.ConsoleServer", side_effect=ValueError("server failure")):
                try:
                    DualArmConsole(pair_path, runtime_factory=runtime)
                except ValueError as exc:
                    assert str(exc) == "server failure"
                else:
                    raise AssertionError("server failure was not raised")
            assert len(opened) == 2
            assert runtimes[-1]._closed
            assert all(link._tcp is None for link in runtimes[-1].links.values())
            print("Post-connection startup failure closes both links: PASS", flush=True)
            original_visuals = dual_arm_console._visuals
            original_meshes = rerun_scene.static_mesh_geoms

            def before_transport():
                assert all(link._tcp is None for link in runtimes[-1].links.values())
                assert not runtimes[-1].hands

            def stalled_visuals(config):
                before_transport()
                # PyDLL retains the GIL. Reproduce native mesh loading longer
                # than the unchanged 1.2 s RT deadline, not time.sleep (which
                # would allow the heartbeat threads to run).
                assert ctypes.PyDLL(None).usleep(1_600_000) == 0
                return original_visuals(config)

            def meshes_before_transport(*args, **kwargs):
                before_transport()
                return original_meshes(*args, **kwargs)

            with patch.object(dual_arm_console, "_visuals", stalled_visuals), patch.object(
                rerun_scene, "static_mesh_geoms", meshes_before_transport
            ):
                app = DualArmConsole(pair_path, start_server=False, runtime_factory=runtime)
            sessions = {a: link.session_id for a, link in app.runtime.links.items()}
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                assert app.runtime.fresh
                assert all(not link.motor_health()["armed"] for link in app.runtime.links.values())
                time.sleep(.05)
            assert sessions == {a: link.session_id for a, link in app.runtime.links.items()}
            np.testing.assert_allclose(
                app.runtime.world.data.qpos[app.runtime.world._qadr],
                app.runtime.joint_state()["position"],
            )
            for a, cfg in config.robots.items():
                for j in gripper_joints(cfg):
                    adr = app.runtime.world.model.joint(f"{a}__{j}").qposadr[0]
                    assert app.runtime.world.data.qpos[adr] == app.runtime.fingers[a]
            print("GIL-blocking asset setup precedes connections; fresh disarmed sessions survive: PASS", flush=True)

            def action(name, **kw):
                s = app.snapshot()
                return app.post("action", dict(session=s["session"], revision=s["revision"],
                                               action=name, **kw))

            def wait(predicate, timeout=20):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    state = app.snapshot()
                    if predicate(state):
                        return state
                    time.sleep(.02)
                raise AssertionError(state)

            state = app.snapshot()
            assert state["backend"] == "remote_rt" and state["connected"]
            assert not any(a["armed"] for a in state["arms"].values())
            assert (state["shared_object"] is None) == (config.shared_object is None)
            try:
                action("set_mode", mode="shared", grasps_confirmed=True)
            except ValueError:
                pass
            else:
                raise AssertionError("hardware admitted unqualified shared-object mode")
            a = next(iter(state["arms"]))
            link = app.runtime.links[a]
            link.invalid = True
            wait(lambda s: not s["connected"])
            assert app.runtime.controllers[a]._last_state_at is None
            with app._lock:
                sample = app.runtime.diagnostic_sample()["arms"][a]
            assert np.isnan(sample["q_rt"][0])
            assert sample["controller_observed_at_pc"] is None
            assert not sample["controller_measurement_valid"]
            link.invalid = False
            wait(lambda s: s["connected"])
            print("Non-finite feedback refuses freshness; valid measurements recover: PASS", flush=True)
            # Hold an authority RPC in flight: repeated UI requests must not
            # accumulate enables behind a disarm.
            entered, release = threading.Event(), threading.Event()
            enable = link.enable_all

            def delayed_enable():
                entered.set()
                assert release.wait(3)
                return enable()

            with patch.object(link, "enable_all", delayed_enable):
                action("arm", arm_id=a)
                assert entered.wait(1)
                action("disarm", arm_id=a)
                for _ in range(10):
                    try:
                        action("arm", arm_id=a)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("pending authority admitted re-arm")
                assert len(app.runtime._pending[a]) <= 2
                release.set()
                wait(lambda s: not app.runtime._pending[a] and not s["arms"][a]["armed"])
            print("Disarm supersedes pending ARM; repeated ARM requests refused: PASS", flush=True)
            for a in state["arms"]:
                action("arm", arm_id=a)
            wait(lambda s: all(a["ready"] for a in s["arms"].values()))
            hand = app.runtime.hands[a]
            action("gripper", arm_id=a, width_mm=78)
            assert hand.busy
            action("cancel")
            assert not hand.busy
            action("gripper", arm_id=a, width_mm=80)
            with app._lock:
                app._client_seen = time.monotonic() - 4
            time.sleep(.1)  # no polling: snapshot itself is a browser heartbeat
            assert not hand.busy
            assert app.snapshot()["error"].startswith("Console disconnected")
            hand.results.append(dict(ok=False, reason="test Hand failure"))
            wait(lambda s: s["telemetry"][a]["hand"]["last_move"] is not None)
            assert app.snapshot()["telemetry"][a]["hand"]["last_move"]["reason"] == "test Hand failure"
            assert not hand.results
            print("Cancel and browser loss stop independent Hand motion: PASS", flush=True)
            state = app.snapshot()
            targets = {}
            for a, measured in state["arms"].items():
                goal = measured["q_deg"].copy()
                goal[0] += 2
                targets[a] = dict(kind="joint", frame="base", values=goal)
            action("plan", targets=targets, active_arms=list(targets))
            state = wait(lambda s: s["status"] in {"preview", "failed"})
            assert state["status"] == "preview", state["error"]
            pair = state["preview_id"]
            action("stage", pair_id=pair, reviewed=True)
            wait(lambda s: s["pair"]["lifecycle"] == "ready")
            action("execute", pair_id=pair)
            state = wait(lambda s: s["pair"]["lifecycle"] in {"complete", "faulted", "cancelled"})
            assert state["pair"]["lifecycle"] == "complete", state
            assert all(np.max(np.abs(np.array(s["q_deg"]) - targets[a]["values"])) < .7
                       for a, s in state["arms"].items())
            print("Two RT sessions: disarmed startup, measured planning, reviewed paired execution: PASS", flush=True)
            if config.shared_object is not None:
                check_virtual(app, action, wait, refinement)
                return
            # Kill one state/control server while a plan is staged. Peer must
            # receive cancellation and stale feedback must refuse new motion.
            action("plan", targets=targets, active_arms=list(targets))
            state = wait(lambda s: s["status"] in {"preview", "failed"})
            assert state["status"] == "preview", state["error"]
            pair = state["preview_id"]
            action("stage", pair_id=pair, reviewed=True)
            wait(lambda s: s["pair"]["lifecycle"] == "ready")
            servers[0].terminate()
            servers[0].wait(timeout=3)
            wait(lambda s: not s["connected"] and s["pair"]["lifecycle"] in {"cancelled", "faulted"})
            wait(lambda s: not s["pair"]["cancellation_pending"])
            print("Lost RT peer cancels pair; no shared-object hardware admission: PASS", flush=True)
            app._stop.set()
            app._thread.join(timeout=2)
            assert not app._thread.is_alive()
            app.runtime.stop()
            survivor = app.runtime.links[config.scene.actors[1].name]
            deadline = time.monotonic() + 3
            while survivor.motor_health()["armed"] and time.monotonic() < deadline:
                time.sleep(.01)
            assert not survivor.motor_health()["armed"]
            print("Runtime stop revokes surviving RT authority without further ticks: PASS", flush=True)
        except BaseException:
            for log in logs:
                print(log.read_text()[-5000:])
            raise
        finally:
            if app:
                app.close()
            for process in servers:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair")
    parser.add_argument("binary")
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    main(args.pair, args.binary, refinement="trajopt" if args.refine else "none")
