"""The dual console over two arm supervisors and two C++ fake RT servers, never robot addresses (CTRL-053).

    python tools/check/check_dual_hardware.py <staged remote_rt pair> <arm_rt_server> [--refine]

The pair is staged as ``multi_arm.launch`` stages it for the bench (``plant rt``); ONLY each arm's RT endpoint moves to a
local fake server, which starts at that arm's scene pose (``--fake-q0``: the FR3 envelope refuses zero). Each arm's
supervisor is started as the launcher starts it; the console reaches them through ``ARM_CONTROL_PAIR_SOCKETS``. Measured
motion, authority and faults use the wire; Hands are a local double (physical Hand transport is not qualified here).
SupervisedHardware's own surface (pending ARM, close while ARM is in flight, trace/1 drops) is check_supervised_hardware.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
import yaml

from arm_control.config import arm_joints, gripper_joints
from arm_control.execution.contract import Lifecycle
from arm_control.multi_arm import diagnostics, supervised_hardware
from arm_control.multi_arm.config import load_pair_config
from arm_control.multi_arm.cooperative import pose_error, virtual_capture_reason
from arm_control.multi_arm.feedback import FeedbackGap
from arm_control.multi_arm.launch import SOCKETS_ENV, stage_pair
from arm_control.multi_arm.supervised_hardware import SupervisedHardware
from arm_control.sdk import ArmClient
from arm_control.simulation import rerun_scene
from arm_control.supervisor.launch import start_supervisor, stop_supervisor
from arm_control.ui import dual_arm_console
from arm_control.ui.dual_arm_console import DualArmConsole, _pose_matrix


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


def free_port(kind):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def tracked_releases(app, delay_s=.02):
    """Patch both members' release RPCs: record each start stamp; the second member's arrives ``delay_s`` late."""
    stamps = []
    patches = []
    for index, handle in enumerate(app.runtime.handles.values()):
        client = handle._inner._arm
        real = client.release

        def release(eid, start_at=None, real=real, late=index == 1):
            if late:
                time.sleep(delay_s)
            stamps.append(start_at)
            return real(eid, start_at=start_at)

        patches.append(patch.object(client, "release", release))
    return stamps, patches


def released_together(app, action, pair):
    """Release a staged pair with the partner's release 20 ms late: both members carry ONE start stamp."""
    stamps, patches = tracked_releases(app)
    for p in patches:
        p.start()
    try:
        action("execute", pair_id=pair)
    finally:
        for p in patches:
            p.stop()
    assert len(stamps) == 2 and stamps[0] is not None and stamps[0] == stamps[1], stamps


def check_virtual(app, action, wait, refinement, probe):
    """Exercise virtual grasps with empty Hands over real RT wire protocols."""
    assert not app.config.scene.welds  # coupling is virtual: nothing is welded (load_pair_config refuses anything else)

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
        released_together(app, action, pair)
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
        # the object is inferred from common-time TCPs now (not the display pose): it agrees to the capture
        np.testing.assert_allclose(app.runtime.object_pose(), object_before, atol=1e-6)
        assert app.runtime.kinematics.grasps is not nominal
        captured = app._virtual_grasps()
        assert captured and app.snapshot()["shared_object"]["captured_offsets"][first]["position_mm"] > 1
        assert app.snapshot()["shared_object"]["closure_mm"] < 1e-6
    assert app.snapshot()["mode"] == "shared" and app.snapshot()["preview_id"] is None
    assert hands == {a: (h.width, h.moves) for a, h in app.runtime.hands.items()}
    assert all(h.width == .08 for h in app.runtime.hands.values())
    assert all(c.controller_state().payload.value.mass_kg == 0 for c in app.runtime.clients.values())
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
    holds = {a: np.asarray(p.state().q, float) for a, p in probe.items()}
    align, invalidate = supervised_hardware.align_arms, app._invalidate
    injected, cancelled = [], []

    def gap_once(clients, **kwargs):
        if not injected:  # the operator's 20.525493 ms bracket, once
            injected.append(True)
            raise FeedbackGap("paired feedback interpolation gap", arm_id=list(clients)[1], interval_s=.020525493)
        return align(clients, **kwargs)

    def observe_cancel(reason):
        if "feedback interrupted" in reason:
            assert not app.runtime.feedback_ready
            cancelled.append(True)
        invalidate(reason)

    with patch.object(supervised_hardware, "align_arms", gap_once), patch.object(app, "_invalidate", observe_cancel):
        state = wait(lambda s: s["status"] in {"cancelled", "faulted"})
        assert state["status"] == "cancelled", state["error"]
    assert injected and cancelled and app.fault_trace is None
    assert state["pair"]["lifecycle"] == "cancelled" and state["preview_id"] is None
    assert all(a["released_at"] is None for a in state["pair"]["members"])
    wait(lambda s: app.runtime.feedback_ready and app.runtime.fresh)
    for a, p in probe.items():
        assert p.authority().lifecycle is Lifecycle.ARMED
        np.testing.assert_allclose(p.state().q, holds[a], atol=1e-3)  # still holding where it was
    assert not any(h.state().executing for h in app.runtime.handles.values())
    refused("execute", pair_id=idle_pair)
    print("Idle 20.525493 ms feedback gap cancels staged pair, preserves hold, requires new plan: PASS", flush=True)
    action("plan", object_target=goal)
    shared = execute()
    state = app.snapshot()
    error = pose_error(_pose_matrix(state["shared_object"]["values"]), _pose_matrix(goal))
    assert error[0] < .001 and error[1] < np.deg2rad(.5), error
    assert state["shared_object"]["closure_mm"] < 3
    assert app._virtual_grasps() == captured, "captured grasps changed during motion"
    print(f"Virtual 10 mm / 2 degree move ({refinement}): PASS; error {error[0]*1000:.4f} mm / {np.rad2deg(error[1]):.5f} degrees", flush=True)
    moving = [s for s in app._trace.samples if s.get("closure") is not None
              and all(a.get("plan_id") == f"{shared}:{name}" and a.get("trajectory_elapsed_s") is not None
                      for name, a in s["arms"].items())]
    assert moving, ("no moving shared-object samples captured", shared, sorted({
        (n, a.get("plan_id"), a.get("trajectory_elapsed_s") is not None, s.get("closure") is not None)
        for s in app._trace.samples for n, a in s["arms"].items()}, key=str)[-12:])
    # Each supervisor's tick: reference start = its PC time minus the elapsed phase. One start stamp => one start.
    starts = {name: np.median([s["arms"][name]["pc_monotonic"] - s["arms"][name]["trajectory_elapsed_s"] for s in moving])
              for name in moving[0]["arms"]}
    spread = max(starts.values()) - min(starts.values())
    assert spread < .002, starts
    peak = max(s["closure"][0] for s in moving) * 1000
    print(f"20 ms late partner release: one start stamp, reference starts {spread * 1000:.2f} ms apart, "
          f"peak closure {peak:.4f} mm: PASS", flush=True)
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
    # Change only the peer's common-time measurement, not either plant: TCP agreement breaks and both arms must stop.
    peer = list(app.runtime.clients)[1]

    def diverged(clients, **kwargs):
        positions, report = align(clients, **kwargs)
        positions, report = dict(positions), copy.deepcopy(report)
        positions[peer] = positions[peer].copy()
        positions[peer][0] += .05
        bracket = report["brackets"][peer]
        bracket["q"] = list(positions[peer])
        return positions, report

    with patch.object(supervised_hardware, "align_arms", diverged):
        state = wait(lambda s: s["status"] == "faulted")
    assert "TCP grasp agreement" in state["error"], state["error"]
    deadline = time.monotonic() + 3
    while any(p.authority().lifecycle is Lifecycle.ARMED for p in probe.values()) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not any(p.authority().lifecycle is Lifecycle.ARMED for p in probe.values())
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
    # CTRL-041: the trace is schema v2 and carries the five closure quantities separately.
    assert trace["schema"] == "arm_control.dual_fault/v2" and trace["schema_version"] == 2
    assert trace["closure_schema"] == "arm_control.closure/v1"
    record = last["closure_v1"]
    assert record["schema"] == "arm_control.closure/v1" and record["window"] in ("motion", "stationary")
    inferred_record = record["common_time_inferred"]["value"]
    np.testing.assert_allclose([inferred_record["distance_m"], inferred_record["angle_rad"]], measured, atol=1e-12)
    assert record["common_time_inferred"]["basis"]["brackets"].keys() == alignment["brackets"].keys()
    assert record["tracking"]["value"].keys() == last["arms"].keys()
    for arm, view in last["arms"].items():  # values, not just keys: max |q_rt - command| per arm
        expected_rt = max(abs(a - b) for a, b in zip(view["q_rt"], view["q_cmd_rt"]))
        expected_pc = max(abs(a - b) for a, b in zip(view["q_rt"], view["q_cmd_pc"]))
        assert abs(record["tracking"]["value"][arm]["vs_rt_command_rad"] - expected_rt) < 1e-12, arm
        assert abs(record["tracking"]["value"][arm]["vs_pc_command_rad"] - expected_pc) < 1e-12, arm
    assert (record["window"] == "motion") == (record["stationary_for_s"] is None)
    assert record["planned_reference"]["basis"]["plan_ids"] == last["pair_plans"] if record["planned_reference"]["value"] else True
    assert trace["closure_assumptions"] and "assumptions" not in record
    # every recorded sample carries a closure_v1 (including any recorded on an alignment fault)
    assert all("closure_v1" in s for s in trace["samples"])
    assert record["rt_selected_reference"]["value"] is None and record["rt_selected_reference"]["absent"]
    assert record["physical"]["value"] is None and record["physical"]["absent"]
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
    print("Virtual release freezes pose; TCP divergence stops both arms through their supervisors: PASS", flush=True)


def start_fakes(pair_path, binary, root):
    """Stage the pair for the bench, move each arm's RT endpoint to a local fake started at the arm's scene pose; return
    ``(pair config path, {arm: supervisor config}, fake servers, logs)``."""
    staged, configs = stage_pair(pair_path, plant="rt")
    raw = yaml.safe_load(staged.read_text())
    poses = {a.name: a.q for a in load_pair_config(staged).scene.actors}
    servers, logs, paths = [], [], {}
    for name, source in configs.items():
        cfg = yaml.safe_load(source.read_text())
        udp, tcp = free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM)
        cfg["rt"] = dict(cfg["rt"], host="127.0.0.1", udp_port=udp, tcp_port=tcp)  # ONLY the RT endpoint moves
        paths[name] = Path(root) / f"{name}.yaml"
        paths[name].write_text(yaml.safe_dump(cfg, sort_keys=False))
        raw["arms"][name]["config"] = str(paths[name])
        joints = arm_joints(cfg)
        log = Path(root) / f"{name}.log"
        logs.append(log)
        with log.open("w") as output:
            servers.append(subprocess.Popen(
                [binary, "--backend", "fake", "--n", str(len(joints)), "--fake-q0", ",".join(map(repr, poses[name])),
                 "--instance-id", cfg["rt"]["instance_id"], "--joint-names", ",".join(joints),
                 "--bind", "127.0.0.1", "--udp-port", str(udp), "--tcp-port", str(tcp)],
                stdout=output, stderr=subprocess.STDOUT))
    pair = Path(root) / "pair.yaml"
    pair.write_text(yaml.safe_dump(raw, sort_keys=False))
    time.sleep(.3)
    return pair, paths, servers, logs


def main(pair_path, binary, refinement="none"):
    # the staged pair's instance ids are the real arms': its supervisors' fault traces go to a scratch root, never theirs
    os.environ.setdefault(diagnostics.FAULT_ROOT_ENV, tempfile.mkdtemp(prefix="dual_hardware_faults_"))
    trace = diagnostics.PairFaultTrace("self-check")
    for i in range(510):
        trace.record(dict(index=i, value=float("nan")))
    assert len(trace.samples) == 500 and trace.samples[0]["index"] == 10
    with tempfile.TemporaryDirectory() as trace_root, patch.object(diagnostics, "CONTROL_ROOT", Path(trace_root)):
        report = json.loads(trace.save("test", {}).read_text())
        assert report["samples"][-1]["value"] is None
    config = load_pair_config(pair_path)
    assert config.backend == "remote_rt"
    if config.shared_object is not None:  # a shared object must be stated virtual: a physical one is refused at load
        raw = yaml.safe_load(Path(pair_path).read_text())
        for value in (False, None):
            obj = dict(raw["shared_object"])
            obj.pop("virtual") if value is None else obj.update(virtual=value)
            probe_path = Path(pair_path).with_name(f".virtual-probe-{os.getpid()}.yaml")  # beside it: relative paths hold
            probe_path.write_text(yaml.safe_dump(dict(raw, shared_object=obj)))
            try:
                load_pair_config(probe_path)
            except ValueError as exc:
                assert "virtual must be true" in str(exc), exc
            else:
                raise AssertionError(f"shared_object.virtual={value} was admitted")
            finally:
                probe_path.unlink()
        print("A shared object not stated virtual is refused at load: PASS", flush=True)
    app = None
    supervisors, servers, probe = [], [], {}
    with tempfile.TemporaryDirectory(prefix="dual-hardware-check-") as root, patch.object(
        diagnostics, "CONTROL_ROOT", Path(root)
    ):
        logs = []
        try:
            pair, configs, servers, logs = start_fakes(pair_path, binary, root)
            env, sockets = dict(os.environ), {}
            for name, path in configs.items():
                proc, out = start_supervisor(path, env, plant="rt")
                supervisors.append(proc)
                sockets[name] = out["ARM_CONTROL_SOCKET"]
            env[SOCKETS_ENV] = ",".join(f"{k}={v}" for k, v in sockets.items())
            config = load_pair_config(pair)
            probe = {a: ArmClient(config.robots[a].get("rt")["instance_id"], socket_path=s) for a, s in sockets.items()}
            runtimes = []

            def runtime(cfg):
                rt = SupervisedHardware(cfg, env=env, hand_factory=FakeHand)
                runtimes.append(rt)
                return rt

            def unclaimed():
                end = time.monotonic() + 3  # a closed session's claim is released by its supervisor
                while time.monotonic() < end:
                    if all(p.authority().lifecycle is Lifecycle.CONNECTED for p in probe.values()):
                        return True
                    time.sleep(.02)
                return False

            with patch("arm_control.ui.dual_arm_console._visuals", side_effect=ValueError("visual failure")):
                try:
                    DualArmConsole(pair, start_server=False, runtime_factory=runtime)
                except ValueError as exc:
                    assert str(exc) == "visual failure"
                else:
                    raise AssertionError("visual failure was not raised")
            assert runtimes[-1]._closed and not runtimes[-1].hands and unclaimed()
            print("Visual setup failure claims no arm and opens no Hand: PASS", flush=True)
            with patch("arm_control.ui.dual_arm_console.ConsoleServer", side_effect=ValueError("server failure")):
                try:
                    DualArmConsole(pair, runtime_factory=runtime)
                except ValueError as exc:
                    assert str(exc) == "server failure"
                else:
                    raise AssertionError("server failure was not raised")
            assert runtimes[-1]._closed and len(runtimes[-1].hands) == 2 and unclaimed()
            print("Post-connection startup failure releases both claims: PASS", flush=True)
            original_visuals = dual_arm_console._visuals
            original_meshes = rerun_scene.static_mesh_geoms

            def before_claim():
                assert all(p.authority().lifecycle is Lifecycle.CONNECTED for p in probe.values())
                assert not runtimes[-1].hands

            def stalled_visuals(config):
                before_claim()
                # PyDLL retains the GIL: native mesh loading longer than the RT deadline, not time.sleep.
                assert ctypes.PyDLL(None).usleep(1_600_000) == 0
                return original_visuals(config)

            def meshes_before_claim(*args, **kwargs):
                before_claim()
                return original_meshes(*args, **kwargs)

            with patch.object(dual_arm_console, "_visuals", stalled_visuals), patch.object(
                rerun_scene, "static_mesh_geoms", meshes_before_claim
            ):
                app = DualArmConsole(pair, start_server=False, runtime_factory=runtime)
            sessions = {a: c.session for a, c in app.runtime.clients.items()}
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                assert app.runtime.fresh
                assert all(p.authority().lifecycle is Lifecycle.CLAIMED for p in probe.values())
                time.sleep(.05)
            assert sessions == {a: c.session for a, c in app.runtime.clients.items()}
            np.testing.assert_allclose(
                app.runtime.world.data.qpos[app.runtime.world._qadr],
                app.runtime.joint_state()["position"],
            )
            for a, cfg in config.robots.items():
                for j in gripper_joints(cfg):
                    adr = app.runtime.world.model.joint(f"{a}__{j}").qposadr[0]
                    assert app.runtime.world.data.qpos[adr] == app.runtime.fingers[a]
            print("GIL-blocking asset setup precedes the claims; claimed, disarmed sessions survive: PASS", flush=True)

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
            assert state["backend"] == "remote_rt" and state["connected"]  # the page's REAL HARDWARE banner keys on it
            assert not any(a["armed"] for a in state["arms"].values())
            assert (state["shared_object"] is None) == (config.shared_object is None)
            try:
                action("set_mode", mode="shared", grasps_confirmed=True)
            except ValueError:
                pass
            else:
                raise AssertionError("shared-object mode admitted without the empty-Hand confirmation")
            a = next(iter(state["arms"]))
            # A Hand command needs the handle's readiness, not only an armed arm.
            handle = app.runtime.handles[a]
            real_state = handle.state
            with patch.object(handle, "state", lambda: replace(real_state(), armed=True, ready=False)):
                try:
                    action("gripper", arm_id=a, width_mm=40)
                except ValueError as exc:
                    assert "arm the controller" in str(exc), exc
                else:
                    raise AssertionError("Hand commanded while the handle was not ready")
            inner_state = handle._inner.state
            with patch.object(handle._inner, "state", lambda: replace(inner_state(), position=None, velocity=None)):
                wait(lambda s: not s["connected"])
                with app._lock:
                    assert not app.runtime.diagnostic_sample()["arms"][a]["measurement_valid"]
            wait(lambda s: s["connected"])
            print("Invalid feedback refuses freshness; valid measurements recover: PASS", flush=True)
            # Hold an ARM in flight: repeated UI requests must not accumulate enables behind a disarm.
            entered, release = threading.Event(), threading.Event()
            client = handle._inner._arm
            enable = client.arm

            def delayed_enable(*args, **kwargs):
                entered.set()
                assert release.wait(3)
                return enable(*args, **kwargs)

            with patch.object(client, "arm", delayed_enable):
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
                assert len(handle._pending) <= 2
                release.set()
                wait(lambda s: not handle._pending and not s["arms"][a]["armed"])
            assert probe[a].authority().lifecycle is not Lifecycle.ARMED
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
            pair_id = state["preview_id"]
            action("stage", pair_id=pair_id, reviewed=True)
            wait(lambda s: s["pair"]["lifecycle"] == "ready")
            released_together(app, action, pair_id)
            state = wait(lambda s: s["pair"]["lifecycle"] in {"complete", "faulted", "cancelled"})
            assert state["pair"]["lifecycle"] == "complete", state
            assert all(np.max(np.abs(np.array(s["q_deg"]) - targets[a]["values"])) < .7
                       for a, s in state["arms"].items())
            with app._lock:
                sample = app.runtime.diagnostic_sample()["arms"][a]
            for key in ("tau_cmd_rt", "tau_meas_rt", "tau_ff_pc", "raw_pc"):  # trace/1: the supervisor's controller tick
                assert sample[key] is not None and np.isfinite(np.asarray(sample[key])).all(), key
            # CTRL-051: reconstruct the executed chain from the bounded trace the console kept -- detached, read-only
            # analysis of the recorded samples. A lineage report is never an executable trajectory.
            lineage = diagnostics.reconstruct_lineage(
                list(app._trace.samples), pair_id=state["pair"]["pair_id"])
            assert lineage["complete"], lineage["missing"]
            assert lineage["ticks"] > 0
            assert set(lineage["links"]) == set(diagnostics.LINE_LINKS)
            assert lineage["declared_plans"] and not lineage["foreign_arms"], lineage
            # Field presence is not correlation: a controller holding a foreign plan, or a pair that never reached an
            # admitted phase, must NOT reconstruct as a complete chain.
            foreign_samples = [
                dict(s, arms={a: dict(v, plan_id="some-other-run")
                              for a, v in (s.get("arms") or {}).items()})
                for s in app._trace.samples
            ]
            foreign = diagnostics.reconstruct_lineage(
                foreign_samples, pair_id=state["pair"]["pair_id"])
            assert not foreign["complete"] and "reference" in foreign["missing"], foreign
            assert foreign["foreign_arms"], foreign
            never_admitted = diagnostics.reconstruct_lineage(
                [dict(s, lifecycle="staging") for s in app._trace.samples],
                pair_id=state["pair"]["pair_id"])
            assert not never_admitted["complete"], never_admitted
            assert "admitted" in never_admitted["missing"], never_admitted
            # A command link must come from a JOINED arm: strip the joined arm's command data and let the foreign-plan
            # arm keep its own -- the chain must stay incomplete, not borrow the foreign numbers.
            declared = lineage["declared_plans"]
            joined_arm, other_arm = sorted(declared)
            command_only_foreign = [
                dict(s, arms={
                    joined_arm: dict((s.get("arms") or {}).get(joined_arm) or {},
                                     plan_id=declared[joined_arm],
                                     **{f: None for _, f in diagnostics.CHAIN_FIELDS}),
                    other_arm: dict((s.get("arms") or {}).get(other_arm) or {},
                                    plan_id="some-other-run"),
                })
                for s in app._trace.samples
            ]
            borrowed = diagnostics.reconstruct_lineage(
                command_only_foreign, pair_id=state["pair"]["pair_id"])
            assert not borrowed["complete"], borrowed
            assert "raw" in borrowed["missing"] and "sent" in borrowed["missing"], borrowed
            assert borrowed["joined_arms"] == [joined_arm], borrowed
            # One of two declared members joining is not the PAIR's reference.
            half_joined = [
                dict(s, arms={
                    arm: dict(view, plan_id=(declared[arm] if arm == joined_arm else None))
                    for arm, view in (s.get("arms") or {}).items()
                })
                for s in app._trace.samples
            ]
            partial = diagnostics.reconstruct_lineage(
                half_joined, pair_id=state["pair"]["pair_id"])
            assert not partial["complete"], partial
            assert "reference" in partial["missing"], partial
            assert partial["joined_arms"] == [joined_arm], partial
            print("Execution lineage reconstructs from the bounded trace: PASS", flush=True)
            print("Two supervisors: disarmed startup, measured planning, one-stamp paired execution: PASS", flush=True)
            if config.shared_object is not None:
                check_virtual(app, action, wait, refinement, probe)
                return
            # Kill one RT server while a plan is staged: the peer must be cancelled and stale feedback refuse new motion.
            action("plan", targets=targets, active_arms=list(targets))
            state = wait(lambda s: s["status"] in {"preview", "failed"})
            assert state["status"] == "preview", state["error"]
            pair_id = state["preview_id"]
            action("stage", pair_id=pair_id, reviewed=True)
            wait(lambda s: s["pair"]["lifecycle"] == "ready")
            servers[0].terminate()
            servers[0].wait(timeout=3)
            wait(lambda s: not s["connected"] and s["pair"]["lifecycle"] in {"cancelled", "faulted"})
            wait(lambda s: not s["pair"]["cancellation_pending"])
            print("Lost RT peer cancels pair: PASS", flush=True)
            app._stop.set()
            app._thread.join(timeout=2)
            assert not app._thread.is_alive()
            app.runtime.stop()
            survivor = probe[config.scene.actors[1].name]
            deadline = time.monotonic() + 3
            while survivor.authority().lifecycle is Lifecycle.ARMED and time.monotonic() < deadline:
                time.sleep(.01)
            assert survivor.authority().lifecycle is not Lifecycle.ARMED
            print("Runtime stop disarms the surviving arm without further ticks: PASS", flush=True)
        except BaseException:
            for log in logs:
                print(log.read_text()[-5000:])
            for proc in supervisors:
                print(Path(proc._workdir, "supervisor.log").read_text()[-5000:])
            raise
        finally:
            if app:
                app.close()
            for p in probe.values():
                p.close()
            for proc in supervisors:
                stop_supervisor(proc)
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
