#!/usr/bin/env python3
"""One hosted world, two supervised actors (scene-hosting D2(a), operator 2026-10-09: the FR3 and the base in one sim world).

    PYTHONPATH=. python tools/check/check_scene_cohost.py

The toy two-actor world of check_sim_supervisor. In process: the ``arm`` link hosts the world (``scene_world``), the ``peer``
link is a co-host (``SimLink.from_config(cfg, cohost_of=...)``) built from a body-list arm (``arm.body_list``, no URDF); both
ARM and track their own targets in the one world, a scene change pauses the world's clock (and is not staleness), the co-host
never steps it (its close leaves the host stepping), and a co-host config without ``sim_actor`` is refused. Over the service (``--cohost``, via ``launch.start_supervisor(cohosts=...)``):
two sockets, two claims, two ARMs, a move COMPLETED on each, and the env names the co-host's config and socket. scene_action/1
on the host only: one lease (another session refused until the holder's connection ends), a stale revision refused, a newer
one applied and returned.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sdk_supervisor import write_config  # noqa: E402
from check_sim_supervisor import SERVO, command, wait, write_scene  # noqa: E402

from arm_control.config import load_robot_config  # noqa: E402
from arm_control.execution.contract import Phase, TrajectoryRequest  # noqa: E402
from arm_control.motion import JointTrajectory  # noqa: E402
from arm_control.plants.sim_link import SimLink  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SERVO_YAML = "sim_servo: {" + ", ".join(f"{k}: {v}" for k, v in SERVO.items()) + "}\nsim_observe: {rate_hz: 50, ring_s: 1}\n"
# The peer's one link as numbers: a hinge about y carrying the capsule (MuJoCo's default density) centred on the joint.
PEER_BODIES = """  body_list:
    joints: [j1]
    inactive_joints: []
    gravity: [0, 0, -9.81]
    joint_limits: {lower: [-3.0], upper: [3.0]}
    bodies:
      - {parent: 0, pos: [1, 0, 0.1], quat: [1, 0, 0, 0], mass: 0.16, com: [0, 0, 0], inertia: [2e-4, 2e-4, 3e-5],
         inertia_quat: [1, 0, 0, 0], joint: {name: j1, kind: 3, axis: [0, 1, 0], pos: [0, 0, 0]}}
"""


def configs(tmp):
    host = tmp / "host.yaml"
    # The toy MJCF's capsules are not three.urdf's links: the URDF gravity leaves a ~0.02 rad sag at 0.2 rad, so this arm's
    # completion tolerance is wider than the fixture's (a fixture mismatch, not the co-hosting under test).
    host.write_text(write_config(tmp, 0, 0).read_text().replace("  native_reference: true\n", "")
                    .replace("  done_pos_tol: 0.02\n", "  done_pos_tol: 0.05\n")
                    + SERVO_YAML + "sim_actor: arm\nidle_timeout_sec: 0.1\n")
    peer = tmp / "peer.yaml"
    peer.write_text(f"""arm:
  joints: [j1]
{PEER_BODIES}  kp: [100]
  kd: [20]
  max_tau: [50]
  done_pos_tol: 0.02
  done_vel_tol: 0.05
  settle_timeout_s: 3.0
  settle_dwell_s: 0.1
num_motors: 1
motor_names: [j1]
joint_names: [j1]
rt: {{host: 127.0.0.1, udp_port: 0, tcp_port: 0, instance_id: peer_arm, state_timeout_s: 0.5}}
execution_policy:
  command_rate_hz: 100
  state_timeout_sec: 0.25
  health_timeout_sec: 1.0
  start_pos_tol_rad: 0.1
  abort_pos_err_rad: 0.35
  hold_relatch_rad: 0.3
  velocity_limits: [1.0]
  acceleration_limits: [2.0]
  cartesian_kp_min: 20.0
{SERVO_YAML}sim_actor: peer
""")
    return host, peer


def pause(shared, seconds: float) -> None:
    """A scene change that holds the world for ``seconds`` (the recompile, without a scene), crediting the stall to the slice
    deadmen as SceneCore.apply_scene does."""
    shared.core.apply_scene = lambda request: (time.sleep(seconds), shared.core.credit_stall(seconds),
                                               {"revision": request["revision"]})[2]
    try:
        shared.apply_scene(dict(revision=1, state={}))
    finally:
        del shared.core.apply_scene  # back to the class's


def peer_command(q):
    return dict(position=np.asarray(q, float), velocity=np.zeros(1), torque=np.zeros(1), kp=np.full(1, 100.),
                kd=np.full(1, 20.))


def in_process(host_path, peer_path) -> None:
    import threading

    from arm_control.plants.remote_rt import protocol as rtp

    host = SimLink.from_config(load_robot_config(host_path))
    host.open()
    peer = None
    try:
        shared = host.scene_world
        assert shared is not None, "the hosting link exposes no world"
        unnamed = peer_path.with_name("unnamed.yaml")
        unnamed.write_text(peer_path.read_text().replace("sim_actor: peer\n", ""))
        try:
            SimLink.from_config(load_robot_config(unnamed), cohost_of=shared)
        except ValueError as exc:
            assert "sim_actor" in str(exc), exc
        else:
            raise AssertionError("a co-host without sim_actor")
        core = shared.core
        span = slice(core.arm_slices["peer"]["start"], core.arm_slices["peer"]["start"] + 1)
        spawn_kp = core.command["kp"][span].copy()
        peer = SimLink.from_config(load_robot_config(peer_path), cohost_of=shared)
        assert peer.scene_world is None, "a co-host must not expose the world it does not host"
        peer.open()
        # disarmed and never driven, a co-hosted actor keeps the world's spawn hold (its release word is not written)
        time.sleep(.3)
        assert np.all(spawn_kp > 0) and np.array_equal(core.command["kp"][span], spawn_kp), core.command["kp"][span]
        host.enable_all()
        peer.enable_all()
        a, b = np.array([.3, -.2, .1]), np.array([.6])
        # the supervisors' command streams, kept up through every step below (a pause must not starve the deadman)
        driven, stop_driving = {host: command(a), peer: peer_command(b)}, threading.Event()

        def drive():
            while not stop_driving.is_set():
                for link, word in list(driven.items()):
                    link.apply_command(word)
                time.sleep(.01)
        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        time.sleep(2.0)
        hs, ps = host.latest_state()[0], peer.latest_state()[0]
        assert hs.armed and not hs.faulted and np.allclose(hs.q, a, atol=.03), hs.q
        assert ps.armed and not ps.faulted and np.allclose(ps.q, b, atol=.03), ps.q
        world = host.observations(0)["samples"][-1]["actor_q"]
        assert abs(list(world["peer"].values())[0] - .6) < .03, world
        # A scene change pauses the world: its clock stands still and its links are not stale meanwhile (a 0.6 s pause, past
        # the links' 0.5 s state timeout), and afterwards the clock has lost exactly the pause against the wall.
        wall0, clock0 = time.monotonic(), shared.clock()
        change = threading.Thread(target=pause, args=(shared, .6))
        change.start()
        time.sleep(.3)
        frozen = shared.clock()
        time.sleep(.2)
        assert abs(shared.clock() - frozen) < 1e-9 and peer.clock() == shared.clock(), "the world clock ran during a pause"
        time.sleep(.2)
        assert host.motor_health()["state_fresh"] and peer.motor_health()["state_fresh"], "a paused world reported stale"
        change.join()
        lost = (time.monotonic() - wall0) - (shared.clock() - clock0)
        assert abs(lost - .6) < .05, lost
        # ... and no actor's slice deadman fires across pauses while its supervisor keeps commanding (review: a co-host
        # that waited out the pause wrote a pre-pause time and was zeroed in 5 of 10 pauses)
        zeroed = []
        core._log = lambda text, **kw: zeroed.append(text)
        changes = threading.Thread(target=lambda: [pause(shared, .4) for _ in range(6)])
        changes.start()
        changes.join()
        core._log = print
        assert not zeroed, zeroed
        assert not peer.latest_state()[0].faulted and not host.latest_state()[0].faulted, (peer.motor_health()["latched_fault"], host.motor_health()["latched_fault"])
        sim_t = shared.backend.data.time
        driven.pop(peer)
        peer.close()
        peer = None
        wait(lambda: shared.backend.data.time > sim_t + .1, 2., "the host stepping after the co-host closed")
        # a world the host no longer steps is a failed plant for its co-host too (review: it stayed fresh and ARMed)
        peer = SimLink.from_config(load_robot_config(peer_path), cohost_of=shared)
        peer.open()
        peer.enable_all()
        driven[peer] = peer_command(b)
        time.sleep(.3)
        core.tick = lambda now: (_ for _ in ()).throw(RuntimeError("world step failed"))
        wait(lambda: host.latest_state()[0].fault_code == rtp.FAULT_PLANT, 2., "the host's FAULT_PLANT")
        wait(lambda: peer.latest_state()[0].fault_code == rtp.FAULT_PLANT, 2., "the co-host's FAULT_PLANT")
        del core.tick
    finally:
        if "stop_driving" in locals():
            stop_driving.set()
            driver.join()
        if peer is not None:
            peer.close()
        host.close()
    print("co-host in process: two links, one world, each tracks its own slice; a co-host keeps its spawn hold until driven; a "
          "scene change pauses the world clock and no slice deadman; a stopped world faults its co-host; the co-host never "
          "steps or closes it: PASS", flush=True)


def stale_after_pause(host_path) -> None:
    """The supervisor's controller on a world clock: after a pause a fresh sample is fresh, and a plant that then dies is
    caught within the policy's state timeout (review MUST: wall-time arrival stamps on a paused clock hid it)."""
    from arm_control.supervisor.runtime import RobotRuntime

    cfg = load_robot_config(host_path)
    host = SimLink.from_config(cfg)
    runtime = RobotRuntime(cfg, link=host)
    runtime.start()
    try:
        c = runtime.controller
        wait(lambda: c._last_state_at is not None, 5., "a fed state")
        pause(host.scene_world, .6)
        time.sleep(.1)
        assert c._observation_error() is None, c._observation_error()
        timeout = c.execution_policy.state_timeout_sec
        host._stop.set()  # the servo dies: no sample after this
        host._thread.join(timeout=2.)
        died = time.monotonic()
        wait(lambda: c._observation_error() is not None, timeout + .5, "the stale-state error")
        assert time.monotonic() - died <= timeout + .1, (time.monotonic() - died, timeout)
    finally:
        runtime.close()
    print("world clock: fresh after a pause, a dead plant caught within the state timeout: PASS", flush=True)


def supervised(host_path, peer_path) -> None:
    from arm_control.sdk import ArmClient
    from arm_control.supervisor.launch import start_supervisor, stop_supervisor

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    proc, graph_env = start_supervisor(host_path, env, cwd=ROOT, cohosts={"PEER": peer_path}, timeout_s=120.)
    clients = []
    try:
        assert graph_env["PEER_CONFIG"] == str(peer_path) and graph_env["PEER_SOCKET"], graph_env
        arm = ArmClient("sdk_arm", socket_path=graph_env["ARM_CONTROL_SOCKET"])
        peer = ArmClient("peer_arm", socket_path=graph_env["PEER_SOCKET"])
        clients += [arm, peer]
        assert arm.extensions.get("scene_action") == 1 and "scene_action" not in peer.extensions, (arm.extensions,
                                                                                                  peer.extensions)
        writer, rival = ArmClient("sdk_arm", socket_path=graph_env["ARM_CONTROL_SOCKET"]), \
            ArmClient("sdk_arm", socket_path=graph_env["ARM_CONTROL_SOCKET"])
        clients += [rival]
        assert writer.scene_claim()
        refused(lambda: rival.scene_claim(), "lease")
        refused(lambda: rival.scene_apply("r0", 1, {}), "lease")
        state = arm.observe()["samples"][-1]
        revision = state["scene_revision"]
        refused(lambda: writer.scene_apply("stale", revision, {}), "stale")
        current = writer.scene_apply("same", revision + 1, {})
        assert current["request_id"] == "same" and current["scene_state"]["revision"] == revision + 1, current
        writer.close()
        wait(lambda: _claims(rival), 2., "the lease released when its holder disconnected")
        for client in (arm, peer):
            client.claim()
            client.arm()
        runs = []
        for client, instance, joints, delta in ((arm, "sdk_arm", ("j1", "j2", "j3"), .2), (peer, "peer_arm", ("j1",), .3)):
            q0 = np.asarray(client.state().q)
            q1 = q0 + delta
            traj = JointTrajectory([0., 1.5], np.stack([q0, q1]), np.zeros((2, len(q0))))
            status = client.execute(TrajectoryRequest(f"cohost_{instance}", instance, joints, traj, model_token=None))
            runs.append((client, status.execution_id, q1))
        for client, eid, q1 in runs:
            wait(lambda: client.status(eid).phase is Phase.COMPLETED, 10., f"{eid} COMPLETED")
            assert np.allclose(client.state().q, q1, atol=.05), (eid, client.state().q, q1)
        for client in (arm, peer):
            client.disarm()
    finally:
        for client in clients:
            client.close()
        stop_supervisor(proc)
    print("co-host service: two sockets in one process, each claimed, ARMed and moved to COMPLETED: PASS", flush=True)


def refused(call, words) -> None:
    from arm_control.sdk import Refused

    try:
        call()
    except Refused as exc:
        assert words in str(exc), (words, exc)
    else:
        raise AssertionError(f"accepted: {words}")


def _claims(client) -> bool:
    from arm_control.sdk import Refused

    try:
        return client.scene_claim()
    except Refused:
        return False


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="scene_cohost_"))
    os.environ["WORKCELL_SCENE"] = str(write_scene(tmp))
    host_path, peer_path = configs(tmp)
    in_process(host_path, peer_path)
    stale_after_pause(host_path)
    supervised(host_path, peer_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
