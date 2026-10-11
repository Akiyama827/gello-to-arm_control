"""Two fake RT instances at once: isolation, identity routing and fault containment.

Spawns two ``arm_rt_server --backend fake`` processes that stand in for two
7-joint arms (same joint names, different instance ids and ports) and checks:

* each commander tracks its own target through its own server simultaneously;
* a config for one instance pointed at the other's ports is refused before any
  UDP is sent (the swapped-cable/swapped-config case);
* an observer of one instance never receives the other's states;
* one instance losing its command stream latches only that instance;
* each confirmed stop matches its own stop epoch.

Fake backends only: this says nothing about FCI timing, CPU contention or a
real robot. No hardware is contacted.

    PYTHONPATH=. python tools/check/rt/check_two_instances.py path/to/arm_rt_server
"""
from __future__ import annotations

# ruff: noqa: E402  (path bootstrap must precede the arm_control import)

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import socket
import subprocess
import time

import numpy as np

from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.client import RtBackend, RtConfig, RtLinkError, RtObserver

JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
PORTS: dict[str, tuple[int, int]] = {}  # ephemeral, chosen per run (fixed ports raced other checks)


def free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn(binary: str, instance: str) -> subprocess.Popen:
    PORTS[instance] = (free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM))
    udp, tcp = PORTS[instance]
    return subprocess.Popen(
        [binary, "--backend", "fake", "--n", "7", "--instance-id", instance,
         "--joint-names", ",".join(JOINTS), "--udp-port", str(udp), "--tcp-port", str(tcp),
         "--bind", "127.0.0.1", "--hold-ms", "100", "--fault-ms", "600", "--slew", "5"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def config(instance: str, ports_of: str) -> RtConfig:
    udp, tcp = PORTS[ports_of]
    return RtConfig(host="127.0.0.1", udp_port=udp, tcp_port=tcp, instance_id=instance)


def stream(arms: dict[str, RtBackend], targets: dict[str, np.ndarray], seconds: float) -> None:
    kp, kd = np.full(7, 60.0), np.full(7, 10.0)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for name, arm in arms.items():
            arm.apply_command({"position": targets[name], "velocity": np.zeros(7),
                               "torque": np.zeros(7), "kp": kp, "kd": kd})
        time.sleep(0.01)


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "rt/build/arm_rt_server")
    servers = [spawn(binary, name) for name in ("fr3_1", "fr3_2")]
    arms: dict[str, RtBackend] = {}
    try:
        for name in PORTS:  # readiness, not a fixed sleep: wait until each server accepts
            deadline = time.monotonic() + 10
            while True:
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    probe.connect(("127.0.0.1", PORTS[name][1]))
                    break
                except OSError:
                    server = servers[list(PORTS).index(name)]
                    assert server.poll() is None, f"{name} exited: {server.stdout.read()}"
                    assert time.monotonic() < deadline, f"{name} did not come up"
                    time.sleep(0.05)
                finally:
                    probe.close()
        # Swapped configuration: the fr3_1 config aimed at fr3_2's ports.
        try:
            RtBackend(config("fr3_1", "fr3_2"), JOINTS).open()
        except RtLinkError as exc:
            assert "server is instance 'fr3_2'" in str(exc), exc
        else:
            raise AssertionError("swapped instance admitted")
        for name in PORTS:
            arms[name] = RtBackend(config(name, name), JOINTS)
            arms[name].open()
            arms[name].enable_all()
        assert arms["fr3_1"].session_id != arms["fr3_2"].session_id
        targets = {"fr3_1": np.linspace(0.1, 0.4, 7), "fr3_2": np.linspace(-0.4, -0.1, 7)}
        watcher = RtObserver("127.0.0.1", PORTS["fr3_2"][0], "fr3_2", JOINTS, lease_ms=500)
        watcher.open()
        stream(arms, targets, 1.5)
        for name, arm in arms.items():
            state, _ = arm.latest_state()
            err = float(np.max(np.abs(np.asarray(state.q) - targets[name])))
            assert state.armed and not state.holding and err < 0.05, (name, err)
        seen, _ = watcher.latest_state()
        assert seen is not None and watcher.identity_mismatches == 0
        assert np.allclose(seen.q, arms["fr3_2"].latest_state()[0].q, atol=0.01)
        assert seen.identity_hash == rtp.identity_hash("fr3_2", rtp.joint_order_hash(JOINTS))
        watcher.close()

        # fr3_1's commander goes quiet; only fr3_1 may latch.
        stream({"fr3_2": arms["fr3_2"]}, targets, 1.0)
        one, _ = arms["fr3_1"].latest_state()
        two, _ = arms["fr3_2"].latest_state()
        assert one.faulted and one.fault_code == rtp.FAULT_CMD_LOST, one
        assert two.armed and not two.faulted and not two.holding, two

        for name, arm in arms.items():
            assert arm.safe_stop(), f"{name}: stop not confirmed"
            state, _ = arm.latest_state()
            assert not state.armed and state.stop_epoch >= 1
        print("two RT instances: PASS (simultaneous tracking, swapped-instance refusal, "
              "identity-filtered observer, per-instance fault latch, per-instance stop epoch)")
        return 0
    finally:
        for arm in arms.values():
            arm.close()
        for server in servers:
            server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
