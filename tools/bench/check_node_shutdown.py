"""Assert clean signal exits and CSV flushes without Dora or hardware services.

Run from the repository root: PYTHONPATH=. python tools/bench/check_node_shutdown.py
"""
from __future__ import annotations

import csv
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace


def child(kind: str, stop: str, directory: Path) -> None:
    import numpy as np

    from arm_control.messages import pack_motor_state

    class Node:
        polls = 0

        def __iter__(self):
            return self

        def __next__(self):
            while (event := self.next(timeout=0.05)) is None:
                pass
            return event

        def next(self, *, timeout):
            assert 0 < timeout <= 0.2, timeout
            self.polls += 1
            if self.polls == 1:
                z = np.zeros(7)
                return {"type": "INPUT", "id": "motor_state",
                        "value": pack_motor_state(z, z, z, z, z, z, z, z)}
            if self.polls == 2:
                if stop == "STOP":
                    return {"type": "STOP"}
                if stop == "ERROR":
                    raise RuntimeError("deliberate node failure")
                signal.raise_signal(getattr(signal, stop))
            assert self.polls < 4, "signal did not stop the event loop"
            return None

        def send_output(self, topic, value):
            assert topic != "motor_command", "disarmed shutdown emitted motion"

    if kind == "logger":
        from arm_control.viz import motor_state_logger as adapter

        adapter._load_cfg = lambda: {"num_motors": 7,
                                    "motor_log_path": str(directory / "state.csv")}
    elif kind == "replay":
        from arm_control.control import replay_adapter as adapter

        names = [f"joint_{i}" for i in range(7)]
        # A REAL RobotConfig, not a hand-written imitation of one. The
        # SimpleNamespace that used to stand here reproduced attribute access
        # and not `.get()`, so replay's own config reads blew up inside the
        # fixture rather than in the code under test -- and the fixture was
        # the only thing claiming this node boots.
        #
        # With `_load_mode_config` returning {} (no mode profile configured,
        # which is legal), this is also the regression test for replay booting
        # from ROBOT configuration alone: a reusable package must not demand a
        # mode-profile file for a limit its caller already supplies.
        from arm_control.config import RobotConfig

        cfg = RobotConfig.from_mapping({
            "arm": {"name": "shutdown-check", "joint_names": names,
                    "motor_names": names},
            "execution_policy": {"acceleration_limits": [1.0] * 7},
        })
        adapter.load_robot_config = lambda: cfg
        adapter.arm_joints = lambda cfg: names
        adapter._load_mode_config = lambda: {}
        adapter.resolve_gains = lambda *args: {"kp": np.ones(7), "kd": np.ones(7)}
        adapter.GO_FILE = directory / "go"
        adapter.STOP_FILE = directory / "stop"
    elif kind == "hand":
        import socket

        # The Hand adapter imports Dora only at the process seam now.
        sys.modules["dora"] = SimpleNamespace(Node=Node)
        from arm_control.end_effectors import franka_adapter as adapter

        closed = threading.Event()
        connected = threading.Event()

        class Socket:
            def settimeout(self, timeout):
                connected.set()

            def recv(self, size):
                threading.Event().wait(0.01)
                raise socket.timeout()

            def close(self):
                closed.set()

        adapter.socket.create_connection = lambda *args, **kwargs: Socket()
        adapter.load_robot_config = lambda: {}
        adapter.HOME_FILE = directory / "home"
        client = adapter.BridgeClient("127.0.0.1", 1)
        adapter.BridgeClient = lambda *args: client
        original_next = Node.next

        def next_connected(self, *, timeout):
            assert connected.wait(2), "bridge thread did not start"
            return original_next(self, timeout=timeout)

        Node.next = next_connected
    else:
        from arm_control.control import adapter
        from arm_control.control.doubles import _FakeExecutor, _GRIPPER

        names = [f"joint_{i}" for i in range(7)]
        adapter.load_robot_config = lambda: SimpleNamespace(
            joint_names=names, motor_names=names, get=lambda key: None)
        adapter.arm_joints = lambda cfg: names
        adapter._load_mode_config = lambda: {}
        adapter.resolve_gains = lambda *args: None
        adapter.build_executor = lambda *args, **kwargs: _FakeExecutor()
        adapter.gripper_command_cfg = lambda cfg: dict(_GRIPPER)
    adapter.Node = Node
    try:
        adapter.main()
    finally:
        if kind == "hand":
            assert closed.is_set(), "Hand socket was not closed"
            assert not client.is_alive(), "Hand thread was not joined"


def main() -> None:
    from arm_control.ui.arm_console import ControlPanel

    # The panel must use ConsoleServer's public close() API, not the old
    # embedded HTTPServer shutdown() method.
    panel = object.__new__(ControlPanel)
    closed = []
    panel._server = SimpleNamespace(close=lambda: closed.append(True))
    panel.close()
    assert closed == [True]
    for kind in ("logger", "controller", "replay", "hand"):
        for stop in ("SIGTERM", "SIGINT", "STOP", "ERROR"):
            with tempfile.TemporaryDirectory(prefix="arm-shutdown-") as directory:
                result = subprocess.run(
                    [sys.executable, __file__, "--child", kind, stop, directory],
                    capture_output=True, text=True, timeout=15,
                )
                if stop == "ERROR":
                    assert result.returncode == 1 and "deliberate node failure" in result.stderr, result
                else:
                    assert result.returncode == 0, (kind, stop, result.returncode, result.stderr)
                    assert "Traceback" not in result.stderr, result.stderr
                if kind == "logger":
                    paths = list(Path(directory).glob("state_*.csv"))
                    assert len(paths) == 1, paths
                    with paths[0].open() as stream:
                        rows = list(csv.DictReader(stream))
                    assert len(rows) == 1 and float(rows[0]["motor_0_pos"]) == 0, rows
                print(f"{kind}: {stop} OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else:
        main()
