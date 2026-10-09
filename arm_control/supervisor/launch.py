"""Start one arm's supervisor as a child process for a graph that talks to it through the SDK (``supervised_arm``,
``scene_observer``), and stop it again. Shared by the library's example launcher and a project's own launcher.

One merged config is written (the entry config, then each overlay, then the control mode as ``control_mode``) and given to the
supervisor AND, through ``ARM_CONTROL_CONFIG``, to every node, so they cannot run different configurations (the bridge node
also checks the supervisor's gains). ``ARM_CONTROL_SOCKET`` names the supervisor's socket.
"""
from __future__ import annotations

import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def start_supervisor(config: Path, env: dict, *, overlays=(), control_mode: Path | None = None, plant: str = "sim",
                     cwd: Path | None = None, stopped=lambda: False, timeout_s: float = 60.0):
    """Start the supervisor and wait until its socket answers; return ``(process, env for the graph)``. RuntimeError (with the
    log's path) if it exits, does not come up within ``timeout_s``, or ``stopped()`` turns true meanwhile."""
    import yaml

    from arm_control.config import load_config_tree

    merged = load_config_tree(config)
    for overlay in overlays:
        merged.update(load_config_tree(overlay))
    if control_mode is not None:
        merged["control_mode"] = load_config_tree(control_mode)
    workdir = Path(tempfile.mkdtemp(prefix="supervised_arm_"))
    merged_path = workdir / "supervised_arm.yaml"
    merged_path.write_text(yaml.safe_dump(merged))
    sock = workdir / "arm.sock"
    log = open(workdir / "supervisor.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "arm_control.supervisor.service", "--config", str(merged_path), "--socket", str(sock),
         "--plant", plant],
        env=dict(env, ARM_CONTROL_CONFIG=str(merged_path)), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)  # its own session: a Ctrl-C reaches the graph, and the supervisor outlives it long enough to be stopped after
    proc._workdir, proc._log = workdir, log

    def _echo():  # the controller's own lines (a fault, a refused ARM, a stale stream) reach the operator's terminal AND the log
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"[supervisor] {line}" if not line.startswith("[") else line, end="", flush=True)
    proc._echo = threading.Thread(target=_echo, name="supervisor-log", daemon=True)
    proc._echo.start()
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"the supervisor exited ({proc.returncode}); see {workdir / 'supervisor.log'}")
            if stopped():
                raise RuntimeError(f"stopped while the supervisor was starting; see {workdir / 'supervisor.log'}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"the supervisor did not come up in {timeout_s:g} s; see {workdir / 'supervisor.log'}")
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(sock))
                break
            except OSError:
                time.sleep(0.2)
            finally:
                probe.close()
    except BaseException:  # a failure or an interrupt (Ctrl-C) mid-start: never leave it running, keep the log it names
        stop_supervisor(proc, keep_log=True)
        raise
    print(f"[supervisor] up on {sock} (config {merged_path})", flush=True)
    return proc, dict(env, ARM_CONTROL_CONFIG=str(merged_path), ARM_CONTROL_SOCKET=str(sock))


def stop_supervisor(proc, *, keep_log: bool = False) -> None:
    """SIGTERM (the supervisor DISARMs the plant on the way out), then kill if it does not stop. Safe with None and twice. A clean
    stop removes its work directory unless ``keep_log`` (hardware runs keep it: the bench's evidence); a failure always keeps it,
    and a kept log's path is printed. Every supervisor line was also echoed to the launcher's terminal while it ran."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    proc._echo.join(timeout=2)  # the pipe closes with the process: the last lines are written before the log closes
    try:
        proc._log.close()
    except Exception:  # noqa: BLE001
        pass
    if proc.returncode in (0, -signal.SIGTERM) and not keep_log:
        import shutil

        shutil.rmtree(proc._workdir, ignore_errors=True)
    else:
        print(f"[supervisor] log kept: {proc._workdir / 'supervisor.log'}", flush=True)
