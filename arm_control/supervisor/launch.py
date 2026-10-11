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
                     cwd: Path | None = None, stopped=lambda: False, timeout_s: float = 60.0, cohosts: dict | None = None):
    """Start the supervisor and wait until its socket answers; return ``(process, env for the graph)``. RuntimeError (with the
    log's path) if it exits, does not come up within ``timeout_s``, or ``stopped()`` turns true meanwhile. ``cohosts``
    (sim): ``{PREFIX: staged config path}``, another actor of the same world served by this process; the graph env gains
    ``PREFIX_CONFIG`` and ``PREFIX_SOCKET`` for each."""
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
    cohosts = dict(cohosts or {})
    co_socks = {prefix: workdir / f"{prefix.lower()}.sock" for prefix in cohosts}
    log = open(workdir / "supervisor.log", "w")
    # The supervisor writes its log FILE directly (never a pipe: a stalled or suspended reader must never block the control
    # loop's prints); the launcher TAILS that file to echo the controller's own lines (a fault, a refused ARM, a stale stream) to
    # the operator's terminal.
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "arm_control.supervisor.service", "--config", str(merged_path), "--socket", str(sock),
         "--plant", plant, *[f"--cohost={cohosts[p]}={co_socks[p]}" for p in cohosts]],
        env=dict(env, ARM_CONTROL_CONFIG=str(merged_path)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)  # its own session: a Ctrl-C reaches the graph, and the supervisor outlives it long enough to be stopped after
    proc._workdir, proc._log = workdir, log
    proc._echo_stop = threading.Event()

    def _echo():
        partial = ""
        with open(workdir / "supervisor.log", errors="replace") as tail:  # one stray byte must not end the echo
            while True:
                partial += tail.readline()
                if partial.endswith("\n") or (partial and proc._echo_stop.is_set()):  # whole lines only (a write may split one)
                    print(partial if partial.startswith("[") else f"[supervisor] {partial}", end="", flush=True)
                    partial = ""
                elif proc._echo_stop.is_set():
                    return
                else:
                    time.sleep(.1)
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
            if all(_answers(path) for path in (sock, *co_socks.values())):
                break
            time.sleep(0.2)
    except BaseException:  # a failure or an interrupt (Ctrl-C) mid-start: never leave it running, keep the log it names
        stop_supervisor(proc, keep_log=True)
        raise
    print(f"[supervisor] up on {sock} (config {merged_path})", flush=True)
    graph_env = dict(env, ARM_CONTROL_CONFIG=str(merged_path), ARM_CONTROL_SOCKET=str(sock))
    for prefix, path in cohosts.items():
        graph_env[f"{prefix}_CONFIG"], graph_env[f"{prefix}_SOCKET"] = str(path), str(co_socks[prefix])
    return proc, graph_env


def _answers(path: Path) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(path))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def stop_supervisors(procs, *, keep_log: bool = False) -> None:
    """Stop several supervisors (a pair): SIGTERM to EVERY one first, so an interrupt while waiting on one (a second Ctrl-C
    during the bounded wait) has already told the rest to DISARM and exit; then each wait runs even if an earlier one raised.
    Stopping them one by one left the second supervisor orphaned on the bench (2026-10-10)."""
    procs = [p for p in procs if p is not None]
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()

    def stop(rest):
        if rest:
            try:
                stop_supervisor(rest[0], keep_log=keep_log)
            finally:
                stop(rest[1:])
    stop(procs)


def stop_supervisor(proc, *, keep_log: bool = False) -> None:
    """SIGTERM (the supervisor DISARMs the plant on the way out), then kill if it does not stop. Safe with None and twice. A clean
    stop removes its work directory unless ``keep_log`` (hardware runs keep it: the bench's evidence); a failure always keeps it,
    and a kept log's path is printed. Every supervisor line was also echoed (tailed from the log) to the launcher's terminal."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    proc._echo_stop.set()  # the process has ended: the tail prints what is left, then stops
    proc._echo.join(timeout=2)
    try:
        proc._log.close()
    except Exception:  # noqa: BLE001
        pass
    if proc.returncode in (0, -signal.SIGTERM) and not keep_log:
        import shutil

        shutil.rmtree(proc._workdir, ignore_errors=True)
    else:
        print(f"[supervisor] log kept: {proc._workdir / 'supervisor.log'}", flush=True)


def _self_check() -> None:
    """stop_supervisors: an interrupt while waiting on the first still stops (and reaps) the second."""
    import tempfile
    import threading

    def proc():
        p = subprocess.Popen(["sleep", "30"])
        p._echo_stop, p._echo, p._workdir = threading.Event(), threading.Thread(target=lambda: None), Path(tempfile.mkdtemp())
        p._echo.start()
        p._log = open(p._workdir / "supervisor.log", "w")
        return p
    first, second = proc(), proc()
    real_wait = first.wait

    def interrupted(timeout=None):
        # every supervisor was signalled before this first wait: the second exits on its own
        assert second.wait(timeout=5) is not None, "the second supervisor was not signalled before the first wait"
        raise KeyboardInterrupt
    first.wait = interrupted
    try:
        stop_supervisors([first, None, second])
        raise AssertionError("the interrupt was swallowed")
    except KeyboardInterrupt:
        pass
    first.wait = real_wait
    assert second.poll() is not None, "the second supervisor survived an interrupt during the first stop"
    assert first.wait(timeout=5) is not None, "the first supervisor was not signalled before the wait"
    import shutil
    for p in (first, second):
        p._log.close()
        shutil.rmtree(p._workdir, ignore_errors=True)
    print("stop_supervisors: every supervisor is signalled before any wait; an interrupted wait still stops the rest: OK")


if __name__ == "__main__":
    _self_check()
