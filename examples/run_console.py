#!/usr/bin/env python3
"""Run the operator console against a simulated arm, from a fresh clone.

Deliberately thin. This is NOT a second scenario registry -- a deployment's
launcher (which knows about that project's scenarios, arms and CAD) stays where
it is. All this does is the three things a bare `dora run` cannot do for you:
check the assets are staged, export the two roots, and pick the graph.

    python examples/run_console.py                 # one arm, console on :7500
    python examples/run_console.py --dual          # shared-world console, :7520
    python examples/run_console.py --config <path> # your own entry config

Meshes are not vendored here (this repo owns the code; a project owns its CAD),
so the first run needs the FR3 description staged once:

    git clone https://github.com/frankarobotics/franka_description
    python tools/assets/setup_fr3.py --source ./franka_description
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SINGLE = REPO_ROOT / "dataflows" / "sim_franka_motion.yml"
SUPERVISED = REPO_ROOT / "dataflows" / "sim_franka_motion_supervised.yml"
SUPERVISED_PROFILE = REPO_ROOT / "examples" / "profiles" / "sim_supervised.yaml"
MODE_PROFILE = REPO_ROOT / "examples" / "profiles" / "motion_franka.yaml"
PAIR_CONFIG = REPO_ROOT / "examples" / "configs" / "dual_arm.yaml"
OBJECT_CONFIG = REPO_ROOT / "examples" / "configs" / "dual_arm_object.yaml"
DEFAULT_CONFIG = REPO_ROOT / "examples" / "configs" / "sim_arm.yaml"


def assets_ready() -> tuple[bool, str]:
    """Is the FR3 description staged? Returns (ready, what to do about it)."""
    urdf = REPO_ROOT / "franka" / "urdf" / "fr3.urdf"
    meshes = REPO_ROOT / "franka" / "meshes"
    if urdf.is_file() and meshes.is_dir() and any(meshes.rglob("*.stl")):
        return True, ""
    return False, (
        f"The FR3 description is not staged at {urdf.parent}.\n"
        f"This repo vendors no meshes -- it owns the code, a project owns its\n"
        f"CAD -- so stage it once:\n\n"
        f"    git clone https://github.com/frankarobotics/franka_description\n"
        f"    python {Path('tools/assets/setup_fr3.py')} "
        f"--source ./franka_description\n\n"
        f"(`python tools/assets/setup_fr3.py --check` reports status and\n"
        f"changes nothing.)"
    )


def ensure_rerun_viewer(port: int = 9876) -> None:
    """Start ONE persistent viewer if none is listening; nodes only connect.

    Every Rerun producer in this repo uses `spawn=False` + `connect_grpc()` --
    visualizer, the scene mirror, the plan preview -- so nothing here starts a
    viewer. Without one the connect succeeds and the data goes NOWHERE, which
    is the worst failure shape: no error, no visuals, nothing to search for.

    Detached on purpose: a viewer spawned as a child of the graph dies with
    every restart, and the window you are looking at ends up belonging to a
    previous run. (Control's own `view.py` does the same thing for the project
    graphs; this is the standalone launcher's copy, which is why the demo can
    run from a fresh clone with no project around it.)
    """
    import socket

    def listening() -> bool:
        probe = socket.socket()
        probe.settimeout(0.3)
        try:
            probe.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            probe.close()

    if listening():
        print("[run_console] rerun viewer already up — reusing it", flush=True)
        return
    try:
        subprocess.Popen(
            ["rerun"], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("[run_console] no `rerun` on PATH — the console still works, "
              "visuals are just disabled (pip install rerun-sdk)", flush=True)
        return
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        if listening():
            print("[run_console] rerun viewer up (persistent — survives restarts)",
                  flush=True)
            return
        time.sleep(0.3)
    print("[run_console] rerun viewer did not come up — visuals may lag", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"entry config (default: {DEFAULT_CONFIG.name})")
    parser.add_argument("--dual", action="store_true",
                        help="one shared-world dual-arm simulation console on :7520")
    parser.add_argument("--pair-config", type=Path, default=PAIR_CONFIG,
                        help="dual-arm composition (used with --dual)")
    parser.add_argument("--port", type=int, default=7520, help="dual console loopback port")
    parser.add_argument("--check", action="store_true",
                        help="report readiness and exit, launching nothing")
    parser.add_argument("--supervised", action="store_true",
                        help="the arm's SUPERVISOR is the control authority (over the same MuJoCo plant): starts it with the graph")
    parser.add_argument("--no-viewer", action="store_true", help="do not start a Rerun viewer (headless runs)")
    args = parser.parse_args()

    if args.dual:
        if args.config != DEFAULT_CONFIG:
            parser.error("--config is single-arm; use --pair-config with --dual")
        if args.pair_config.resolve() in (PAIR_CONFIG, OBJECT_CONFIG):
            os.environ["ARM_CONTROL_ROOT"] = str(REPO_ROOT)
        from arm_control.multi_arm.config import load_pair_config
        config_path = args.pair_config.resolve()
        if args.check:
            pair = load_pair_config(config_path)
            print(f"{pair.backend} configuration: {config_path}, arms={pair.scene.actor_names}; no transports opened")
            return 0
        from arm_control.ui.dual_arm_console import main as dual_main
        dual_main(config_path, port=args.port)
        return 0

    ready, hint = assets_ready()
    config = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    graph = SUPERVISED if args.supervised else SINGLE

    if args.check or not ready:
        print(f"config: {config}  {'ok' if config.is_file() else 'MISSING'}")
        print(f"graph:  {graph}  {'ok' if graph.is_file() else 'MISSING'}")
        print(f"assets: {'ok' if ready else 'NOT STAGED'}")
        if not ready:
            print("\n" + hint, file=sys.stderr)
        return 0 if (ready and args.check) else (0 if args.check else 1)
    if not config.is_file():
        print(f"entry config not found: {config}", file=sys.stderr)
        return 1

    busy = _port_holder(config)
    if busy:
        print(
            f"[run_console] console port {busy} is already in use — another "
            f"graph from this checkout is probably still running.\n"
            f"  Find it:  ss -ltnp | grep {busy}\n"
            f"  A launch now would come up half-working and you would be "
            f"driving the OLD console on that port.",
            file=sys.stderr,
        )
        return 1

    env = dict(os.environ)
    # Both roots, because a bare `dora run` rebases each node's cwd to the
    # dataflow's directory: a relative ARM_CONTROL_CONFIG would then resolve
    # against dataflows/ and a relative mesh path against nothing useful.
    env["ARM_CONTROL_ROOT"] = str(REPO_ROOT)
    env["ARM_CONTROL_CONFIG"] = str(config)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    supervisor = None
    live: dict = {"graph": None}
    stopped_by: list[int] = []

    def _stop(signum, _frame):
        # Signal only; never wait here. The handler runs on the main thread, which may be blocked in a wait holding Popen's
        # _waitpid_lock, so a handler that waits re-enters that lock on the same thread and hangs. Signalling dora is enough --
        # it reaps its nodes, the wait returns, and the real teardown runs in `finally`, off the handler. Installed BEFORE
        # anything is started: a signal during the supervisor's start-up or the viewer's must not orphan either.
        stopped_by.append(int(signum))
        graph_proc = live["graph"]
        if graph_proc is not None:
            try:
                os.killpg(os.getpgid(graph_proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _stop)
    if args.supervised:
        try:
            supervisor, env = start_supervisor(config, env, stopped=lambda: bool(stopped_by))
        except RuntimeError as exc:
            print(f"[run_console] {exc}", file=sys.stderr)
            return 128 + stopped_by[0] if stopped_by else 1
    if stopped_by:
        _stop_supervisor(supervisor)
        return 128 + stopped_by[0]
    if not args.no_viewer:
        ensure_rerun_viewer()
    print(f"[run_console] {config.name} -> {graph.name}", flush=True)
    print("[run_console] console on http://127.0.0.1:7500", flush=True)
    print("[run_console] the arm comes up DISARMED — press ARM on the page", flush=True)
    try:
        # start_new_session: the graph gets its OWN session, so teardown can
        # reach every node at once. `subprocess.call` here left orphans --
        # dora reaps its nodes correctly when it is signalled, but nothing was
        # signalling it, so killing the launcher (or any single node) left the
        # whole graph running: a controller still streaming motor_command and a
        # console still holding its port, which the NEXT launch then talks to.
        proc = subprocess.Popen(
            ["dora", "run", str(graph)], env=env, cwd=REPO_ROOT,
            start_new_session=True,
        )
    except FileNotFoundError:
        print("`dora` is not on PATH — is the dora-rs CLI installed?",
              file=sys.stderr)
        _stop_supervisor(supervisor)
        return 1

    live["graph"] = proc
    if stopped_by:  # a signal arrived between the graph's start and now
        _terminate_graph(proc)
    try:
        if supervisor is None:
            return proc.wait()
        # With a supervisor the graph is only as alive as the thing that owns control: watch both.
        while proc.poll() is None:
            if supervisor.poll() is not None:
                print(f"[run_console] the supervisor exited ({supervisor.returncode}): stopping the graph", file=sys.stderr, flush=True)
                _terminate_graph(proc)
                return 1
            time.sleep(0.5)
        return proc.returncode
    finally:
        _terminate_graph(proc)
        _stop_supervisor(supervisor)  # after the graph: its SIGTERM DISARMs the plant through the link
        if stopped_by:
            raise SystemExit(128 + stopped_by[0])


def start_supervisor(config: Path, env: dict, stopped=lambda: False):
    """Start the arm's supervisor over the simulated plant and return ``(process, env for the graph)``.

    One merged config is written (the entry config, the stated sim servo parameters, the mode profile as ``control_mode``) and given to
    the supervisor AND every node, so they cannot run different configurations; the bridge node also checks the supervisor's gains.
    """
    import tempfile

    import yaml

    from arm_control.config import load_config_tree

    merged = load_config_tree(config)
    merged.update(yaml.safe_load(SUPERVISED_PROFILE.read_text()))
    merged["control_mode"] = yaml.safe_load(MODE_PROFILE.read_text())
    workdir = Path(tempfile.mkdtemp(prefix="supervised_console_"))
    merged_path = workdir / "supervised_arm.yaml"
    merged_path.write_text(yaml.safe_dump(merged))
    sock = workdir / "arm.sock"
    log = open(workdir / "supervisor.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "arm_control.supervisor.service", "--config", str(merged_path), "--socket", str(sock), "--plant", "sim"],
        env=dict(env, ARM_CONTROL_CONFIG=str(merged_path)), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    proc._workdir, proc._log = workdir, log
    import socket as _socket

    deadline = time.monotonic() + 60
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"the supervisor exited ({proc.returncode}); see {workdir / 'supervisor.log'}")
        if stopped():
            _stop_supervisor(proc)
            raise RuntimeError("stopped while the supervisor was starting")
        if time.monotonic() > deadline:
            _stop_supervisor(proc)
            raise RuntimeError(f"the supervisor did not come up; see {workdir / 'supervisor.log'}")
        probe = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        try:
            probe.connect(str(sock))
            break
        except OSError:
            time.sleep(0.2)
        finally:
            probe.close()
    print(f"[run_console] supervisor up on {sock} (config {merged_path})", flush=True)
    return proc, dict(env, ARM_CONTROL_CONFIG=str(merged_path), ARM_CONTROL_SOCKET=str(sock))


def _stop_supervisor(proc) -> None:
    """SIGTERM (the supervisor DISARMs the plant on the way out), then kill if it does not stop. Safe with None and twice."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        proc._log.close()
    except Exception:  # noqa: BLE001
        pass
    if proc.returncode in (0, -signal.SIGTERM):  # a clean stop leaves nothing behind; a failure keeps its log
        import shutil

        shutil.rmtree(proc._workdir, ignore_errors=True)


def _port_holder(config) -> int | None:
    """The console port from ``config``, if something is already listening.

    Cheap preflight for the failure that wastes the most time: a stale graph
    still holding the port, so the new run's console never binds and every
    click you make lands on the PREVIOUS process. Nothing about that looks
    wrong on screen.
    """
    import socket

    try:
        from arm_control.config import load_config_tree

        port = int(load_config_tree(config).get("console", {}).get("http_port", 0))
    except Exception:
        return None                      # not our problem; the graph will say
    if not port:
        return None
    probe = socket.socket()
    probe.settimeout(0.2)
    try:
        probe.connect(("127.0.0.1", port))
        return port
    except OSError:
        return None
    finally:
        probe.close()


def _terminate_graph(proc, grace_s: float = 8.0) -> None:
    """Stop the graph and every node it spawned. Safe to call twice.

    dora puts each node in its OWN process group inside the graph's session,
    so neither `kill <launcher>` nor a kill of dora's process group reaches
    them. Two things do, and this uses both: a SIGTERM to dora, which reaps its
    nodes properly (measured -- a graceful TERM cleared every node in 6 s), and
    then a session-wide kill for anything that ignored it.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    print("[run_console] graph did not stop in "
          f"{grace_s:.0f}s — killing the session", flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _self_check() -> None:
    """The launcher's own claims: the files it points at have to exist."""
    assert SINGLE.is_file(), SINGLE
    assert SUPERVISED.is_file() and SUPERVISED_PROFILE.is_file() and MODE_PROFILE.is_file()
    assert PAIR_CONFIG.is_file(), PAIR_CONFIG
    assert OBJECT_CONFIG.is_file(), OBJECT_CONFIG
    assert DEFAULT_CONFIG.is_file(), DEFAULT_CONFIG
    # The entry config must actually load through the include machinery, and
    # name a console port -- the whole reason it lives with the robot.
    from arm_control.config import load_config_tree

    cfg = load_config_tree(DEFAULT_CONFIG)
    assert cfg["console"]["http_port"], cfg.get("console")
    dual_cfg = load_config_tree(REPO_ROOT / "examples/configs/sim_arm_b.yaml")
    assert dual_cfg["console"]["http_port"] != cfg["console"]["http_port"], (
        "the two demo arms must not share a console port -- that collision is "
        "the exact thing per-instance config exists to prevent"
    )
    _check_entry_keys()
    ready, hint = assets_ready()
    assert ready or "franka_description" in hint, hint
    print(f"run_console: ok (assets {'staged' if ready else 'not staged'})")


def _check_entry_keys() -> None:
    """Every key the entry configs state must be read by somebody.

    A config key nothing consumes is worse than a missing one: it reads as a
    setting, documents itself in a comment, and does nothing. Caught in the
    wild -- `sim_damping_ratio: 0.025` sat in sim_demo.yaml with a paragraph
    explaining why the sim plant needs damping, and no code ever looked it up,
    so the demo shipped kp 1200 against kd 0 (the config's own measurements
    call that "unusable"). A missing key would have failed loudly; this one
    just quietly meant nothing.

    Top-level keys only, and only those the entry file states ITSELF --
    inherited keys are the included fragment's business.
    """
    import re

    sources = "\n".join(
        path.read_text()
        for directory in ("arm_control", "nodes", "tools", "examples")
        for path in sorted((REPO_ROOT / directory).rglob("*.py"))
    )
    for entry in sorted((REPO_ROOT / "examples" / "configs").glob("sim_arm*.yaml")):
        # The file's own top-level keys, read as text: parsing the tree back
        # would hand us the merged result and lose exactly this distinction.
        own = [
            m.group(1)
            for m in re.finditer(r"^([a-z_][a-z0-9_]*):", entry.read_text(), re.M)
        ]
        for key in own:
            if key in ("include", "arm"):
                continue  # structural: the loader's own, and the shared table
            assert f'"{key}"' in sources or f"'{key}'" in sources, (
                f"{entry.name} sets {key!r}, which no code in arm_control/, "
                f"nodes/ or scripts/ ever reads -- a key that does nothing "
                f"still reads like a setting"
            )


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
