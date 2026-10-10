#!/usr/bin/env python3
"""Build and run the device-free RT, SDK, and fast regression checks, and every other device-free check in this tree.

Run from any directory: python tools/check/suite.py
The only server spawned by these checks is the local fake backend.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
REGRESSIONS = (
    ("torque_cap", "torque_cap.py", "server"),
    ("check_two_instances", "check_two_instances.py", "server"),
    ("clock_domain", "clock_domain.py", "server"),
    ("group_execution", "group_execution.py", "server"),
    ("group_coordinator", "group_coordinator.py", "server"),
    ("reference_numerics", "reference_numerics.py", "probe"),
    ("reference_protocol", "reference_protocol.py", "reference_protocol_selfcheck"),
    ("mode_select", "mode_select.py", "mode_select_selfcheck"),
    ("mode_select_supervisor", "mode_select_supervisor.py", "server"),
    ("reference_horizon", "reference_horizon.py", "reference_runtime_selfcheck"),
    ("reference_supervisor", "reference_supervisor.py", "reference_runtime_selfcheck"),
    ("pc_stall", "pc_stall.py", "server"),
    ("stale_envelope_scope", "stale_envelope_scope.py", "server"),
    ("stream_races", "stream_races.py", "reference_runtime_selfcheck"),
    ("stream_supervisor", "stream_supervisor.py", "reference_runtime_selfcheck"),
    ("streamed_hold", "streamed_hold.py", "server"),
    ("controller_extension", "controller_extension.py", "reference_runtime_selfcheck"),
    ("pending_commit_cancel", "pending_commit_cancel.py", "reference_runtime_selfcheck"),
    ("selection_unresolved", "selection_unresolved.py", "reference_runtime_selfcheck"),
    ("selection_expiry_precedence", "selection_expiry_precedence.py", "reference_runtime_selfcheck"),
    ("selection_unresolved_cancel", "selection_unresolved_cancel.py", "reference_runtime_selfcheck"),
    ("console_adapter", "console_adapter.py", "reference_runtime_selfcheck"),
    ("console_differential", "console_differential.py", "reference_runtime_selfcheck"),
    ("clients", "clients.py", "reference_runtime_selfcheck"),
    ("sdk_channel", "sdk_channel.py", "server"),
    ("cancel_on_ack", "cancel_on_ack.py", "reference_runtime_selfcheck"),
    ("lifecycle_states", "lifecycle_states.py", "reference_runtime_selfcheck"),
)
# Device-free checks that need no build: run whether or not the build passed. A device-free check no suite runs is deleted.
# An entry is a script in tools/check/ and its arguments.
CHECKS = (
    "check_arm_controller", "check_bimanual_contract", "check_cartesian_sim", "check_bimanual_sync", "check_closure_record",
    "check_console_authority", "check_console_grasp", "check_cooperative", "check_dual_console",
    "check_execution_policy", "check_hand_grasp", "check_load_model", "check_motion_profiles", "check_node_shutdown",
    "check_pair_planning", "check_pose_hold", "check_pose_hold_sim", "check_refinement", "check_retiming", "check_rt_telemetry", "check_scene_cohost",
    "check_stream", "check_supervisor_ledger",
    "check_console_controls", "check_console_page", "check_controller_mode_feedback", "check_ui_scene_injection",
)
# Wall-clock assertions (check_refinement: parent heartbeat gap < 0.25 s, teardown < 2 s): run alone, not in the pool.
TIMED = {"check_refinement"}
# Simulated supervisors serve their sim Hand on the profile's FIXED loopback port (examples/profiles/sim_supervised.yaml:
# 47812, +10 per pair member): two at once collide ("Address already in use"), so these also run one at a time.
TIMED |= {"check_cooperative", "check_dual_console", "check_load_model", "check_pair_planning", "check_scene_cohost"}
# These simulate the staged FR3 (franka/, gitignored, staged by tools/assets/setup_fr3.py). Unstaged, they do not run and one
# fr3_assets FAIL names them: a fresh checkout is red until the assets are staged, never silently smaller.
FR3_ASSET = ROOT / "franka/urdf/fr3.urdf"
NEEDS_FR3 = {"check_cartesian_sim", "check_cooperative", "check_dual_console", "check_pair_planning",
             "check_pose_hold_sim", "check_refinement"}
# Module self-checks: `python -m <module> [args]`. Not here: node CLIs without a self-check flag (they run a node), kinematic_fit
# and model_bundle (they read the project's FT description; the project suite P1 creates is to run them, nothing runs them until
# then).
MODULE_CHECKS = (
    "arm_control.assets", "arm_control.calibration_console", "arm_control.config", "arm_control.console_assets",
    "arm_control.console_server", "arm_control.contracts.impedance", "arm_control.control.factory", "arm_control.control.gains", "arm_control.control.integral",
    "arm_control.control.jog_hold", "arm_control.control.leg_settling", "arm_control.control.reference",
    "arm_control.end_effectors.franka_adapter --demo", "arm_control.end_effectors.hand_move", "arm_control.execution.codec", "arm_control.execution.contract", "arm_control.frames",
    "arm_control.contracts", "arm_control.motion.types", "arm_control.mukca", "arm_control.node_utils",
    "arm_control.operator_console", "arm_control.planning.planner", "arm_control.planning.ik", "arm_control.planning.jog",
    "arm_control.planning.mujoco_collision", "arm_control.viz.preview", "arm_control.planning.retiming",
    "arm_control.planning.scene_geoms", "arm_control.planning.trajopt",
    "arm_control.planning.trajopt --native", "arm_control.plants.base", "arm_control.plants.dm.backend",
    "arm_control.plants.franka.backend", "arm_control.plants.mujoco.backend --self-check", "arm_control.plants.mujoco.backend --demo",
    "arm_control.plants.mujoco.backend --scene-demo",
    "arm_control.plants.remote_rt.observation", "arm_control.plants.remote_rt.protocol",
    "arm_control.scene", "arm_control.simulation.rerun_scene", "arm_control.supervisor.launch", "arm_control.supervisor.scene_observer", "arm_control.ui.hand_panel", "arm_control.ui.jog_panel",
    "arm_control.ui.operator_arm --self-check", "arm_control.viz.visualizer --self-check",
)
# Scripts with their own device-free self-check (an hw/ tool's estimator runs here; the tool itself never does).
SCRIPT_CHECKS = (
    "examples/run_console.py --self-check", "tools/assets/setup_fr3.py --self-check", "rt/bindings/pose_hold_check.py",
    "tools/hw/pair_skew.py --self-check",
)

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run(name: str, command: list[str], *, logs: Path, env: dict[str, str],
        timeout: int) -> dict[str, object]:
    start = time.monotonic()
    log = logs / f"{name}.log"
    try:
        with log.open("w") as output:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=timeout,
                                    check=False)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
        with log.open("a") as output:
            output.write(f"\nTIMEOUT after {timeout}s\n")
    elapsed = round(time.monotonic() - start, 3)
    status = "PASS" if code == 0 else "FAIL"
    print(f"{status:4} {name:27} {elapsed:8.3f}s  {log}", flush=True)
    if code:
        print("".join(log.read_text(errors="replace").splitlines(keepends=True)[-20:]),
              file=sys.stderr, flush=True)
    return {"name": name, "status": status, "returncode": code,
            "seconds": elapsed, "command": command, "log": str(log)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "rt/build-offline")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="concurrent timing-free checks (timing-sensitive ones always run alone); more than ~8 "
                             "gains nothing (the sequential part dominates) and only shrinks timing margins")
    args = parser.parse_args()
    if args.jobs < 1 or args.workers < 1:
        parser.error("--jobs and --workers must be positive")
    build = args.build_dir.resolve()
    logs = build / "offline-logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ARM_CONTROL_ROOT"] = str(ROOT)
    env["ARM_CONTROL_FAULT_ROOT"] = tempfile.mkdtemp(prefix="suite_faults_")  # checks' fault traces stay out of real directories
    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    print(f"RT OFFLINE SUITE revision={revision} dirty={dirty} root={ROOT}", flush=True)

    results, pool = [], []  # pool: (name, command, env, timeout) entries free of timing, run concurrently
    results.append(run("configure", ["cmake", "-S", str(ROOT / "rt"), "-B",
                                     str(build), "-G", "Ninja", "-DWITH_FRANKA=OFF",
                                     "-DBUILD_TESTING=ON"], logs=logs, env=env,
                       timeout=120))
    if results[-1]["status"] == "PASS":
        results.append(run("build", ["cmake", "--build", str(build), "--parallel",
                                     str(args.jobs)], logs=logs, env=env, timeout=600))
    if results[-1]["status"] == "PASS":
        results.append(run("ctest", ["ctest", "--test-dir", str(build),
                                     "--output-on-failure", "--timeout", "120"],
                           logs=logs, env=env, timeout=900))
        server = str(build / "arm_rt_server")
        for name, extra in (("sdk", []), ("sdk_native", ["--native"])):
            results.append(run(name, [sys.executable,
                                      str(ROOT / "tools/check/check_sdk_supervisor.py"),
                                      "--server", server, *extra],
                               logs=logs, env=env, timeout=180))
        # The FR3 build (libfranka, found by CMake like the box's build; this PC's /usr is 0.21.2): backend_franka, hand_bridge and
        # flange_probe compile, and CTest adds hand_bridge_selfcheck. Without libfranka it FAILs with that reason, never silently.
        franka = build.parent / (build.name + "-franka")
        results.append(run("configure_franka", ["cmake", "-S", str(ROOT / "rt"), "-B", str(franka), "-G", "Ninja",
                                                "-DWITH_FRANKA=ON", "-DBUILD_TESTING=ON"], logs=logs, env=env, timeout=120))
        if results[-1]["status"] == "PASS":
            results.append(run("build_franka", ["cmake", "--build", str(franka), "--parallel", str(args.jobs)],
                               logs=logs, env=env, timeout=600))
        if results[-1]["status"] == "PASS":
            results.append(run("ctest_franka", ["ctest", "--test-dir", str(franka), "--output-on-failure", "--timeout", "120"],
                               logs=logs, env=env, timeout=900))
        results.append(run("dual_supervisor", [sys.executable,
                                               str(ROOT / "tools/check/check_dual_supervisor.py"),
                                               "--server", server],
                           logs=logs, env=env, timeout=180))
        results.append(run("dual_group", [sys.executable,
                                          str(ROOT / "tools/check/check_dual_group.py"),
                                          "--server", server],
                           logs=logs, env=env, timeout=240))
        # every console graph has exactly one authority feedback source
        pool.append(("console_graphs", [sys.executable, str(ROOT / "arm_control/ui/arm_console.py"), "--self-check"], env, 180))
        # dora_arm (supervisor deadman windows) and sim_supervisor (rate windows, >= .95x real time under its own readers)
        # assert wall-clock behaviour: they run alone.
        results.append(run("dora_arm", [sys.executable, str(ROOT / "tools/check/check_dora_arm.py")], logs=logs, env=env,
                           timeout=180))
        results.append(run("sim_supervisor", [sys.executable, str(ROOT / "tools/check/check_sim_supervisor.py")], logs=logs,
                           env=env, timeout=180))
        pool.append(("scene_core", [sys.executable, str(ROOT / "tools/check/check_scene_core.py")], env, 120))
        results.append(run("trace_retention", [sys.executable,
                                               str(ROOT / "tools/check/check_trace_retention.py")],
                           logs=logs, env=env, timeout=120))
        results.append(run("supervised_hardware", [sys.executable,
                                                   str(ROOT / "tools/check/check_supervised_hardware.py"),
                                                   "--server", server],
                           logs=logs, env=env, timeout=180))
        results.append(run("installed_sdk", [sys.executable, str(ROOT / "tools/check/check_installed_sdk.py"),
                                             "--server", server], logs=logs, env=env, timeout=300))
        results.append(run("state_history", [sys.executable,
                                             str(ROOT / "tools/check/check_state_history.py"),
                                             "--server", server],
                           logs=logs, env=env, timeout=120))
        # The loopback rung: a live client against the built fake server (it SKIPs, rc 0, when no server is given).
        results.append(run("remote_rt_client", [sys.executable, "-m", "arm_control.plants.remote_rt.client"], logs=logs,
                           env={**env, "ARM_RT_SERVER_BIN": server}, timeout=180))
        binaries = {
            "server": server,
            "probe": str(build / "reference_probe.so"),
            "reference_protocol_selfcheck": str(build / "reference_protocol_selfcheck"),
            "mode_select_selfcheck": str(build / "mode_select_selfcheck"),
            "reference_runtime_selfcheck": str(build / "reference_runtime_selfcheck"),
        }
        for name, script, target in REGRESSIONS:
            command = [sys.executable, str(ROOT / "tools/check/rt" / script),
                       binaries[target]]
            if name == "stale_envelope_scope":
                command += ["--repetitions", "3"]  # the 50-run network gate is separate
            results.append(run(name, command, logs=logs, env=env, timeout=300))

    for check in CHECKS:
        if check in NEEDS_FR3 and not FR3_ASSET.is_file():
            continue
        script, *extra = check.split()
        path = ROOT / "tools/check" / script
        if check in TIMED:
            results.append(run(check, [sys.executable, f"{path}.py", *extra], logs=logs, env=env, timeout=180))
            continue
        pool.append((check.replace(" --", "_"), [sys.executable, f"{path}.py", *extra], env, 180))
    if not FR3_ASSET.is_file():
        results.append(dict(name="fr3_assets", status="FAIL", returncode=None, seconds=0.0, command=[], log="",
                            reason=f"{FR3_ASSET} not staged (tools/assets/setup_fr3.py); not run: {sorted(NEEDS_FR3)}"))
        print(f"FAIL fr3_assets: {results[-1]['reason']}", flush=True)
    for check in MODULE_CHECKS:
        module, *extra = check.split()
        pool.append((check.replace(" --", "_"), [sys.executable, "-m", module, *extra], env, 180))
    for check in SCRIPT_CHECKS:
        script, *extra = check.split()
        pool.append((Path(script).stem + "".join(e.replace("--", "_") for e in extra),
                     [sys.executable, str(ROOT / script), *extra], env, 120))

    # The timing-sensitive entries above (fake RT servers, protocol deadlines) ran one at a time on a quiet machine; the rest
    # are independent processes on ephemeral ports and run concurrently. Results keep submission order.
    with ThreadPoolExecutor(args.workers) as executor:
        results += list(executor.map(lambda job: run(job[0], job[1], logs=logs, env=job[2], timeout=job[3]), pool))

    summary = {"revision": revision, "dirty": dirty, "root": str(ROOT),
               "build_dir": str(build), "results": results,
               "passed": sum(r["status"] == "PASS" for r in results),
               "failed": sum(r["status"] != "PASS" for r in results)}
    summary_path = build / "offline-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"SUMMARY revision={revision} dirty={dirty} "
          f"PASS={summary['passed']} FAIL={summary['failed']} {summary_path}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
