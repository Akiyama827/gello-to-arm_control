#!/usr/bin/env python3
"""Build and run the device-free RT, SDK, and fast regression checks.

Run from any directory: python tools/bench/rt_offline_suite.py
The only server spawned by these checks is the local fake backend.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
REGRESSIONS = (
    ("torque_cap", "torque_cap.py", "server"),
    ("check_two_instances", "check_two_instances.py", "server"),
    ("clock_domain", "clock_domain.py", "server"),
    ("group_execution", "group_execution.py", "server"),
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
    ("cancel_on_ack", "cancel_on_ack.py", "reference_runtime_selfcheck"),
    ("lifecycle_states", "lifecycle_states.py", "reference_runtime_selfcheck"),
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
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build = args.build_dir.resolve()
    logs = build / "offline-logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ARM_CONTROL_ROOT"] = str(ROOT)
    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    print(f"RT OFFLINE SUITE revision={revision} dirty={dirty} root={ROOT}", flush=True)

    results = []
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
                                      str(ROOT / "tools/bench/check_sdk_supervisor.py"),
                                      "--server", server, *extra],
                               logs=logs, env=env, timeout=180))
        binaries = {
            "server": server,
            "probe": str(build / "reference_probe.so"),
            "reference_protocol_selfcheck": str(build / "reference_protocol_selfcheck"),
            "mode_select_selfcheck": str(build / "mode_select_selfcheck"),
            "reference_runtime_selfcheck": str(build / "reference_runtime_selfcheck"),
        }
        for name, script, target in REGRESSIONS:
            command = [sys.executable, str(ROOT / "tools/bench/rt" / script),
                       binaries[target]]
            if name == "stale_envelope_scope":
                command += ["--repetitions", "3"]  # the 50-run network gate is separate
            results.append(run(name, command, logs=logs, env=env, timeout=300))

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
