"""Servo-tick period distribution of a fake ``arm_rt_server`` (measurement, not a pass/fail gate).

Reads the RT state stream (``--state-hz 1000``, each packet stamped with the tick's own host
CLOCK_MONOTONIC time and sequence) and reports the period between CONSECUTIVE ticks (sequence
difference exactly 1: an unskipped, true period) and the largest gap seen including skipped ticks
(a lower bound on the gap). This is the input to the CTRL-040 constants proposal: the peer heartbeat is
written by the servo tick, so the tick's worst gap is the floor under any peer-loss age.

    PYTHONPATH=. python tools/bench/rt/tick_jitter.py --server /path/arm_rt_server [--host H --udp P --tcp P] \
        [--seconds 60] [--load N]

With ``--host`` the server is expected to be running already (started by the caller, pinned as needed);
otherwise a local fake server is started. ``--load N`` starts N busy-loop child processes (stopped by PID).
"""
from __future__ import annotations

# ruff: noqa: E402

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from arm_control.plants.remote_rt.client import RtBackend, RtConfig, RtLinkError

JOINTS = ["j0", "j1", "j2"]


def free_port(kind):
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server")
    ap.add_argument("--host")
    ap.add_argument("--udp", type=int)
    ap.add_argument("--tcp", type=int)
    ap.add_argument("--instance", default="jitter")
    ap.add_argument("--seconds", type=float, default=60.)
    ap.add_argument("--load", type=int, default=0)
    args = ap.parse_args()
    procs, server = [], None
    try:
        if args.host:
            host, udp, tcp = args.host, args.udp, args.tcp
        else:
            host, udp, tcp = "127.0.0.1", free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM)
            server = subprocess.Popen(
                [args.server, "--backend", "fake", "--n", "3", "--instance-id", args.instance,
                 "--joint-names", ",".join(JOINTS), "--udp-port", str(udp), "--tcp-port", str(tcp),
                 "--bind", "127.0.0.1", "--state-hz", "1000"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(args.load):
            procs.append(subprocess.Popen([sys.executable, "-c", "while True: pass"]))
        link = None
        deadline = time.monotonic() + 10
        while link is None:
            try:
                link = RtBackend(RtConfig(host=host, udp_port=udp, tcp_port=tcp,
                                          instance_id=args.instance), JOINTS)
                link.open()
            except (RtLinkError, OSError):
                link = None
                assert time.monotonic() < deadline, "server not ready"
                time.sleep(0.1)
        link.enable_all()
        stop = threading.Event()

        def hold():  # keep the command deadman satisfied
            while not stop.is_set():
                link.apply_command({"position": np.zeros(3), "velocity": np.zeros(3),
                                    "torque": np.zeros(3), "kp": np.ones(3), "kd": np.ones(3) * .1})
                stop.wait(.005)
        threading.Thread(target=hold, daemon=True).start()
        seen, last_seq, last_t, periods, gaps = 0, None, None, [], []
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            state, _ = link.latest_state()
            if state is not None and state.state_seq != last_seq:
                if last_seq is not None:
                    dt = (state.t_mono_ns - last_t) / 1e6
                    (periods if state.state_seq - last_seq == 1 else gaps).append(dt)
                last_seq, last_t = state.state_seq, state.t_mono_ns
                seen += 1
            time.sleep(0.0002)
        stop.set()
        link.close()
        p = np.array(periods)
        allgaps = np.concatenate([p, np.array(gaps)]) if gaps else p
        q = lambda a, x: float(np.percentile(a, x))  # noqa: E731
        print(f"host={host} load={args.load} ticks_seen={seen} unskipped_periods={len(p)} "
              f"skipped_intervals={len(gaps)}")
        print("period_ms  p50 %.3f  p99 %.3f  p99.9 %.3f  p99.99 %.3f  max %.3f" % (
            q(p, 50), q(p, 99), q(p, 99.9), q(p, 99.99), p.max()))
        print("largest interval incl. skipped ticks (ms): %.3f" % allgaps.max())
        return 0
    finally:
        for proc in procs + ([server] if server else []):
            proc.terminate()
            try:
                proc.wait(3)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
