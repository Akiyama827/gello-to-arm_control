"""Trace the pc_stall local-cancel section's first native envelope on loopback.

Each iteration uses a new control session against one fake RT process. This
probe preserves the pc_stall stage/poll/approve/envelope ordering; it adds only
timestamps around the link operations and observed state. No FCI is used.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from arm_control.plants.remote_rt.client import RtConfig
from arm_control.plants.remote_rt.reference import ReferenceHorizon
from first_envelope_trace import TraceBackend
from pc_stall import ramp, unused_port


def one(rt, events, run, *, cancel=True):
    command = dict(position=np.zeros(3), velocity=np.zeros(3),
                   torque=np.zeros(3), kp=np.ones(3) * 2, kd=np.ones(3) * .1)
    h = ReferenceHorizon(rt)
    try:
        state, _ = rt.latest_state()
        h.stage("local-cancel", ramp(np.asarray(state.q), 2.0), stream=True)
        rt.record("stage_call")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (h.selected and not h.error):
            state, received = rt.latest_state()
            h.poll(state, received)
            if (state and state.armed and not h.selected
                    and not (state.reference or {}).get("reference_id")):
                rt.apply_command(dict(position=np.asarray(state.q),
                                      velocity=np.zeros(3), torque=np.zeros(3),
                                      kp=np.ones(3) * 2, kd=np.ones(3) * .1))
            if h.ready and not h.selected:
                h.approve()
            if h.selected:
                rt.apply_reference_envelope(
                    command, reference_id=h.id, revision=h.feedback["revision"],
                    phase=h.point.time, sample_received_at=received,
                    valid_for_s=.1, phase_window_s=.1)
            time.sleep(0.005)
        if not h.selected or h.error:
            rt.record("selection_failed", error=h.error)
            return False
        rt.record("selection_reached")
        if not cancel:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not h.error:
                h.poll(*rt.latest_state()[:2])
                time.sleep(0.005)
            rt.record("expiry_done", error=h.error)
            return h.error == "REFERENCE_EXPIRED"
        rt.record("cancel_requested")
        h.cancel()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and h.selected:
            h.poll(*rt.latest_state()[:2])
            time.sleep(0.005)
        result = not h.selected and h.error != "REFERENCE_EXPIRED"
        rt.record("cancel_done", success=result, error=h.error)
        return result
    finally:
        h.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--remote", help="operator-controlled scratch fake-RT host")
    parser.add_argument("--udp-port", type=int)
    parser.add_argument("--tcp-port", type=int)
    parser.add_argument("--instance-id", default="fake")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-cpu", type=int)
    parser.add_argument("--expiry-prelude", action="store_true",
                        help="precede each local cancel with the bench's expiry session")
    args = parser.parse_args()
    if args.remote:
        if not args.udp_port or not args.tcp_port or args.instance_id == "fake":
            parser.error("remote needs --udp-port, --tcp-port, and a scratch --instance-id")
        udp, tcp = args.udp_port, args.tcp_port
    else:
        if args.binary is None:
            parser.error("local loopback needs --binary")
        udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    argv = [str(args.binary), "--backend", "fake", "--n", "3",
            "--instance-id", args.instance_id, "--joint-names", "j0,j1,j2",
            "--bind", "127.0.0.1", "--udp-port", str(udp),
            "--tcp-port", str(tcp), "--hold-ms", "100", "--fault-ms", "1000",
            "--state-hz", "250", "--enable-reference"]
    if args.server_cpu is not None:
        argv = ["taskset", "-c", str(args.server_cpu), *argv]
    events = []
    with args.out.open("w") as output, args.out.with_suffix(".server.log").open("w") as server_log:
        proc = None if args.remote else subprocess.Popen(
            argv, stdout=server_log, stderr=server_log)
        try:
            time.sleep(.2)
            for run in range(1, args.runs + 1):
                success = True
                for cancel in ((False, True) if args.expiry_prelude else (True,)):
                    rt = TraceBackend(RtConfig(host=args.remote or "127.0.0.1", udp_port=udp,
                                                tcp_port=tcp, instance_id=args.instance_id),
                                      ["j0", "j1", "j2"], events=events, run=run)
                    try:
                        rt.open()
                        assert rt.backend_name == "fake" and rt.hold_ms == 100
                        assert rt.negotiate_reference("pc-stall-fixture-model")
                        rt.enable_all()
                        success = one(rt, events, run, cancel=cancel)
                    finally:
                        rt.close()
                    if not success:
                        break
                output.write(json.dumps(dict(run=run, success=success,
                                             events=events)) + "\n")
                output.flush()
                events.clear()
                if run % 20 == 0 or not success:
                    print(f"run {run}/{args.runs}: {'PASS' if success else 'FAIL'}",
                          flush=True)
        finally:
            if proc is not None:
                proc.terminate()
                proc.wait(3)


if __name__ == "__main__":
    main()
