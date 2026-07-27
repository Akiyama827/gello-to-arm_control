"""Passive timing bench for a running arm_rt_server — states only, no control.

Listens to the disarmed server's state stream and measures the thing that
makes an RT box worth having: the servo thread's per-tick wakeup jitter.
``t_mono_ns`` is stamped right after the backend's absolute-deadline wakeup,
so ``t - seq * tick`` is the wakeup latency (up to a constant); its spread is
the loop's timing precision, measured on the server's own clock — the wire
only decides which ticks we get to see (state-hz samples of the 1 kHz loop).

Read-only instrument by construction: the one packet it sends (to teach the
server our address) carries kp = kd = tau_ff = 0, and the bench aborts if the
server reports ARMED.

Usage (rung 1 has the service on the box, rung 0 a local server):

    python -m arm_control.rt_timing_bench --host 172.16.1.2 --seconds 60
"""
from __future__ import annotations

import argparse
import socket
import time

from . import rt_protocol as rtp

TICK_NS = 1_000_000  # servo tick, all backends run 1 kHz


def _pct(sorted_vals: list[int], p: float) -> int:
    return sorted_vals[min(len(sorted_vals) - 1, int(p / 100.0 * len(sorted_vals)))]


def collect(host: str, port: int, seconds: float) -> list[tuple[int, int, int]]:
    """Return (state_seq, t_server_ns, t_arrival_pc_ns) samples."""
    zeros = [0.0] * 7
    prime = rtp.pack_command(
        n=7, seq=1, t_mono_ns=0, q_des=zeros, qd_des=zeros,
        tau_ff=zeros, kp=zeros, kd=zeros,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    samples: list[tuple[int, int, int]] = []
    deadline = time.monotonic() + seconds
    try:
        for attempt in range(5):
            sock.sendto(prime, (host, port))
            try:
                data, _ = sock.recvfrom(rtp.STATE_SIZE)
                break
            except socket.timeout:
                if attempt == 4:
                    raise SystemExit(f"no state stream from {host}:{port}")
        st = rtp.unpack_state(data)
        if st.armed:
            raise SystemExit("server is ARMED — bench is read-only, refusing")
        samples.append((st.state_seq, st.t_mono_ns, time.monotonic_ns()))
        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(rtp.STATE_SIZE)
            except socket.timeout:
                continue
            t_pc = time.monotonic_ns()
            st = rtp.unpack_state(data)
            if st.state_seq > samples[-1][0]:  # drop reorders/dupes
                samples.append((st.state_seq, st.t_mono_ns, t_pc))
    finally:
        sock.close()
    return samples


def report(samples: list[tuple[int, int, int]], label: str) -> None:
    n = len(samples)
    if n < 100:
        raise SystemExit(f"only {n} samples — not enough to report")
    seq0, t0, _ = samples[0]
    span_ticks = samples[-1][0] - seq0
    span_s = (samples[-1][1] - t0) / 1e9

    # Servo wakeup jitter (server clock, absolute deadlines -> slope is TICK_NS
    # exactly; residual = wakeup latency up to the unknown latency of sample 0).
    resid = [(t - t0) - (seq - seq0) * TICK_NS for seq, t, _ in samples]
    lo = min(resid)
    jit = sorted(r - lo for r in resid)

    # Stream cadence as seen at this end (tx pacing + wire + our own stack).
    gaps = sorted(
        samples[i][2] - samples[i - 1][2] for i in range(1, n)
    )
    med_gap = gaps[len(gaps) // 2]
    losses = sum(1 for g in gaps if g > 2.5 * med_gap)

    print(f"== rt_timing_bench: {label} ==")
    print(f"samples {n}  span {span_s:.1f}s  ticks {span_ticks}  "
          f"(every ~{span_ticks // (n - 1)}th tick)")
    print("servo wakeup jitter, ns above best-case (server clock):")
    print(f"  p50 {_pct(jit, 50):>8,}   p90 {_pct(jit, 90):>8,}   "
          f"p99 {_pct(jit, 99):>8,}")
    print(f"  p99.9 {_pct(jit, 99.9):>6,}   max {jit[-1]:>8,}   "
          f"peak-to-peak {jit[-1]:,}")
    print("stream cadence at receiver, us:")
    print(f"  median {med_gap / 1e3:.0f}   p99 {_pct(gaps, 99) / 1e3:.0f}   "
          f"max {gaps[-1] / 1e3:.0f}   gaps>2.5x-median {losses}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47800)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    label = args.label or f"{args.host}:{args.port}"
    report(collect(args.host, args.port, args.seconds), label)


if __name__ == "__main__":
    main()
