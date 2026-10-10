"""Physical start skew of a released pair, observed read-only (bench 2026-10-10, open item: "no instrument").

    python tools/hw/pair_skew.py --socket fr3_1=<sock> --socket fr3_2=<sock> [--seconds 15] [--out FILE]
    python tools/hw/pair_skew.py --analyze FILE        # re-run the estimate on a saved record
    python tools/hw/pair_skew.py --self-check

Start it, then press Execute pair (the supervisors' sockets are the "[supervisor] up on <sock>" lines). It never claims,
ARMs or commands: it polls each supervisor's trace/1 (the PC controller's ticks: reference phase, commanded q) and
state_history/1 (every RT packet the PC received: measured q, stamped on the RT box's clock, which both servers share).

Per arm it fits ONE delay d: the measured joints over the leg's first second against the arm's own reference, shifted by d
from that leg's anchor (its reference time zero on the PC clock; a scheduled pair gives both the same one). d holds
everything between the anchor and the arm actually moving -- the PC tick phase of its first command, the RT tick that
applies it, the robot's tracking lag -- so each arm moves at anchor + d, and the PHYSICAL skew is the difference of those. Packets come at 250 Hz (4 ms); the fit uses every packet of the window, so it resolves
well below that on a smooth leg, and the residual it prints says whether the fit holds. The box->PC clock mapping is each
arm's minimum (PC receipt - RT stamp), i.e. one network latency, common to both arms to ~0.1 ms (printed per arm).
It reports the PC-side skew of the first commanded tick too (the supervisors' "first sample" lines).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

FIT_WINDOW_S = 1.0           # of the leg, from its anchor
MOVING_RAD = 0.005           # a joint takes part in the fit if its reference moves this much inside the window
DELAYS_S = np.arange(-0.020, 0.150, 0.0001)


def record(sockets: dict[str, str], seconds: float, poll_s: float = 0.05) -> dict:
    from arm_control.sdk import ArmClient

    clients = {arm: ArmClient(arm, socket_path=path) for arm, path in sockets.items()}
    out = {arm: dict(trace=[], packets={}, trace_dropped=0) for arm in clients}
    cursors = dict.fromkeys(clients, 0)
    end = time.monotonic() + seconds
    print(f"[pair_skew] recording {seconds:g} s: press Execute pair now", flush=True)
    try:
        while time.monotonic() < end:
            for arm, client in clients.items():
                batch = client.trace_samples(cursors[arm])
                cursors[arm] = batch["latest_seq"]
                out[arm]["trace_dropped"] += batch["dropped"]
                out[arm]["trace"] += [{k: s.get(k) for k in ("seq", "pc_monotonic", "trajectory_elapsed_s", "q_cmd_pc",
                                                            "plan_id", "executing", "state_t_mono_ns", "received_at_pc")}
                                      for s in batch["samples"]]
                history = client.state_history()
                now_pc = history["now_ns"] / 1e9
                out[arm]["clock_domain"] = history["clock_domain"]
                for p in history["samples"]:  # keyed by seq: overlapping polls deduplicate
                    out[arm]["packets"][str(p["state_seq"])] = dict(t_rt=p["t_mono_ns"] / 1e9,
                                                                    rx_pc=now_pc - p["age_ns"] / 1e9, q=p["q"])
            time.sleep(poll_s)
    finally:
        for client in clients.values():
            client.close()
    for arm in out:
        out[arm]["packets"] = sorted(out[arm]["packets"].values(), key=lambda p: p["t_rt"])
    return out


def _leg(rec: dict):
    """The executing samples of the first leg in the record: (plan id, anchor on the PC clock, phases, commanded q)."""
    moving = [s for s in rec["trace"] if s["executing"] and s["trajectory_elapsed_s"] is not None and s["q_cmd_pc"]]
    if not moving:
        raise ValueError("no executing tick in the record (was Execute pair pressed inside the window?)")
    plan = moving[0]["plan_id"]
    leg = [s for s in moving if s["plan_id"] == plan]
    e = np.array([s["trajectory_elapsed_s"] for s in leg])
    anchor = float(np.median([s["pc_monotonic"] - s["trajectory_elapsed_s"] for s in leg]))
    first = next(s["pc_monotonic"] for s in leg if s["trajectory_elapsed_s"] >= 0)
    order = np.argsort(e)
    return plan, anchor, first, e[order], np.array([leg[i]["q_cmd_pc"] for i in order])


def estimate(record_: dict) -> dict:
    result = {}
    for arm, rec in record_.items():
        plan, anchor, first, e, q_ref = _leg(rec)
        packets = rec["packets"]
        if len(packets) < 20:
            raise ValueError(f"{arm}: {len(packets)} RT packets recorded; polling fell behind")
        t_rt = np.array([p["t_rt"] for p in packets])
        q = np.array([p["q"] for p in packets])
        offset = float(np.min([p["rx_pc"] - p["t_rt"] for p in packets]))  # box -> PC, one latency included
        phase = t_rt + offset - anchor                                      # each packet's reference phase at d = 0
        window = (e >= 0) & (e <= FIT_WINDOW_S)
        joints = np.flatnonzero(np.ptp(q_ref[window], axis=0) > MOVING_RAD) if window.any() else []
        if len(joints) == 0:
            raise ValueError(f"{arm}: no joint moves {MOVING_RAD} rad in the leg's first {FIT_WINDOW_S:g} s")
        use = (phase >= -0.05) & (phase <= FIT_WINDOW_S + 0.05)
        best = None
        for d in DELAYS_S:
            ref = np.stack([np.interp(phase[use] - d, e, q_ref[:, j]) for j in joints], axis=1)
            err = float(np.sqrt(np.mean((q[use][:, joints] - ref) ** 2)))
            if best is None or err < best[1]:
                best = (float(d), err)
        result[arm] = dict(plan_id=plan, delay_ms=best[0] * 1e3, fit_rms_mrad=best[1] * 1e3, joints=[int(j) + 1 for j in joints],
                           packets=int(use.sum()), clock_offset_s=offset, first_command_pc=first, anchor_pc=anchor,
                           trace_dropped=rec["trace_dropped"], clock_domain=rec.get("clock_domain"))
    arms = list(result)
    if len(arms) == 2:
        a, b = arms
        domains = {result[a]["clock_domain"], result[b]["clock_domain"]}
        result["skew"] = dict(
            physical_ms=(result[a]["anchor_pc"] - result[b]["anchor_pc"]) * 1e3 + result[a]["delay_ms"] - result[b]["delay_ms"],
            pc_first_command_ms=(result[a]["first_command_pc"] - result[b]["first_command_pc"]) * 1e3,
            anchor_difference_ms=(result[a]["anchor_pc"] - result[b]["anchor_pc"]) * 1e3,
            clock_offset_difference_ms=(result[a]["clock_offset_s"] - result[b]["clock_offset_s"]) * 1e3,
            one_rt_clock=len(domains) == 1 and None not in domains)
    return result


def _synthetic(delays_s: dict, anchors=(100.0, 100.0), offset=-50.0, seed=0) -> dict:
    """Two arms on one RT clock: a smooth 2 s leg each (different amplitudes and joints), 100 Hz PC trace, 250 Hz packets
    with 0.2 mrad noise and a 0.1-0.4 ms receipt latency; each arm moves ``delays_s[arm]`` after the anchor."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, (arm, d) in enumerate(delays_s.items()):
        anchor = anchors[k]
        amp = np.zeros(7)
        amp[[0, 3] if k == 0 else [1, 5]] = (.3, -.2) if k == 0 else (.05, .4)
        s = lambda t: np.clip(t / 2.0, 0, 1)  # noqa: E731
        ref = lambda t: np.outer(10 * s(t) ** 3 - 15 * s(t) ** 4 + 6 * s(t) ** 5, amp)  # noqa: E731
        e = np.arange(-0.5, 2.5, 0.01) + rng.uniform(0, .01)
        trace = [dict(seq=i, pc_monotonic=anchor + x, trajectory_elapsed_s=float(x), q_cmd_pc=ref(x)[0].tolist(),
                      plan_id="pair-1", executing=True, state_t_mono_ns=None, received_at_pc=None)
                 for i, x in enumerate(e)]
        t_rt = np.arange(anchors[0] - offset - 0.5, anchors[0] - offset + 2.5, 0.004) + rng.uniform(0, .004)
        q = ref(t_rt + offset - anchor - d) + rng.normal(0, 2e-4, (len(t_rt), 7))
        packets = [dict(t_rt=float(t), rx_pc=float(t + offset + rng.uniform(1e-4, 4e-4)), q=row.tolist())
                   for t, row in zip(t_rt, q)]
        out[arm] = dict(trace=trace, packets=packets, trace_dropped=0, clock_domain="box:1")
    return out


def _self_check() -> None:
    got = estimate(_synthetic({"fr3_1": 0.0062, "fr3_2": 0.0021}))
    assert abs(got["skew"]["physical_ms"] - 4.1) < 0.4, got["skew"]
    assert got["skew"]["one_rt_clock"] and all(got[a]["fit_rms_mrad"] < 0.5 for a in ("fr3_1", "fr3_2")), got
    swapped = estimate(_synthetic({"fr3_1": 0.0021, "fr3_2": 0.0062}, seed=1))
    assert abs(swapped["skew"]["physical_ms"] + 4.1) < 0.4, swapped["skew"]
    # released apart (no common anchor): each arm moves at its own anchor + d
    apart = estimate(_synthetic({"fr3_1": 0.0062, "fr3_2": 0.0021}, anchors=(100.0, 100.003), seed=2))
    assert abs(apart["skew"]["physical_ms"] - 1.1) < 0.4 and abs(apart["skew"]["anchor_difference_ms"] + 3) < .1, apart
    still = _synthetic({"fr3_1": 0.0, "fr3_2": 0.0})
    for s in still["fr3_2"]["trace"]:
        s["executing"] = False
    try:
        estimate(still)
    except ValueError as exc:
        assert "no executing tick" in str(exc)
    else:
        raise AssertionError("a record without a leg was estimated")
    print("pair_skew: recovers a 4.1 ms physical skew (either sign, common or separate anchors) from 250 Hz packets "
          "on one RT clock; no leg refused OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", action="append", default=[], help="ARM=SOCKET, twice")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--out", type=Path, help="save the raw record (JSON)")
    parser.add_argument("--analyze", type=Path, help="estimate from a saved record instead of recording")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.analyze:
        rec = json.loads(args.analyze.read_text())
    else:
        sockets = dict(item.split("=", 1) for item in args.socket)
        if len(sockets) != 2:
            sys.exit("give --socket ARM=PATH for both arms")
        rec = record(sockets, args.seconds)
        if args.out:
            args.out.write_text(json.dumps(rec))
            print(f"[pair_skew] record saved: {args.out}")
    print(json.dumps(estimate(rec), indent=2))


if __name__ == "__main__":
    main()
