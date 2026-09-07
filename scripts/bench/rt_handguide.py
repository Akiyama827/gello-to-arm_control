"""Rung-2 bench tool: gravity-float / impedance-hold under the RT server.

Sends NO motion commands, ever — the whole test is server reflexes:

* server launched with ``--hold-kp 0  --hold-kd 0``  -> gravity float
  (armed + no commands = hold with zero gains = zero torque on top of the
  robot's own gravity compensation; push the arm around by hand)
* server launched with ``--hold-kp 30 --hold-kd 2``  -> impedance hold
  (the arm is a spring around the pose captured at ARM)

This script only ARMs (after an explicit Enter, operator at the stop),
streams/logs the state the server publishes, and DISARMs on Enter/Ctrl-C.
A collision reflex latches the server; recover with another run
(DISARM->ARM). Everything between your hand and the motors is the path
under test: FCI -> backend -> servo law -> UDP state -> this console.

Launch the bench server with ``--fault-ms 3600000``: armed-with-no-commands
is this test's steady state, and the command-staleness deadman exists to
catch a commander that died — there is none here, and letting it latch
would make a real fault (collision reflex) indistinguishable in the flags.
The TCP-session deadman and the physical stop stay live.

Also the INSERTION-FORCE bench tool: float the arm with the module in the
gripper, push it into the dock by hand, and read the logged external wrench
(``fx..tz``, robot base frame) for the axial seating force and the lateral
force at a given misalignment — the two numbers that size the Cartesian
impedance stiffness. Needs a backend that estimates a wrench (franka does;
fake/DM log zeros with ``wrench_valid`` 0).

    python scripts/rt_handguide.py --host 172.16.1.2 --log guide.csv
"""
from __future__ import annotations

# ruff: noqa: E402  (path bootstrap must precede the arm_control import)

import sys
from pathlib import Path

# scripts/ is not a package -- bench tools are run as `python scripts/<tool>.py`,
# so put the repo root on the path the way the sibling tools do.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import argparse
import csv
import threading
import time

from arm_control.hardware.rt_backend import RtBackend, RtConfig

LOG_HZ = 50.0
CONSOLE_EVERY_S = 1.0


def _log_loop(backend: RtBackend, path: str, stop: threading.Event) -> None:
    n = backend.n
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # wrench_valid is logged alongside the six numbers on purpose: an
        # all-zero wrench is otherwise ambiguous between "no external force"
        # and "this backend does not estimate one" (fake/DM report zeros).
        w.writerow(
            ["t_mono", "flags", "fault"]
            + [f"{k}{j}" for k in ("q", "dq", "tau", "tau_cmd") for j in range(n)]
            + ["wrench_valid", "fx", "fy", "fz", "tx", "ty", "tz"]
        )
        last_console = 0.0
        last_fault = ""
        while not stop.is_set():
            state, rx_t = backend.latest_state()
            now = time.monotonic()
            if state is not None:
                wrench = list(state.wrench) or [0.0] * 6
                w.writerow(
                    [f"{rx_t:.4f}", state.flags, ""]
                    + [f"{v:.5f}" for v in (*state.q, *state.dq,
                                            *state.tau, *state.tau_cmd)]
                    + [int(state.wrench_valid)]
                    + [f"{v:.5f}" for v in wrench]
                )
                if now - last_console >= CONSOLE_EVERY_S:
                    last_console = now
                    dq = max(abs(v) for v in state.dq) if state.dq else 0.0
                    tc = max(abs(v) for v in state.tau_cmd) if state.tau_cmd else 0.0
                    mode = ("FAULTED" if state.faulted else
                            "holding" if state.holding else
                            "armed" if state.armed else "disarmed")
                    # Live force readout is the point of the insertion test:
                    # you need to see what you are pushing WHILE you push.
                    force = (f"F [{wrench[0]:+6.1f} {wrench[1]:+6.1f} "
                             f"{wrench[2]:+6.1f}] N" if state.wrench_valid
                             else "F (none)")
                    print(f"[guide] {mode:9s} max|dq| {dq:6.3f} rad/s   "
                          f"max|tau_cmd| {tc:6.2f} N.m   {force}", flush=True)
                fault = backend.motor_health()["latched_fault"]
                if fault and fault != last_fault:
                    last_fault = fault
                    print(f"[guide] SERVER FAULT: {fault} — arm is parked/latched; "
                          "disarm (Enter) and re-run to recover", flush=True)
            stop.wait(1.0 / LOG_HZ)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="172.16.1.2")
    ap.add_argument("--udp-port", type=int, default=47800)
    ap.add_argument("--tcp-port", type=int, default=47801)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--log", default="handguide.csv")
    args = ap.parse_args()

    cfg = RtConfig(host=args.host, udp_port=args.udp_port, tcp_port=args.tcp_port)
    backend = RtBackend(cfg, [f"j{i}" for i in range(args.n)])
    backend.open()
    try:
        time.sleep(0.3)  # let the state stream arrive
        state, _ = backend.latest_state()
        if state is None:
            raise SystemExit("no state stream — is the server up?")
        if state.armed:
            raise SystemExit("server already ARMED — refusing (another client?)")
        print(f"[guide] backend '{backend.backend_name}', q = "
              + " ".join(f"{v:+.3f}" for v in state.q))
        print("[guide] float vs impedance is the SERVER's --hold-kp/--hold-kd "
              "launch flags — confirm which server is running before arming.")
        input("[guide] ENTER to ARM (operator at the stop, Ctrl-C aborts): ")
        backend.enable_all()
        print(f"[guide] ARMED — guide the arm by hand; logging to {args.log}")
        stop = threading.Event()
        logger = threading.Thread(
            target=_log_loop, args=(backend, args.log, stop), daemon=True
        )
        logger.start()
        try:
            input("[guide] ENTER to DISARM and finish: ")
        finally:
            stop.set()
            logger.join(timeout=1.0)
    finally:
        disarmed = backend.safe_stop()  # verified: ack + state stream
        backend.close()
        if disarmed:
            print(f"[guide] disarmed (confirmed); log: {args.log}")
        else:
            print("[guide] DISARM NOT CONFIRMED — check the server console "
                  f"before approaching the arm; log: {args.log}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()  # ^C is a normal exit here; main's finally already disarmed
