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

    python -m arm_control.rt_handguide --host 172.16.1.2 --log guide.csv
"""
from __future__ import annotations

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
        w.writerow(
            ["t_mono", "flags", "fault"]
            + [f"{k}{j}" for k in ("q", "dq", "tau", "tau_cmd") for j in range(n)]
        )
        last_console = 0.0
        last_fault = ""
        while not stop.is_set():
            state, rx_t = backend.latest_state()
            now = time.monotonic()
            if state is not None:
                w.writerow(
                    [f"{rx_t:.4f}", state.flags, ""]
                    + [f"{v:.5f}" for v in (*state.q, *state.dq,
                                            *state.tau, *state.tau_cmd)]
                )
                if now - last_console >= CONSOLE_EVERY_S:
                    last_console = now
                    dq = max(abs(v) for v in state.dq) if state.dq else 0.0
                    tc = max(abs(v) for v in state.tau_cmd) if state.tau_cmd else 0.0
                    mode = ("FAULTED" if state.faulted else
                            "holding" if state.holding else
                            "armed" if state.armed else "disarmed")
                    print(f"[guide] {mode:9s} max|dq| {dq:6.3f} rad/s   "
                          f"max|tau_cmd| {tc:6.2f} N.m", flush=True)
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
        backend.close()  # safe_stop (disarm) + sockets
        print(f"[guide] disarmed; log: {args.log}")


if __name__ == "__main__":
    main()
