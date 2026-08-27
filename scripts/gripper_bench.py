"""Franka Hand bench: grasp-force threshold + slip testing.

WHAT THE HAND CAN AND CANNOT TELL YOU. The Franka Hand has no force sensor.
``GripperState`` carries width + is_grasped + temperature, and that is all —
so there is no live "current force" to display, on this device, ever. The
``force_n`` in ``franka.gripper`` is a COMMANDED setpoint (how hard the jaw
motor is allowed to squeeze), and ``grasp()`` answers one question: did the
final width land inside the epsilon band, i.e. is something held.

That makes slip a PHYSICAL measurement here, not a telemetry read:

  1. put the object in the jaws, ``g 20`` (grasp at 20 N),
  2. pull on it by hand — or hang a known weight off it,
  3. watch ``is_grasped``. It falling is the drop event.
  4. ``sweep 10 100 10`` walks force up so you can find the lowest force
     that survives your pull. THAT number is your real grip threshold.

Talks the hand_bridge TCP line protocol directly (rt/src/hand_bridge.cpp) —
no dora graph, no arm, no FCI session. Deliberately gripper-only: benching
the jaws must not require the whole motion stack to be up.

Usage:
    python scripts/gripper_bench.py --config configs/real/franka.yaml
        [--host 172.16.1.2] [--port 47802]

Commands:
    g [N]              grasp at the configured width, optional force override
    o                  open to the configured open width
    w <mm>             move to a width in mm
    sweep [lo hi step] grasp repeatedly, stepping force (default 30 70 10 —
                       the Hand's full adjustable range)
    s                  print the live sample once
    q                  quit (leaves the jaws where they are)
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

from arm_control.config import load_robot_config

# No GDONE arrives for a GRASP killed mid-flight (GSTOP, bridge reconnect) —
# same reason nodes/franka_gripper.py carries a timeout. Without this the
# bench would spin forever on a grasp the Hand already abandoned.
GRASP_TIMEOUT_S = 8.0

# Franka Hand Product Manual 1.2 (April 2022), Technical Data:
#   "Grasping (continuous) force adjustable [N] 30-70"
# It is a RANGE, not just a ceiling — 30 N is the floor, and commanding below it
# is not a gentler grasp, it is out of spec. The 140 N "max force" that reseller
# spec pages quote appears NOWHERE in that manual; do not design against it.
# libfranka itself documents no limit and does not clamp (checked
# /usr/include/franka/gripper.h), so nothing warns you at the API level.
MIN_FORCE_N = 30.0
MAX_CONTINUOUS_FORCE_N = 70.0


def grip_cfg(cfg) -> dict:
    """The ``franka.gripper`` block, defaults matching FrankaConfig."""
    g = dict((dict(cfg.get("franka") or {})).get("gripper") or {})
    return {
        "open_width_m": float(g.get("open_width_m", 0.075)),
        "grasp_width_m": float(g.get("grasp_width_m", 0.045)),
        "speed_mps": float(g.get("speed_mps", 0.05)),
        "force_n": float(g.get("force_n", 40.0)),
        "eps_in": float(g.get("epsilon_inner_m", 0.02)),
        "eps_out": float(g.get("epsilon_outer_m", 0.02)),
    }


class Bench:
    """One hand_bridge socket; parses its STATE/GDONE stream."""

    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=3.0)
        self.buf = b""
        self.state: dict | None = None   # {"width": m, "is_grasped": bool}
        self.gdone: bool | None = None   # verdict of the most recent GRASP

    def send(self, line: str) -> None:
        self.sock.sendall((line + "\n").encode())

    def drain(self, timeout_s: float) -> None:
        """Absorb the stream for up to timeout_s, updating state/gdone."""
        self.sock.settimeout(timeout_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data = self.sock.recv(256)
            except socket.timeout:
                return
            if not data:
                raise OSError("hand_bridge closed the connection")
            self.buf += data
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                parts = line.decode(errors="replace").split()
                if len(parts) == 3 and parts[0] == "STATE":
                    self.state = {"width": float(parts[1]),
                                  "is_grasped": parts[2] == "1"}
                elif len(parts) == 2 and parts[0] == "GDONE":
                    self.gdone = parts[1] == "1"

    def await_verdict(self) -> bool | None:
        """Block for this grasp's GDONE; None if the Hand never answered."""
        self.gdone = None
        deadline = time.monotonic() + GRASP_TIMEOUT_S
        while self.gdone is None and time.monotonic() < deadline:
            self.drain(0.2)
        return self.gdone

    def show(self, prefix: str = "  ") -> None:
        self.drain(0.3)
        s = self.state
        if s is None:
            print(f"{prefix}(no STATE yet from hand_bridge)")
            return
        print(f"{prefix}width={s['width'] * 1e3:6.1f} mm   "
              f"is_grasped={'YES' if s['is_grasped'] else 'no '}")

    def close(self) -> None:
        self.sock.close()


def grasp(b: Bench, g: dict, force_n: float) -> bool | None:
    b.send(f"GRASP {g['grasp_width_m']:.5f} {g['speed_mps']:.3f} {force_n:.1f} "
           f"{g['eps_in']:.4f} {g['eps_out']:.4f}")
    return b.await_verdict()


def step(b: Bench, cmd: str, args: list[str], g: dict) -> bool:
    """Run one REPL command. False ends the session."""
    try:
        if cmd in ("q", "quit"):
            return False
        elif cmd == "g":
            force = float(args[0]) if args else g["force_n"]
            print(f"> GRASP {g['grasp_width_m'] * 1e3:.1f} mm at {force:.0f} N "
                  f"(band +{g['eps_out'] * 1e3:.0f}/-{g['eps_in'] * 1e3:.0f} mm)")
            held = grasp(b, g, force)
            print(f"  grasp() -> {'HELD' if held else 'no object' if held is False else 'NO VERDICT (timeout)'}")
            print("  now pull on it — is_grasped falling is the slip:")
        elif cmd == "o":
            b.send(f"MOVE {g['open_width_m']:.5f} {g['speed_mps']:.3f}")
            print(f"> open -> {g['open_width_m'] * 1e3:.0f} mm")
        elif cmd == "w":
            width_m = float(args[0]) / 1e3
            b.send(f"MOVE {width_m:.5f} {g['speed_mps']:.3f}")
            print(f"> move -> {width_m * 1e3:.1f} mm")
        elif cmd == "sweep":
            lo = float(args[0]) if len(args) > 0 else MIN_FORCE_N
            hi = float(args[1]) if len(args) > 1 else MAX_CONTINUOUS_FORCE_N
            inc = float(args[2]) if len(args) > 2 else 10.0
            if lo < MIN_FORCE_N or hi > MAX_CONTINUOUS_FORCE_N:
                print(f"  ! outside the Hand's spec range "
                      f"{MIN_FORCE_N:.0f}-{MAX_CONTINUOUS_FORCE_N:.0f} N — "
                      "firmware may clamp silently or throw")
            print(f"> sweep {lo:.0f}..{hi:.0f} N step {inc:.0f} — pull on the "
                  "object at each step; the lowest force that survives is your "
                  "threshold")
            f = lo
            while f <= hi + 1e-9:
                b.send(f"MOVE {g['open_width_m']:.5f} {g['speed_mps']:.3f}")
                time.sleep(1.0)
                held = grasp(b, g, f)
                b.drain(0.3)
                w = b.state["width"] * 1e3 if b.state else float("nan")
                print(f"  {f:5.0f} N  grasp()="
                      f"{'HELD    ' if held else 'no object' if held is False else 'timeout '}"
                      f"  width={w:6.1f} mm")
                f += inc
        elif cmd == "s":
            pass  # the show() below is the whole command
        else:
            print(f"  ? '{cmd}' — commands: g [N] | o | w <mm> | sweep | s | q")
            return True
        b.show()
    except (ValueError, IndexError):
        print("  ? bad arguments")
    return True


def _demo() -> None:
    """Self-check: stream parsing, the verdict wait, and its timeout."""
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    seen: list[bytes] = []

    def bridge() -> None:
        conn, _ = srv.accept()
        conn.settimeout(5.0)
        conn.sendall(b"STATE 0.07500 0\n")
        buf = b""
        while True:
            try:
                chunk = conn.recv(256)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                seen.append(line)
                if line.startswith(b"GRASP"):
                    # 40 N holds in this fake plant; 10 N closes on air; the
                    # third grasp is the killed-mid-flight case (no GDONE).
                    if b" 40.0 " in line:
                        conn.sendall(b"GDONE 1\nSTATE 0.04510 1\n")
                    elif b" 10.0 " in line:
                        conn.sendall(b"GDONE 0\nSTATE 0.00050 0\n")
                elif line.startswith(b"MOVE"):
                    w = float(line.split()[1])
                    conn.sendall(f"STATE {w:.5f} 0\n".encode())
        conn.close()

    threading.Thread(target=bridge, daemon=True).start()
    g = grip_cfg({"franka": {"gripper": {}}})
    assert g["force_n"] == 40.0 and g["grasp_width_m"] == 0.045, g

    b = Bench("127.0.0.1", srv.getsockname()[1])
    # a held grasp: GDONE 1, and the post-grasp STATE parses through
    assert grasp(b, g, 40.0) is True
    assert b.state == {"width": 0.04510, "is_grasped": True}, b.state
    assert seen and seen[-1].startswith(b"GRASP 0.04500"), seen[-1]
    # closed on air: a real False verdict, distinct from a missing one
    assert grasp(b, g, 10.0) is False
    assert b.state["is_grasped"] is False, b.state
    # a MOVE re-opens, and width tracks it
    b.send(f"MOVE {g['open_width_m']:.5f} {g['speed_mps']:.3f}")
    b.drain(0.5)
    assert abs(b.state["width"] - 0.075) < 1e-6, b.state
    # no GDONE (grasp killed mid-flight) must time out, not hang forever
    global GRASP_TIMEOUT_S
    GRASP_TIMEOUT_S = 0.5
    t0 = time.monotonic()
    assert grasp(b, g, 99.0) is None
    assert 0.4 < time.monotonic() - t0 < 3.0, "verdict timeout misbehaved"
    b.close()
    srv.close()
    print("gripper_bench demo: ok")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Franka Hand grasp-force / slip bench.",
        epilog="The Hand reports no force — see the module docstring.",
    )
    p.add_argument("--config", default="configs/real/franka.yaml")
    p.add_argument("--host", default=None, help="hand_bridge host (default: rt.host)")
    p.add_argument("--port", type=int, default=None, help="default: rt.hand_port")
    args = p.parse_args(argv)

    cfg = load_robot_config(args.config)
    rt = dict(cfg.get("rt") or {})
    host = args.host or str(rt.get("host", "172.16.1.2"))
    port = args.port or int(rt.get("hand_port", 47802))
    g = grip_cfg(cfg)

    print(f"[bench] force threshold (COMMANDED, not measured): {g['force_n']:.0f} N")
    print(f"[bench] grasp width {g['grasp_width_m'] * 1e3:.0f} mm, band "
          f"+{g['eps_out'] * 1e3:.0f}/-{g['eps_in'] * 1e3:.0f} mm, "
          f"open {g['open_width_m'] * 1e3:.0f} mm, speed {g['speed_mps']} m/s")
    print("[bench] the Hand has NO force sensor: is_grasped is the slip signal")
    try:
        b = Bench(host, port)
    except OSError as exc:
        print(f"[bench] cannot reach hand_bridge at {host}:{port} ({exc})")
        print("        is it running on the RT box?")
        return 1
    print(f"[bench] connected to {host}:{port} — g [N] | o | w <mm> | sweep | s | q")
    try:
        b.show("[bench] ")
        while True:
            sys.stdout.write("> ")
            sys.stdout.flush()
            raw = sys.stdin.readline()
            if not raw:
                break
            parts = raw.split()
            if parts and not step(b, parts[0], parts[1:], g):
                break
    except OSError as exc:
        print(f"[bench] link lost: {exc}")
        return 1
    finally:
        b.close()
    return 0


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        sys.exit(main())
