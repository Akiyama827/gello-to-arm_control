"""Bench tool for the Dock_Control CAN nodes (active latch + passive sensor).

Drives the dock protocol (PROTOCOL.md in the Dock_Control tree) through a
SocketCAN FD interface — no robot, no orchestrator. This is the E1b
verification tool and the instrument for the hand-mate FSM dry-run: it
measures the enumeration debounce, the latch travel time, and the one-hot
pin↔angle map that E2/E3 consume as constants.

Transport: the dmcan USB2FDCAN enumerates as a candleLight/gs_usb device
(1d50:606f), so the kernel owns it as ``can0``. Bring the link up once with
the dock bus timing (1 Mbps nominal / 4 Mbps data, from the firmware's
MX_FDCAN1_Init):

    sudo ip link set can0 up type can bitrate 1000000 dbitrate 4000000 fd on

Then:

    python -m arm_control.dock_bench watch  --active 0xF1 --passive 0xF3
    python -m arm_control.dock_bench latch  --active 0xF1
    python -m arm_control.dock_bench unlatch --active 0xF1
    python -m arm_control.dock_bench state  --active 0xF1
    python -m arm_control.dock_bench sense  --passive 0xF3
    python -m arm_control.dock_bench scan

`watch` prints a line on every observed CHANGE with a relative timestamp —
read the debounce directly off the flap timestamps during a hand insert.
`latch`/`unlatch` command the servo and poll the 0x02 state query until the
target state lands (exit 0) or --timeout expires (exit 1). `scan` is
read-only (state + sensor queries, never latch verbs).
"""
from __future__ import annotations

import argparse
import sys
import time

# Wire constants from Dock_Control/PROTOCOL.md.
CMD_UNLATCH = 0x00
CMD_LATCH = 0x01
CMD_STATE_QUERY = 0x02
ACTIVE_CMD_OFFSET = 0x10   # command ID = DOCK_ID - 0x10
PASSIVE_QUERY_OFFSET = 0x20  # query ID = DOCK_ID - 0x20
DOCK_ID_LO, DOCK_ID_HI = 0xF0, 0xFF

STATE_NAMES = {0: "UNKNOWN", 1: "LATCHED", 2: "UNLATCHED", 3: "READ_ERROR"}
REPLY_WAIT_S = 0.05   # per-query reply deadline (read takes ~3 ms + loop slack)


def describe_bits(bits: int | None) -> str:
    """Human name for a passive sensor byte (None = no reply)."""
    if bits is None:
        return "no node (query timeout)"
    if bits == 0:
        return "0x00 (contact, NOT seated)"
    if bits in (0x01, 0x02, 0x04, 0x08):
        return f"0x{bits:02X} (seated, one-hot bit {bits.bit_length() - 1})"
    return f"0x{bits:02X} (MULTI-BIT — partial contact, NOT seated)"


def parse_state_reply(payload: bytes) -> tuple[str, float | None] | None:
    """Decode a [0x02, state, pos_lo, pos_hi] reply -> (name, degrees|None)."""
    if len(payload) < 4 or payload[0] != CMD_STATE_QUERY:
        return None
    state = STATE_NAMES.get(payload[1], f"?{payload[1]}")
    pos = payload[2] | (payload[3] << 8)
    deg = None if pos == 0xFFFF else pos * 270.0 / 1024.0
    return state, deg


class DockBus:
    """Dock-protocol master over a SocketCAN FD interface."""

    def __init__(self, iface: str) -> None:
        import can

        self._can = can
        try:
            # Kernel-side filter: only the response band 0xF0-0xFF reaches us
            # (standard 11-bit IDs; our own 0xDn/0xEn TX is not looped back).
            self.bus = can.Bus(
                interface="socketcan",
                channel=iface,
                fd=True,
                can_filters=[{"can_id": 0x0F0, "can_mask": 0x7F0}],
            )
        except OSError as exc:
            raise SystemExit(
                f"[bench] cannot open {iface}: {exc}\n"
                "  is the link up?  sudo ip link set "
                f"{iface} up type can bitrate 1000000 dbitrate 4000000 fd on"
            ) from exc

    def close(self) -> None:
        self.bus.shutdown()

    def send(self, can_id: int, payload: bytes) -> None:
        self.bus.send(self._can.Message(
            arbitration_id=can_id,
            data=payload,
            is_extended_id=False,
            is_fd=True,
            bitrate_switch=True,
        ))

    def _await(self, dock_id: int, want, deadline: float):
        """First frame from dock_id satisfying want(payload), or None."""
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            msg = self.bus.recv(timeout=left)
            if msg is None:
                return None
            if msg.arbitration_id == dock_id and want(bytes(msg.data)):
                return bytes(msg.data)

    def query_passive(self, dock_id: int) -> int | None:
        self.send(dock_id - PASSIVE_QUERY_OFFSET, b"\x00")
        payload = self._await(
            dock_id, lambda p: len(p) >= 1, time.monotonic() + REPLY_WAIT_S
        )
        return None if payload is None else payload[0]

    def query_state(self, dock_id: int) -> tuple[str, float | None] | None:
        self.send(dock_id - ACTIVE_CMD_OFFSET, bytes([CMD_STATE_QUERY]))
        payload = self._await(
            dock_id,
            lambda p: parse_state_reply(p) is not None,
            time.monotonic() + REPLY_WAIT_S,
        )
        return None if payload is None else parse_state_reply(payload)

    def command(self, dock_id: int, verb: int) -> bool:
        """Send latch/unlatch; True when the 1-byte echo confirms reception."""
        self.send(dock_id - ACTIVE_CMD_OFFSET, bytes([verb]))
        echo = self._await(
            dock_id,
            lambda p: len(p) == 1 and p[0] == verb,
            time.monotonic() + REPLY_WAIT_S,
        )
        return echo is not None


def _dock_id(text: str) -> int:
    value = int(text, 0)
    if not DOCK_ID_LO <= value <= DOCK_ID_HI:
        raise argparse.ArgumentTypeError(f"dock id 0x{value:02X} outside 0xF0-0xFF")
    return value


def cmd_watch(bus: DockBus, args) -> int:
    t0 = time.monotonic()
    last: dict[str, str] = {}
    print("[bench] watching — hand-mate now; lines print on CHANGE (Ctrl-C ends)")
    while True:
        readings: list[tuple[str, str]] = []
        if args.active is not None:
            st = bus.query_state(args.active)
            name = "no node (query timeout)" if st is None else (
                st[0] if st[1] is None else f"{st[0]} {st[1]:.1f}deg"
            )
            readings.append((f"active 0x{args.active:02X}", name))
        if args.passive is not None:
            readings.append(
                (f"passive 0x{args.passive:02X}",
                 describe_bits(bus.query_passive(args.passive)))
            )
        for who, now_s in readings:
            if last.get(who) != now_s:
                print(f"[bench] t+{time.monotonic() - t0:7.3f}s  {who}: "
                      f"{last.get(who, '(start)')} -> {now_s}", flush=True)
                last[who] = now_s
        time.sleep(max(0.0, 1.0 / args.rate))


def cmd_move(bus: DockBus, args, verb: int, target: str) -> int:
    if not bus.command(args.active, verb):
        print(f"[bench] no echo from active 0x{args.active:02X} — node absent "
              "or old firmware", flush=True)
        return 1
    t0 = time.monotonic()
    st = None
    while time.monotonic() - t0 < args.timeout:
        st = bus.query_state(args.active)
        if st is not None and st[0] == target:
            took = time.monotonic() - t0
            pos = "" if st[1] is None else f" (pos {st[1]:.1f}deg)"
            print(f"[bench] {target} in {took:.2f}s{pos}", flush=True)
            return 0
        time.sleep(0.05)
    print(f"[bench] TIMEOUT after {args.timeout:.1f}s — last: "
          f"{st[0] if st else 'no reply'}", flush=True)
    return 1


def cmd_scan(bus: DockBus, _args) -> int:
    """Probe every dock ID band once; read-only (0x02 + passive queries)."""
    found = 0
    for dock_id in range(DOCK_ID_LO, DOCK_ID_HI + 1):
        st = bus.query_state(dock_id)
        if st is not None:
            deg = "" if st[1] is None else f" {st[1]:.1f}deg"
            print(f"[bench] 0x{dock_id:02X} ACTIVE  {st[0]}{deg}", flush=True)
            found += 1
            continue
        bits = bus.query_passive(dock_id)
        if bits is not None:
            print(f"[bench] 0x{dock_id:02X} PASSIVE {describe_bits(bits)}",
                  flush=True)
            found += 1
    print(f"[bench] scan complete: {found} node(s)", flush=True)
    return 0


def _demo() -> None:
    assert describe_bits(None) == "no node (query timeout)"
    assert "NOT seated" in describe_bits(0x00)
    assert describe_bits(0x04) == "0x04 (seated, one-hot bit 2)"
    assert "MULTI-BIT" in describe_bits(0x05)
    assert parse_state_reply(bytes([0x02, 1, 0x1C, 0x01])) == (
        "LATCHED", 284 * 270.0 / 1024.0)
    assert parse_state_reply(bytes([0x02, 3, 0xFF, 0xFF])) == ("READ_ERROR", None)
    assert parse_state_reply(bytes([0x01])) is None      # plain echo, not a state
    assert parse_state_reply(bytes([0x03, 1, 0, 0])) is None
    assert _dock_id("0xF3") == 0xF3
    try:
        _dock_id("0x10")
        raise AssertionError("accepted out-of-band id")
    except argparse.ArgumentTypeError:
        pass
    print("dock_bench demo ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true", help="run the offline self-check")
    ap.add_argument("--active", type=_dock_id, default=None,
                    help="ACTIVE node DOCK_ID (e.g. 0xF1)")
    ap.add_argument("--passive", type=_dock_id, default=None,
                    help="PASSIVE node DOCK_ID (e.g. 0xF3)")
    ap.add_argument("--iface", default="can0", help="SocketCAN interface")
    ap.add_argument("--rate", type=float, default=20.0, help="watch poll Hz")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="latch/unlatch settle timeout (s)")
    ap.add_argument("command", nargs="?",
                    choices=("watch", "latch", "unlatch", "state", "sense", "scan"))
    args = ap.parse_args()

    if args.demo:
        _demo()
        return 0
    if args.command is None:
        ap.error("a command is required (or --demo)")
    if args.command in ("latch", "unlatch", "state") and args.active is None:
        ap.error(f"{args.command} needs --active")
    if args.command == "sense" and args.passive is None:
        ap.error("sense needs --passive")
    if args.command == "watch" and args.active is None and args.passive is None:
        ap.error("watch needs --active and/or --passive")

    bus = DockBus(args.iface)
    try:
        if args.command == "watch":
            return cmd_watch(bus, args)
        if args.command == "latch":
            return cmd_move(bus, args, CMD_LATCH, "LATCHED")
        if args.command == "unlatch":
            return cmd_move(bus, args, CMD_UNLATCH, "UNLATCHED")
        if args.command == "state":
            st = bus.query_state(args.active)
            print(f"[bench] {'no reply' if st is None else st}", flush=True)
            return 0 if st is not None else 1
        if args.command == "sense":
            print(f"[bench] {describe_bits(bus.query_passive(args.passive))}",
                  flush=True)
            return 0
        return cmd_scan(bus, args)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        bus.close()


if __name__ == "__main__":
    sys.exit(main())
