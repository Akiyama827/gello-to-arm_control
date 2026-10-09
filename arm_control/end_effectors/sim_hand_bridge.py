"""A simulated Franka Hand behind the real ``hand_bridge`` line protocol (scene hosting D1; operator 2026-10-08).

The real Hand is its own device: ``rt/src/hand_bridge.cpp`` on the RT box owns it and re-exports a TCP line protocol that
``franka_adapter``'s ``BridgeClient`` drives. A world hosted by the supervisor serves THE SAME protocol, backed by the world's
finger actuators, so the Hand path (``franka_adapter`` + ``HandGraspFsm``) is one code path in simulation and on hardware --
``SimLink``'s principle, applied to the Hand. Ported from the C++ server, rule for rule:

* admission (``HandAdmission``): one command, never a queue; the client must echo the current epoch; the epoch bumps on every
  admission, cancellation and client change; REJECT reasons ``offline``, ``stopping``, ``stale_epoch``, ``busy``,
  ``stale_state``, ``invalid_command``;
* a new action needs a measured sample under 0.6 s old, taken after the previous action ended;
* every 0.1 s ``LIMITS <width_max_m> <speed_max_mps> <force_min_n> <force_max_n>`` (the envelope it admits; the Franka
  Hand's unless constructed with another), ``STATE <width> <grasped>``, ``CAPS active_stop`` and
  ``META 2 <epoch> <available> <busy> <measured> <seq>``;
  ``ACCEPT|REJECT <token> <reason>`` per command, ``DONE <token> <ok>`` per finished action, ``GDONE <ok>`` per finished
  GRASP, ``STOPPING <epoch>`` per GSTOP; a stopped action reports nothing (its epoch is gone);
* a disconnect stops committed work, and otherwise only cancels (a completed grasp keeps holding); one client at a time.

Not modelled, because it cannot happen here: a lost device session and a failed stop (``stop_failed``). The sim's stop
completes on the next plant tick.

The finger physics is the project's measured SimHand model (from its ``sim_bridge``, deleted 2026-10-09), moved here so the hosted
world owns it: the width command RAMPS at the commanded speed (a raw target slams the pads off a curved hull); a GRASP's
verdict is width-only like ``franka::Gripper::grasp`` (settled within the epsilon band AND blocked short of the command,
which an empty close is not); a held grasp hands the fingers to force mode at the commanded force (a position target only
pinches a centred object); ``grasped`` needs the band and servo load (a dropped module parks the servos unloaded). The
project's SimHand/SimHandMove retire when ``sim_bench`` runs on this server (step 2c).
"""
from __future__ import annotations

import math
import re
import select
import socket
import threading
import time

from arm_control.end_effectors.franka_hand import FRANKA_HAND, HandLimits
from arm_control.joint_motor_map import finger_from_width, width_from_finger

CLOSED_TOL_M = 1e-5         # the C++ observe_state's sample envelope: below -1e-5 or above 0.082 the sample is rejected
OPEN_TOL_M = 2e-3           # (no STATE, not fresh); inside it the width is clamped to [0, width_max_m]
SEND_TIMEOUT_S = 0.2        # the C++ server's SO_SNDTIMEO: a client that stops reading is disconnected
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\Z", re.ASCII)   # what `istream >> double` reads, ASCII only
HOME_SPEED_MPS = 0.05
FRESH_S = 0.6               # a measured sample older than this is stale (the C++ server's freshness bound)
STATE_PERIOD_S = 0.1
POLL_S = 0.05
SETTLE_WINDOW_S = 0.2       # SimHand's measured settle detection
SETTLE_TOL_M = 0.0005
CONTACT_TOL_M = 0.003       # blocked this far short of the command = contact
MOVE_TOL_M = 0.001          # a MOVE that settles within this of its goal succeeded
EFFORT_FLOOR_N = 2.0        # a held module keeps the servos loaded (5-12 N measured)


def parse_command(line: str, limits: HandLimits = FRANKA_HAND) -> dict | None:
    """``CMD <token> MOVE w s`` | ``CMD <token> HOME`` | ``CMD <token> GRASP w s f eps_in eps_out``, as the C++ parser,
    inside ``limits`` (``force_max_n == 0``: no GRASP)."""
    parts = line.split(" ")
    if len(parts) < 3 or parts[0] != "CMD" or not (parts[1].isascii() and parts[1].isdigit()):
        return None
    verb, args = parts[2], parts[3:]
    if not all(_NUMBER.match(v) for v in args):
        return None
    values = [float(v) for v in args]
    action = dict(token=int(parts[1]), kind=verb, w=0., s=0., f=0., ei=0., eo=0.)
    if verb == "HOME":
        return action if not values else None
    if verb == "MOVE" and len(values) == 2:
        action.update(w=values[0], s=values[1])
    elif verb == "GRASP" and len(values) == 5:
        action.update(w=values[0], s=values[1], f=values[2], ei=values[3], eo=values[4])
    else:
        return None
    wmax = limits.width_max_m
    if not (math.isfinite(action["w"]) and 0 <= action["w"] <= wmax
            and math.isfinite(action["s"]) and 0 < action["s"] <= limits.speed_max_mps):
        return None
    if verb == "GRASP" and not (limits.force_grasp and math.isfinite(action["f"])
                                and limits.force_min_n <= action["f"] <= limits.force_max_n
                                and all(math.isfinite(action[k]) and 0 <= action[k] <= wmax for k in ("ei", "eo"))):
        return None
    return action


class _Admission:
    """The C++ ``HandAdmission`` without the device-session and stop-failure paths (see the module docstring)."""

    def __init__(self) -> None:
        self.pending: dict | None = None
        self.epoch = 1
        self.client = False
        self.acting = False
        self.dispatch_committed = False
        self.stopping = False

    def cancel(self) -> None:
        self.pending = None
        self.epoch += 1

    def request_stop(self) -> None:
        self.cancel()
        self.stopping = True

    def admit(self, action: dict, fresh: bool) -> str | None:
        if not self.client:
            return "offline"
        if self.stopping:
            return "stopping"
        if action["token"] != self.epoch:
            return "stale_epoch"
        if self.acting or self.pending is not None:
            return "busy"
        if not fresh:
            return "stale_state"
        self.epoch += 1
        self.pending = dict(action, generation=self.epoch)
        return None

    def take(self) -> dict | None:
        if not self.client or self.pending is None or self.acting or self.stopping:
            return None
        action, self.pending = self.pending, None
        if not self.current(action):
            return None
        self.acting = self.dispatch_committed = True  # the sim dispatches in the same tick it takes
        return action

    def current(self, action: dict) -> bool:
        return self.client and action["generation"] == self.epoch


class _Fingers:
    """SimHand's finger physics (plant thread only): per tick, the finger command to apply, and an action's verdict."""

    def __init__(self, width_max_m: float) -> None:
        self.width_max_m = width_max_m
        self.mode = "idle"          # idle | move | grasp | holding
        self.action: dict | None = None
        self.cmd_w: float | None = None
        self.force_n = 0.0          # > 0: force mode (the plant closes under this cap)
        self._last_t: float | None = None
        self._hist: list[tuple[float, float]] = []

    def start(self, action: dict, width: float, now: float) -> None:
        self.action = dict(action)
        if action["kind"] == "HOME":
            self.action.update(w=self.width_max_m, s=HOME_SPEED_MPS)
        self.mode = "grasp" if action["kind"] == "GRASP" else "move"
        self.cmd_w = width           # ramp from where the fingers ARE
        self.force_n = 0.0
        self._last_t, self._hist = now, []

    def stop(self, width: float) -> None:
        """``franka::Gripper::stop``: the action ends where the fingers are, and a held grasp lets go."""
        self.mode, self.action, self.cmd_w, self.force_n = "idle", None, width, 0.0

    def grasped(self, width: float, efforts) -> bool:
        if self.mode != "holding":
            return False
        band = self.action["w"] - self.action["ei"] <= width <= self.action["w"] + self.action["eo"]
        return band and sum(abs(float(e)) for e in efforts) > EFFORT_FLOOR_N

    def tick(self, width: float, now: float) -> bool | None:
        """Advance the ramp; the action's verdict once it settles (None while it runs)."""
        if self.mode not in ("move", "grasp"):
            return None
        target, dt = self.action["w"], max(0.0, now - (self._last_t if self._last_t is not None else now))
        self._last_t = now
        step = self.action["s"] * dt
        self.cmd_w = max(target, self.cmd_w - step) if self.cmd_w > target else min(target, self.cmd_w + step)
        if self.cmd_w != target:
            return None
        self._hist = [(t, w) for t, w in [*self._hist, (now, width)] if now - t <= SETTLE_WINDOW_S]
        settled = (len(self._hist) >= 2 and self._hist[-1][0] - self._hist[0][0] >= 0.5 * SETTLE_WINDOW_S
                   and max(w for _, w in self._hist) - min(w for _, w in self._hist) < SETTLE_TOL_M)
        if not settled:
            return None
        if self.mode == "move":
            self.mode = "idle"
            if abs(width - target) <= MOVE_TOL_M:
                return True
            self.cmd_w = width   # blocked short of the goal: franka's move() fails and the fingers stop where they are
            return False
        ok = (target - self.action["ei"] <= width <= target + self.action["eo"]) and width > target + CONTACT_TOL_M
        if ok:  # hold under force: both fingers drive shut at the commanded force, contact decides where they stop
            self.mode, self.cmd_w, self.force_n = "holding", 0.0, self.action["f"]
        else:
            self.mode = "idle"
        return ok

    def command(self):
        """``(per-finger targets, force_n)`` for the plant, or None before any action."""
        if self.cmd_w is None:
            return None
        finger = finger_from_width(self.cmd_w)
        return [finger, finger], self.force_n


class SimHandBridge:
    """The ``hand_bridge`` protocol over a hosted world's fingers. ``on_sample`` runs on the plant thread, once per tick, after
    the step; the server thread only parses, admits and reports. One lock, as the C++ server's ``mx``."""

    def __init__(self, port: int, *, bind: str = "127.0.0.1", limits: HandLimits = FRANKA_HAND) -> None:
        self.addr = (bind, int(port))
        self.limits = limits
        self._lock = threading.Lock()
        self._adm = _Admission()
        self._fingers = _Fingers(limits.width_max_m)
        self._acting: dict | None = None
        self.have_real, self.real_t, self.real_w, self.real_grasped, self.sample_seq = False, 0.0, 0.0, False, 0
        self.grasp_seq, self.grasp_ok, self.done_seq, self.done_token, self.done_ok = 0, False, 0, 0, False
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------------------------------------
    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(self.addr)
            server.listen(1)
        except OSError:
            server.close()
            raise
        self._stop.clear()
        server.settimeout(0.2)
        self._server = server
        self._thread = threading.Thread(target=self._accept_loop, name="sim-hand-bridge", daemon=True)
        self._thread.start()
        print(f"[sim_hand] hand_bridge protocol at {self.addr[0]}:{self.addr[1]}", flush=True)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._server is not None:
            self._server.close()
        self._server = self._thread = None

    # -- the plant thread ------------------------------------------------------------------------------------
    def on_sample(self, positions, efforts, now: float):
        """One measured finger sample after a plant step. Returns ``(targets, force_n)`` to apply, or None."""
        width = float(sum(width_from_finger(float(p)) for p in positions) / max(len(positions), 1))
        with self._lock:
            adm = self._adm
            if adm.stopping:
                # GSTOP (or a disconnect with committed work): the action ends where the fingers are; its epoch is
                # already gone, so it reports nothing. A fresh sample is required before the next action.
                self._fingers.stop(width)
                adm.acting = adm.dispatch_committed = False
                self._acting, adm.stopping, self.have_real = None, False, False
                return self._fingers.command()
            if not math.isfinite(width) or width < -CLOSED_TOL_M or width > self.limits.width_max_m + OPEN_TOL_M:
                self.have_real = False   # the C++ observe_state: no STATE, not fresh, until a usable sample
                if now - getattr(self, "_rejected_at", -1e9) > 2.0:
                    self._rejected_at = now
                    print(f"[sim_hand] rejected measured width {width:.6g} m (no STATE until a sample is usable)", flush=True)
            else:
                self.have_real, self.real_t, self.real_w = True, now, min(max(width, 0.0), self.limits.width_max_m)
                self.real_grasped = self._fingers.grasped(width, efforts)
                self.sample_seq += 1
            if self._acting is not None:
                ok = self._fingers.tick(width, now)
                if ok is not None:
                    action, self._acting = self._acting, None
                    adm.acting = adm.dispatch_committed = False
                    self.have_real = False  # a new measured sample before another action
                    if adm.current(action):
                        self.done_token, self.done_ok, self.done_seq = action["token"], ok, self.done_seq + 1
                        if action["kind"] == "GRASP":
                            self.grasp_ok, self.grasp_seq = ok, self.grasp_seq + 1
            else:
                action = adm.take()
                if action is not None:
                    self._acting = action
                    self._fingers.start(action, width, now)
                    self._fingers.tick(width, now)
            return self._fingers.command()

    # -- the server ------------------------------------------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except (socket.timeout, OSError):
                continue
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.settimeout(SEND_TIMEOUT_S)   # a send that cannot complete disconnects, as SO_SNDTIMEO does
            try:
                self._serve(client)
            except OSError:  # includes the send timeout
                pass
            finally:
                client.close()
                with self._lock:  # disconnect interrupts outstanding work, not an already completed grasp
                    self._adm.client = False
                    if self._adm.dispatch_committed or self._adm.stopping:
                        self._adm.request_stop()
                        self.have_real = False
                    else:
                        self._adm.cancel()

    def _serve(self, client: socket.socket) -> None:
        with self._lock:  # results from before this client connected are nobody's to consume
            seen_grasp, seen_done = self.grasp_seq, self.done_seq
            self._adm.cancel()
            self._adm.client = True
        buf, last_state = b"", 0.0
        while not self._stop.is_set():
            ready, _, _ = select.select([client], [], [], POLL_S)
            if ready:
                data = client.recv(256)
                if not data:
                    return  # client gone
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    if len(raw) >= 255:  # the C++ reads at most 255 bytes per line
                        print("[sim_hand] oversized line -- disconnecting", flush=True)
                        return
                    client.sendall(self._handle(raw.decode(errors="replace")).encode())
                if len(buf) >= 255:
                    print("[sim_hand] oversized line -- disconnecting", flush=True)
                    return
            out = []
            with self._lock:
                if self.grasp_seq != seen_grasp:
                    seen_grasp = self.grasp_seq
                    out.append(f"GDONE {int(self.grasp_ok)}\n")
                if self.done_seq != seen_done:
                    seen_done = self.done_seq
                    out.append(f"DONE {self.done_token} {int(self.done_ok)}\n")
                now = time.monotonic()
                if now - last_state >= STATE_PERIOD_S:
                    last_state = now
                    adm = self._adm
                    busy = adm.acting or adm.pending is not None or adm.stopping
                    fresh = self.have_real and now - self.real_t < FRESH_S
                    out.append("LIMITS {:.5f} {:.5f} {:.3f} {:.3f}\n".format(*self.limits))  # before STATE, as the C++
                    if self.have_real or adm.acting:
                        out.append(f"STATE {self.real_w:.5f} {int(self.real_grasped)}\n")
                    out.append(f"CAPS active_stop\nMETA 2 {adm.epoch} 1 {int(busy)} {int(fresh and not busy)} "
                               f"{self.sample_seq}\n")
            if out:
                client.sendall("".join(out).encode())

    def _handle(self, line: str) -> str:
        if line == "GSTOP":
            with self._lock:
                self._adm.request_stop()
                self.have_real = False
                return f"STOPPING {self._adm.epoch}\n"
        action = parse_command(line, self.limits)
        if action is None:
            # as the C++: the token is reported only once `CMD <token> <verb>` read and the token parsed
            parts = line.split(" ")
            token = parts[1] if (len(parts) >= 3 and parts[0] == "CMD" and parts[1].isascii()
                                 and parts[1].isdigit()) else "0"
            return f"REJECT {token} invalid_command\n"
        with self._lock:
            error = self._adm.admit(action, self.have_real and time.monotonic() - self.real_t < FRESH_S)
        return f"{'REJECT' if error else 'ACCEPT'} {action['token']} {error or 'accepted'}\n"
