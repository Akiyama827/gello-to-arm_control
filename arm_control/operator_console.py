"""The generic operator panel: review what the arm is about to do, then allow it.

Nothing here knows what an arm is FOR. It shows an ordered list of phases, which
one is current, which are done, what is held for review, and why something
failed -- all injected as an :class:`OperatorWorkspace` by whoever does know
(``assembly/operator_view.py`` in Control, the way ``GraspEditorWorkspace`` feeds
``GraspEditorPanel``). There is no Row Module, no dock, no stack and no bench in
this file, and there must not be.

The panel accepts exactly three operator actions -- GO, STOP, PLAN -- and turns
them into ONE signal on ONE callback. It has no motor command, no trajectory and
no plant handle to reach for. That is not an oversight to be fixed later when
something needs to move faster: an operator gate that can also drive is not a
gate. The check at the bottom asserts it.

It binds loopback-only, for the same reason the grasp editor does: this is the
control the whole safety story rests on, and a panel reachable from the network
is a robot anyone on the network can start.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Sequence

from arm_control.console_assets import console_asset

#: The only actions the panel will accept. GO is the confirm/arm press (the
#: same signal the terminal gate sends); STOP is the abort; PLAN re-plans the
#: held phase without moving anything. Anything else is refused with a 400 --
#: an unknown action must never be silently read as one of these.
ACTIONS = ("go", "stop", "plan")


@dataclass(frozen=True)
class OperatorStep:
    """What the arm proposes to do next, in terms an operator can judge.

    Deliberately not a trajectory: the panel shows a decision, not a plot. A
    caller with a real plan fills ``summary`` and the optional numbers; one
    without leaves them empty and the page shows the phase alone.
    """

    phase: str
    summary: str = ""
    duration_s: float | None = None
    #: Free-form rows rendered as a name/value table. The adapter decides what
    #: is worth an operator's attention; this file has no opinion on it.
    detail: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorWorkspace:
    """Everything the panel renders and the two things it can call.

    ``phases`` is the whole sequence in order, so the page can show progress
    rather than just a current label. ``accepted`` is the trace of what has
    actually been achieved -- which is NOT the same list (a phase can run
    without its gate accepting), and showing them as one would hide exactly the
    disagreement an operator is there to catch.
    """

    phases: Sequence[str]
    current: str
    accepted: Sequence[str] = ()
    holding: str | None = None
    step: OperatorStep | None = None
    failure: str = ""
    started: bool = False
    stopped: bool = False

    def json(self) -> dict:
        step = self.step
        return {
            "phases": list(self.phases),
            "current": self.current,
            "accepted": list(self.accepted),
            "holding": self.holding,
            "failure": self.failure,
            "started": bool(self.started),
            "stopped": bool(self.stopped),
            "step": None if step is None else {
                "phase": step.phase,
                "summary": step.summary,
                "duration_s": step.duration_s,
                "detail": dict(step.detail),
            },
        }


class OperatorPanel:
    """A loopback HTTP page that can produce one signal and nothing else."""

    def __init__(
        self,
        read: Callable[[], OperatorWorkspace],
        signal: Callable[[str], None],
        *,
        bind: str = "127.0.0.1",
        port: int = 7503,
    ) -> None:
        try:
            loopback = ipaddress.ip_address(bind).is_loopback
        except ValueError as exc:
            raise ValueError("operator panel bind must be a loopback address") from exc
        if not loopback:
            raise ValueError("operator panel bind must be a loopback address")
        self._read = read
        self._signal = signal
        self._lock = threading.Lock()
        self.log: list[str] = ["operator panel ready — nothing moves without a press"]
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def json(self, value: object, code: int = 200) -> None:
                body = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                route = self.path.split("?", 1)[0].strip("/")
                if route == "":
                    route = "operator.html"
                if route == "operator.html" or route.startswith("static/"):
                    try:
                        content_type, body, cache = console_asset(route)
                    except KeyError:
                        self.json({"error": "not found"}, 404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", cache)
                    self.send_header("ETag", hashlib.sha256(body).hexdigest())
                    self.end_headers()
                    self.wfile.write(body)
                elif route == "state":
                    self.json(panel.state())
                else:
                    self.json({"error": "not found"}, 404)

            def do_POST(self) -> None:
                origin = self.headers.get("Origin")
                host = self.headers.get("Host", "")
                if origin is not None and origin not in (
                    f"http://{host}", f"https://{host}"
                ):
                    self.json({"error": "cross-origin refused"}, 403)
                    return
                if self.path.split("?", 1)[0].strip("/") != "action":
                    self.json({"error": "not found"}, 404)
                    return
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length > 4096:
                    self.json({"error": "body too large"}, 413)
                    return
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    result = panel.act(payload)
                except (TypeError, ValueError) as exc:
                    self.json({"error": str(exc)}, 400)
                    return
                self.json(result)

        self.server = ThreadingHTTPServer((bind, int(port)), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    # -- surface --------------------------------------------------------------
    def state(self) -> dict:
        payload = self._read().json()
        with self._lock:
            payload["log"] = list(self.log[-40:])
        return payload

    def act(self, payload) -> dict:
        """Run one of ACTIONS. An unknown action is an error, never a default."""
        if not isinstance(payload, dict):
            raise TypeError("action payload must be an object")
        action = str(payload.get("action", "")).strip().lower()
        if action not in ACTIONS:
            raise ValueError(
                f"unknown operator action {action!r} (expected one of "
                f"{', '.join(ACTIONS)})"
            )
        self._signal(action)
        self.note(f"operator: {action.upper()}")
        return {"ok": True, "action": action}

    def note(self, message: str) -> None:
        with self._lock:
            self.log.append(str(message))

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _self_check() -> None:
    """The three properties that make this a gate rather than a controller."""
    import urllib.error
    import urllib.request

    # 1. It refuses to listen anywhere a second machine could reach.
    for bad in ("0.0.0.0", "192.168.1.10", "not-an-address"):
        try:
            OperatorPanel(lambda: OperatorWorkspace(phases=(), current=""),
                          lambda _a: None, bind=bad, port=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"panel bound to {bad!r}")

    fired: list[str] = []
    workspace = OperatorWorkspace(
        phases=("move", "grasp", "dock"),
        current="grasp",
        accepted=("move",),
        holding="grasp",
        step=OperatorStep("grasp", "close on part_a", 1.5, {"width": "40 mm"}),
        started=True,
    )
    panel = OperatorPanel(lambda: workspace, fired.append, port=0)
    try:
        base = f"http://127.0.0.1:{panel.port}"

        def post(body: dict, expect: int) -> dict:
            request = urllib.request.Request(
                f"{base}/action", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    assert response.status == expect, response.status
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                assert exc.code == expect, (exc.code, expect)
                return json.loads(exc.read())

        # 2. Exactly three actions, and an unknown one is refused -- not
        #    silently treated as a GO, which is the failure that matters.
        for action in ACTIONS:
            assert post({"action": action}, 200)["action"] == action
        assert fired == list(ACTIONS), fired
        for bogus in ("execute", "GO ", "", "arm", "resume"):
            if bogus.strip().lower() in ACTIONS:
                continue
            body = post({"action": bogus}, 400)
            assert "unknown operator action" in body["error"], body
        assert fired == list(ACTIONS), f"a refused action still signalled: {fired}"

        # 3. There is no route that could command motion, and no method that
        #    could reach one. Only /state and the static page answer at all.
        with urllib.request.urlopen(f"{base}/state", timeout=5) as response:
            state = json.loads(response.read())
        assert state["holding"] == "grasp" and state["accepted"] == ["move"]
        assert state["step"]["detail"] == {"width": "40 mm"}, state["step"]
        for route in ("motor_command", "plan", "trajectory", "command", "arm"):
            try:
                urllib.request.urlopen(f"{base}/{route}", timeout=5)
            except urllib.error.HTTPError as exc:
                assert exc.code == 404, (route, exc.code)
            else:
                raise AssertionError(f"panel answered a motion route: {route}")
        # A POST anywhere but /action is a 404 too.
        request = urllib.request.Request(
            f"{base}/motor_command", data=b"{}", method="POST"
        )
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, exc.code
        else:
            raise AssertionError("panel accepted a POST to a motion route")
    finally:
        panel.close()
    print("operator_console self-check ok")


if __name__ == "__main__":
    _self_check()
