"""Dora node: operator arm/confirm gate for the hardware assembler graphs.

Comes up IDLE and publishes ``arm {"armed": true}`` each time the human operator
explicitly triggers it — press the button on the operator panel
(``http://<arm-pc>:7502``), press Enter on the launch terminal, or (headless)
``touch`` the trigger file. The FIRST signal arms/starts the sequence; every
LATER signal is a plan-review CONFIRM: when the orchestrator is holding at a
gated phase (``confirm_phases``) with the planned motion streamed to Rerun,
the next trigger releases it. Nothing on the arm moves until the first fire,
and gated phases never proceed without one. The graph must never self-trigger.

The node feeds the orchestrator's ``operator_arm`` input, not the bridge's ``arm``
input, so the orchestrator stays the single producer of the bridge ``arm`` topic
(it enables the bridge on this signal and disarms on fault). No multi-producer.
"""
from __future__ import annotations

# ruff: noqa: E402

import json
import os
import select
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dora import Node


from arm_control.messages import pack_json_message
from arm_control.node_utils import next_event_gil_friendly

_PANEL_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>operator gate</title><style>
body{font-family:sans-serif;margin:2em;max-width:34em}
button{font-size:1.3em;padding:.6em 1em;width:100%;cursor:pointer;margin-top:.5em}
button:active{background:#c8e6c9}
#gate{display:none;gap:.6em;margin-top:.5em}
#gate button{margin-top:0}
#plan{background:#fff3e0}#play{background:#e8f5e9}#exec{background:#ffe0e0}
#status{font-size:1.1em;padding:.5em .8em;background:#eef;border-radius:6px}
#log{font-family:monospace;white-space:pre-wrap;color:#444;margin-top:1.2em}
.hint{color:#777;font-size:.85em;margin-top:.8em}
</style></head><body>
<h3>operator gate</h3>
<div id="status">waiting for orchestrator&hellip;</div>
<button id="fire" onclick="fetch('fire',{method:'POST'})">ARM &amp; START</button>
<div id="gate">
 <button id="plan" onclick="fetch('replan',{method:'POST'})">PLAN</button>
 <button id="play" onclick="fetch('replay',{method:'POST'})">PLAY</button>
 <button id="exec" onclick="fetch('fire',{method:'POST'})">EXECUTE</button>
</div>
<p class="hint">At a held gate the three controls are independent: PLAN
re-plans the held phase from the current pose, PLAY flies the green ghost
through the plan once in Rerun, EXECUTE runs it on the arm. Rerun colors:
real STL = measured arm, green = planned motion, orange = planning target.
Only EXECUTE (or ARM) ever moves the arm — same channel as Enter / the
trigger file.</p>
<div id="log"></div>
<script>
setInterval(async()=>{
 const s=await (await fetch('state')).json();
 document.getElementById('log').textContent=s.log.join('\\n');
 const st=s.status||{}; let line, showFire=true;
 if(st.stopped){line='STOPPED (fault) — see logs';}
 else if(st.holding){line='HOLDING at '+st.holding+' — PLAN / PLAY / EXECUTE';
   showFire=false;
   document.getElementById('exec').textContent='EXECUTE '+st.holding;}
 else if(st.started){line='running: '+(st.phase||'...');}
 else if(st.phase!==undefined){line='DISARMED — idle';}
 else{line='waiting for orchestrator…';}
 document.getElementById('status').textContent=line;
 document.getElementById('fire').style.display=showFire?'block':'none';
 document.getElementById('gate').style.display=st.holding?'flex':'none';
},700);
</script></body></html>"""


class OperatorPanel:
    """One-button browser panel: each click queues one trigger (thread-safe).

    Same semantics and trust level as the trigger file — the panel binds on the
    arm PC's LAN like the teleop panel; it can only fire the gate, never bypass
    it (the orchestrator still holds at every confirm gate until a fire).
    """

    def __init__(self, port: int) -> None:
        self._lock = threading.Lock()
        self._pending = 0
        self._replay_pending = 0
        self._replan_pending = 0
        self._status: dict = {}
        self._log: list[str] = []
        self.evicted = False
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # quiet
                pass

            def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.endswith("state"):
                    with panel._lock:
                        payload = {"log": panel._log[-12:], "status": panel._status}
                    self._send(json.dumps(payload).encode(), "application/json")
                else:
                    self._send(_PANEL_PAGE.encode(), "text/html; charset=utf-8")

            def do_POST(self) -> None:
                # EVERY route, not just takeover. `fire` is the signal that
                # releases a gated motion leg -- the single most consequential
                # thing this process can emit -- and it had no peer check at
                # all while the socket listened on every interface. Anyone who
                # could reach the port could start the arm. The bind below is
                # loopback now; this is the second lock, so a future bind
                # change cannot silently re-open the gate.
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    self._send(b'{"ok": false}', "application/json", 403)
                    return
                if self.path.endswith("fire"):
                    with panel._lock:
                        panel._pending += 1
                elif self.path.endswith("replay"):
                    with panel._lock:
                        panel._replay_pending += 1
                elif self.path.endswith("replan"):
                    with panel._lock:
                        panel._replan_pending += 1
                elif self.path.endswith("takeover"):
                    # A NEWER session is claiming the port: this panel belongs
                    # to a dead/killed graph (the recurring orphan trap — its
                    # buttons look alive but publish into nothing). Release the
                    # port. (Peer check is now at the top of do_POST.)
                    panel.evicted = True
                    threading.Timer(0.2, panel.close).start()
                self._send(b'{"ok": true}', "application/json")

        self._server = None
        for attempt in range(4):
            try:
                # Loopback ONLY, matching arm_control.operator_console.OperatorPanel,
                # which REFUSES a non-loopback bind outright. Two operator
                # gates with two answers to "who may release motion" is one
                # answer too many. For a remote desk, tunnel the port
                # (ssh -L) rather than listening on the network.
                self._server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
                break
            except OSError:
                if attempt == 3 or int(port) == 0:
                    raise
                # Evict the stale holder (an orphaned operator_arm from a
                # killed session) and retry.
                import urllib.request

                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({})
                )
                try:
                    opener.open(
                        urllib.request.Request(
                            f"http://127.0.0.1:{int(port)}/takeover", method="POST"
                        ),
                        timeout=1.0,
                    )
                except Exception:
                    pass
                import time as _time

                _time.sleep(0.5)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def take_trigger(self) -> bool:
        """Consume ONE queued click (one click = one gate fire, never batched)."""
        with self._lock:
            if self._pending > 0:
                self._pending -= 1
                return True
            return False

    def take_replay(self) -> bool:
        """Consume ONE queued PLAY request (view-only channel)."""
        with self._lock:
            if self._replay_pending > 0:
                self._replay_pending -= 1
                return True
            return False

    def take_replan(self) -> bool:
        """Consume ONE queued PLAN request (plan-only channel, no motion)."""
        with self._lock:
            if self._replan_pending > 0:
                self._replan_pending -= 1
                return True
            return False

    def set_status(self, status: dict) -> None:
        with self._lock:
            self._status = dict(status)

    def status_holding(self) -> bool:
        with self._lock:
            return bool(self._status.get("holding"))

    def log(self, message: str) -> None:
        with self._lock:
            self._log.append(message)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()  # release the port for the taker-over


def _trigger_file() -> Path:
    return Path(os.environ.get("ARM_OPERATOR_ARM_FILE", "/tmp/arm_operator_arm"))


def _file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def main() -> None:
    trigger = _trigger_file()
    # Only honour stdin on a real terminal: a non-tty stdin (e.g. /dev/null) reports
    # readable at EOF, which would auto-arm — the exact failure this gate prevents.
    use_stdin = sys.stdin.isatty()
    poll_s = 0.1
    panel: OperatorPanel | None = None
    panel_port = os.environ.get("ARM_OPERATOR_PANEL_PORT", "7502").strip()
    if panel_port and panel_port != "0":
        try:
            panel = OperatorPanel(int(panel_port))
        except OSError as exc:  # port busy — Enter/touch still work
            print(f"[operator_arm] panel disabled ({exc})", flush=True)
    print(
        "[operator_arm] DISARMED — arm gate ready. "
        + (f"Button panel at http://127.0.0.1:{panel.port} " if panel else "")
        + ("| Enter to ARM/CONFIRM " if use_stdin else "")
        + f"(or in another shell: touch {trigger})",
        flush=True,
    )
    node = Node()
    # Ignore a stale pre-existing trigger file: only a create/touch after startup fires.
    baseline = _file_mtime(trigger)
    fired = 0
    while True:
        event = next_event_gil_friendly(node, idle_sleep=poll_s)
        if event is not None and event["type"] == "STOP":
            return
        if (
            event is not None
            and event["type"] == "INPUT"
            and event["id"] == "status"
            and panel is not None
        ):
            from arm_control.messages import unpack_json_message

            panel.set_status(unpack_json_message(event["value"]))
        if panel is not None and panel.evicted:
            print(
                "[operator_arm] panel port taken over by a NEWER session — "
                "this graph's panel is gone (Enter/trigger file still work)",
                flush=True,
            )
            panel = None
        triggered = False
        if use_stdin and select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            triggered = True
        mtime = _file_mtime(trigger)
        if mtime is not None and mtime != baseline:
            baseline = mtime  # each fresh touch is one trigger
            triggered = True
        if panel is not None and panel.take_trigger():
            triggered = True
        if panel is not None and panel.take_replay():
            # View-only channel, sent ONLY while orchestrator status shows a
            # held plan. In graphs where this node feeds the real bridge's
            # `arm` input directly (real_motion) there is never a status feed,
            # so a replay can never reach a consumer that would read the
            # non-armed message as a DISARM.
            if panel.status_holding():
                node.send_output("arm", pack_json_message("arm", {"replay": True}))
                panel.log("PLAY — flying the plan preview")
            else:
                panel.log("play ignored — no plan held")
        if panel is not None and panel.take_replan():
            # Plan-only channel, same holding guard as PLAY for the same
            # reason (a non-armed message must never reach a raw bridge).
            if panel.status_holding():
                node.send_output("arm", pack_json_message("arm", {"replan": True}))
                panel.log("PLAN — re-planning the held phase")
            else:
                panel.log("plan ignored — no plan held")
        if triggered:
            fired += 1
            node.send_output("arm", pack_json_message("arm", {"armed": True}))
            message = (
                "ARM signal sent — sequence starting" if fired == 1 else "CONFIRM sent"
            )
            print(f"[operator_arm] {message}", flush=True)
            if panel is not None:
                panel.log(message)


def _self_check() -> None:
    """The gate must not be reachable from the network. Assert it, do not read it.

    This panel can emit the signal that RELEASES a gated motion leg. It used to
    bind 0.0.0.0 with a peer check on /takeover only, so /fire -- the arm
    signal itself -- was open to anyone who could reach the port, while
    arm_control.operator_console.OperatorPanel refuses a non-loopback bind
    outright. Two gates, two answers. This pins the one answer.
    """
    import urllib.request

    panel = OperatorPanel(0)
    try:
        host, port = panel._server.server_address[:2]
        assert host == "127.0.0.1", f"operator gate bound {host}, not loopback"

        def post(path: str) -> int:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/{path}", method="POST"
            )
            return opener.open(req, timeout=2.0).status

        # The gate still works from loopback -- a lock that breaks the button
        # is not a fix.
        assert post("fire") == 200
        assert panel.take_trigger(), "a loopback click must still fire the gate"
        assert not panel.take_trigger(), "one click is one fire, never batched"

        # And the guard sits at the TOP of do_POST, so it covers fire/replay/
        # replan and not just takeover. Checked by source rather than by
        # spoofing a peer, which needs a second interface.
        import inspect

        body = inspect.getsource(OperatorPanel.__init__)
        guard = body.index('self.client_address[0] not in ("127.0.0.1", "::1")')
        assert guard < body.index('self.path.endswith("fire")'), (
            "the peer check must precede every route, not just takeover"
        )
    finally:
        panel.close()
    print("operator_arm: OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
