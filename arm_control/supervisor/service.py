"""The supervisor process: one robot, one local socket (CTRL-022).

    python -m arm_control.supervisor.service --config <robot.yaml> [--socket PATH]

Serves ``arm_control.sdk.ArmClient``: newline-delimited JSON over a Unix
socket in an owner-only directory. One connection is one client session; its
thread only parses and waits, so a stalled client blocks nobody but itself.
Starting the service claims the RT server and never arms.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import threading
import uuid
from pathlib import Path

from arm_control.execution import codec
from arm_control.execution.contract import CONTRACT_VERSION
from arm_control.sdk import MAX_MESSAGE, default_socket_path
from arm_control.supervisor.runtime import Refused, RobotRuntime

MAX_CLIENTS = 8
#: Longest a single request may block its connection thread. Clients wait
#: longer by repeating short waits, so a dropped owner is noticed within this.
MAX_WAIT_S = 1.0


def dispatch(runtime: RobotRuntime, session: str, op: str, args: dict):
    """One named supervisor operation -> a JSON-able result. The same names
    and arguments whatever the transport."""
    wait = min(max(float(args.get("wait_for_s", 0.0) or 0.0), 0.0), MAX_WAIT_S)
    enc = codec.encode
    if op == "hello":
        return dict(session=session, contract_version=CONTRACT_VERSION,
                    identity=enc(runtime.identity), capabilities=enc(runtime.capabilities),
                    now=enc(runtime.now()), extensions=runtime.extensions)
    if op.startswith('controller_'):
        if runtime.extensions.get('controller') != 1:
            raise Refused('controller/1 is unavailable')
        if op == 'controller_presets':
            return enc(runtime.controller_presets())
        if op == 'controller_state':
            return enc(runtime.controller_state())
        if op == 'controller_select':
            return enc(runtime.select_control_preset(session, str(args['preset_id'])))
        if op == 'controller_hold':
            return enc(runtime.hold(session, str(args.get('reason') or 'hold')))
        if op == 'controller_payload':
            return enc(runtime.set_payload(session, codec.payload_from(args['payload'])))
    if op.startswith('stream_'):
        if runtime.extensions.get('stream') != 1:
            raise Refused('stream/1 is unavailable')
        if op == 'stream_open':
            return enc(runtime.open_stream(session, codec.stream_open_from(args['request'])))
        if op == 'stream_revise':
            return enc(runtime.revise_stream(session, codec.stream_revision_from(args['request'])))
        methods = {'stream_state': 'stream_state', 'stream_anchor': 'stream_anchor',
                   'stream_revision': 'stream_revision', 'stream_abort': 'abort_stream_revision'}
        if op in methods:
            extra = [str(args['request_id'])] if op in ('stream_revision', 'stream_abort') else []
            return enc(getattr(runtime, methods[op])(session, str(args['execution_id']), *extra))
    if op.startswith('group_'):
        if runtime.extensions.get('group') != 1:
            raise Refused('group/1 is unavailable')
        if op == 'group_join':
            return runtime.group_join(session, int(args['group_id']), int(args['member_index']),
                                      int(args['member_count']))
        if op == 'group_release':
            return enc(runtime.group_release(session, str(args['execution_id']), int(args['group_id']),
                                             int(args['budget_ns'])))
        if op == 'group_cancel':
            return runtime.group_cancel(session, int(args['group_id']))
        if op == 'group_report':
            return runtime.group_report(str(args['execution_id']))
        if op == 'group_events':
            return runtime.group_events(session)
    if op == 'setpoint':
        if runtime.extensions.get('setpoint') != 1:
            raise Refused('setpoint/1 is unavailable (this supervisor plays the native reference: use stream/1)')
        return runtime.setpoint(session, args['q'])
    if op == 'trace_samples':
        if runtime.extensions.get('trace') != 1:
            raise Refused('trace/1 is unavailable')
        return runtime.trace_samples(max(0, int(args.get('after') or 0)))
    if op == 'observe':
        if runtime.extensions.get('observe') != 1:
            raise Refused('observe/1 is unavailable')
        limit = args.get('limit')
        return runtime.observe(max(0, int(args.get('after') or 0)), 64 if limit is None else int(limit))  # capped by the link
    if op == 'state_history':
        if runtime.extensions.get('state_history') != 1:
            raise Refused('state_history/1 is unavailable')
        return runtime.state_history()
    if op == "now":
        return enc(runtime.now())
    if op == "state":
        return enc(runtime.state())
    if op == "authority":
        return enc(runtime.authority())
    if op in ("claim", "unclaim"):
        return enc(getattr(runtime, op)(session))
    if op in ("arm", "disarm"):
        return {"gen": getattr(runtime, op)(session)}
    if op == "await_authority":
        auth = runtime.await_authority(int(args["gen"]), armed=bool(args["armed"]), wait_s=wait)
        return {"settled": auth is not None, "authority": enc(auth)}
    if op == "submit":
        return enc(runtime.submit(session, codec.request_from(args["request"])))
    if op == "release":
        return enc(runtime.release(session, str(args["execution_id"])))
    if op == "cancel":
        return enc(runtime.cancel(session, str(args["execution_id"]),
                                  str(args.get("reason") or "cancelled")))
    if op == "status":
        return enc(runtime.status(str(args["execution_id"]), wait_s=wait))
    if op == "receipts":
        return enc(runtime.receipts(str(args["execution_id"])))
    raise ValueError(f"unknown operation {op!r}")


class Service:
    def __init__(self, runtime: RobotRuntime, path: str | os.PathLike) -> None:
        self.runtime = runtime
        self.path = Path(path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        if self.path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(self.path))
                raise RuntimeError(f"another supervisor already serves {self.path}")
            except (ConnectionRefusedError, FileNotFoundError):
                self.path.unlink()  # stale socket from a dead process
            finally:
                probe.close()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.path))
        self._server.listen(MAX_CLIENTS)
        self._server.settimeout(0.2)
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()

    def serve(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            with self._lock:
                if len(self._clients) >= MAX_CLIENTS:
                    conn.close()
                    continue
                self._clients.add(conn)
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        session = uuid.uuid4().hex
        rx = conn.makefile("rb")
        try:
            while True:
                line = rx.readline(MAX_MESSAGE + 1)
                if not line.endswith(b"\n"):
                    return  # closed, or an oversized message: drop the session
                reply = self._handle(session, line)
                conn.sendall(json.dumps(reply).encode() + b"\n")
        except OSError:
            return
        finally:
            self.runtime.close_session(session)
            with self._lock:
                self._clients.discard(conn)
            rx.close()
            conn.close()

    def _handle(self, session: str, line: bytes) -> dict:
        rid = None
        try:
            msg = json.loads(line)
            rid = msg.get("id")
            result = dispatch(self.runtime, session, str(msg["op"]), dict(msg.get("args") or {}))
            return {"id": rid, "ok": True, "result": result}
        except Refused as exc:
            return {"id": rid, "ok": False, "kind": "refused", "error": str(exc)}
        except TimeoutError as exc:
            return {"id": rid, "ok": False, "kind": "timeout",
                    "error": str(exc) or "supervisor loop did not answer in time"}
        except (KeyError, TypeError, ValueError) as exc:
            return {"id": rid, "ok": False, "kind": "invalid", "error": f"{exc!r}"}
        except Exception as exc:  # noqa: BLE001 -- reported to the client, session survives
            return {"id": rid, "ok": False, "kind": "internal", "error": f"{exc!r}"}

    def close(self) -> None:
        self._server.close()
        with self._lock:
            for conn in list(self._clients):
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def main(argv=None, *, stream_admission=None) -> None:
    """Serve one robot.

    ``stream_admission`` is the composition-owned trusted provider for the
    frozen stream/1 interface (``arm_control.supervisor.admission``). Default
    launches keep today's neutral path: without a provider the stream
    extension is not advertised and stream methods refuse.
    """
    from arm_control.config import load_robot_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="robot config (with an rt: block)")
    parser.add_argument("--socket", help="socket path (default: per-user, per-instance)")
    parser.add_argument("--plant", choices=("rt", "sim"), default="rt",
                        help="rt: an arm_rt_server (default); sim: the config's MuJoCo plant behind the same supervisor")
    args = parser.parse_args(argv)
    cfg = load_robot_config(args.config)
    link = None
    if args.plant == "sim":
        from arm_control.plants.sim_link import SimLink

        link = SimLink.from_config(cfg)
    runtime = RobotRuntime(cfg, link=link, stream_admission=stream_admission)
    if not runtime.instance_id:
        raise SystemExit("config has no rt.instance_id")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    runtime.start()
    service = Service(runtime, args.socket or default_socket_path(runtime.instance_id))
    print(f"[supervisor] {runtime.instance_id} serving {service.path} -- DISARMED, no owner",
          flush=True)
    try:
        service.serve(stop)
    finally:
        service.close()
        runtime.close()
        print("[supervisor] stopped", flush=True)


if __name__ == "__main__":
    main()
