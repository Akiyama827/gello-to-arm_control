"""One loopback HTTP server for every offline console in this package.

There were four hand-rolled copies of this: the operator panel, the grasp
editor, the teleop control panel, and the arm button panel. They agreed on the
shape and disagreed on the details that matter -- one capped a POST body at
4 KiB and another at 64 KiB, one answered 304 to a matching ETag and the others
re-sent the whole file, one set ``Cache-Control: no-store`` on its JSON and the
others let a browser cache ``/state``. Every copy re-implemented the two gates
that keep a page which can ARM a torque-controlled arm from being clicked by
any website the operator happens to have open.

``console_assets`` already extracted the asset allowlist from this same pattern,
for this same reason, back when there were two. This is the server half.

What is shared here is exactly the security-critical part -- the loopback
refusal, the same-origin check, the body cap, the allowlisted static serving --
plus the boilerplate. What each console does with a route stays in that console:
callers pass two functions and get a running server.

    server = ConsoleServer(
        name="operator panel", bind="127.0.0.1", port=7503,
        get=lambda route: panel.state() if route == "state" else None,
        post=lambda route, body: panel.act(body) if route == "action" else None,
    )

``get`` returns a JSON-able object, an :class:`Asset` for bytes, or ``None`` for
404. ``post`` is the same, and a ``KeyError``/``TypeError``/``ValueError`` out
of either becomes a 400 carrying its message -- which is how a console reports
a bad request without writing any HTTP.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from arm_control.console_assets import CONSOLE_ASSETS, console_asset

__all__ = ["Asset", "ConsoleServer", "file_asset", "DEFAULT_MAX_BODY"]

#: POST body cap. A panel that can DISARM must not be OOM-able by a declared
#: multi-GB body, and no console here posts anything near this.
DEFAULT_MAX_BODY = 64 * 1024


@dataclass(frozen=True)
class Asset:
    """Raw bytes with the headers to serve them under."""

    content_type: str
    body: bytes
    cache: str = "no-store"


def file_asset(
    path: str | Path | None,
    *,
    content_type: str = "application/octet-stream",
    cache: str = "max-age=86400",
) -> Asset | None:
    """One file off disk, or ``None`` (404) for a path the console had none for.

    The mesh routes on the teleop panel and the grasp editor each open-coded
    this three times over; ``None`` in means 404 out, so a caller can pass a
    lookup straight through without branching.
    """
    if path is None:
        return None
    return Asset(content_type, Path(path).read_bytes(), cache)


class ConsoleServer:
    """A threaded HTTP server serving allowlisted assets plus two callbacks."""

    def __init__(
        self,
        *,
        name: str,
        bind: str,
        port: int,
        get: Callable[[str], object] | None = None,
        post: Callable[[str, dict], object] | None = None,
        index: str = "index.html",
        require_loopback: bool = True,
        max_body: int = DEFAULT_MAX_BODY,
    ) -> None:
        if require_loopback:
            try:
                loopback = ipaddress.ip_address(bind).is_loopback
            except ValueError as exc:
                raise ValueError(f"{name} bind must be a loopback address") from exc
            if not loopback:
                raise ValueError(f"{name} bind must be a loopback address")
        self._get = get
        self._post = post
        # Which page ``/`` means. NOT a constant: two consoles share this
        # allowlist and this server, and CONSOLE_ASSETS[""] is the grasp
        # editor's index -- the operator panel serving that on / would hand an
        # operator the wrong page under the right URL.
        self._index = index
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            # -- writers ------------------------------------------------------
            def _send(self, asset: Asset, code: int = 200) -> None:
                self.send_response(code)
                self.send_header("Content-Type", asset.content_type)
                self.send_header("Content-Length", str(len(asset.body)))
                self.send_header("Cache-Control", asset.cache)
                self.end_headers()
                self.wfile.write(asset.body)

            def _json(self, value: object, code: int = 200) -> None:
                self._send(
                    Asset("application/json", json.dumps(value).encode(), "no-store"),
                    code,
                )

            def _reply(self, result: object) -> None:
                """A handler's return value -> a response. None is a 404."""
                if result is None:
                    self._json({"error": "not found"}, 404)
                elif isinstance(result, Asset):
                    # Content-addressed: a matching ETag means the client
                    # already has these exact bytes.
                    etag = hashlib.sha256(result.body).hexdigest()
                    if self.headers.get("If-None-Match") == etag:
                        self.send_response(304)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", result.content_type)
                    self.send_header("Content-Length", str(len(result.body)))
                    self.send_header("Cache-Control", result.cache)
                    self.send_header("ETag", etag)
                    self.end_headers()
                    self.wfile.write(result.body)
                else:
                    self._json(result)

            def _route(self) -> str:
                return self.path.split("?", 1)[0].strip("/")

            # -- methods ------------------------------------------------------
            def do_GET(self) -> None:
                route = self._route() or server._index
                if route in CONSOLE_ASSETS:
                    content_type, body, cache = console_asset(route)
                    self._reply(Asset(content_type, body, cache))
                    return
                if server._get is None:
                    self._json({"error": "not found"}, 404)
                    return
                try:
                    self._reply(server._get(route))
                except (KeyError, TypeError, ValueError) as exc:
                    self._json({"error": str(exc)}, 400)

            def do_POST(self) -> None:
                # Two cheap gates on a server that can ARM a torque-controlled
                # arm. A cross-origin browser POST always carries an Origin
                # that will not match ours, which kills the CSRF class -- any
                # site the operator visits could otherwise click ARM. And a
                # declared oversize body must not OOM the only node that can
                # DISARM.
                origin = self.headers.get("Origin")
                host = self.headers.get("Host", "")
                if origin is not None and origin not in (
                    f"http://{host}", f"https://{host}"
                ):
                    self._json({"error": "cross-origin refused"}, 403)
                    return
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length > max_body:
                    self._json({"error": "body too large"}, 413)
                    return
                if server._post is None:
                    self._json({"error": "not found"}, 404)
                    return
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    self._reply(server._post(self._route(), payload))
                except (KeyError, TypeError, ValueError) as exc:
                    self._json({"error": str(exc)}, 400)

        self.server = ThreadingHTTPServer((str(bind), int(port)), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _self_check() -> None:
    """The gates, not the plumbing: what each hand-rolled copy had to get right."""
    import urllib.error
    import urllib.request

    # 1. A console that requires loopback refuses anything else, by name.
    for bad in ("0.0.0.0", "10.1.1.50", "not-an-address"):
        try:
            ConsoleServer(name="test panel", bind=bad, port=0)
        except ValueError as exc:
            assert "test panel" in str(exc), exc
        else:
            raise AssertionError(f"bound to {bad!r}")
    # ...and one that opts out does not. Nothing shipped here opts out any
    # more -- the control panel gained jog and took the refusal (2026-09-07) --
    # but the flag stays for a consumer whose page cannot command motion.
    open_server = ConsoleServer(
        name="open", bind="0.0.0.0", port=0, require_loopback=False
    )
    open_server.close()

    posted: list[tuple[str, dict]] = []

    def get(route: str):
        if route == "state":
            return {"ok": True}
        if route == "blob":
            return Asset("application/octet-stream", b"\x00\x01\x02", "max-age=60")
        if route == "boom":
            raise ValueError("bad route argument")
        return None

    def post(route: str, body: dict):
        if route != "action":
            return None
        posted.append((route, body))
        return {"ok": True}

    server = ConsoleServer(name="t", bind="127.0.0.1", port=0, get=get, post=post)
    base = f"http://127.0.0.1:{server.port}"
    try:
        def fetch(path, *, headers=None, data=None, expect=200):
            request = urllib.request.Request(
                f"{base}/{path}", data=data, headers=headers or {},
                method="POST" if data is not None else "GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    assert response.status == expect, (path, response.status)
                    # headers, not dict(headers): the client normalises
                    # header case, and Message lookups are insensitive.
                    return response.status, response.read(), response.headers
            except urllib.error.HTTPError as exc:
                assert exc.code == expect, (path, exc.code, expect)
                return exc.code, exc.read(), exc.headers

        # 2. Handler results map to responses: object -> JSON, None -> 404,
        #    a raised ValueError -> 400 carrying its own message.
        assert json.loads(fetch("state")[1]) == {"ok": True}
        assert fetch("nope", expect=404)
        assert "bad route argument" in json.loads(fetch("boom", expect=400)[1])["error"]

        # 3. Bytes carry an ETag, and a client holding it gets 304 -- the mesh
        #    routes re-sent whole STLs on every page load without this.
        _, body, headers = fetch("blob")
        assert body == b"\x00\x01\x02", body
        etag = headers["ETag"]
        assert fetch("blob", headers={"If-None-Match": etag}, expect=304)[1] == b""

        # 4. JSON is never cached: a stale /state is a lying panel.
        assert fetch("state")[2]["Cache-Control"] == "no-store"

        # 5. The CSRF gate. A same-origin POST lands; a foreign Origin is
        #    refused BEFORE the handler runs, so nothing is recorded.
        host = f"127.0.0.1:{server.port}"
        assert fetch("action", data=b"{}", headers={"Origin": f"http://{host}"})
        assert len(posted) == 1, posted
        fetch("action", data=b"{}", headers={"Origin": "http://evil.test"}, expect=403)
        assert len(posted) == 1, f"a cross-origin POST reached the handler: {posted}"

        # 6. The body cap is declared-length, so it costs nothing to enforce.
        big = urllib.request.Request(
            f"{base}/action", data=b"{}", method="POST",
            headers={"Content-Length": str(DEFAULT_MAX_BODY + 1)},
        )
        try:
            urllib.request.urlopen(big, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 413, exc.code
        except urllib.error.URLError:
            pass  # the server closed on the short body; the cap still fired
        assert len(posted) == 1, "an oversize POST reached the handler"

        # 7. An unrouted POST is a 404, not a silent success.
        fetch("motor_command", data=b"{}", expect=404)
    finally:
        server.close()

    # 8. `/` is the console's OWN index. Two consoles share this allowlist, so
    #    a server that hardcoded CONSOLE_ASSETS[""] would serve the grasp
    #    editor's page from the operator panel's port.
    for index in ("index.html", "operator.html"):
        server = ConsoleServer(name="t", bind="127.0.0.1", port=0, index=index)
        try:
            root = urllib.request.urlopen(
                f"http://127.0.0.1:{server.port}/", timeout=5
            ).read()
            named = urllib.request.urlopen(
                f"http://127.0.0.1:{server.port}/{index}", timeout=5
            ).read()
            assert root == named, index
        finally:
            server.close()
    print("console_server self-check ok")


if __name__ == "__main__":
    _self_check()
