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
        index="operator.html",
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
from urllib.parse import urlsplit

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
        index: str,
        require_loopback: bool = True,
        max_body: int = DEFAULT_MAX_BODY,
        extra_assets: dict[str, tuple[str, str, str]] | None = None,
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
        # Which page ``/`` means. REQUIRED, with no default: three consoles
        # share this allowlist and this server, and a default is exactly how
        # the arm console ended up serving the grasp editor's page -- the
        # right URL, the wrong application, for months. A console that does not
        # name its page gets a TypeError at construction, not a wrong page at
        # runtime.
        self._index = index
        # A CONSUMER's own page, served through this server without its files
        # living here. Same shape as every other injection in this package: the
        # project supplies the thing that is its own. The allowlist is still the
        # boundary -- these entries are absolute paths the caller vouched for,
        # and nothing outside the merged map can be read.
        self._assets = dict(CONSOLE_ASSETS)
        for route, entry in (extra_assets or {}).items():
            content_type, path, cache = entry
            if not Path(path).is_absolute():
                raise ValueError(
                    f"extra asset {route!r} must give an absolute path, got {path!r} "
                    "-- a relative one would resolve against this package"
                )
            if not Path(path).is_file():
                raise FileNotFoundError(f"extra asset {route!r}: no file at {path}")
            self._assets[route] = (content_type, path, cache)
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

            def _trusted_host(self) -> bool:
                # Origin==Host alone trusts a rebinding website's own hostname.
                # SSH tunnels use localhost or the literal bound loopback IP.
                if not require_loopback:
                    return True
                try:
                    host = urlsplit("http://" + self.headers.get("Host", ""))
                    # A tunnel may expose a different browser-facing port.
                    trusted = (host.hostname in {bind, "localhost"}
                               and not (host.username or host.password or host.path
                                        or host.query or host.fragment)
                               and (host.port is None or 0 < host.port <= 65535))
                except ValueError:
                    trusted = False
                if not trusted:
                    self._json({"error": "untrusted Host refused"}, 403)
                    return False
                return True

            # -- methods ------------------------------------------------------
            def do_GET(self) -> None:
                if not self._trusted_host():
                    return
                route = self._route() or server._index
                if route in server._assets:
                    content_type, path, cache = server._assets[route]
                    body = (
                        Path(path).read_bytes() if Path(path).is_absolute()
                        else console_asset(route)[1]
                    )
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
                if not self._trusted_host():
                    return
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
                try:
                    length = int(self.headers.get("Content-Length", 0) or 0)
                except ValueError:
                    self._json({"error": "invalid body length"}, 400)
                    return
                if length < 0:
                    self._json({"error": "invalid body length"}, 400)
                    return
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
            ConsoleServer(name="test panel", bind=bad, port=0,
                          index="console.html")
        except ValueError as exc:
            assert "test panel" in str(exc), exc
        else:
            raise AssertionError(f"bound to {bad!r}")
    # ...and one that opts out does not. Nothing shipped here opts out any
    # more -- the control panel gained jog and took the refusal (2026-09-07) --
    # but the flag stays for a consumer whose page cannot command motion.
    open_server = ConsoleServer(
        name="open", bind="0.0.0.0", port=0, require_loopback=False,
        index="console.html",
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

    server = ConsoleServer(name="t", bind="127.0.0.1", port=0, get=get, post=post,
                           index="console.html")
    base = f"http://127.0.0.1:{server.port}"
    local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        def fetch(path, *, headers=None, data=None, expect=200):
            request = urllib.request.Request(
                f"{base}/{path}", data=data, headers=headers or {},
                method="POST" if data is not None else "GET",
            )
            try:
                with local.open(request, timeout=5) as response:
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
        fetch("action", data=b"{}", headers={
            "Host": "evil.test", "Origin": "http://evil.test"}, expect=403)
        assert len(posted) == 1, "an untrusted Host reached the action handler"
        fetch("state", headers={"Host": "evil.test"}, expect=403)
        assert fetch("state", headers={"Host": f"localhost:{server.port}"})
        assert fetch("state", headers={"Host": "localhost:17500"})  # Remapped SSH tunnel.
        for invalid in ("localhost:0", "localhost:65536", "evil@localhost:7500",
                        "localhost:7500/path", "localhost.evil.test:7500"):
            fetch("state", headers={"Host": invalid}, expect=403)
        # Invalid lengths must fail before reading, even when the peer closes
        # its write side (read(-1) used to consume the whole stream).
        import socket
        for length in ("-1", "invalid"):
            with socket.create_connection(("127.0.0.1", server.port), timeout=2) as conn:
                conn.sendall((f"POST /action HTTP/1.1\r\nHost: {host}\r\n"
                              f"Content-Length: {length}\r\n\r\n{{}}").encode())
                conn.shutdown(socket.SHUT_WR)
                assert b" 400 " in conn.recv(4096).split(b"\r\n", 1)[0]
        assert len(posted) == 1, "an invalid-length POST reached the handler"

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

    # 8. `/` is the console's OWN index. This is the regression that shipped:
    #    the arm console and the grasp editor shared one page, so driving a
    #    robot rendered the editor's Calibration rack with every control
    #    404ing. Each page must answer only on its own port.
    for index in ("console.html", "operator.html"):
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
    # 9. A consumer's own page, registered rather than vendored -- how the
    #    grasp editor keeps its page beside its node in the project that owns
    #    it, without this package shipping the assembly task's UI.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "mine.html"
        page.write_text("<!doctype html><title>mine</title>")
        server = ConsoleServer(
            name="t", bind="127.0.0.1", port=0, index="mine.html",
            extra_assets={
                "mine.html": ("text/html; charset=utf-8", str(page), "no-store")
            },
        )
        try:
            body = urllib.request.urlopen(
                f"http://127.0.0.1:{server.port}/", timeout=5
            ).read()
            assert b"<title>mine</title>" in body, body
            shared = urllib.request.urlopen(
                f"http://127.0.0.1:{server.port}/static/core.js", timeout=5
            ).read()
            assert b"export" in shared, "core.js did not come through"
        finally:
            server.close()
    # A relative extra asset would resolve against THIS package's directory,
    # which is the one thing the caller cannot have meant.
    for bad in ({"x.html": ("text/html", "relative.html", "no-store")},
                {"x.html": ("text/html", "/nonexistent/page.html", "no-store")}):
        try:
            ConsoleServer(name="t", bind="127.0.0.1", port=0, index="console.html",
                          extra_assets=bad)
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError(f"accepted {bad}")

    # 10. A console that names no page is a TypeError, never a wrong page.
    try:
        ConsoleServer(name="t", bind="127.0.0.1", port=0)
    except TypeError:
        pass
    else:
        raise AssertionError("ConsoleServer built without an index page")
    print("console_server self-check ok")


if __name__ == "__main__":
    _self_check()
