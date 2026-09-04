"""One allowlist for the offline console files, and one loader for them.

Both consoles serve static files off ``nodes/console/`` over a loopback HTTP
server: the calibration/grasp editor and the operator panel. The allowlist is
the security boundary -- a route not named here cannot be read, so no path
traversal is possible regardless of what a request asks for -- and it belongs
in the library rather than in one of the two node files that use it, or the
second console would have to import from the first.
"""

from __future__ import annotations

from pathlib import Path

CONSOLE_DIR = Path(__file__).resolve().parents[1] / "nodes" / "console"

_IMMUTABLE = "public, max-age=31536000, immutable"  # vendored, content-addressed
_NO_STORE = "no-store"  # our own pages: always revalidate, they change per build

CONSOLE_ASSETS: dict[str, tuple[str, str, str]] = {
    "": ("text/html; charset=utf-8", "index.html", _NO_STORE),
    "index.html": ("text/html; charset=utf-8", "index.html", _NO_STORE),
    "static/style.css": ("text/css; charset=utf-8", "style.css", _NO_STORE),
    "static/app.js": ("text/javascript", "app.js", _NO_STORE),
    # The operator panel (the caller's ``nodes/operator_console.py``). Same
    # directory, same loader, same ETag path -- a different page.
    "operator.html": ("text/html; charset=utf-8", "operator.html", _NO_STORE),
    "static/operator.css": ("text/css; charset=utf-8", "operator.css", _NO_STORE),
    "static/operator.js": ("text/javascript", "operator.js", _NO_STORE),
    "static/vendor/three.module.js": (
        "text/javascript", "vendor/three.module.js", _IMMUTABLE,
    ),
    "static/vendor/OrbitControls.js": (
        "text/javascript", "vendor/OrbitControls.js", _IMMUTABLE,
    ),
    "static/vendor/TransformControls.js": (
        "text/javascript", "vendor/TransformControls.js", _IMMUTABLE,
    ),
    "static/vendor/STLLoader.js": (
        "text/javascript", "vendor/STLLoader.js", _IMMUTABLE,
    ),
}


def console_asset(route: str) -> tuple[str, bytes, str]:
    """Return one allowlisted offline console asset: (content type, body, cache)."""
    try:
        content_type, relative, cache = CONSOLE_ASSETS[route.strip("/")]
    except KeyError as exc:
        raise KeyError(f"unknown console asset: {route}") from exc
    return content_type, (CONSOLE_DIR / relative).read_bytes(), cache


__all__ = ["CONSOLE_ASSETS", "CONSOLE_DIR", "console_asset"]
