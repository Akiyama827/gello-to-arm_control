"""One allowlist for the offline console files, and one loader for them.

Three consoles serve static files off ``nodes/console/`` over a loopback HTTP
server: the arm console, the grasp editor, and the operator gate. The allowlist
is the security boundary -- a route not named here cannot be read, so no path
traversal is possible regardless of what a request asks for -- and it belongs
in the library rather than in one of the node files that use it, or the second
console would have to import from the first.

There is deliberately no ``""`` entry any more. A bare ``/`` used to map to
``index.html``, which meant "whichever page happened to be first in this dict"
rather than "this console's page"; ConsoleServer resolves ``/`` through its own
``index=`` instead, so a console cannot serve another console's page.
"""

from __future__ import annotations

from pathlib import Path

CONSOLE_DIR = Path(__file__).resolve().parents[1] / "nodes" / "console"

_IMMUTABLE = "public, max-age=31536000, immutable"  # vendored, content-addressed
_NO_STORE = "no-store"  # our own pages: always revalidate, they change per build

CONSOLE_ASSETS: dict[str, tuple[str, str, str]] = {
    # THREE pages, one allowlist. They were two until 2026-09-07, because the
    # arm console and the grasp editor shared `index.html` -- so driving a robot
    # rendered the editor's Calibration rack, whose every control answered 404,
    # under a tab titled "Workcell calibration console". Each console now names
    # its own page through ConsoleServer(index=...).
    #
    # `console.js` and `editor.js` are page modules; `core.js` is what they
    # share (scene, meshes, gizmo, sliders, deadman) and belongs to neither.
    "console.html": ("text/html; charset=utf-8", "console.html", _NO_STORE),
    "static/console.js": ("text/javascript", "console.js", _NO_STORE),
    "static/console.css": ("text/css; charset=utf-8", "console.css", _NO_STORE),
    "editor.html": ("text/html; charset=utf-8", "editor.html", _NO_STORE),
    "static/editor.js": ("text/javascript", "editor.js", _NO_STORE),
    "static/core.js": ("text/javascript", "core.js", _NO_STORE),
    "static/style.css": ("text/css; charset=utf-8", "style.css", _NO_STORE),
    # The operator GATE (the caller's ``nodes/operator_console.py``): a third,
    # much smaller page -- ARM/DISARM and nothing that can plan or jog.
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


def _check_pages() -> None:
    """Every element a page's JS reaches for must exist in that page's HTML.

    This is the failure that has no symptom worth the name: `$("jog-pad")`
    returning null throws deep inside a build function, the page renders half
    of itself, and the console log is the only place it shows. Splitting one
    page into two made it a live risk -- a moved element is now a moved element
    in ONE of two files.

    Also asserts the pages do not reach across: console.js must not touch an
    id that only editor.html has, and vice versa.
    """
    import re

    def ids_in(name: str) -> set[str]:
        return set(re.findall(r'id="([^"]+)"', (CONSOLE_DIR / name).read_text()))

    def ids_used(name: str) -> set[str]:
        src = (CONSOLE_DIR / name).read_text()
        # `$("literal")` only; a template literal ($(`${x}-ik`)) is dynamic and
        # cannot be checked here -- those ids are asserted by their page below.
        return set(re.findall(r'\$\("([^"]+)"\)', src))

    shared = ids_used("core.js")
    for page, script in (("console.html", "console.js"), ("editor.html", "editor.js")):
        have = ids_in(page)
        want = ids_used(script) | shared
        missing = sorted(want - have)
        assert not missing, f"{script} reaches for ids absent from {page}: {missing}"

    # The module graph: a name imported from core.js must be exported by it,
    # and an import nothing uses is dead weight. The browser reports the first
    # as a hard SyntaxError at load -- a blank page with one console line --
    # which is exactly the failure a headless check should catch instead.
    core_src = (CONSOLE_DIR / "core.js").read_text()
    exported = set(re.findall(r"export (?:const|let|function|async function) ([\w$]+)", core_src))
    exported |= {
        name.strip()
        for group in re.findall(r"export \{([^}]+)\}", core_src)
        for name in group.split(",")
    }
    for script in ("console.js", "editor.js"):
        src = (CONSOLE_DIR / script).read_text()
        match = re.search(r'import \{([^}]+)\} from "\./core\.js"', src, re.S)
        assert match, f"{script} does not import from core.js"
        imported = {n.strip() for n in match.group(1).split(",") if n.strip()}
        absent = sorted(imported - exported)
        assert not absent, f"{script} imports names core.js does not export: {absent}"
        body = src[match.end():]
        unused = sorted(
            n for n in imported
            if not re.search(rf"(?<![\w$]){re.escape(n)}(?![\w$])", body)
        )
        assert not unused, f"{script} imports but never uses: {unused}"

    # The dynamic ones, stated explicitly because the regex cannot see them.
    for dynamic in ("storage-ik", "dock-ik"):
        assert dynamic in ids_in("editor.html"), dynamic

    # No page may serve as another's index, and every asset must exist.
    for route, (_ctype, relative, _cache) in CONSOLE_ASSETS.items():
        assert (CONSOLE_DIR / relative).is_file(), f"{route} -> missing {relative}"
    assert "" not in CONSOLE_ASSETS, (
        'a "" route makes `/` mean whichever page is first in this dict; '
        "ConsoleServer(index=...) is the only thing that should resolve /"
    )
    print(f"console_assets: {len(CONSOLE_ASSETS)} assets, pages cross-check ok")


if __name__ == "__main__":
    _check_pages()
