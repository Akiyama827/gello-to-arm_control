"""Operator export of persisted fault traces (CTRL-051): verify and copy, read-only.

    python -m arm_control.multi_arm.trace_export --verify [--root DIR]
    python -m arm_control.multi_arm.trace_export --dest DIR [--root DIR] [--include-unverified]
    python -m arm_control.multi_arm.trace_export --prune [--keep-files N] [--yes --confirm-exported DIR] [--root DIR]

Reads the fault-trace directory and does not modify it, except for an explicit ``--prune --yes --confirm-exported DIR`` (dry run
without ``--yes``; only traces with a byte-identical copy in DIR are deleted), the only way traces are ever deleted. A trace is verified when its SHA-256 sidecar matches its bytes,
it parses as ``arm_control.dual_fault`` JSON and its recorded ``samples_kept`` equals the number of samples it holds.
``--dest`` copies the verified traces (and their sidecars) plus ``EXPORT_MANIFEST.json`` (name, size, sha256, reason,
observed time, samples kept, dropped samples) into a directory that must not exist or must be empty; files are created
exclusively, so nothing is overwritten. This module imports no transport and opens no device, socket or robot address:
a command log is not a safe executable trajectory, and replay is detached analysis only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from arm_control.multi_arm.diagnostics import RETAIN_FILES, TRACE_GLOB, fault_root, prune_traces


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path) -> dict:
    """One trace's verdict. Never raises for a bad file: the problem is the result."""
    info = {"name": path.name, "ok": False, "legacy": False, "problem": None}
    try:
        info["size"] = path.stat().st_size
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.is_file():
            # A trace written before sidecars existed (real fault records): it cannot be hash-verified, but it can be
            # parsed and it is data: report it as legacy, export it, and never lose it to a "failed verification".
            info["sha256"] = _digest(path)
            report = json.loads(path.read_text())
            if isinstance(report, dict) and ("samples_kept" in report or "retention" in report):
                # this code always writes a sidecar with these keys: a missing one means it was removed, not legacy
                info["problem"] = "sidecar missing for a trace written with one (removed or tampered)"
            elif isinstance(report, dict) and str(report.get("schema", "")).startswith("arm_control.dual_fault/"):
                info.update(ok=True, legacy=True, problem="no sha256 sidecar (written before sidecars existed)",
                            reason=report.get("reason"), observed_at_utc=report.get("observed_at_utc"),
                            samples_kept=len(report.get("samples", [])), schema=report.get("schema"))
            else:
                info["problem"] = "no sha256 sidecar and not a dual_fault trace"
            return info
        recorded = sidecar.read_text().split()[0]
        info["sha256"] = _digest(path)
        if info["sha256"] != recorded:
            info["problem"] = "sha256 does not match the sidecar (modified or truncated)"
            return info
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or not isinstance(report.get("samples", []), list):
            info["problem"] = "not a trace object with a samples list"
            return info
        if not str(report.get("schema", "")).startswith("arm_control.dual_fault/"):
            info["problem"] = f"unknown schema {report.get('schema')!r}"
            return info
        held = len(report.get("samples", []))
        if "samples_kept" in report and report["samples_kept"] != held:
            info["problem"] = f"samples_kept {report['samples_kept']} != {held} samples present"
            return info
        info.update(ok=True, reason=report.get("reason"), observed_at_utc=report.get("observed_at_utc"),
                    samples_kept=held, dropped_samples=report.get("dropped_samples"),
                    schema=report.get("schema"))
    except (OSError, ValueError, IndexError, TypeError, AttributeError) as exc:
        info["problem"] = f"{type(exc).__name__}: {exc}"
    return info


def list_traces(root: Path) -> list[dict]:
    infos = [verify(p) for p in sorted(root.glob(TRACE_GLOB)) if p.is_file() and not p.is_symlink()]
    # A crash between writing and renaming leaves *.partial / orphan sidecars: never a trace, but say so.
    for p in sorted(root.glob("*.partial")):
        infos.append({"name": p.name, "ok": False, "legacy": False, "problem": "incomplete write (crash leftover)"})
    return infos


def export(root: Path, dest: Path, *, include_unverified: bool = False) -> dict:
    root, dest = Path(root).resolve(), Path(dest).resolve()
    if dest == root or root in dest.parents:
        raise ValueError("destination must be outside the fault-trace directory")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("destination exists and is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for info in list_traces(root):
        if not info["ok"] and not include_unverified:
            entries.append(dict(info, exported=False))
            continue
        for name in (info["name"], info["name"] + ".sha256"):
            src = root / name
            if src.is_file():
                with src.open("rb") as read, (dest / name).open("xb") as write:
                    shutil.copyfileobj(read, write)
        entries.append(dict(info, exported=True))
    manifest = {"schema": "arm_control.trace_export/v1", "source": str(root), "traces": entries}
    with (dest / "EXPORT_MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=1, allow_nan=False)
        stream.write("\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None, help="fault-trace directory (default: the console's)")
    ap.add_argument("--verify", action="store_true", help="list traces and their verification")
    ap.add_argument("--dest", type=Path, help="copy verified traces and a manifest here")
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--prune", action="store_true", help="remove the oldest traces beyond the capacity policy")
    ap.add_argument("--keep-files", type=int, default=RETAIN_FILES)
    ap.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    ap.add_argument("--confirm-exported", type=Path, help="with --prune --yes: only traces with a byte-identical copy in this directory are deleted")
    args = ap.parse_args(argv)
    root = fault_root() if args.root is None else args.root
    if args.prune:
        if args.keep_files < 0:
            print("--keep-files must be >= 0")
            return 2
        only = None
        if args.yes and args.confirm_exported is None:
            print("refusing to delete: give --confirm-exported DIR (a directory holding byte-identical exported copies)")
            return 2
        if args.confirm_exported is not None:  # the dry run shows what the real run would delete
            only = {p.name for p in root.glob(TRACE_GLOB) if p.is_file() and not p.is_symlink()
                    and (args.confirm_exported / p.name).is_file() and _digest(args.confirm_exported / p.name) == _digest(p)}
        names = prune_traces(root, keep_files=args.keep_files, dry_run=not args.yes, only=only)
        print(("removed " if args.yes else "would remove ") + f"{len(names)} trace(s): {names}")
        return 0
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 2
    if args.dest is not None:
        if args.dest.exists() and not args.dest.is_dir():
            print(f"--dest is not a directory: {args.dest}")
            return 2
        manifest = export(root, args.dest, include_unverified=args.include_unverified)
        bad = [t for t in manifest["traces"] if not t["ok"]]
        print(f"exported {sum(t['exported'] for t in manifest['traces'])} trace(s) to {args.dest}; "
              f"{len(bad)} failed verification")
        return 1 if bad and not args.include_unverified else 0
    infos = list_traces(root)
    for info in infos:
        mark = "LEG  " if info.get("legacy") else "OK   " if info["ok"] else "BAD  "
        print(mark + info["name"] + ("" if info["ok"] and not info.get("legacy") else f"  {info['problem']}"))
    print(f"{len(infos)} trace(s), {sum(not i['ok'] for i in infos)} failed verification")
    return 1 if any(not i["ok"] for i in infos) else 0


if __name__ == "__main__":
    sys.exit(main())
